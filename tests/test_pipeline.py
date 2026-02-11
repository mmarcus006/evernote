"""Comprehensive tests for the PDF-to-Markdown + Vector DB pipeline.

All tests use real PDF files from output/Properties/.
Expensive resources (VLM converter, sentiment model) are session-scoped fixtures.
"""

from __future__ import annotations

import json
from pathlib import Path


from pipeline import (
    DocRecord,
    analyze_sentiment,
    chunk_documents,
    convert_pdfs,
    convert_single_pdf,
    create_chroma_collection,
    discover_pdfs,
    ensure_output_dirs,
    extract_entities,
    extract_keywords_rake,
    extract_tfidf_topics,
    extractive_summary,
    insert_chunks,
    is_colab,
    run_nlp_analysis,
    save_manifest,
    search_collection,
)


# =========================================================================
# 1. Path helpers
# =========================================================================


class TestIsColab:
    def test_returns_false_locally(self):
        assert is_colab() is False


class TestDiscoverPdfs:
    def test_finds_real_pdfs(self, all_pdf_paths: list[Path]):
        folder = all_pdf_paths[0].parent
        found = discover_pdfs(folder)
        assert len(found) >= 3
        assert all(p.suffix == ".pdf" for p in found)

    def test_returns_sorted(self, all_pdf_paths: list[Path]):
        folder = all_pdf_paths[0].parent
        found = discover_pdfs(folder)
        assert found == sorted(found)

    def test_empty_for_nonexistent_folder(self):
        found = discover_pdfs(Path("/nonexistent/folder/abc123"))
        assert found == []

    def test_empty_for_folder_without_pdfs(self, tmp_path: Path):
        (tmp_path / "not_a_pdf.txt").write_text("hello")
        found = discover_pdfs(tmp_path)
        assert found == []


class TestEnsureOutputDirs:
    def test_creates_subdirectories(self, tmp_path: Path):
        md_dir, db_dir = ensure_output_dirs(tmp_path / "out")
        assert md_dir.exists()
        assert db_dir.exists()
        assert md_dir.name == "markdown"
        assert db_dir.name == "vector_db"

    def test_idempotent(self, tmp_path: Path):
        out = tmp_path / "out"
        ensure_output_dirs(out)
        ensure_output_dirs(out)  # should not raise
        assert (out / "markdown").exists()


# =========================================================================
# 2. DocRecord dataclass
# =========================================================================


class TestDocRecord:
    def test_defaults(self):
        r = DocRecord(filename="test.pdf", filepath="/tmp/test.pdf")
        assert r.status == "pending"
        assert r.markdown == ""
        assert r.error is None
        assert r.num_pages == 0

    def test_fields_settable(self):
        r = DocRecord(
            filename="a.pdf",
            filepath="/a.pdf",
            markdown="# Title",
            num_pages=5,
            status="success",
        )
        assert r.num_pages == 5
        assert r.markdown == "# Title"


# =========================================================================
# 3. VLM Converter
# =========================================================================


class TestCreateVlmConverter:
    def test_returns_converter_and_name(self, vlm_converter):
        # vlm_converter fixture already called create_vlm_converter()
        assert vlm_converter is not None


class TestConvertSinglePdf:
    def test_small_pdf_success(self, converted_small):
        record, doc = converted_small
        assert record.status == "success"
        assert record.error is None
        assert len(record.markdown) > 0
        assert record.conversion_time_s > 0
        assert record.filename == "2014_07_02_15_28_10.pdf"

    def test_small_pdf_has_markdown_content(self, converted_small):
        record, _ = converted_small
        # Should have some actual text, not just whitespace
        stripped = record.markdown.strip()
        assert len(stripped) > 10

    def test_medium_pdf_success(self, converted_medium):
        record, doc = converted_medium
        assert record.status == "success"
        assert len(record.markdown) > 0

    def test_medium_pdf_has_title(self, converted_medium):
        record, _ = converted_medium
        assert record.title  # non-empty

    def test_returns_docling_document(self, converted_small):
        _, doc = converted_small
        assert doc is not None
        assert hasattr(doc, "export_to_markdown")

    def test_handles_nonexistent_file_gracefully(self, vlm_converter):
        record, doc = convert_single_pdf(vlm_converter, Path("/nonexistent/fake.pdf"))
        assert record.status == "error"
        assert record.error is not None
        assert doc is None


class TestConvertPdfs:
    def test_batch_converts_and_saves_markdown(
        self, vlm_converter, pdf_small, tmp_path
    ):
        md_dir = tmp_path / "md"
        md_dir.mkdir()
        records, docs = convert_pdfs(vlm_converter, [pdf_small], md_dir)

        assert len(records) == 1
        assert records[0].status == "success"

        # Markdown file should exist on disk
        md_files = list(md_dir.glob("*.md"))
        assert len(md_files) == 1
        assert md_files[0].read_text(encoding="utf-8") == records[0].markdown

    def test_batch_stores_docling_docs(self, vlm_converter, pdf_small, tmp_path):
        md_dir = tmp_path / "md"
        md_dir.mkdir()
        records, docs = convert_pdfs(vlm_converter, [pdf_small], md_dir)
        assert pdf_small.name in docs

    def test_batch_handles_mix_of_valid_and_invalid(
        self, vlm_converter, pdf_small, tmp_path
    ):
        md_dir = tmp_path / "md"
        md_dir.mkdir()
        files = [pdf_small, Path("/fake/does_not_exist.pdf")]
        records, docs = convert_pdfs(vlm_converter, files, md_dir)

        assert len(records) == 2
        statuses = {r.status for r in records}
        assert "success" in statuses
        assert "error" in statuses


# =========================================================================
# 4. NLP Analysis
# =========================================================================


class TestExtractKeywordsRake:
    def test_returns_list_of_strings(self, converted_small):
        record, _ = converted_small
        keywords = extract_keywords_rake(record.markdown)
        assert isinstance(keywords, list)
        assert all(isinstance(k, str) for k in keywords)

    def test_respects_top_n(self, converted_small):
        record, _ = converted_small
        keywords = extract_keywords_rake(record.markdown, top_n=3)
        assert len(keywords) <= 3

    def test_empty_text_returns_empty(self):
        keywords = extract_keywords_rake("")
        assert keywords == []


class TestExtractEntities:
    def test_returns_dict_of_entity_lists(self, converted_small):
        record, _ = converted_small
        entities = extract_entities(record.markdown)
        assert isinstance(entities, dict)
        for label, ents in entities.items():
            assert isinstance(label, str)
            assert isinstance(ents, list)
            assert all(isinstance(e, str) for e in ents)

    def test_respects_max_chars(self):
        # Should not crash on truncation
        entities = extract_entities(
            "Apple Inc. was founded by Steve Jobs." * 100, max_chars=50
        )
        assert isinstance(entities, dict)

    def test_empty_text(self):
        entities = extract_entities("")
        assert entities == {}


class TestExtractiveSummary:
    def test_returns_string(self, converted_small):
        record, _ = converted_small
        summary = extractive_summary(record.markdown)
        assert isinstance(summary, str)

    def test_respects_num_sentences(self, converted_medium):
        record, _ = converted_medium
        summary_1 = extractive_summary(record.markdown, num_sentences=1)
        summary_5 = extractive_summary(record.markdown, num_sentences=5)
        # With 1 sentence, result should be shorter (or equal if only 1 sentence exists)
        assert len(summary_1) <= len(summary_5) or len(summary_5) == 0

    def test_empty_text(self):
        summary = extractive_summary("")
        assert summary == ""


class TestAnalyzeSentiment:
    def test_returns_label_and_score(self, sentiment_analyzer, converted_small):
        record, _ = converted_small
        result = analyze_sentiment(record.markdown, analyzer=sentiment_analyzer)
        assert "label" in result
        assert "score" in result
        assert result["label"] in ("POSITIVE", "NEGATIVE")
        assert 0.0 <= result["score"] <= 1.0

    def test_short_text(self, sentiment_analyzer):
        result = analyze_sentiment("This is great!", analyzer=sentiment_analyzer)
        assert result["label"] == "POSITIVE"


class TestExtractTfidfTopics:
    def test_returns_topics_per_document(self, converted_small, converted_medium):
        r1, _ = converted_small
        r2, _ = converted_medium
        texts = [r1.markdown, r2.markdown]
        topics = extract_tfidf_topics(texts, top_n=5)

        assert len(topics) == 2
        assert all(isinstance(t, list) for t in topics)
        assert all(len(t) <= 5 for t in topics)

    def test_single_document(self, converted_small):
        record, _ = converted_small
        topics = extract_tfidf_topics([record.markdown])
        assert len(topics) == 1
        assert len(topics[0]) > 0

    def test_empty_input(self):
        topics = extract_tfidf_topics([])
        assert topics == []


class TestRunNlpAnalysis:
    def test_full_analysis_on_successful_records(
        self, converted_all, sentiment_analyzer
    ):
        records, _ = converted_all
        successful = [r for r in records if r.status == "success"]
        results = run_nlp_analysis(records, sentiment_analyzer=sentiment_analyzer)

        assert len(results) == len(successful)
        for filename, analysis in results.items():
            assert "keywords_rake" in analysis
            assert "named_entities" in analysis
            assert "summary" in analysis
            assert "sentiment" in analysis
            assert "tfidf_topics" in analysis
            assert "word_count" in analysis
            assert "char_count" in analysis
            assert analysis["word_count"] > 0
            assert analysis["char_count"] > 0

    def test_skips_failed_records(self, sentiment_analyzer):
        records = [
            DocRecord(filename="bad.pdf", filepath="/bad.pdf", status="error"),
            DocRecord(
                filename="good.pdf",
                filepath="/good.pdf",
                status="success",
                markdown="The quick brown fox jumps over the lazy dog. " * 20,
            ),
        ]
        results = run_nlp_analysis(records, sentiment_analyzer=sentiment_analyzer)
        assert "bad.pdf" not in results
        assert "good.pdf" in results


# =========================================================================
# 5. Chunking
# =========================================================================


class TestCreateChunker:
    def test_creates_hybrid_chunker(self, chunker):
        assert chunker is not None
        assert hasattr(chunker, "chunk")
        assert hasattr(chunker, "contextualize")


class TestChunkDocuments:
    def test_chunks_real_document(self, chunker, converted_small):
        record, doc = converted_small
        docs_dict = {record.filename: doc}
        nlp_data = {
            record.filename: {
                "keywords_rake": ["test", "keyword"],
                "tfidf_topics": ["topic1"],
                "sentiment": {"label": "POSITIVE", "score": 0.95},
                "summary": "Test summary.",
                "word_count": 100,
            }
        }
        chunks = chunk_documents(chunker, [record], docs_dict, nlp_data)

        assert len(chunks) > 0
        for c in chunks:
            assert c["source_file"] == record.filename
            assert c["text"]  # non-empty contextualized text
            assert c["id"].startswith(record.filename)
            assert "chunk_index" in c
            assert "keywords" in c
            assert "sentiment_label" in c

    def test_chunk_ids_are_unique(self, chunker, converted_small):
        record, doc = converted_small
        chunks = chunk_documents(chunker, [record], {record.filename: doc}, {})
        ids = [c["id"] for c in chunks]
        assert len(ids) == len(set(ids))

    def test_skips_missing_docs(self, chunker):
        record = DocRecord(
            filename="missing.pdf", filepath="/missing.pdf", status="success"
        )
        chunks = chunk_documents(chunker, [record], {}, {})
        assert chunks == []

    def test_skips_failed_records(self, chunker, converted_small):
        record, doc = converted_small
        failed = DocRecord(filename="fail.pdf", filepath="/fail.pdf", status="error")
        chunks = chunk_documents(chunker, [failed, record], {record.filename: doc}, {})
        # Should only have chunks from the successful record
        assert all(c["source_file"] == record.filename for c in chunks)

    def test_multiple_docs_produce_chunks(self, chunker, converted_all):
        records, docs = converted_all
        successful = [r for r in records if r.status == "success"]
        chunks = chunk_documents(chunker, successful, docs, {})
        source_files = {c["source_file"] for c in chunks}
        assert len(source_files) >= 2  # at least 2 of 3 should produce chunks


# =========================================================================
# 6. ChromaDB Vector Database
# =========================================================================


class TestCreateChromaCollection:
    def test_creates_persistent_collection(self, tmp_path):
        db_dir = tmp_path / "db"
        db_dir.mkdir()
        client, collection = create_chroma_collection(db_dir)
        assert collection is not None
        assert collection.count() == 0
        assert collection.name == "docling_documents"

    def test_custom_collection_name(self, tmp_path):
        db_dir = tmp_path / "db"
        db_dir.mkdir()
        _, coll = create_chroma_collection(db_dir, collection_name="test_coll")
        assert coll.name == "test_coll"

    def test_idempotent_get_or_create(self, tmp_path):
        db_dir = tmp_path / "db"
        db_dir.mkdir()
        _, c1 = create_chroma_collection(db_dir, collection_name="reuse")
        _, c2 = create_chroma_collection(db_dir, collection_name="reuse")
        assert c1.name == c2.name


class TestInsertChunks:
    def _make_chunks(self, n: int = 3) -> list[dict]:
        return [
            {
                "id": f"file.pdf::chunk_{i:04d}",
                "text": f"This is test chunk number {i} with enough content to embed.",
                "raw_text": f"Raw text {i}",
                "source_file": "file.pdf",
                "source_path": "/tmp/file.pdf",
                "doc_title": "Test Doc",
                "num_pages": 2,
                "num_tables": 0,
                "num_figures": 0,
                "chunk_index": i,
                "headings": "Section > Subsection",
                "keywords": "test, keyword",
                "tfidf_topics": "topic1, topic2",
                "sentiment_label": "POSITIVE",
                "sentiment_score": 0.9,
                "doc_summary": "A test summary.",
                "word_count": 50,
            }
            for i in range(n)
        ]

    def test_inserts_and_returns_count(self, tmp_path):
        db_dir = tmp_path / "db"
        db_dir.mkdir()
        _, collection = create_chroma_collection(db_dir)
        chunks = self._make_chunks(5)
        count = insert_chunks(collection, chunks)
        assert count == 5

    def test_upsert_is_idempotent(self, tmp_path):
        db_dir = tmp_path / "db"
        db_dir.mkdir()
        _, collection = create_chroma_collection(db_dir)
        chunks = self._make_chunks(3)
        insert_chunks(collection, chunks)
        insert_chunks(collection, chunks)  # same IDs again
        assert collection.count() == 3  # not 6

    def test_respects_batch_size(self, tmp_path):
        db_dir = tmp_path / "db"
        db_dir.mkdir()
        _, collection = create_chroma_collection(db_dir)
        chunks = self._make_chunks(7)
        count = insert_chunks(collection, chunks, batch_size=2)
        assert count == 7

    def test_empty_chunks(self, tmp_path):
        db_dir = tmp_path / "db"
        db_dir.mkdir()
        _, collection = create_chroma_collection(db_dir)
        count = insert_chunks(collection, [])
        assert count == 0


class TestSearchCollection:
    def test_search_returns_results(self, tmp_path):
        db_dir = tmp_path / "db"
        db_dir.mkdir()
        _, collection = create_chroma_collection(db_dir)
        chunks = [
            {
                "id": "doc.pdf::chunk_0000",
                "text": "Machine learning is a subset of artificial intelligence.",
                "raw_text": "Machine learning is a subset of artificial intelligence.",
                "source_file": "doc.pdf",
                "source_path": "/doc.pdf",
                "doc_title": "ML Paper",
                "num_pages": 1,
                "num_tables": 0,
                "num_figures": 0,
                "chunk_index": 0,
                "headings": "Introduction",
                "keywords": "machine learning, AI",
                "tfidf_topics": "ml, ai",
                "sentiment_label": "POSITIVE",
                "sentiment_score": 0.8,
                "doc_summary": "About ML.",
                "word_count": 9,
            }
        ]
        insert_chunks(collection, chunks)
        results = search_collection(collection, "artificial intelligence", n_results=1)
        assert len(results["ids"][0]) == 1
        assert results["ids"][0][0] == "doc.pdf::chunk_0000"

    def test_search_with_where_filter(self, tmp_path):
        db_dir = tmp_path / "db"
        db_dir.mkdir()
        _, collection = create_chroma_collection(db_dir)
        chunks = [
            {
                "id": f"file{i}.pdf::chunk_0000",
                "text": f"Content about topic {i} with enough words for embedding.",
                "raw_text": f"Content {i}",
                "source_file": f"file{i}.pdf",
                "source_path": f"/file{i}.pdf",
                "doc_title": f"Doc {i}",
                "num_pages": 1,
                "num_tables": 0,
                "num_figures": 0,
                "chunk_index": 0,
                "headings": "",
                "keywords": "",
                "tfidf_topics": "",
                "sentiment_label": "POSITIVE",
                "sentiment_score": 0.5,
                "doc_summary": "",
                "word_count": 10,
            }
            for i in range(3)
        ]
        insert_chunks(collection, chunks)
        results = search_collection(
            collection, "topic", n_results=5, where={"source_file": "file1.pdf"}
        )
        assert len(results["ids"][0]) == 1
        assert results["metadatas"][0][0]["source_file"] == "file1.pdf"

    def test_search_empty_collection(self, tmp_path):
        db_dir = tmp_path / "db"
        db_dir.mkdir()
        _, collection = create_chroma_collection(db_dir)
        results = search_collection(collection, "anything", n_results=5)
        assert results["ids"][0] == []


# =========================================================================
# 7. Manifest I/O
# =========================================================================


class TestSaveManifest:
    def test_writes_valid_json(self, tmp_path):
        records = [
            DocRecord(
                filename="a.pdf",
                filepath="/a.pdf",
                status="success",
                title="Doc A",
                num_pages=3,
            ),
            DocRecord(
                filename="b.pdf",
                filepath="/b.pdf",
                status="error",
                error="bad file",
            ),
        ]
        nlp_results = {
            "a.pdf": {
                "keywords_rake": ["kw1"],
                "summary": "Summary A",
                "word_count": 100,
            }
        }
        path = save_manifest(tmp_path, records, nlp_results)

        assert path.exists()
        data = json.loads(path.read_text(encoding="utf-8"))
        assert len(data) == 2

    def test_manifest_includes_nlp_for_successful(self, tmp_path):
        records = [
            DocRecord(filename="a.pdf", filepath="/a.pdf", status="success"),
        ]
        nlp_results = {"a.pdf": {"keywords_rake": ["k1", "k2"]}}
        path = save_manifest(tmp_path, records, nlp_results)
        data = json.loads(path.read_text())
        assert "nlp_analysis" in data[0]
        assert data[0]["nlp_analysis"]["keywords_rake"] == ["k1", "k2"]

    def test_manifest_excludes_nlp_for_failed(self, tmp_path):
        records = [
            DocRecord(filename="b.pdf", filepath="/b.pdf", status="error"),
        ]
        path = save_manifest(tmp_path, records, {})
        data = json.loads(path.read_text())
        assert "nlp_analysis" not in data[0]

    def test_manifest_path_location(self, tmp_path):
        path = save_manifest(tmp_path, [], {})
        assert path == tmp_path / "document_manifest.json"


# =========================================================================
# 8. Integration: end-to-end on real PDFs
# =========================================================================


class TestEndToEnd:
    """Full pipeline integration test using the small real PDF."""

    def test_convert_analyze_chunk_store_search(
        self,
        vlm_converter,
        pdf_small,
        sentiment_analyzer,
        chunker,
        tmp_path,
    ):
        # 1. Convert
        md_dir, db_dir = ensure_output_dirs(tmp_path)
        records, docling_docs = convert_pdfs(vlm_converter, [pdf_small], md_dir)
        assert len(records) == 1
        assert records[0].status == "success"

        # 2. NLP analysis
        nlp_results = run_nlp_analysis(records, sentiment_analyzer=sentiment_analyzer)
        assert pdf_small.name in nlp_results

        # 3. Chunk
        chunks = chunk_documents(chunker, records, docling_docs, nlp_results)
        assert len(chunks) > 0

        # 4. Store in ChromaDB
        _, collection = create_chroma_collection(db_dir)
        count = insert_chunks(collection, chunks)
        assert count == len(chunks)

        # 5. Search
        results = search_collection(collection, "document content", n_results=3)
        assert len(results["ids"][0]) > 0

        # 6. Manifest
        manifest_path = save_manifest(tmp_path, records, nlp_results)
        assert manifest_path.exists()
        manifest_data = json.loads(manifest_path.read_text())
        assert manifest_data[0]["status"] == "success"
        assert "nlp_analysis" in manifest_data[0]
