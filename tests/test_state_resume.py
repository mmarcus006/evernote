from __future__ import annotations

import hashlib
import json
import sys
import types

from pipeline.chunking import chunk_documents
from pipeline.cli import main, parse_args
from pipeline.models import DocRecord
from pipeline.utils import (
    file_fingerprint,
    is_unchanged_file,
    load_pipeline_state,
    save_manifest,
    save_pipeline_state,
)
from pipeline.vectorstores import insert_chunks


def test_pipeline_state_roundtrip(tmp_path):
    out_dir = tmp_path / "out"
    state = {"files": {"/tmp/a.pdf": {"status": "success", "size": 10, "mtime_ns": 20}}}
    path = save_pipeline_state(out_dir, state)
    assert path.exists()

    loaded = load_pipeline_state(out_dir)
    assert loaded["files"]["/tmp/a.pdf"]["status"] == "success"


def test_load_pipeline_state_handles_invalid_json(tmp_path):
    out_dir = tmp_path / "out"
    out_dir.mkdir(parents=True)
    (out_dir / "pipeline_state.json").write_text("{invalid json", encoding="utf-8")
    assert load_pipeline_state(out_dir) == {"files": {}}


def test_load_pipeline_state_repairs_invalid_files_key(tmp_path):
    out_dir = tmp_path / "out"
    save_pipeline_state(out_dir, {"files": []})
    loaded = load_pipeline_state(out_dir)
    assert loaded["files"] == {}


def test_is_unchanged_file_detects_changes(tmp_path):
    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(b"alpha")

    same = {
        "status": "success",
        "size": pdf.stat().st_size,
        "mtime_ns": pdf.stat().st_mtime_ns,
    }
    assert is_unchanged_file(pdf, same) is True

    pdf.write_bytes(b"beta-updated")
    assert is_unchanged_file(pdf, same) is False


def test_is_unchanged_file_requires_previous_success(tmp_path):
    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(b"alpha")
    fp = file_fingerprint(pdf)

    previous_error = {"status": "error", **fp}
    assert is_unchanged_file(pdf, previous_error) is False

    assert is_unchanged_file(pdf, None) is False


def test_manifest_merge_keeps_previous_entries(tmp_path):
    first_records = [
        DocRecord(filename="a.pdf", filepath="/a.pdf", status="success"),
    ]
    save_manifest(tmp_path, first_records, {"a.pdf": {"summary": "one"}})

    second_records = [
        DocRecord(filename="b.pdf", filepath="/b.pdf", status="success"),
    ]
    path = save_manifest(tmp_path, second_records, {"b.pdf": {"summary": "two"}})

    data = json.loads(path.read_text(encoding="utf-8"))
    by_file = {entry["filename"]: entry for entry in data}
    assert "a.pdf" in by_file
    assert "b.pdf" in by_file
    assert by_file["a.pdf"]["nlp_analysis"]["summary"] == "one"
    assert by_file["b.pdf"]["nlp_analysis"]["summary"] == "two"


def test_parse_args_supports_resume_flags():
    args = parse_args(
        [
            "--local-dir",
            "./output",
            "--force-reprocess",
            "--rebuild-index",
        ]
    )
    assert args.force_reprocess is True
    assert args.rebuild_index is True


def test_main_skips_unchanged_pdf_when_outputs_exist(tmp_path, monkeypatch):
    local_dir = tmp_path / "pdfs"
    local_dir.mkdir()
    pdf = local_dir / "doc.pdf"
    pdf.write_bytes(b"%PDF-sample")

    output_dir = tmp_path / "out"
    md_dir = output_dir / "markdown"
    md_dir.mkdir(parents=True)
    (md_dir / "doc.md").write_text("existing markdown", encoding="utf-8")

    previous = {"status": "success", **file_fingerprint(pdf)}
    save_pipeline_state(output_dir, {"files": {str(pdf.resolve()): previous}})

    calls = {"create_converter": 0, "convert": 0}

    tqdm_stub = types.SimpleNamespace(tqdm=lambda iterable, **kwargs: iterable)
    monkeypatch.setitem(sys.modules, "tqdm", tqdm_stub)

    def _create_converter(**kwargs):
        calls["create_converter"] += 1
        return object(), "stub"

    def _convert_single_pdf(converter, pdf_path):
        calls["convert"] += 1
        record = DocRecord(
            filename=pdf_path.name,
            filepath=str(pdf_path),
            status="error",
        )
        return record, None

    monkeypatch.setattr("pipeline.conversion.create_ocr_converter", _create_converter)
    monkeypatch.setattr("pipeline.conversion.convert_single_pdf", _convert_single_pdf)
    monkeypatch.setattr("pipeline.chunking.create_chunker", lambda: object())
    monkeypatch.setattr(
        "pipeline.vectorstores.build_qdrant_collection",
        lambda *args, **kwargs: 0,
    )

    main(
        [
            "--local-dir",
            str(local_dir),
            "--output-dir",
            str(output_dir),
            "--skip-nlp",
        ]
    )

    assert calls["create_converter"] == 0
    assert calls["convert"] == 0

    loaded = load_pipeline_state(output_dir)
    assert str(pdf.resolve()) in loaded["files"]
    assert loaded["files"][str(pdf.resolve())]["status"] == "success"


def test_main_force_reprocess_overrides_resume_skip(tmp_path, monkeypatch):
    local_dir = tmp_path / "pdfs"
    local_dir.mkdir()
    pdf = local_dir / "doc.pdf"
    pdf.write_bytes(b"%PDF-sample")

    output_dir = tmp_path / "out"
    md_dir = output_dir / "markdown"
    md_dir.mkdir(parents=True)
    (md_dir / "doc.md").write_text("stale markdown", encoding="utf-8")

    previous = {"status": "success", **file_fingerprint(pdf)}
    save_pipeline_state(output_dir, {"files": {str(pdf.resolve()): previous}})

    calls = {"create_converter": 0, "convert": 0, "vector_records": -1}

    tqdm_stub = types.SimpleNamespace(tqdm=lambda iterable, **kwargs: iterable)
    monkeypatch.setitem(sys.modules, "tqdm", tqdm_stub)

    class _Doc:
        pass

    def _create_converter(**kwargs):
        calls["create_converter"] += 1
        return object(), "stub"

    def _convert_single_pdf(converter, pdf_path):
        calls["convert"] += 1
        return (
            DocRecord(
                filename=pdf_path.name,
                filepath=str(pdf_path),
                status="success",
                markdown="# refreshed",
            ),
            _Doc(),
        )

    def _build_qdrant_collection(
        qdrant_path,
        chunker,
        records,
        docling_docs,
        nlp_results,
        **kwargs,
    ):
        calls["vector_records"] = len(records)
        return 0

    monkeypatch.setattr("pipeline.conversion.create_ocr_converter", _create_converter)
    monkeypatch.setattr("pipeline.conversion.convert_single_pdf", _convert_single_pdf)
    monkeypatch.setattr("pipeline.chunking.create_chunker", lambda: object())
    monkeypatch.setattr(
        "pipeline.vectorstores.build_qdrant_collection",
        _build_qdrant_collection,
    )

    main(
        [
            "--local-dir",
            str(local_dir),
            "--output-dir",
            str(output_dir),
            "--skip-nlp",
            "--force-reprocess",
        ]
    )

    assert calls["create_converter"] == 1
    assert calls["convert"] == 1
    assert calls["vector_records"] == 1
    md_text = (output_dir / "markdown" / "doc.md").read_text(encoding="utf-8")
    assert md_text == "# refreshed"

    loaded = load_pipeline_state(output_dir)
    state_entry = loaded["files"][str(pdf.resolve())]
    assert state_entry["status"] == "success"
    assert "processed_at" in state_entry


def test_insert_chunks_deletes_old_chunks_by_source_path():
    class FakeCollection:
        def __init__(self):
            self.deleted = []
            self.upsert_batches = []

        def delete(self, where):
            self.deleted.append(where)

        def upsert(self, ids, documents, metadatas):
            self.upsert_batches.append(
                {
                    "ids": ids,
                    "documents": documents,
                    "metadatas": metadatas,
                }
            )

        def count(self):
            return 7

    collection = FakeCollection()
    chunks = [
        {
            "id": "a::0001",
            "text": "one",
            "source_file": "a.pdf",
            "source_path": "/docs/a.pdf",
            "doc_title": "A",
            "num_pages": 1,
            "num_tables": 0,
            "num_figures": 0,
            "chunk_index": 0,
            "headings": "",
            "keywords": "",
            "tfidf_topics": "",
            "sentiment_label": "",
            "sentiment_score": 0.0,
            "doc_summary": "",
            "word_count": 1,
        },
        {
            "id": "a::0002",
            "text": "two",
            "source_file": "a.pdf",
            "source_path": "/docs/a.pdf",
            "doc_title": "A",
            "num_pages": 1,
            "num_tables": 0,
            "num_figures": 0,
            "chunk_index": 1,
            "headings": "",
            "keywords": "",
            "tfidf_topics": "",
            "sentiment_label": "",
            "sentiment_score": 0.0,
            "doc_summary": "",
            "word_count": 1,
        },
        {
            "id": "b::0001",
            "text": "three",
            "source_file": "b.pdf",
            "source_path": "/docs/b.pdf",
            "doc_title": "B",
            "num_pages": 1,
            "num_tables": 0,
            "num_figures": 0,
            "chunk_index": 0,
            "headings": "",
            "keywords": "",
            "tfidf_topics": "",
            "sentiment_label": "",
            "sentiment_score": 0.0,
            "doc_summary": "",
            "word_count": 1,
        },
    ]

    count = insert_chunks(collection, chunks, batch_size=2)
    assert count == 7
    assert len(collection.deleted) == 2
    assert {"source_path": "/docs/a.pdf"} in collection.deleted
    assert {"source_path": "/docs/b.pdf"} in collection.deleted
    assert len(collection.upsert_batches) == 2


def test_chunk_documents_uses_path_hash_for_chunk_ids():
    class DummyMeta:
        def __init__(self, headings):
            self.headings = headings

    class DummyChunk:
        def __init__(self, text, headings):
            self.text = text
            self.meta = DummyMeta(headings)

    class DummyChunker:
        def chunk(self, dl_doc):
            return [DummyChunk("alpha", ["H1"]), DummyChunk("beta", [])]

        def contextualize(self, chunk):
            return f"CTX::{chunk.text}"

    record = DocRecord(
        filename="doc.pdf",
        filepath="/tmp/my/path/doc.pdf",
        title="Doc",
        status="success",
    )
    chunks = chunk_documents(
        DummyChunker(),
        [record],
        {"doc.pdf": object()},
        {},
    )
    assert len(chunks) == 2

    expected_prefix = hashlib.sha1(record.filepath.encode("utf-8")).hexdigest()[:16]
    assert chunks[0]["id"] == f"{expected_prefix}::chunk_0000"
    assert chunks[1]["id"] == f"{expected_prefix}::chunk_0001"
