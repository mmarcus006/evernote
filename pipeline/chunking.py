"""Document chunking with Docling HybridChunker."""

from __future__ import annotations

import hashlib
import logging
from typing import Any

from .models import DocRecord
from .utils import EMBEDDING_MODEL, MAX_TOKENS

log = logging.getLogger(__name__)


def create_chunker(
    embedding_model: str = EMBEDDING_MODEL,
    max_tokens: int = MAX_TOKENS,
) -> Any:
    """Create a Docling HybridChunker aligned to the embedding model tokenizer."""
    from docling.chunking import HybridChunker

    return HybridChunker(
        tokenizer=embedding_model,
        max_tokens=max_tokens,
        merge_peers=True,
    )


def chunk_documents(
    chunker: Any,
    records: list[DocRecord],
    docling_docs: dict[str, Any],
    nlp_results: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Chunk all successful documents and return flat list of chunk dicts."""
    all_chunks: list[dict[str, Any]] = []

    for record in records:
        if record.status != "success":
            continue
        doc = docling_docs.get(record.filename)
        if doc is None:
            continue

        analysis = nlp_results.get(record.filename, {})

        try:
            source_key = record.filepath or record.filename
            source_hash = hashlib.sha1(source_key.encode("utf-8")).hexdigest()[:16]
            for chunk_idx, chunk in enumerate(chunker.chunk(dl_doc=doc)):
                contextualized_text = chunker.contextualize(chunk)
                chunk_data = {
                    "id": f"{source_hash}::chunk_{chunk_idx:04d}",
                    "text": contextualized_text,
                    "raw_text": chunk.text,
                    "source_file": record.filename,
                    "source_path": record.filepath,
                    "doc_title": record.title,
                    "num_pages": record.num_pages,
                    "num_tables": record.num_tables,
                    "num_figures": record.num_figures,
                    "chunk_index": chunk_idx,
                    "headings": (
                        " > ".join(chunk.meta.headings) if chunk.meta.headings else ""
                    ),
                    "keywords": ", ".join(analysis.get("keywords_rake", [])[:10]),
                    "tfidf_topics": ", ".join(analysis.get("tfidf_topics", [])[:10]),
                    "sentiment_label": analysis.get("sentiment", {}).get("label", ""),
                    "sentiment_score": analysis.get("sentiment", {}).get("score", 0.0),
                    "doc_summary": analysis.get("summary", "")[:500],
                    "word_count": analysis.get("word_count", 0),
                }
                all_chunks.append(chunk_data)
        except Exception as exc:
            log.warning(f"Chunking failed for {record.filename}: {exc}")
            continue

    return all_chunks
