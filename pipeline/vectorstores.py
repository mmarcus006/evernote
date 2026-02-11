"""Vector database backends: ChromaDB and Qdrant."""

from __future__ import annotations

import hashlib
import importlib
import logging
from pathlib import Path
from typing import Any

from .models import DocRecord
from .utils import COLLECTION_NAME, EMBEDDING_DIM, EMBEDDING_MODEL

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ChromaDB
# ---------------------------------------------------------------------------


def create_chroma_collection(
    db_dir: Path,
    collection_name: str = COLLECTION_NAME,
    embedding_model: str = EMBEDDING_MODEL,
) -> tuple[Any, Any]:
    """Create persistent ChromaDB client + collection.

    Returns (client, collection).
    """
    chromadb = importlib.import_module("chromadb")
    embedding_functions = importlib.import_module("chromadb.utils.embedding_functions")

    client = chromadb.PersistentClient(path=str(db_dir))
    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=embedding_model,
    )
    collection = client.get_or_create_collection(
        name=collection_name,
        embedding_function=embedding_fn,
        metadata={"hnsw:space": "cosine"},
    )
    return client, collection


def insert_chunks(
    collection: Any,
    chunks: list[dict[str, Any]],
    batch_size: int = 50,
) -> int:
    """Upsert chunks into a ChromaDB collection. Returns total count."""
    if not chunks:
        return collection.count()

    # Remove prior chunks for documents being updated so chunk-count changes
    # don't leave stale vectors behind.
    source_paths = sorted({c["source_path"] for c in chunks if c.get("source_path")})
    for source_path in source_paths:
        collection.delete(where={"source_path": source_path})

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        ids = [c["id"] for c in batch]
        documents = [c["text"] for c in batch]
        metadatas = [
            {
                "source_file": c["source_file"],
                "source_path": c["source_path"],
                "doc_title": c["doc_title"],
                "num_pages": c["num_pages"],
                "num_tables": c["num_tables"],
                "num_figures": c["num_figures"],
                "chunk_index": c["chunk_index"],
                "headings": c["headings"],
                "keywords": c["keywords"],
                "tfidf_topics": c["tfidf_topics"],
                "sentiment_label": c["sentiment_label"],
                "sentiment_score": c["sentiment_score"],
                "doc_summary": c["doc_summary"],
                "word_count": c["word_count"],
            }
            for c in batch
        ]
        collection.upsert(ids=ids, documents=documents, metadatas=metadatas)
    return collection.count()


def search_collection(
    collection: Any,
    query: str,
    n_results: int = 5,
    where: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Semantic search against a ChromaDB collection."""
    kwargs: dict[str, Any] = {
        "query_texts": [query],
        "n_results": n_results,
    }
    if where:
        kwargs["where"] = where
    return collection.query(**kwargs)


# ---------------------------------------------------------------------------
# Qdrant
# ---------------------------------------------------------------------------


def build_qdrant_collection(
    qdrant_path: Path,
    chunker: Any,
    records: list[DocRecord],
    docling_docs: dict[str, Any],
    nlp_results: dict[str, dict[str, Any]],
    collection_name: str = COLLECTION_NAME,
    embedding_model_id: str = EMBEDDING_MODEL,
    embedding_dim: int = EMBEDDING_DIM,
    batch_size: int = 128,
    embedding_threads: int | None = None,
    recreate_collection: bool = False,
) -> int:
    """Chunk all documents and insert into Qdrant with FastEmbed embeddings.

    Returns the total vector count.
    """
    from fastembed import TextEmbedding
    from qdrant_client import QdrantClient
    from qdrant_client.models import (
        Distance,
        FieldCondition,
        Filter,
        MatchValue,
        PointStruct,
        VectorParams,
    )

    client = QdrantClient(path=str(qdrant_path))

    collection_exists = client.collection_exists(collection_name)
    if recreate_collection and collection_exists:
        client.delete_collection(collection_name)
        collection_exists = False

    if not collection_exists:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=embedding_dim,
                distance=Distance.COSINE,
            ),
        )

    successful = [r for r in records if r.status == "success"]

    # Remove all existing chunks for docs being refreshed.
    for record in successful:
        source_path = record.filepath
        if not source_path:
            continue
        client.delete(
            collection_name=collection_name,
            points_selector=Filter(
                must=[
                    FieldCondition(
                        key="source_path",
                        match=MatchValue(value=source_path),
                    )
                ]
            ),
            wait=True,
        )

    all_documents: list[str] = []
    all_metadata: list[dict[str, Any]] = []
    all_ids: list[str] = []

    for record in successful:
        doc = docling_docs.get(record.filename)
        if doc is None:
            continue
        analysis = nlp_results.get(record.filename, {})

        for chunk_idx, chunk in enumerate(chunker.chunk(dl_doc=doc)):
            ctx_text = chunker.contextualize(chunk)
            metadata = {
                "text": ctx_text,
                "source_file": record.filename,
                "source_path": record.filepath,
                "doc_title": record.title,
                "chunk_index": chunk_idx,
                "headings": (
                    " > ".join(chunk.meta.headings) if chunk.meta.headings else ""
                ),
                "keywords": ", ".join(analysis.get("keywords_rake", [])[:10]),
                "tfidf_topics": ", ".join(analysis.get("tfidf_topics", [])[:10]),
                "document_type": analysis.get("document_type", ""),
                "sentiment_label": analysis.get("sentiment", {}).get("label", ""),
                "sentiment_compound": analysis.get("sentiment", {}).get(
                    "compound", 0.0
                ),
                "doc_summary": analysis.get("summary", "")[:500],
                "people": ", ".join(analysis.get("people", [])[:10]),
                "organizations": ", ".join(analysis.get("organizations", [])[:10]),
                "dates": ", ".join(analysis.get("dates", [])[:10]),
                "amounts": ", ".join(analysis.get("amounts", [])[:10]),
                "word_count": analysis.get("word_count", 0),
            }
            id_source = f"{record.filepath}::{chunk_idx}"
            all_documents.append(ctx_text)
            all_metadata.append(metadata)
            all_ids.append(hashlib.sha1(id_source.encode("utf-8")).hexdigest())

    if not all_documents:
        count = client.count(collection_name=collection_name).count
        client.close()
        return count

    log.info(f"Embedding {len(all_documents)} chunks with {embedding_model_id}...")
    embedding_model = TextEmbedding(
        embedding_model_id,
        threads=embedding_threads,
    )
    embeddings = list(embedding_model.embed(all_documents))

    normalized_batch_size = max(1, batch_size)
    for i in range(0, len(all_documents), normalized_batch_size):
        batch_end = min(i + normalized_batch_size, len(all_documents))
        points = [
            PointStruct(
                id=all_ids[i + j],
                vector=embeddings[i + j].tolist(),
                payload=all_metadata[i + j],
            )
            for j in range(batch_end - i)
        ]
        client.upsert(collection_name=collection_name, points=points)

    count = client.count(collection_name=collection_name).count
    client.close()
    return count
