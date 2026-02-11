"""PDF-to-Markdown + NLP + Vector Database pipeline.

Public API -- all symbols that tests and external code import live here.
Internally the code is split across focused submodules; this file
re-exports the stable public surface so ``from pipeline import X`` works.
"""

from .chunking import chunk_documents, create_chunker
from .conversion import (
    convert_pdf,
    convert_pdfs,
    convert_single_pdf,
    create_ocr_converter,
    create_vlm_converter,
)
from .models import DocRecord
from .nlp import (
    analyze_sentiment,
    classify_document_type,
    compute_tfidf_topics,
    extract_entities,
    extract_keywords,
    extract_keywords_rake,
    extract_tfidf_topics,
    extractive_summary,
    init_nlp,
    run_nlp_analysis,
)
from .sources import (
    authenticate_drive,
    discover_pdfs,
    download_pdf,
    list_drive_pdfs,
)
from .utils import (
    COLLECTION_NAME,
    EMBEDDING_DIM,
    EMBEDDING_MODEL,
    MAX_TOKENS,
    ensure_output_dirs,
    is_colab,
    save_manifest,
)
from .vectorstores import (
    build_qdrant_collection,
    create_chroma_collection,
    insert_chunks,
    search_collection,
)

__all__ = [
    # Models
    "DocRecord",
    # Constants
    "EMBEDDING_MODEL",
    "EMBEDDING_DIM",
    "MAX_TOKENS",
    "COLLECTION_NAME",
    # Utils
    "is_colab",
    "ensure_output_dirs",
    "save_manifest",
    # Sources
    "discover_pdfs",
    "authenticate_drive",
    "list_drive_pdfs",
    "download_pdf",
    # Conversion
    "create_ocr_converter",
    "create_vlm_converter",
    "convert_single_pdf",
    "convert_pdf",
    "convert_pdfs",
    # NLP
    "init_nlp",
    "extract_keywords_rake",
    "extract_keywords",
    "extract_entities",
    "extractive_summary",
    "analyze_sentiment",
    "extract_tfidf_topics",
    "compute_tfidf_topics",
    "classify_document_type",
    "run_nlp_analysis",
    # Chunking
    "create_chunker",
    "chunk_documents",
    # Vector stores
    "create_chroma_collection",
    "insert_chunks",
    "search_collection",
    "build_qdrant_collection",
]
