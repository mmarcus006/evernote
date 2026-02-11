"""Cross-cutting helpers: constants, path utilities, manifest I/O."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from .models import DocRecord

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384
MAX_TOKENS = 512
COLLECTION_NAME = "docling_documents"


# ---------------------------------------------------------------------------
# Environment helpers
# ---------------------------------------------------------------------------


def is_colab() -> bool:
    """Detect if running inside Google Colab."""
    return "google.colab" in sys.modules


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------


def ensure_output_dirs(output_dir: Path) -> tuple[Path, Path]:
    """Create and return (markdown_dir, db_dir) under *output_dir*."""
    md_dir = output_dir / "markdown"
    db_dir = output_dir / "vector_db"
    md_dir.mkdir(parents=True, exist_ok=True)
    db_dir.mkdir(parents=True, exist_ok=True)
    return md_dir, db_dir


# ---------------------------------------------------------------------------
# Manifest I/O
# ---------------------------------------------------------------------------


def save_manifest(
    output_dir: Path,
    records: list[DocRecord],
    nlp_results: dict[str, dict[str, Any]],
) -> Path:
    """Write document_manifest.json and return its path."""
    manifest: list[dict[str, Any]] = []
    for record in records:
        entry: dict[str, Any] = {
            "filename": record.filename,
            "filepath": record.filepath,
            "title": record.title,
            "status": record.status,
            "num_pages": record.num_pages,
            "num_tables": record.num_tables,
            "num_figures": record.num_figures,
            "conversion_time_s": record.conversion_time_s,
            "error": record.error,
        }
        if record.filename in nlp_results:
            entry["nlp_analysis"] = nlp_results[record.filename]
        manifest.append(entry)

    manifest_path = output_dir / "document_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, ensure_ascii=False, default=str)
    return manifest_path
