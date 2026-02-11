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
STATE_FILE_NAME = "pipeline_state.json"


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


def _state_path(output_dir: Path) -> Path:
    return output_dir / STATE_FILE_NAME


def file_fingerprint(path: Path) -> dict[str, int]:
    """Return a cheap fingerprint for local change detection."""
    stat = path.stat()
    return {
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def is_unchanged_file(path: Path, previous: dict[str, Any] | None) -> bool:
    """Check whether *path* matches a previous successful state entry."""
    if not previous or previous.get("status") != "success":
        return False
    current = file_fingerprint(path)
    return (
        previous.get("size") == current["size"]
        and previous.get("mtime_ns") == current["mtime_ns"]
    )


def load_pipeline_state(output_dir: Path) -> dict[str, Any]:
    """Load persistent pipeline state from ``pipeline_state.json``."""
    path = _state_path(output_dir)
    if not path.exists():
        return {"files": {}}
    try:
        with open(path, "r", encoding="utf-8") as fh:
            state = json.load(fh)
    except (json.JSONDecodeError, OSError, ValueError):
        return {"files": {}}

    if not isinstance(state, dict):
        return {"files": {}}

    files = state.get("files")
    if not isinstance(files, dict):
        state["files"] = {}
    return state


def save_pipeline_state(output_dir: Path, state: dict[str, Any]) -> Path:
    """Persist pipeline state and return the state file path."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = _state_path(output_dir)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(state, fh, indent=2, ensure_ascii=False, default=str)
    return path


# ---------------------------------------------------------------------------
# Manifest I/O
# ---------------------------------------------------------------------------


def save_manifest(
    output_dir: Path,
    records: list[DocRecord],
    nlp_results: dict[str, dict[str, Any]],
    *,
    merge_existing: bool = True,
) -> Path:
    """Write document_manifest.json and return its path."""
    manifest_path = output_dir / "document_manifest.json"
    manifest_by_path: dict[str, dict[str, Any]] = {}

    if merge_existing and manifest_path.exists():
        try:
            with open(manifest_path, "r", encoding="utf-8") as fh:
                existing = json.load(fh)
            if isinstance(existing, list):
                for item in existing:
                    if not isinstance(item, dict):
                        continue
                    key = str(item.get("filepath") or item.get("filename") or "")
                    if key:
                        manifest_by_path[key] = item
        except (json.JSONDecodeError, OSError, ValueError):
            manifest_by_path = {}

    for record in records:
        key = record.filepath or record.filename
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
        elif key in manifest_by_path and "nlp_analysis" in manifest_by_path[key]:
            entry["nlp_analysis"] = manifest_by_path[key]["nlp_analysis"]
        manifest_by_path[key] = entry

    manifest = sorted(
        manifest_by_path.values(),
        key=lambda item: (str(item.get("filename", "")), str(item.get("filepath", ""))),
    )
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, ensure_ascii=False, default=str)
    return manifest_path
