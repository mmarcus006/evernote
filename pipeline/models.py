"""Shared data models for the pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class DocRecord:
    """Tracks conversion results and metadata for a single PDF."""

    filename: str
    filepath: str
    drive_file_id: str = ""
    markdown: str = ""
    num_pages: int = 0
    num_tables: int = 0
    num_figures: int = 0
    title: str = ""
    conversion_time_s: float = 0.0
    status: str = "pending"
    error: Optional[str] = None
