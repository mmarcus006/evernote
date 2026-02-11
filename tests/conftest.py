"""Shared fixtures for the pipeline test suite.

Uses three real PDF files from the project's output folder.
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Configure verbose logging for test debugging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stderr,
    force=True,
)
log = logging.getLogger("conftest")

# Real PDF test fixtures — paths to actual documents in the project
PDF_DIR = Path(__file__).resolve().parent.parent / "output" / "Properties"

PDF_SMALL = PDF_DIR / "2014_07_02_15_28_10.pdf"  # ~93 KB
PDF_MEDIUM = PDF_DIR / "34 Bunsen Impound Letter.pdf"  # ~510 KB
PDF_LARGE = PDF_DIR / "3 Goodyear.pdf"  # ~8.5 MB

ALL_PDFS = [PDF_SMALL, PDF_MEDIUM, PDF_LARGE]


@pytest.fixture(scope="session")
def pdf_small() -> Path:
    assert PDF_SMALL.exists(), f"Test PDF missing: {PDF_SMALL}"
    return PDF_SMALL


@pytest.fixture(scope="session")
def pdf_medium() -> Path:
    assert PDF_MEDIUM.exists(), f"Test PDF missing: {PDF_MEDIUM}"
    return PDF_MEDIUM


@pytest.fixture(scope="session")
def pdf_large() -> Path:
    assert PDF_LARGE.exists(), f"Test PDF missing: {PDF_LARGE}"
    return PDF_LARGE


@pytest.fixture(scope="session")
def all_pdf_paths() -> list[Path]:
    for p in ALL_PDFS:
        assert p.exists(), f"Test PDF missing: {p}"
    return list(ALL_PDFS)


@pytest.fixture
def tmp_output(tmp_path: Path) -> Path:
    """Return a temporary output directory for a single test."""
    return tmp_path / "output"


@pytest.fixture(scope="session")
def vlm_converter():
    """Session-scoped VLM converter (expensive — load once)."""
    log.info(">>> FIXTURE vlm_converter: starting create_vlm_converter()")
    t0 = time.time()

    log.info("  Importing pipeline.create_vlm_converter ...")
    from pipeline import create_vlm_converter

    log.info(f"  Import done in {time.time() - t0:.2f}s")

    t1 = time.time()
    log.info("  Calling create_vlm_converter() ...")
    converter, vlm_name = create_vlm_converter()
    log.info(
        f"  create_vlm_converter() returned in {time.time() - t1:.2f}s "
        f"(vlm_name={vlm_name})"
    )
    log.info(f">>> FIXTURE vlm_converter: total = {time.time() - t0:.2f}s")
    return converter


@pytest.fixture(scope="session")
def converted_small(vlm_converter, pdf_small) -> tuple:
    """Convert the small PDF once for the entire session.

    Returns (DocRecord, DoclingDocument).
    """
    log.info(f">>> FIXTURE converted_small: converting {pdf_small.name} ...")
    t0 = time.time()

    from pipeline import convert_single_pdf

    record, doc = convert_single_pdf(vlm_converter, pdf_small)

    log.info(
        f">>> FIXTURE converted_small: done in {time.time() - t0:.2f}s — "
        f"status={record.status}, error={record.error}"
    )
    assert record.status == "success", f"Conversion failed: {record.error}"
    return record, doc


@pytest.fixture(scope="session")
def converted_medium(vlm_converter, pdf_medium) -> tuple:
    """Convert the medium PDF once for the entire session."""
    log.info(f">>> FIXTURE converted_medium: converting {pdf_medium.name} ...")
    t0 = time.time()

    from pipeline import convert_single_pdf

    record, doc = convert_single_pdf(vlm_converter, pdf_medium)

    log.info(
        f">>> FIXTURE converted_medium: done in {time.time() - t0:.2f}s — "
        f"status={record.status}, error={record.error}"
    )
    assert record.status == "success", f"Conversion failed: {record.error}"
    return record, doc


@pytest.fixture(scope="session")
def converted_all(vlm_converter, all_pdf_paths) -> tuple:
    """Convert all three PDFs once.

    Returns (list[DocRecord], dict[filename, DoclingDocument]).
    """
    log.info(f">>> FIXTURE converted_all: converting {len(all_pdf_paths)} PDFs ...")
    t0 = time.time()

    from pipeline import convert_single_pdf

    records = []
    docs = {}
    for i, pdf_path in enumerate(all_pdf_paths):
        log.info(f"  [{i + 1}/{len(all_pdf_paths)}] Converting {pdf_path.name} ...")
        t1 = time.time()
        record, doc = convert_single_pdf(vlm_converter, pdf_path)
        log.info(
            f"  [{i + 1}/{len(all_pdf_paths)}] Done in {time.time() - t1:.2f}s — "
            f"status={record.status}"
        )
        records.append(record)
        if record.status == "success" and doc is not None:
            docs[record.filename] = doc

    log.info(f">>> FIXTURE converted_all: total = {time.time() - t0:.2f}s")
    return records, docs


@pytest.fixture(scope="session")
def sentiment_analyzer():
    """Session-scoped sentiment analyzer (expensive — load once)."""
    log.info(">>> FIXTURE sentiment_analyzer: loading HF pipeline ...")
    t0 = time.time()

    from transformers import pipeline as hf_pipeline

    analyzer = hf_pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english",
        truncation=True,
        max_length=512,
    )
    log.info(f">>> FIXTURE sentiment_analyzer: loaded in {time.time() - t0:.2f}s")
    return analyzer


@pytest.fixture(scope="session")
def chunker():
    """Session-scoped HybridChunker."""
    log.info(">>> FIXTURE chunker: creating HybridChunker ...")
    t0 = time.time()

    from pipeline import create_chunker

    c = create_chunker()
    log.info(f">>> FIXTURE chunker: created in {time.time() - t0:.2f}s")
    return c
