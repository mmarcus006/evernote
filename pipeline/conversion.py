"""Docling OCR converter and PDF conversion."""

from __future__ import annotations

import logging
import time
import traceback
from pathlib import Path
from typing import Any

from .models import DocRecord

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Docling converter factory (OCR-based, no VLM)
# ---------------------------------------------------------------------------


def create_ocr_converter() -> tuple[Any, str]:
    """Build a Docling ``DocumentConverter`` with the standard OCR pipeline.

    Returns:
        (converter, ocr_engine) -- the converter instance and the name of
        the OCR engine used.
    """
    import time as _time

    t0 = _time.time()
    log.info("create_ocr_converter: importing docling modules ...")

    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import (
        EasyOcrOptions,
        OcrMacOptions,
        PdfPipelineOptions,
        TesseractOcrOptions,
    )
    from docling.document_converter import DocumentConverter, PdfFormatOption

    log.info(f"create_ocr_converter: imports done in {_time.time() - t0:.2f}s")

    # Try EasyOCR first (best quality, GPU-accelerated if available),
    # fall back to Tesseract, then to default.
    t1 = _time.time()
    ocr_engine = "easyocr"
    try:
        log.info("create_ocr_converter: configuring EasyOCR ...")
        ocr_options = EasyOcrOptions()
        pipeline_options = PdfPipelineOptions(do_ocr=True, ocr_options=ocr_options)
        log.info(f"create_ocr_converter: EasyOCR configured in {_time.time() - t1:.2f}s")
    except Exception as exc:
        log.warning(f"create_ocr_converter: EasyOCR unavailable ({exc}), trying Tesseract ...")
        try:
            ocr_options = TesseractOcrOptions()
            pipeline_options = PdfPipelineOptions(do_ocr=True, ocr_options=ocr_options)
            ocr_engine = "tesseract"
            log.info(f"create_ocr_converter: Tesseract configured in {_time.time() - t1:.2f}s")
        except Exception as exc2:
            log.warning(f"create_ocr_converter: Tesseract unavailable ({exc2}), using defaults ...")
            pipeline_options = PdfPipelineOptions(do_ocr=True)
            ocr_engine = "default"

    t2 = _time.time()
    log.info("create_ocr_converter: creating DocumentConverter ...")
    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=pipeline_options,
            ),
        }
    )
    log.info(f"create_ocr_converter: DocumentConverter created in {_time.time() - t2:.2f}s")
    log.info(f"Docling OCR converter initialized ({ocr_engine}) — total {_time.time() - t0:.2f}s.")
    return converter, ocr_engine


# Backward-compatible alias
create_vlm_converter = create_ocr_converter


# ---------------------------------------------------------------------------
# PDF conversion
# ---------------------------------------------------------------------------


def convert_single_pdf(converter: Any, pdf_path: Path) -> tuple[DocRecord, Any]:
    """Convert one PDF and return (DocRecord, DoclingDocument | None).

    Never raises -- conversion errors are captured inside the record.
    """
    log.info(
        f"convert_single_pdf: START — {pdf_path.name} ({pdf_path.stat().st_size if pdf_path.exists() else 'MISSING'} bytes)"
    )
    record = DocRecord(filename=pdf_path.name, filepath=str(pdf_path))
    doc = None
    t0 = time.time()
    try:
        log.info(
            f"convert_single_pdf: calling converter.convert(source={pdf_path}) ..."
        )
        result = converter.convert(source=str(pdf_path))
        log.info(
            f"convert_single_pdf: converter.convert() returned in {time.time() - t0:.2f}s"
        )

        doc = result.document

        t1 = time.time()
        log.info("convert_single_pdf: exporting markdown ...")
        record.markdown = doc.export_to_markdown()
        log.info(
            f"convert_single_pdf: markdown exported ({len(record.markdown)} chars) in {time.time() - t1:.2f}s"
        )

        record.num_pages = len(result.pages) if hasattr(result, "pages") else 0
        record.num_tables = sum(1 for _ in doc.tables) if hasattr(doc, "tables") else 0
        record.num_figures = (
            sum(1 for _ in doc.pictures) if hasattr(doc, "pictures") else 0
        )
        record.title = doc.name if hasattr(doc, "name") and doc.name else pdf_path.stem
        record.status = "success"
        log.info(
            f"convert_single_pdf: SUCCESS — pages={record.num_pages}, "
            f"tables={record.num_tables}, figures={record.num_figures}"
        )
    except Exception:
        record.status = "error"
        record.error = traceback.format_exc()
        log.error(f"convert_single_pdf: ERROR — {record.error}")
    finally:
        record.conversion_time_s = round(time.time() - t0, 2)
        log.info(
            f"convert_single_pdf: DONE — {pdf_path.name} in {record.conversion_time_s}s"
        )
    return record, doc


# Alias for backward compatibility
convert_pdf = convert_single_pdf


def convert_pdfs(
    converter: Any,
    pdf_files: list[Path],
    md_output_dir: Path,
) -> tuple[list[DocRecord], dict[str, Any]]:
    """Batch-convert PDFs. Returns (records, {filename: DoclingDocument})."""
    records: list[DocRecord] = []
    docling_docs: dict[str, Any] = {}

    for pdf_path in pdf_files:
        record, doc = convert_single_pdf(converter, pdf_path)
        records.append(record)
        if record.status == "success" and doc is not None:
            md_file = md_output_dir / f"{pdf_path.stem}.md"
            md_file.write_text(record.markdown, encoding="utf-8")
            docling_docs[record.filename] = doc

    success = [r for r in records if r.status == "success"]
    failed = [r for r in records if r.status == "error"]
    log.info(f"Conversion: {len(success)} succeeded, {len(failed)} failed")
    return records, docling_docs
