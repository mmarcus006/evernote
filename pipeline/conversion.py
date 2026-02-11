"""Docling OCR converter and PDF conversion."""

from __future__ import annotations

import importlib.util
import logging
import time
import traceback
from pathlib import Path
from typing import Any

from .models import DocRecord

log = logging.getLogger(__name__)


def create_ocr_converter(
    *,
    num_threads: int = 8,
    ocr_batch_size: int = 8,
    enable_ocr: bool = True,
) -> tuple[Any, str]:
    """Build a Docling ``DocumentConverter`` with the OCR pipeline.

    Args:
        num_threads: Thread count used by Docling accelerator options.
        ocr_batch_size: Batch size for OCR processing.
        enable_ocr: Whether OCR is enabled.

    Returns:
        (converter, ocr_engine) where ``ocr_engine`` is a best-effort profile.
    """
    import shutil as _shutil
    import time as _time

    t0 = _time.time()
    log.info("create_ocr_converter: importing docling modules ...")

    from docling.datamodel.accelerator_options import AcceleratorOptions
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import (
        EasyOcrOptions,
        OcrAutoOptions,
        PdfPipelineOptions,
        TesseractOcrOptions,
    )
    from docling.document_converter import DocumentConverter, PdfFormatOption

    log.info("create_ocr_converter: imports done in %.2fs", _time.time() - t0)

    accelerator_options = AcceleratorOptions(num_threads=max(1, num_threads))

    t1 = _time.time()
    ocr_engine = "disabled"

    has_easyocr = importlib.util.find_spec("easyocr") is not None
    has_tesseract = _shutil.which("tesseract") is not None

    if enable_ocr and (has_easyocr or has_tesseract):
        ocr_options = None

        if has_easyocr:
            try:
                log.info("create_ocr_converter: configuring EasyOCR ...")
                ocr_options = EasyOcrOptions()
                ocr_engine = "easyocr"
                log.info(
                    "create_ocr_converter: EasyOCR configured in %.2fs",
                    _time.time() - t1,
                )
            except Exception as exc:
                log.warning(
                    "create_ocr_converter: EasyOCR configuration failed (%s)",
                    exc,
                )

        if ocr_options is None and has_tesseract:
            try:
                log.info("create_ocr_converter: configuring Tesseract ...")
                ocr_options = TesseractOcrOptions()
                ocr_engine = "tesseract"
                log.info(
                    "create_ocr_converter: Tesseract configured in %.2fs",
                    _time.time() - t1,
                )
            except Exception as exc:
                log.warning(
                    "create_ocr_converter: Tesseract configuration failed (%s)",
                    exc,
                )

        if ocr_options is None:
            log.info("create_ocr_converter: using auto OCR configuration")
            ocr_options = OcrAutoOptions()
            ocr_engine = "auto"

        pipeline_options = PdfPipelineOptions(
            do_ocr=True,
            ocr_options=ocr_options,
            accelerator_options=accelerator_options,
            ocr_batch_size=max(1, ocr_batch_size),
        )
    else:
        if enable_ocr:
            log.warning(
                "create_ocr_converter: no OCR engine found; OCR disabled for speed"
            )
            ocr_engine = "disabled-no-engine"
        else:
            log.info("create_ocr_converter: OCR disabled")
        pipeline_options = PdfPipelineOptions(
            do_ocr=False,
            accelerator_options=accelerator_options,
        )

    t2 = _time.time()
    log.info("create_ocr_converter: creating DocumentConverter ...")
    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=pipeline_options,
            ),
        }
    )
    log.info(
        "create_ocr_converter: DocumentConverter created in %.2fs",
        _time.time() - t2,
    )
    log.info(
        "Docling OCR converter initialized (%s) total %.2fs",
        ocr_engine,
        _time.time() - t0,
    )
    return converter, ocr_engine


create_vlm_converter = create_ocr_converter


def convert_single_pdf(converter: Any, pdf_path: Path) -> tuple[DocRecord, Any]:
    """Convert one PDF and return (DocRecord, DoclingDocument | None).

    Never raises; conversion errors are captured inside the returned record.
    """
    file_size = pdf_path.stat().st_size if pdf_path.exists() else "MISSING"
    log.info("convert_single_pdf: START - %s (%s bytes)", pdf_path.name, file_size)

    record = DocRecord(filename=pdf_path.name, filepath=str(pdf_path))
    doc = None
    t0 = time.time()

    try:
        log.info(
            "convert_single_pdf: calling converter.convert(source=%s)...",
            pdf_path,
        )
        result = converter.convert(source=str(pdf_path))
        log.info(
            "convert_single_pdf: converter.convert() returned in %.2fs",
            time.time() - t0,
        )

        doc = result.document

        t1 = time.time()
        log.info("convert_single_pdf: exporting markdown ...")
        record.markdown = doc.export_to_markdown()
        log.info(
            "convert_single_pdf: markdown exported (%s chars) in %.2fs",
            len(record.markdown),
            time.time() - t1,
        )

        record.num_pages = len(result.pages) if hasattr(result, "pages") else 0
        record.num_tables = sum(1 for _ in doc.tables) if hasattr(doc, "tables") else 0
        record.num_figures = (
            sum(1 for _ in doc.pictures) if hasattr(doc, "pictures") else 0
        )
        record.title = doc.name if hasattr(doc, "name") and doc.name else pdf_path.stem
        record.status = "success"
        log.info(
            "convert_single_pdf: SUCCESS - pages=%s tables=%s figures=%s",
            record.num_pages,
            record.num_tables,
            record.num_figures,
        )
    except Exception:
        record.status = "error"
        record.error = traceback.format_exc()
        log.error("convert_single_pdf: ERROR - %s", record.error)
    finally:
        record.conversion_time_s = round(time.time() - t0, 2)
        log.info(
            "convert_single_pdf: DONE - %s in %ss",
            pdf_path.name,
            record.conversion_time_s,
        )

    return record, doc


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
    log.info("Conversion: %s succeeded, %s failed", len(success), len(failed))
    return records, docling_docs
