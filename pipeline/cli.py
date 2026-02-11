"""CLI entrypoint for the PDF -> Markdown -> NLP -> Vector DB pipeline.

Usage:
    python -m pipeline
    python -m pipeline --folder-id <DRIVE_FOLDER_ID>
    python -m pipeline --local-dir ./output/Properties
    python -m pipeline --local-dir ./output/Properties --skip-nlp
    python -m pipeline --local-dir ./output/Properties --backend qdrant
    python -m pipeline --local-dir ./output/Properties --force-reprocess
    python -m pipeline --local-dir ./output/Properties --rebuild-index
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import logging
from logging.handlers import RotatingFileHandler
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


def _detect_gpu_vram_mib() -> int:
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.total",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except (subprocess.SubprocessError, FileNotFoundError):
        return 0

    first_line = completed.stdout.strip().splitlines()
    if not first_line:
        return 0

    try:
        return int(first_line[0].strip())
    except ValueError:
        return 0


def _recommended_runtime_defaults() -> dict[str, int]:
    cpu_count = max(1, os.cpu_count() or 1)
    gpu_vram_mib = _detect_gpu_vram_mib()
    has_gpu = gpu_vram_mib > 0

    max_workers = min(64, max(16, cpu_count * 2))
    num_threads = min(24, max(8, cpu_count))
    embed_threads = min(24, max(8, cpu_count))
    ocr_batch_size = 12 if gpu_vram_mib >= 8 * 1024 else (8 if has_gpu else 4)
    vector_batch_size = 256 if has_gpu else 128

    return {
        "max_workers": max_workers,
        "num_threads": num_threads,
        "embed_threads": embed_threads,
        "ocr_batch_size": ocr_batch_size,
        "vector_batch_size": vector_batch_size,
    }


def _setup_logging(
    *,
    verbose: bool,
    detailed_logging: bool,
    output_dir: Path,
    log_file: Path | None,
) -> None:
    root_logger = logging.getLogger()
    root_logger.handlers.clear()

    root_level = logging.DEBUG if verbose else logging.INFO
    root_logger.setLevel(root_level)

    console_fmt = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    detailed_fmt = (
        "%(asctime)s | %(levelname)-8s | %(name)s | "
        "%(threadName)s | %(filename)s:%(lineno)d | %(message)s"
    )
    formatter = logging.Formatter(
        detailed_fmt if detailed_logging else console_fmt,
        "%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler()
    console_handler.setLevel(root_level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    resolved_log_file = log_file
    if resolved_log_file is None and detailed_logging:
        resolved_log_file = output_dir / "pipeline.log"

    if resolved_log_file is not None:
        resolved_log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            resolved_log_file,
            maxBytes=20 * 1024 * 1024,
            backupCount=5,
            encoding="utf-8",
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter(detailed_fmt, "%Y-%m-%d %H:%M:%S"))
        root_logger.addHandler(file_handler)

    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    tuned_defaults = _recommended_runtime_defaults()

    parser = argparse.ArgumentParser(
        description="PDF -> Markdown -> NLP -> Vector DB pipeline"
    )
    src = parser.add_mutually_exclusive_group(required=False)
    src.add_argument("--folder-id", help="Google Drive folder ID containing PDFs")
    src.add_argument(
        "--local-dir",
        type=Path,
        help=(
            "Local directory with PDFs (skip Drive). "
            "Defaults to --output-dir when no source argument is provided."
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output"),
        help="Output directory (default: output/)",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("cache/pdfs"),
        help="Cache for downloaded PDFs (default: cache/pdfs/)",
    )
    parser.add_argument(
        "--backend",
        choices=["chroma", "qdrant"],
        default="qdrant",
        help="Vector DB backend (default: qdrant)",
    )
    parser.add_argument(
        "--qdrant-path",
        type=Path,
        default=Path("qdrant_data"),
        help="Qdrant storage path (default: qdrant_data/)",
    )
    parser.add_argument(
        "--credentials",
        type=Path,
        default=Path("credentials.json"),
        help="Google OAuth2 credentials file (default: credentials.json)",
    )
    parser.add_argument(
        "--skip-nlp",
        action="store_true",
        help="Skip NLP analysis (faster, less metadata)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=tuned_defaults["max_workers"],
        help="Worker threads for parallel I/O and ingestion",
    )
    parser.add_argument(
        "--num-threads",
        type=int,
        default=tuned_defaults["num_threads"],
        help="Docling/FastEmbed internal thread count",
    )
    parser.add_argument(
        "--ocr-batch-size",
        type=int,
        default=tuned_defaults["ocr_batch_size"],
        help="OCR batch size for Docling",
    )
    parser.add_argument(
        "--embed-threads",
        type=int,
        default=tuned_defaults["embed_threads"],
        help="FastEmbed embedding thread count",
    )
    parser.add_argument(
        "--vector-batch-size",
        type=int,
        default=tuned_defaults["vector_batch_size"],
        help="Vector DB upsert batch size",
    )
    parser.add_argument(
        "--disable-ocr",
        action="store_true",
        help="Disable OCR for faster conversion on text PDFs",
    )
    parser.add_argument(
        "--force-reprocess",
        action="store_true",
        help="Process every PDF even if unchanged in saved pipeline state",
    )
    parser.add_argument(
        "--rebuild-index",
        action="store_true",
        help="Drop and recreate the Qdrant collection before upserting",
    )
    parser.add_argument(
        "--detailed-logging",
        action="store_true",
        help="Enable detailed logging (thread, file/line, rotating log file)",
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=None,
        help=(
            "Optional log file path "
            "(default: <output-dir>/pipeline.log in detailed mode)"
        ),
    )
    args = parser.parse_args(argv)
    if args.folder_id is None and args.local_dir is None:
        args.local_dir = args.output_dir

    return args


def main(argv: list[str] | None = None) -> None:
    """Run the full pipeline."""
    from tqdm import tqdm

    from .chunking import chunk_documents, create_chunker
    from .conversion import convert_single_pdf, create_ocr_converter
    from .nlp import init_nlp, run_nlp_analysis
    from .sources import (
        authenticate_drive,
        discover_pdfs,
        download_pdf,
        list_drive_pdfs,
    )
    from .utils import (
        ensure_output_dirs,
        file_fingerprint,
        is_unchanged_file,
        load_pipeline_state,
        save_manifest,
        save_pipeline_state,
    )
    from .vectorstores import (
        build_qdrant_collection,
        create_chroma_collection,
        insert_chunks,
    )

    args = parse_args(argv)
    _setup_logging(
        verbose=args.verbose,
        detailed_logging=args.detailed_logging,
        output_dir=args.output_dir,
        log_file=args.log_file,
    )

    overall_t0 = time.perf_counter()
    log.info(
        "Runtime tuning: max_workers=%s num_threads=%s embed_threads=%s "
        "ocr_batch_size=%s vector_batch_size=%s",
        args.max_workers,
        args.num_threads,
        args.embed_threads,
        args.ocr_batch_size,
        args.vector_batch_size,
    )
    log.debug(
        "Logging setup: verbose=%s detailed=%s log_file=%s",
        args.verbose,
        args.detailed_logging,
        args.log_file,
    )

    # Create directories
    md_dir, db_dir = ensure_output_dirs(args.output_dir)
    nlp_dir = args.output_dir / "nlp"
    nlp_dir.mkdir(parents=True, exist_ok=True)

    # --- Step 1: Get PDF files ---
    step1_t0 = time.perf_counter()
    if args.local_dir:
        log.info(f"Using local PDFs from: {args.local_dir}")
        pdf_files = discover_pdfs(args.local_dir)
    else:
        args.cache_dir.mkdir(parents=True, exist_ok=True)
        log.info("Authenticating with Google Drive...")
        token_file = args.credentials.parent / "token.json"
        creds = authenticate_drive(args.credentials, token_file)

        from googleapiclient.discovery import build

        service = build("drive", "v3", credentials=creds)

        log.info(f"Listing PDFs in folder: {args.folder_id}")
        drive_files = list_drive_pdfs(service, args.folder_id)
        log.info(f"Found {len(drive_files)} PDFs")

        pdf_files: list[Path] = []
        max_workers = max(1, args.max_workers)
        if max_workers == 1 or len(drive_files) <= 1:
            for drive_file in tqdm(drive_files, desc="Downloading"):
                path = download_pdf(
                    service,
                    drive_file["id"],
                    drive_file["name"],
                    args.cache_dir,
                )
                pdf_files.append(path)
        else:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(
                        download_pdf,
                        service,
                        drive_file["id"],
                        drive_file["name"],
                        args.cache_dir,
                    ): drive_file
                    for drive_file in drive_files
                }
                for future in tqdm(as_completed(futures), total=len(futures)):
                    pdf_files.append(future.result())

    log.info("Input discovery completed in %.2fs", time.perf_counter() - step1_t0)
    log.info(f"Total PDFs discovered: {len(pdf_files)}")
    if pdf_files:
        sample_files = ", ".join(path.name for path in pdf_files[:5])
        log.debug("Sample input files: %s", sample_files)

    if not pdf_files:
        log.warning("No PDFs found. Exiting.")
        sys.exit(0)

    # --- Resume check: skip unchanged PDFs ---
    state = load_pipeline_state(args.output_dir)
    state_files: dict[str, dict[str, Any]] = state.setdefault("files", {})
    pdf_files_to_process: list[Path] = []
    skipped_unchanged = 0
    for pdf_path in pdf_files:
        state_key = str(pdf_path.resolve())
        previous = state_files.get(state_key)
        md_file = md_dir / f"{pdf_path.stem}.md"
        nlp_file = nlp_dir / f"{pdf_path.stem}_nlp.json"
        has_output_artifacts = md_file.exists() and (args.skip_nlp or nlp_file.exists())
        if (
            not args.force_reprocess
            and has_output_artifacts
            and is_unchanged_file(pdf_path, previous)
        ):
            skipped_unchanged += 1
            continue
        pdf_files_to_process.append(pdf_path)

    log.info(
        "Resume check: %s unchanged skipped, %s queued for processing",
        skipped_unchanged,
        len(pdf_files_to_process),
    )

    # --- Step 2: Convert PDFs to Markdown ---
    step2_t0 = time.perf_counter()
    docling_docs: dict[str, object] = {}
    records = []

    if pdf_files_to_process:
        log.info("Initializing Docling OCR converter...")
        converter, ocr_engine = create_ocr_converter(
            num_threads=max(1, args.num_threads),
            ocr_batch_size=max(1, args.ocr_batch_size),
            enable_ocr=not args.disable_ocr,
        )
        log.info("Converter OCR profile: %s", ocr_engine)

        for pdf_path in tqdm(pdf_files_to_process, desc="Converting PDFs"):
            record, doc = convert_single_pdf(converter, pdf_path)
            records.append(record)
            if doc is not None:
                docling_docs[record.filename] = doc
                md_file = md_dir / f"{pdf_path.stem}.md"
                md_file.write_text(record.markdown, encoding="utf-8")
    else:
        log.info("No new/changed PDFs detected; conversion stage skipped.")

    step2_elapsed = time.perf_counter() - step2_t0
    success = [r for r in records if r.status == "success"]
    failed = [r for r in records if r.status == "error"]
    log.info(
        "Conversion: %s succeeded, %s failed (%.2fs)",
        len(success),
        len(failed),
        step2_elapsed,
    )

    # --- Step 3: NLP Analysis ---
    step3_t0 = time.perf_counter()
    nlp_results: dict[str, dict[str, Any]] = {}
    if args.skip_nlp:
        log.info("NLP stage disabled via --skip-nlp")
    elif success:
        log.info("Running NLP analysis...")
        init_nlp()
        nlp_results = run_nlp_analysis(records)
        for filename, analysis in nlp_results.items():
            nlp_file = nlp_dir / f"{Path(filename).stem}_nlp.json"
            with open(nlp_file, "w") as f:
                json.dump(analysis, f, indent=2, default=str)
    else:
        log.info("No successful new/changed docs; NLP stage skipped.")
    log.info("NLP stage completed in %.2fs", time.perf_counter() - step3_t0)

    # --- Step 4: Chunk and store in vector DB ---
    step4_t0 = time.perf_counter()
    log.info("Chunking and building vector database...")
    chunker = create_chunker()

    if args.backend == "qdrant":
        count = build_qdrant_collection(
            args.qdrant_path,
            chunker,
            records,
            docling_docs,
            nlp_results,
            batch_size=max(1, args.vector_batch_size),
            embedding_threads=max(1, args.embed_threads),
            recreate_collection=args.rebuild_index,
        )
        log.info(f"Qdrant collection: {count} vectors stored")
    else:
        chunks = chunk_documents(chunker, records, docling_docs, nlp_results)
        _, collection = create_chroma_collection(db_dir)
        count = insert_chunks(
            collection,
            chunks,
            batch_size=max(1, args.vector_batch_size),
        )
        log.info(f"ChromaDB collection: {count} vectors stored")
    log.info("Vector stage completed in %.2fs", time.perf_counter() - step4_t0)

    # --- Step 5: Save manifest ---
    step5_t0 = time.perf_counter()
    manifest_path = save_manifest(args.output_dir, records, nlp_results)
    log.info("Manifest stage completed in %.2fs", time.perf_counter() - step5_t0)

    # --- Step 6: Save pipeline state ---
    step6_t0 = time.perf_counter()
    current_keys = {str(path.resolve()) for path in pdf_files}
    for key in list(state_files):
        if key not in current_keys:
            state_files.pop(key, None)

    processed_at = datetime.now(timezone.utc).isoformat()
    for record in records:
        state_key = str(Path(record.filepath).resolve())
        entry: dict[str, Any] = {
            "filename": record.filename,
            "status": record.status,
            "processed_at": processed_at,
        }
        file_path = Path(record.filepath)
        if file_path.exists():
            try:
                entry.update(file_fingerprint(file_path))
            except OSError:
                pass
        if record.status == "error" and record.error:
            entry["error"] = record.error[:500]
        state_files[state_key] = entry

    state_path = save_pipeline_state(args.output_dir, state)
    log.info("State stage completed in %.2fs", time.perf_counter() - step6_t0)

    # --- Summary ---
    total_time = sum(r.conversion_time_s for r in records)
    log.info("=" * 60)
    log.info("PIPELINE COMPLETE")
    log.info(f"  PDFs discovered: {len(pdf_files)}")
    log.info(f"  Processed now:   {len(records)}")
    log.info(f"  Skipped cached:  {skipped_unchanged}")
    log.info(f"  Succeeded:       {len(success)}")
    log.info(f"  Failed:          {len(failed)}")
    log.info(f"  Vectors stored:  {count}")
    log.info(f"  Markdown dir:    {md_dir}")
    log.info(f"  Manifest:        {manifest_path}")
    log.info(f"  State:           {state_path}")
    log.info(f"  Total conv time: {total_time:.1f}s")
    log.info(f"  Total runtime:   {time.perf_counter() - overall_t0:.1f}s")
    if failed:
        log.warning("Failed files:")
        for r in failed:
            log.warning(f"  - {r.filename}: {(r.error or 'unknown')[:200]}")
