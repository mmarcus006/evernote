"""CLI entrypoint for the PDF -> Markdown -> NLP -> Vector DB pipeline.

Usage:
    python -m pipeline --folder-id <DRIVE_FOLDER_ID>
    python -m pipeline --local-dir ./output/Properties
    python -m pipeline --local-dir ./output/Properties --skip-nlp
    python -m pipeline --local-dir ./output/Properties --backend qdrant
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

log = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="PDF -> Markdown -> NLP -> Vector DB pipeline"
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--folder-id", help="Google Drive folder ID containing PDFs")
    src.add_argument(
        "--local-dir", type=Path, help="Local directory with PDFs (skip Drive)"
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
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Run the full pipeline."""
    from tqdm import tqdm

    from .chunking import create_chunker
    from .conversion import convert_single_pdf, create_ocr_converter
    from .nlp import init_nlp, run_nlp_analysis
    from .sources import (
        authenticate_drive,
        discover_pdfs,
        download_pdf,
        list_drive_pdfs,
    )
    from .utils import ensure_output_dirs, save_manifest
    from .vectorstores import (
        build_qdrant_collection,
        create_chroma_collection,
        insert_chunks,
    )

    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    # Create directories
    md_dir, db_dir = ensure_output_dirs(args.output_dir)
    nlp_dir = args.output_dir / "nlp"
    nlp_dir.mkdir(parents=True, exist_ok=True)

    # --- Step 1: Get PDF files ---
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

        pdf_files = []
        for f in tqdm(drive_files, desc="Downloading"):
            path = download_pdf(service, f["id"], f["name"], args.cache_dir)
            pdf_files.append(path)

    log.info(f"Total PDFs to process: {len(pdf_files)}")
    if not pdf_files:
        log.warning("No PDFs found. Exiting.")
        sys.exit(0)

    # --- Step 2: Convert PDFs to Markdown ---
    log.info("Initializing Docling OCR converter...")
    converter, ocr_engine = create_ocr_converter()
    docling_docs: dict[str, object] = {}
    records = []

    for pdf_path in tqdm(pdf_files, desc="Converting PDFs"):
        record, doc = convert_single_pdf(converter, pdf_path)
        records.append(record)
        if doc is not None:
            docling_docs[record.filename] = doc
            md_file = md_dir / f"{pdf_path.stem}.md"
            md_file.write_text(record.markdown, encoding="utf-8")

    success = [r for r in records if r.status == "success"]
    failed = [r for r in records if r.status == "error"]
    log.info(f"Conversion: {len(success)} succeeded, {len(failed)} failed")

    # --- Step 3: NLP Analysis ---
    nlp_results: dict[str, dict] = {}
    if not args.skip_nlp:
        log.info("Running NLP analysis...")
        init_nlp()
        nlp_results = run_nlp_analysis(records)
        for filename, analysis in nlp_results.items():
            nlp_file = nlp_dir / f"{Path(filename).stem}_nlp.json"
            with open(nlp_file, "w") as f:
                json.dump(analysis, f, indent=2, default=str)

    # --- Step 4: Chunk and store in vector DB ---
    log.info("Chunking and building vector database...")
    chunker = create_chunker()

    if args.backend == "qdrant":
        from .chunking import chunk_documents

        count = build_qdrant_collection(
            args.qdrant_path, chunker, records, docling_docs, nlp_results
        )
        log.info(f"Qdrant collection: {count} vectors stored")
    else:
        from .chunking import chunk_documents

        chunks = chunk_documents(chunker, records, docling_docs, nlp_results)
        _, collection = create_chroma_collection(db_dir)
        count = insert_chunks(collection, chunks)
        log.info(f"ChromaDB collection: {count} vectors stored")

    # --- Step 5: Save manifest ---
    manifest_path = save_manifest(args.output_dir, records, nlp_results)

    # --- Summary ---
    total_time = sum(r.conversion_time_s for r in records)
    log.info("=" * 60)
    log.info("PIPELINE COMPLETE")
    log.info(f"  PDFs processed:  {len(records)}")
    log.info(f"  Succeeded:       {len(success)}")
    log.info(f"  Failed:          {len(failed)}")
    log.info(f"  Vectors stored:  {count}")
    log.info(f"  Markdown dir:    {md_dir}")
    log.info(f"  Manifest:        {manifest_path}")
    log.info(f"  Total conv time: {total_time:.1f}s")
    if failed:
        log.warning("Failed files:")
        for r in failed:
            log.warning(f"  - {r.filename}: {(r.error or 'unknown')[:200]}")
