import logging
import time
from pathlib import Path

from enex_parser import parse_enex_dir

log = logging.getLogger(__name__)


def extract_pdfs(enex_dir: str = "enex_files", output_dir: str = "output") -> None:
    t0 = time.perf_counter()
    out = Path(output_dir)
    out.mkdir(exist_ok=True)

    exports = parse_enex_dir(enex_dir)
    saved = 0
    skipped = 0

    log.info("Found %s ENEX export files", len(exports))

    for export in exports:
        src = Path(export.source_file or "unknown").stem
        subfolder = out / src
        subfolder.mkdir(exist_ok=True)
        used_names: dict[str, int] = {}

        export_saved = 0

        log.info(
            "%s: notes=%s resources=%s",
            src,
            export.note_count,
            export.resource_count,
        )

        for note in export.notes:
            for resource in note.resources:
                if resource.mime != "application/pdf":
                    skipped += 1
                    continue

                # Build a unique filename
                base = resource.resource_attributes.file_name or f"{note.title}.pdf"
                if not base.lower().endswith(".pdf"):
                    base += ".pdf"

                # Deduplicate within this subfolder
                if base in used_names:
                    used_names[base] += 1
                    stem = base.rsplit(".", 1)[0]
                    base = f"{stem}_{used_names[base]}.pdf"
                else:
                    used_names[base] = 0

                resource.save(subfolder / base)
                saved += 1
                export_saved += 1

        log.info("  output=%s saved=%s", subfolder, export_saved)

    log.info(
        "Done: saved=%s subfolders=%s output=%s elapsed=%.2fs",
        saved,
        len(exports),
        out,
        time.perf_counter() - t0,
    )
    if skipped:
        log.info("Skipped non-PDF resources=%s", skipped)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    extract_pdfs()
