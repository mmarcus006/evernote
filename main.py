from pathlib import Path

from enex_parser import parse_enex_dir


def extract_pdfs(enex_dir: str = "enex_files", output_dir: str = "output") -> None:
    out = Path(output_dir)
    out.mkdir(exist_ok=True)

    exports = parse_enex_dir(enex_dir)
    saved = 0
    skipped = 0

    for export in exports:
        src = Path(export.source_file or "unknown").stem
        subfolder = out / src
        subfolder.mkdir(exist_ok=True)
        used_names: dict[str, int] = {}

        print(f"\n{src}: {export.note_count} notes, {export.resource_count} resources")

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

        print(f"  -> {subfolder}/")

    print(f"\nDone: {saved} PDFs saved across {len(exports)} subfolders in {out}/")
    if skipped:
        print(f"  ({skipped} non-PDF resources skipped)")


if __name__ == "__main__":
    extract_pdfs()
