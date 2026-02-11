"""CLI shim -- delegates to pipeline.cli.main().

Usage:
    python pdf_pipeline.py --local-dir ./output/Properties
    python pdf_pipeline.py --folder-id <DRIVE_FOLDER_ID>
"""

from pipeline.cli import main

if __name__ == "__main__":
    main()
