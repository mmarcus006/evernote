"""PDF file discovery and Google Drive integration."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

log = logging.getLogger(__name__)

SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]


# ---------------------------------------------------------------------------
# Local filesystem
# ---------------------------------------------------------------------------


def discover_pdfs(folder: Path) -> list[Path]:
    """Recursively find all PDF files under *folder*, sorted by name."""
    if not folder.exists():
        return []
    return sorted(folder.rglob("*.pdf"))


# ---------------------------------------------------------------------------
# Google Drive API
# ---------------------------------------------------------------------------


def authenticate_drive(credentials_file: Path, token_file: Path):
    """OAuth2 authentication with token caching."""
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow

    creds = None
    if token_file.exists():
        creds = Credentials.from_authorized_user_file(str(token_file), SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            if not credentials_file.exists():
                log.error(
                    f"Credentials file not found: {credentials_file}\n"
                    "  1. Go to Google Cloud Console -> APIs & Services -> Credentials\n"
                    "  2. Create OAuth 2.0 Client ID (Desktop app)\n"
                    "  3. Download JSON and save as credentials.json in project root"
                )
                sys.exit(1)
            flow = InstalledAppFlow.from_client_secrets_file(
                str(credentials_file), SCOPES
            )
            creds = flow.run_local_server(port=0)
        token_file.write_text(creds.to_json())
    return creds


def list_drive_pdfs(service, folder_id: str) -> list[dict]:
    """List all PDF files in a Google Drive folder (non-recursive)."""
    results = []
    page_token = None
    query = f"'{folder_id}' in parents and mimeType='application/pdf' and trashed=false"
    while True:
        response = (
            service.files()
            .list(
                q=query,
                spaces="drive",
                fields="nextPageToken, files(id, name, size, modifiedTime)",
                pageToken=page_token,
                pageSize=100,
            )
            .execute()
        )
        results.extend(response.get("files", []))
        page_token = response.get("nextPageToken")
        if not page_token:
            break
    return results


def download_pdf(service, file_id: str, file_name: str, cache_dir: Path) -> Path:
    """Download a PDF from Drive, skipping if already cached."""
    import io

    from googleapiclient.http import MediaIoBaseDownload

    dest = cache_dir / file_name
    if dest.exists():
        log.debug(f"  Cached: {file_name}")
        return dest
    request = service.files().get_media(fileId=file_id)
    with open(dest, "wb") as f:
        downloader = MediaIoBaseDownload(f, request)
        done = False
        while not done:
            _status, done = downloader.next_chunk()
    log.info(f"  Downloaded: {file_name}")
    return dest
