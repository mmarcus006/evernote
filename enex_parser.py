"""
Evernote .enex file parser.

Parses Evernote export files (.enex) into Python dataclasses.
Supports all documented elements from the evernote-export3/4 DTD including
notes, resources, tags, note-attributes, and resource-attributes.

Usage:
    from enex_parser import parse_enex, parse_enex_dir

    # Parse a single file
    export = parse_enex("path/to/file.enex")
    for note in export.notes:
        print(note.title, note.created)

    # Parse all .enex files in a directory
    exports = parse_enex_dir("path/to/enex_files/")
"""

from __future__ import annotations

import base64
import hashlib
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path


# ---------------------------------------------------------------------------
# Evernote datetime helpers
# ---------------------------------------------------------------------------

_EN_DT_RE = re.compile(r"^(\d{4})(\d{2})(\d{2})T(\d{2})(\d{2})(\d{2})Z$")


def parse_en_datetime(value: str | None) -> datetime | None:
  """Parse Evernote datetime string (``YYYYMMDDTHHMMSSz``) into a UTC datetime."""
  if not value:
    return None
  value = value.strip()
  m = _EN_DT_RE.match(value)
  if not m:
    return None
  return datetime(
    int(m.group(1)), int(m.group(2)), int(m.group(3)),
    int(m.group(4)), int(m.group(5)), int(m.group(6)),
    tzinfo=timezone.utc,
  )


# ---------------------------------------------------------------------------
# ENML → plain-text helper
# ---------------------------------------------------------------------------

class _ENMLStripper(HTMLParser):
  """Minimal HTML/ENML tag stripper that preserves readable text."""

  def __init__(self) -> None:
    super().__init__()
    self._pieces: list[str] = []
    self._block_tags = {
      "div", "p", "br", "h1", "h2", "h3", "h4", "h5", "h6",
      "li", "tr", "blockquote", "hr", "en-note",
    }

  def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
    if tag in self._block_tags:
      self._pieces.append("\n")
    if tag == "en-todo":
      checked = dict(attrs).get("checked", "false")
      self._pieces.append("[x] " if checked == "true" else "[ ] ")

  def handle_endtag(self, tag: str) -> None:
    if tag in self._block_tags:
      self._pieces.append("\n")

  def handle_data(self, data: str) -> None:
    self._pieces.append(data)

  def get_text(self) -> str:
    raw = "".join(self._pieces)
    # Collapse runs of blank lines
    return re.sub(r"\n{3,}", "\n\n", raw).strip()


def enml_to_text(enml: str | None) -> str:
  """Convert ENML/HTML content to plain text."""
  if not enml:
    return ""
  # Strip XML declaration and DOCTYPE if present
  enml = re.sub(r"<\?xml[^?]*\?>", "", enml)
  enml = re.sub(r"<!DOCTYPE[^>]*>", "", enml)
  stripper = _ENMLStripper()
  stripper.feed(enml)
  return stripper.get_text()


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ResourceAttributes:
  """Metadata about an attached resource."""
  source_url: str | None = None
  timestamp: datetime | None = None
  latitude: float | None = None
  longitude: float | None = None
  altitude: float | None = None
  camera_make: str | None = None
  camera_model: str | None = None
  reco_type: str | None = None
  file_name: str | None = None
  attachment: bool | None = None


@dataclass
class Resource:
  """A file attachment (image, PDF, audio, etc.) embedded in a note."""
  data_b64: str | None = None
  mime: str | None = None
  width: int | None = None
  height: int | None = None
  duration: int | None = None
  recognition: str | None = None
  resource_attributes: ResourceAttributes = field(default_factory=ResourceAttributes)
  alternate_data_b64: str | None = None

  @property
  def data(self) -> bytes | None:
    """Decode and return the raw resource bytes."""
    if not self.data_b64:
      return None
    return base64.b64decode(self.data_b64)

  @property
  def md5(self) -> str | None:
    """Return the hex MD5 hash of the resource data (used by en-media references)."""
    raw = self.data
    if raw is None:
      return None
    return hashlib.md5(raw).hexdigest()

  @property
  def size(self) -> int | None:
    """Approximate size in bytes (from base64 length, without decoding)."""
    if not self.data_b64:
      return None
    # base64 encodes 3 bytes into 4 chars; strip whitespace first
    clean = self.data_b64.replace("\n", "").replace("\r", "").replace(" ", "")
    padding = clean.count("=")
    return (len(clean) * 3 // 4) - padding

  def save(self, path: str | Path) -> Path:
    """Write the decoded resource to *path* and return it."""
    p = Path(path)
    raw = self.data
    if raw is None:
      raise ValueError("Resource has no data to save")
    p.write_bytes(raw)
    return p


@dataclass
class NoteAttributes:
  """Optional metadata associated with a note."""
  subject_date: datetime | None = None
  latitude: float | None = None
  longitude: float | None = None
  altitude: float | None = None
  author: str | None = None
  source: str | None = None
  source_url: str | None = None
  source_application: str | None = None
  reminder_time: datetime | None = None
  reminder_order: str | None = None
  reminder_done_time: datetime | None = None
  content_class: str | None = None


@dataclass
class Note:
  """A single Evernote note."""
  title: str = ""
  content: str | None = None
  created: datetime | None = None
  updated: datetime | None = None
  tags: list[str] = field(default_factory=list)
  note_attributes: NoteAttributes = field(default_factory=NoteAttributes)
  resources: list[Resource] = field(default_factory=list)

  @property
  def plain_text(self) -> str:
    """Return the note content as plain text (ENML tags stripped)."""
    return enml_to_text(self.content)

  def resource_by_hash(self, md5_hash: str) -> Resource | None:
    """Find a resource by its MD5 hash (as referenced in ``<en-media>``)."""
    for r in self.resources:
      if r.md5 == md5_hash:
        return r
    return None


@dataclass
class EnexExport:
  """Top-level container representing a parsed ``.enex`` file."""
  export_date: datetime | None = None
  application: str | None = None
  version: str | None = None
  source_file: str | None = None
  notes: list[Note] = field(default_factory=list)

  @property
  def note_count(self) -> int:
    return len(self.notes)

  @property
  def resource_count(self) -> int:
    return sum(len(n.resources) for n in self.notes)

  @property
  def all_tags(self) -> set[str]:
    """Return every unique tag across all notes."""
    tags: set[str] = set()
    for n in self.notes:
      tags.update(n.tags)
    return tags


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def _text(elem: ET.Element | None) -> str | None:
  """Return stripped text content of *elem*, or None."""
  if elem is None:
    return None
  t = elem.text
  return t.strip() if t else None


def _float(elem: ET.Element | None) -> float | None:
  t = _text(elem)
  if t is None:
    return None
  try:
    return float(t)
  except ValueError:
    return None


def _int(elem: ET.Element | None) -> int | None:
  t = _text(elem)
  if t is None:
    return None
  try:
    return int(t)
  except ValueError:
    return None


def _bool(elem: ET.Element | None) -> bool | None:
  t = _text(elem)
  if t is None:
    return None
  return t.lower() in ("true", "1", "yes")


def _parse_resource_attributes(elem: ET.Element) -> ResourceAttributes:
  return ResourceAttributes(
    source_url=_text(elem.find("source-url")),
    timestamp=parse_en_datetime(_text(elem.find("timestamp"))),
    latitude=_float(elem.find("latitude")),
    longitude=_float(elem.find("longitude")),
    altitude=_float(elem.find("altitude")),
    camera_make=_text(elem.find("camera-make")),
    camera_model=_text(elem.find("camera-model")),
    reco_type=_text(elem.find("reco-type")),
    file_name=_text(elem.find("file-name")),
    attachment=_bool(elem.find("attachment")),
  )


def _parse_resource(elem: ET.Element) -> Resource:
  data_elem = elem.find("data")
  data_b64 = None
  if data_elem is not None and data_elem.text:
    data_b64 = data_elem.text.strip()

  alt_elem = elem.find("alternate-data")
  alt_b64 = None
  if alt_elem is not None and alt_elem.text:
    alt_b64 = alt_elem.text.strip()

  ra_elem = elem.find("resource-attributes")
  ra = _parse_resource_attributes(ra_elem) if ra_elem is not None else ResourceAttributes()

  reco_elem = elem.find("recognition")
  reco = None
  if reco_elem is not None:
    # recognition can contain nested XML; grab raw text or serialize
    reco = ET.tostring(reco_elem, encoding="unicode") if len(reco_elem) else _text(reco_elem)

  return Resource(
    data_b64=data_b64,
    mime=_text(elem.find("mime")),
    width=_int(elem.find("width")),
    height=_int(elem.find("height")),
    duration=_int(elem.find("duration")),
    recognition=reco,
    resource_attributes=ra,
    alternate_data_b64=alt_b64,
  )


def _parse_note_attributes(elem: ET.Element) -> NoteAttributes:
  return NoteAttributes(
    subject_date=parse_en_datetime(_text(elem.find("subject-date"))),
    latitude=_float(elem.find("latitude")),
    longitude=_float(elem.find("longitude")),
    altitude=_float(elem.find("altitude")),
    author=_text(elem.find("author")),
    source=_text(elem.find("source")),
    source_url=_text(elem.find("source-url")),
    source_application=_text(elem.find("source-application")),
    reminder_time=parse_en_datetime(_text(elem.find("reminder-time"))),
    reminder_order=_text(elem.find("reminder-order")),
    reminder_done_time=parse_en_datetime(_text(elem.find("reminder-done-time"))),
    content_class=_text(elem.find("content-class")),
  )


def _parse_note(elem: ET.Element) -> Note:
  tags = [tag.text.strip() for tag in elem.findall("tag") if tag.text]
  na_elem = elem.find("note-attributes")
  na = _parse_note_attributes(na_elem) if na_elem is not None else NoteAttributes()
  resources = [_parse_resource(r) for r in elem.findall("resource")]

  return Note(
    title=_text(elem.find("title")) or "",
    content=_text(elem.find("content")),
    created=parse_en_datetime(_text(elem.find("created"))),
    updated=parse_en_datetime(_text(elem.find("updated"))),
    tags=tags,
    note_attributes=na,
    resources=resources,
  )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def parse_enex(path: str | Path) -> EnexExport:
  """Parse a single ``.enex`` file and return an :class:`EnexExport`.

  Uses ``iterparse`` to keep memory usage low for large exports — each
  ``<note>`` element is parsed and then discarded from the tree.
  """
  path = Path(path)
  export = EnexExport(source_file=str(path))
  context = ET.iterparse(str(path), events=("start", "end"))

  root = None
  for event, elem in context:
    if event == "start" and root is None:
      root = elem
      export.export_date = parse_en_datetime(elem.get("export-date"))
      export.application = elem.get("application")
      export.version = elem.get("version")

    if event == "end" and elem.tag == "note":
      export.notes.append(_parse_note(elem))
      # Free memory — the note subtree is no longer needed
      root.remove(elem)  # type: ignore[union-attr]

  return export


def parse_enex_dir(directory: str | Path) -> list[EnexExport]:
  """Parse every ``.enex`` file in *directory* and return a list of exports."""
  directory = Path(directory)
  exports: list[EnexExport] = []
  for enex_path in sorted(directory.glob("*.enex")):
    exports.append(parse_enex(enex_path))
  return exports


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _cli() -> None:
  """Quick summary when run directly: ``python enex_parser.py [path]``."""
  import sys

  target = sys.argv[1] if len(sys.argv) > 1 else "enex_files"
  target_path = Path(target)

  if target_path.is_file():
    exports = [parse_enex(target_path)]
  elif target_path.is_dir():
    exports = parse_enex_dir(target_path)
  else:
    print(f"Error: {target} is not a file or directory", file=sys.stderr)
    sys.exit(1)

  total_notes = 0
  total_resources = 0

  for export in exports:
    total_notes += export.note_count
    total_resources += export.resource_count
    print(f"\n{'─' * 60}")
    print(f"  File:       {export.source_file}")
    print(f"  App:        {export.application} v{export.version}")
    print(f"  Exported:   {export.export_date}")
    print(f"  Notes:      {export.note_count}")
    print(f"  Resources:  {export.resource_count}")
    tags = export.all_tags
    if tags:
      print(f"  Tags:       {', '.join(sorted(tags))}")

    print()
    for i, note in enumerate(export.notes[:5], 1):
      res_info = f"  [{len(note.resources)} attachment(s)]" if note.resources else ""
      tag_info = f"  tags={note.tags}" if note.tags else ""
      print(f"    {i}. {note.title}{res_info}{tag_info}")
      if note.note_attributes.author:
        print(f"       author: {note.note_attributes.author}")
      print(f"       created: {note.created}  updated: {note.updated}")
      text_preview = note.plain_text[:120].replace("\n", " ")
      if text_preview:
        print(f"       text: {text_preview}...")
      for r in note.resources[:2]:
        print(f"       resource: {r.resource_attributes.file_name} ({r.mime}, ~{r.size:,} bytes)")

    if export.note_count > 5:
      print(f"    ... and {export.note_count - 5} more notes")

  print(f"\n{'═' * 60}")
  print(f"  TOTAL: {total_notes} notes, {total_resources} resources across {len(exports)} file(s)")
  print(f"{'═' * 60}\n")


if __name__ == "__main__":
  _cli()
