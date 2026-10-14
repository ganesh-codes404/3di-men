"""Orchestrates parse -> chunk -> index for one file or a whole directory
(section 4's user workflow). The single entry point a CLI, a future UI
upload, or a test calls -- document_parser.py and chunking.py are never
called directly from outside this module plus cli.py, so duplicate
detection and error reporting are enforced in exactly one place.

document_id is the SHA-256 of the raw file bytes (not a counter, not a
path) -- re-ingesting an identical file, even under a different filename or
from a different directory, is recognized as the same document (section 4:
"duplicate detection") rather than silently double-indexed.
"""

from __future__ import annotations

import hashlib
import shutil
from pathlib import Path
from typing import Callable

from knowledge.chunking import chunk_document
from knowledge.config import KnowledgeConfig
from knowledge.document_parser import SUPPORTED_SUFFIXES, DocumentParseError, parse_document
from knowledge.index import KnowledgeIndex
from knowledge.schemas import DocumentRecord, IngestionIssue, IngestionReport


def _file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            h.update(block)
    return h.hexdigest()


def add_document(
    index: KnowledgeIndex,
    path: Path,
    *,
    source_or_publisher: str | None = None,
    license_notes: str | None = None,
    config: KnowledgeConfig | None = None,
    source_id: str | None = None,
    source_url: str | None = None,
    source_revision: str | None = None,
) -> DocumentRecord:
    """Raises DocumentParseError (an unprocessable file) or ValueError (an
    exact duplicate already indexed) -- both are the CALLER's signal to
    report the problem, never swallowed here (section 4: "report the
    problem and let the user decide").

    `source_id`/`source_url`/`source_revision` are trailing-defaulted
    (None for the plain manual-ingestion case, unchanged from before) --
    set by knowledge/sources.py for a document acquired automatically, so
    its provenance is traceable and so sources.py's prune logic can later
    find and remove exactly this document without touching anything a
    user imported directly (source_id=None)."""
    config = config or index.config
    if not path.is_file():
        raise DocumentParseError(f"{path}: not a file")

    document_id = _file_hash(path)
    if index.has_document(document_id):
        existing = index.get_document(document_id)
        raise ValueError(f"{path}: duplicate of already-indexed document {existing.filename!r} ({document_id[:12]}...)")

    parsed = parse_document(path, config)  # raises DocumentParseError on anything unprocessable

    stored_path = config.documents_dir / f"{document_id}{path.suffix.lower()}"
    config.documents_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, stored_path)

    chunks = chunk_document(
        parsed, document_id=document_id, source_path=str(stored_path), config=config,
        source_id=source_id, source_url=source_url, source_revision=source_revision,
    )
    if not chunks:
        stored_path.unlink(missing_ok=True)
        raise DocumentParseError(f"{path}: parsed successfully but produced zero chunks")

    record = DocumentRecord(
        document_id=document_id,
        filename=path.name,
        source_path=str(path.resolve()),
        title=parsed.title,
        source_or_publisher=source_or_publisher,
        license_notes=license_notes,
        extraction_method=parsed.extraction_method,
        page_count=parsed.page_count,
        chunk_count=len(chunks),
        source_id=source_id, source_url=source_url, source_revision=source_revision,
    )
    index.add_document(record, chunks)
    return record


def add_directory(
    index: KnowledgeIndex,
    dir_path: Path,
    *,
    source_or_publisher: str | None = None,
    license_notes: str | None = None,
    config: KnowledgeConfig | None = None,
    recursive: bool = True,
    source_id: str | None = None,
    source_revision: str | None = None,
    url_for_path: Callable[[Path], str] | None = None,
) -> IngestionReport:
    """One file failing to parse never aborts the rest of the batch --
    every file gets its own attempt and its own place in the report.

    `url_for_path`, if given, maps each LOCAL file path to its real
    upstream URL (knowledge/sources.py supplies this for an acquired
    source, since a whole directory tree maps to many different URLs, one
    per file -- a single shared `source_url` wouldn't be accurate here the
    way it can be for a single `add_document` call)."""
    config = config or index.config
    if not dir_path.is_dir():
        raise DocumentParseError(f"{dir_path}: not a directory")

    pattern = "**/*" if recursive else "*"
    paths = sorted(p for p in dir_path.glob(pattern) if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES)

    report = IngestionReport()
    for path in paths:
        try:
            record = add_document(
                index, path, source_or_publisher=source_or_publisher, license_notes=license_notes, config=config,
                source_id=source_id, source_revision=source_revision,
                source_url=url_for_path(path) if url_for_path else None,
            )
            report.added.append(record)
        except ValueError as exc:
            if "duplicate" in str(exc):
                report.skipped_duplicates.append(str(path))
            else:
                report.failed.append(IngestionIssue(source_path=str(path), reason=str(exc)))
        except DocumentParseError as exc:
            report.failed.append(IngestionIssue(source_path=str(path), reason=str(exc)))
    return report


def remove_document(index: KnowledgeIndex, document_id: str, *, config: KnowledgeConfig | None = None) -> bool:
    config = config or index.config
    record = index.get_document(document_id)
    removed = index.remove_document(document_id)
    if removed and record is not None:
        for stored in config.documents_dir.glob(f"{document_id}.*"):
            stored.unlink(missing_ok=True)
    return removed
