"""Typed records for the knowledge subsystem. Plain dataclasses, no ORM --
matches this project's existing convention (see llm_decompose.py's
ActionablePoint/DecomposeResult, sandbox.py's ProgramExecutionResult).

Keeps 3 things explicitly separate, per this subsystem's own design
constraint: (1) DocumentRecord/Chunk -- knowledge EXTRACTED from a document
the user supplied, with provenance; (2) KNOWLEDGE_DOMAINS/CONCEPT_TYPES --
engineering taxonomy authored and validated in this file, not inferred from
any document; (3) RetrievalResult -- what retrieval.py hands back for a
given query, always traceable to a real Chunk, never synthesized text.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Literal

# Section 6's taxonomy -- a closed, authored vocabulary (not inferred from
# documents) so tagging is a lookup, never a fabrication. Deliberately only
# the domains section 6 says to prioritize first (solid modeling, parametric
# CAD, mechanical design, assembly reasoning); product-modeling/artistry and
# manufacturing are included because the keyword lists cost nothing to
# define now, but are not specifically prioritized in retrieval ranking.
KNOWLEDGE_DOMAINS = (
    "solid_modeling",
    "parametric_cad",
    "mechanical_design",
    "product_modeling",
    "assembly_reasoning",
    "manufacturing",
)

ConceptType = Literal[
    "definition",
    "design_principle",
    "geometric_operation",
    "modeling_procedure",
    "constraint",
    "formula",
    "failure_mode",
    "troubleshooting_procedure",
    "manufacturing_rule",
    "example",
    "general",  # the honest fallback -- most prose doesn't cleanly fit a
                # narrower type, and guessing one anyway would be exactly
                # the kind of conservative-inference violation section 5
                # warns against ("infer metadata conservatively").
]

ExtractionMethod = Literal["pdf_text", "pdf_ocr", "markdown", "plain_text", "html", "rst"]


@dataclass
class DocumentRecord:
    """One ingested source file's provenance -- everything section 4 says
    must be preserved per document. `document_id` is a stable content hash
    (not a counter), so re-ingesting the same file is recognized as a
    duplicate rather than creating a second copy (see ingestion.py)."""

    document_id: str  # sha256 of the raw file bytes, hex
    filename: str
    source_path: str  # absolute path at ingestion time, for user reference only
    title: str | None
    source_or_publisher: str | None
    license_notes: str | None
    extraction_method: ExtractionMethod
    page_count: int | None
    chunk_count: int
    imported_at: float = field(default_factory=time.time)
    # Trailing-defaulted (matches this project's established convention --
    # see llm_decompose.py's ActionablePoint.id/relationship -- so every
    # existing manual-ingestion call site, test, and pickled index keeps
    # working unchanged). None for a document the user ingested directly
    # (knowledge.cli's `add`/`add-dir`); set for one acquired automatically
    # via knowledge.sources -- this is the field sources.py's prune logic
    # uses to find and remove ONLY that source's stale documents, never a
    # user's own imported ones (sources.py's own explicit guarantee).
    source_id: str | None = None
    source_url: str | None = None  # the canonical page/file URL, for citation
    source_revision: str | None = None  # the git commit (or doc version) this copy came from


@dataclass
class Chunk:
    """One retrievable unit of text. `topic`/`concept_type`/`domains` are
    always INFERRED (never present in the source document itself), so they
    are kept visibly separate from provenance fields that come straight
    from the document (section/page/title)."""

    chunk_id: str  # f"{document_id}:{chunk_index}" -- stable, derivable, no counter state needed
    document_id: str
    chunk_index: int
    text: str
    title: str | None  # the document's own title, carried onto every chunk for display
    section: str | None  # nearest enclosing heading, when the format has one
    page_start: int | None
    page_end: int | None
    source_path: str
    content_hash: str  # sha256 of `text` -- used for near-duplicate-chunk detection
    concept_type: ConceptType = "general"
    domains: tuple[str, ...] = ()  # subset of KNOWLEDGE_DOMAINS, possibly empty
    # Mirrors DocumentRecord's own trailing-defaulted source_id/source_url/
    # source_revision -- carried onto the CHUNK too (not just the document)
    # so retrieval.format_context_block can cite a real, clickable URL
    # directly, not just a title/page. None for a manually-ingested document.
    source_id: str | None = None
    source_url: str | None = None
    source_revision: str | None = None


@dataclass
class RetrievalResult:
    """What retrieval.py hands the caller: a real Chunk plus how it was
    found, never a rewritten/summarized version of the text (section 9:
    "concise passages and structured rules, not entire textbooks" means
    SELECTING less, not rewriting what's selected)."""

    chunk: Chunk
    score: float
    matched_terms: tuple[str, ...] = ()  # the query terms that actually overlapped this chunk's text


@dataclass
class IngestionIssue:
    source_path: str
    reason: str


@dataclass
class IngestionReport:
    """Returned by ingestion.py for one add_document/add_directory call --
    section 4's "see ingestion status and failures" requirement. A failure
    here is always reported, never silently dropped (section 4: "If a
    document cannot be processed reliably, report the problem and let the
    user decide")."""

    added: list[DocumentRecord] = field(default_factory=list)
    skipped_duplicates: list[str] = field(default_factory=list)  # source_paths
    failed: list[IngestionIssue] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Automated documentation acquisition (knowledge/sources.py)
# ---------------------------------------------------------------------------

LicenseStatus = Literal["approved", "needs_review", "rejected", "unverified"]
AcquisitionMethod = Literal["git_sparse_checkout", "http_fetch", "local"]
SourceSyncStatus = Literal["not_synced", "synced", "sync_failed"]
SourceIngestStatus = Literal["not_ingested", "ingested", "ingest_failed", "partially_ingested"]


@dataclass
class SourceEntry:
    """One row of the source manifest (section 3). A source only ever
    acquires content when `license_status == "approved"` -- `sync()`
    refuses (not silently skips) any other status, since an unverified or
    flagged license is exactly the case section 1 says to "skip or flag,"
    never proceed past on a default/implicit basis."""

    source_id: str
    title: str
    canonical_url: str  # the human-facing docs site, for citation
    repo_url: str | None  # the git/API endpoint actually fetched from
    license: str  # the SPDX-ish name, e.g. "CC0-1.0", "Apache-2.0"
    license_status: LicenseStatus
    license_note: str  # WHY it's approved/flagged -- never a bare status with no reasoning
    acquisition_method: AcquisitionMethod
    include_paths: tuple[str, ...]  # repo-relative glob patterns or explicit paths
    exclude_paths: tuple[str, ...] = ()
    branch: str | None = None  # git sources only
    revision: str | None = None  # last-synced commit hash (git) or a content snapshot id (http)
    last_synced_at: float | None = None
    last_ingested_at: float | None = None
    sync_status: SourceSyncStatus = "not_synced"
    ingest_status: SourceIngestStatus = "not_ingested"
    file_count: int = 0
    error: str | None = None


@dataclass
class SourceSyncResult:
    source_id: str
    ok: bool
    files_fetched: int = 0
    files_unchanged: int = 0
    revision: str | None = None
    error: str | None = None


@dataclass
class SourceIngestResult:
    source_id: str
    report: IngestionReport = field(default_factory=IngestionReport)
    pruned_stale_documents: list[str] = field(default_factory=list)  # document_ids removed
