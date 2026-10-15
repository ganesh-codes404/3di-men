"""Automated documentation acquisition for the knowledge subsystem.

Extends knowledge/, does not replace it: this module's only job is getting
real files from an approved upstream source onto local disk (under
KnowledgeConfig.store_dir, see `_raw_dir`) and handing them to the SAME
ingestion.py/chunking.py/embeddings.py/index.py pipeline every manually
ingested document already goes through -- there is exactly one ingestion
path in this project, this module just gives it a second way to find
files besides a user pointing the CLI at a local path.

DEFAULT_SOURCES below is a small, hand-verified registry, not a crawler
output. Every entry's `license`/`license_status`/`license_note` reflects an
ACTUAL check performed against the live repository during this work (the
repo's own LICENSE file or README license statement, read directly, not
assumed from the project's reputation) -- see each entry's license_note for
what was actually checked and when a license is unclear, scoped to
documentation content specifically, not just the surrounding software's
license (sections 1 and the mission's own explicit "software licenses do
not automatically apply to documentation" warning).

A source whose license_status is not "approved" is never synced by
sync_source()/sync_all() -- not filtered out quietly, but refused with a
clear, specific reason, so a reviewer always sees exactly why a source
didn't come in.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path

from knowledge.config import KnowledgeConfig
from knowledge.index import KnowledgeIndex
from knowledge.ingestion import add_directory
from knowledge.schemas import IngestionIssue, IngestionReport, SourceEntry, SourceIngestResult, SourceSyncResult


class SourceAcquisitionError(Exception):
    """A sync attempt failed for a reason worth surfacing specifically
    (network failure, git not available, license not approved, ...) --
    never silently degraded to "0 files synced" with no explanation."""


# ---------------------------------------------------------------------------
# The verified source registry
# ---------------------------------------------------------------------------
# Content-format note: every include_paths entry below targets the
# documentation's own SOURCE format (Markdown or reStructuredText), not
# rendered HTML -- the mission's own stated preference -- since all 4
# approved sources publish their docs' source files directly in their git
# repository rather than only as a built HTML site.

DEFAULT_SOURCES: tuple[SourceEntry, ...] = (
    SourceEntry(
        source_id="freecad_docs",
        title="FreeCAD Documentation Wiki",
        canonical_url="https://wiki.freecad.org/",
        repo_url="https://github.com/FreeCAD/FreeCAD-documentation",
        license="CC0-1.0",
        license_status="approved",
        license_note=(
            "Verified directly against the FreeCAD-documentation GitHub repo (not the live wiki.freecad.org "
            "site, which blocks automated fetches): the repo's own LICENSE file and GitHub-displayed license "
            "badge both state CC0-1.0 (public domain dedication) for the mirrored wiki content under wiki/."
        ),
        acquisition_method="git_sparse_checkout",
        # A curated subset (not the whole ~1000+-page wiki) directly matching
        # the mission's own stated FreeCAD topics: Part Design, sketches,
        # constraints, parametric features, booleans, fillets, chamfers,
        # assemblies, Python scripting.
        include_paths=(
            "wiki/Part_Design_Workbench.md",
            "wiki/Part_Workbench.md",
            "wiki/Sketcher_Workbench.md",
            "wiki/PartDesign_Boolean.md",
            "wiki/Part_Boolean.md",
            "wiki/Part_BooleanFragments.md",
            "wiki/PartDesign_Fillet.md",
            "wiki/Part_Fillet.md",
            "wiki/PartDesign_Chamfer.md",
            "wiki/Part_Chamfer.md",
            "wiki/PartDesign_Pad.md",
            "wiki/PartDesign_Pocket.md",
            "wiki/PartDesign_Revolution.md",
            "wiki/PartDesign_AdditiveLoft.md",
            "wiki/PartDesign_Sketch.md",
            "wiki/Sketcher_ConstrainDistance.md",
            "wiki/Assembly_workbench.md",
            "wiki/Introduction_to_API.md",
            "wiki/FreeCAD_Scripting_Basics.md",
        ),
        branch="main",
    ),
    SourceEntry(
        source_id="cadquery_docs",
        title="CadQuery Documentation",
        canonical_url="https://cadquery.readthedocs.io/",
        repo_url="https://github.com/CadQuery/cadquery",
        license="Apache-2.0",
        license_status="approved",
        license_note=(
            "Verified via the CadQuery GitHub repo's README ('licensed under the terms of the Apache Public "
            "License, version 2.0') and GitHub's own detected license badge; the doc/ directory is part of the "
            "same repo under the same license, no separate docs-specific license file exists."
        ),
        acquisition_method="git_sparse_checkout",
        include_paths=(
            "doc/intro.rst", "doc/primer.rst", "doc/quickstart.rst", "doc/workplane.rst",
            "doc/sketch.rst", "doc/assy.rst", "doc/designprinciples.rst", "doc/selectors.rst",
            "doc/classreference.rst",
        ),
        branch="master",
    ),
    SourceEntry(
        source_id="build123d_docs",
        title="build123d Documentation",
        canonical_url="https://build123d.readthedocs.io/",
        repo_url="https://github.com/gumyr/build123d",
        license="Apache-2.0",
        license_status="approved",
        license_note=(
            "Verified via GitHub's license API for gumyr/build123d, which reports 'Apache License 2.0' "
            "(spdx_id Apache-2.0) as the repository license; docs/ is part of the same repository."
        ),
        acquisition_method="git_sparse_checkout",
        include_paths=(
            "docs/introduction.rst", "docs/key_concepts.rst", "docs/builders.rst",
            "docs/build_part.rst", "docs/build_sketch.rst", "docs/build_line.rst",
            "docs/operations.rst", "docs/objects.rst", "docs/assemblies.rst",
            "docs/joints.rst", "docs/topology_selection.rst",
        ),
        branch="dev",
    ),
    SourceEntry(
        source_id="blender_manual",
        title="Blender Manual",
        canonical_url="https://docs.blender.org/manual/en/latest/",
        repo_url="https://projects.blender.org/blender/blender-manual",
        license="CC-BY-SA-4.0",
        license_status="approved",
        license_note=(
            "Verified by fetching the repository's actual LICENSE file content via the Gitea content API: it is "
            "the full Creative Commons Attribution-ShareAlike 4.0 International legal text. Attribution + "
            "share-alike are satisfied by this subsystem's own citation (source_url/source_revision on every "
            "retrieved chunk, see schemas.Chunk) -- content is indexed locally for retrieval grounding, not "
            "redistributed or republished."
        ),
        # NOT git_sparse_checkout: the repo is ~876MB (mostly screenshot
        # images unrelated to this subsystem's text-retrieval use), and
        # Blender's infra (projects.blender.org, Gitea) doesn't support git
        # partial-clone filtering the way GitHub does -- fetching individual
        # .rst files directly via the Gitea raw-content HTTP API avoids
        # downloading any of that image weight at all (the mission's own
        # "do not clone enormous repositories when only a documentation
        # directory is needed" rule, applied via a different mechanism).
        acquisition_method="http_fetch",
        include_paths=(
            "manual/modeling/introduction.rst",
            "manual/modeling/meshes/primitives.rst",
            "manual/modeling/meshes/selecting/introduction.rst",
            "manual/modeling/modifiers/generate/booleans.rst",
            "manual/modeling/modifiers/generate/bevel.rst",
            "manual/modeling/modifiers/generate/solidify.rst",
            "manual/scene_layout/object/properties/transforms.rst",
        ),
        branch="main",
    ),
    SourceEntry(
        source_id="occt_docs",
        title="Open CASCADE Technology (OCCT) Overview Documentation",
        canonical_url="https://dev.opencascade.org/doc/overview/html/",
        repo_url="https://github.com/Open-Cascade-SAS/OCCT",
        license="LGPL-2.1-or-later (software license; see note)",
        license_status="needs_review",
        license_note=(
            "FLAGGED, not approved: the repository's stated license (LGPL-2.1, per its README, with a separate "
            "OCCT_LGPL_EXCEPTION.txt) is a SOFTWARE copyleft license. No separate, explicit license statement "
            "for the dox/ documentation content itself was found during this check. The mission's own guidance "
            "is explicit that a software license does not automatically extend to documentation -- this source "
            "is kept in the manifest for visibility but sync_source() refuses to acquire it until a human "
            "confirms (e.g. by asking the OCCT maintainers, or finding an explicit docs license statement) that "
            "dox/ content may be locally indexed this way."
        ),
        acquisition_method="git_sparse_checkout",
        include_paths=("dox/user_guides/modeling_algos/modeling_algos.md", "dox/user_guides/modeling_data/modeling_data.md"),
        branch="master",
    ),
    SourceEntry(
        source_id="wikipedia_engineering",
        title="Wikipedia: Mechanical/Product Design Reference Articles",
        canonical_url="https://en.wikipedia.org/",
        repo_url="https://en.wikipedia.org/w/api.php",
        license="CC-BY-SA-4.0",
        license_status="approved",
        license_note=(
            "Verified directly against Wikipedia's own API (action=query&meta=siteinfo&siprop=rightsinfo), "
            "which reports 'Creative Commons Attribution-Share Alike 4.0' as the content license -- the "
            "standard, well-documented Wikipedia content license, not assumed from reputation."
        ),
        # Added specifically to close a gap this session's own real retrieval
        # testing found (MISSION_STATUS.md Addendum 1, section 5): the other
        # 4 approved sources are CAD-TOOL API/operation documentation, which
        # serves tool-operation-phrased prompts well but product-shape
        # prompts (a water bottle, a container) only weakly. These articles
        # are product/mechanical-DESIGN-concept content specifically, not
        # another tool manual. `include_paths` holds Wikipedia ARTICLE
        # TITLES here (not repo-relative file paths) -- the field's meaning
        # is "the specific items to fetch from this source," which this
        # acquisition method fulfills differently than a git checkout does.
        acquisition_method="wikipedia_api",
        include_paths=(
            "Fillet (mechanics)", "Chamfer", "Screw thread", "Flange", "Fastener",
            "Boss (engineering)", "Bottle", "Design for manufacturability", "Gear",
        ),
        branch=None,
    ),
)

_GITHUB_HOSTS = ("github.com",)


def _raw_dir(config: KnowledgeConfig, source_id: str) -> Path:
    return config.store_dir / "sources" / "raw" / source_id


def _manifest_path(config: KnowledgeConfig) -> Path:
    return config.store_dir / "sources" / "manifest.json"


# ---------------------------------------------------------------------------
# Manifest persistence
# ---------------------------------------------------------------------------

class SourceManifest:
    """Persisted (JSON, human-readable -- section 3's "machine-readable
    source manifest") registry of SourceEntry rows. Seeded from
    DEFAULT_SOURCES the first time it's loaded for a given store_dir, then
    mutated in place as sync/ingest commands run -- the manifest file IS
    the record of what's been acquired, not a cache of it."""

    def __init__(self, config: KnowledgeConfig, entries: dict[str, SourceEntry]):
        self.config = config
        self.entries = entries

    @staticmethod
    def load(config: KnowledgeConfig | None = None) -> "SourceManifest":
        config = config or KnowledgeConfig()
        path = _manifest_path(config)
        if not path.exists():
            entries = {s.source_id: s for s in DEFAULT_SOURCES}
            return SourceManifest(config, entries)
        raw = json.loads(path.read_text())
        entries = {sid: SourceEntry(**fields) for sid, fields in raw.items()}
        # Any NEW default source added to this file since the manifest was
        # last saved appears too (so `discover`/`sync` always sees the full
        # current candidate list), without clobbering an existing entry's
        # already-recorded sync/ingest state.
        for default in DEFAULT_SOURCES:
            entries.setdefault(default.source_id, default)
        return SourceManifest(config, entries)

    def save(self) -> None:
        path = _manifest_path(self.config)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({sid: asdict(e) for sid, e in self.entries.items()}, indent=2, sort_keys=True))

    def get(self, source_id: str) -> SourceEntry | None:
        return self.entries.get(source_id)

    def list(self) -> list[SourceEntry]:
        return sorted(self.entries.values(), key=lambda e: e.source_id)


# ---------------------------------------------------------------------------
# Validation (network-light: reachability + recorded license, no download)
# ---------------------------------------------------------------------------

def validate_source(entry: SourceEntry, timeout: float = 10.0) -> tuple[bool, str]:
    """Checks the source's URL is actually reachable right now -- a cheap
    HEAD-equivalent check, not a download. Returns (ok, message). Does NOT
    re-derive the license (that was a one-time human-verified check
    recorded in license_note) -- it reports what's already recorded."""
    url = entry.canonical_url
    try:
        req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "3di-men-knowledge-sync/1.0"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            reachable = 200 <= resp.status < 400
    except Exception:
        # Some docs sites (e.g. Anubis-protected wikis) reject HEAD/bot
        # traffic even when genuinely reachable -- fall back to the repo
        # URL, which is what sync_source() actually fetches from anyway.
        try:
            req = urllib.request.Request(
                entry.repo_url or url, method="HEAD", headers={"User-Agent": "3di-men-knowledge-sync/1.0"}
            )
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                reachable = 200 <= resp.status < 400
        except Exception as exc2:
            return False, f"unreachable: {type(exc2).__name__}: {exc2}"
    status_msg = f"license={entry.license} status={entry.license_status}"
    return reachable, (f"reachable; {status_msg}" if reachable else f"unreachable; {status_msg}")


# ---------------------------------------------------------------------------
# Acquisition
# ---------------------------------------------------------------------------

def _git_sparse_checkout(entry: SourceEntry, dest: Path) -> str:
    """Partial clone (--filter=blob:none) + sparse-checkout limited to
    include_paths -- fetches git history/tree metadata but only the blob
    content for the files actually selected, never the whole repository
    (the mission's own "shallow clones or sparse checkouts" + "do not
    clone enormous repositories" guidance). Returns the checked-out commit
    hash. Raises SourceAcquisitionError with the real git stderr on any
    failure -- never silently returns an empty/partial checkout as success."""
    if shutil.which("git") is None:
        raise SourceAcquisitionError("git is not available on PATH")
    if dest.exists():
        shutil.rmtree(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)

    clone = subprocess.run(
        ["git", "clone", "--filter=blob:none", "--sparse", "--depth", "1",
         "--branch", entry.branch or "main", "--no-tags", entry.repo_url, str(dest)],
        capture_output=True, text=True, timeout=180,
    )
    if clone.returncode != 0:
        raise SourceAcquisitionError(f"git clone of {entry.repo_url} failed: {clone.stderr.strip()[-1000:]}")

    # sparse-checkout set takes directory CONES by default; our
    # include_paths are specific files, so --no-cone (plain gitignore-style
    # patterns) lets us list exact files instead of whole directories.
    sparse = subprocess.run(
        ["git", "sparse-checkout", "set", "--no-cone", *entry.include_paths],
        cwd=str(dest), capture_output=True, text=True, timeout=60,
    )
    if sparse.returncode != 0:
        raise SourceAcquisitionError(f"git sparse-checkout for {entry.source_id} failed: {sparse.stderr.strip()[-1000:]}")

    rev = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(dest), capture_output=True, text=True, timeout=30)
    if rev.returncode != 0:
        raise SourceAcquisitionError(f"git rev-parse for {entry.source_id} failed: {rev.stderr.strip()}")
    return rev.stdout.strip()


def _http_fetch_files(entry: SourceEntry, dest: Path, config: KnowledgeConfig) -> tuple[int, int, str | None]:
    """Fetches each include_paths file individually over HTTP (Gitea's
    raw-content API for blender_manual today) -- no git clone at all, so
    none of the repository's non-documentation weight is ever downloaded.
    Returns (files_fetched, files_unchanged, revision)."""
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True, exist_ok=True)

    # Resolve the branch's current commit once, so every file in this sync
    # is attributed to the same revision (even though each is fetched
    # independently) -- matches the git path's "one revision per sync" shape.
    parsed_repo_url = urllib.parse.urlsplit(entry.repo_url.rstrip("/"))
    owner_repo = "/".join(parsed_repo_url.path.strip("/").split("/")[-2:])
    api_base = f"{parsed_repo_url.scheme}://{parsed_repo_url.netloc}/api/v1/repos/{owner_repo}"
    revision = None
    try:
        with urllib.request.urlopen(
            urllib.request.Request(f"{api_base}/branches/{entry.branch or 'main'}",
                                    headers={"User-Agent": "3di-men-knowledge-sync/1.0"}),
            timeout=15,
        ) as resp:
            revision = json.loads(resp.read())["commit"]["id"]
    except Exception:
        revision = None  # non-fatal -- files still fetch, just without a verified single revision id

    fetched = 0
    for rel_path in entry.include_paths:
        # Gitea's raw-file content is served from the repo's own WEB path
        # (.../<owner>/<repo>/raw/branch/<branch>/<path>), not under
        # /api/v1/repos/ -- confirmed directly against this source, not
        # assumed from generic Gitea API docs (api_base above is only used
        # for the branch/commit lookup, a genuine API endpoint).
        url = f"{entry.repo_url.rstrip('/')}/raw/branch/{entry.branch or 'main'}/{rel_path}"
        local_path = dest / rel_path
        local_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with urllib.request.urlopen(
                urllib.request.Request(url, headers={"User-Agent": "3di-men-knowledge-sync/1.0"}), timeout=20
            ) as resp:
                local_path.write_bytes(resp.read())
            fetched += 1
        except urllib.error.HTTPError as exc:
            raise SourceAcquisitionError(f"fetching {url} failed: HTTP {exc.code}") from exc
        except Exception as exc:
            raise SourceAcquisitionError(f"fetching {url} failed: {type(exc).__name__}: {exc}") from exc
    return fetched, 0, revision


def _wikipedia_fetch_pages(entry: SourceEntry, dest: Path) -> tuple[int, int, str | None]:
    """Fetches each include_paths entry as a Wikipedia ARTICLE TITLE via
    the plain-text extracts API (explaintext=1) -- no wikitext markup to
    parse, no HTML to scrape, just clean prose with "== Heading =="
    markers (see knowledge/document_parser.py's parse_mediawiki_text).
    No per-page incremental-change detection (unlike the git path's
    commit-hash comparison): Wikipedia pages don't share one "repo
    revision," and re-fetching ~9 short articles on every sync is cheap
    enough at this source's actual scale not to need it -- `revision` is a
    snapshot timestamp for display/provenance only, not a skip-logic key."""
    import time as _time

    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True, exist_ok=True)

    fetched = 0
    for title in entry.include_paths:
        params = urllib.parse.urlencode({
            "action": "query", "titles": title, "prop": "extracts",
            "explaintext": 1, "format": "json",
        })
        url = f"{entry.repo_url}?{params}"
        try:
            with urllib.request.urlopen(
                urllib.request.Request(url, headers={"User-Agent": "3di-men-knowledge-sync/1.0"}), timeout=20
            ) as resp:
                data = json.loads(resp.read())
        except Exception as exc:
            raise SourceAcquisitionError(f"fetching Wikipedia article {title!r} failed: {type(exc).__name__}: {exc}") from exc

        pages = data.get("query", {}).get("pages", {})
        page = next(iter(pages.values()), None)
        if page is None or "missing" in page:
            raise SourceAcquisitionError(f"Wikipedia article {title!r} does not exist")
        extract = (page.get("extract") or "").strip()
        if not extract:
            raise SourceAcquisitionError(f"Wikipedia article {title!r} has no extractable text")

        filename = title.replace(" ", "_").replace("/", "_") + ".wikitext"
        (dest / filename).write_text(extract)
        fetched += 1

    return fetched, 0, f"snapshot-{int(_time.time())}"


def sync_source(source_id: str, manifest: SourceManifest) -> SourceSyncResult:
    """Refuses (does not silently skip) any source whose license_status is
    not 'approved' -- section 1/3's "skip or flag any source whose terms
    are unclear" means sync() must actually enforce that gate, not just
    display a warning somewhere a user might not see."""
    entry = manifest.get(source_id)
    if entry is None:
        return SourceSyncResult(source_id=source_id, ok=False, error=f"unknown source_id {source_id!r}")
    if entry.license_status != "approved":
        return SourceSyncResult(
            source_id=source_id, ok=False,
            error=f"license_status={entry.license_status!r}, not 'approved' -- refusing to sync. {entry.license_note}",
        )

    dest = _raw_dir(manifest.config, source_id)
    try:
        if entry.acquisition_method == "git_sparse_checkout":
            new_revision = _git_sparse_checkout(entry, dest)
            if new_revision == entry.revision and entry.sync_status == "synced":
                result = SourceSyncResult(source_id=source_id, ok=True, files_fetched=0,
                                           files_unchanged=entry.file_count, revision=new_revision)
            else:
                file_count = sum(1 for p in dest.rglob("*") if p.is_file() and ".git" not in p.parts)
                result = SourceSyncResult(source_id=source_id, ok=True, files_fetched=file_count, revision=new_revision)
        elif entry.acquisition_method == "http_fetch":
            fetched, unchanged, new_revision = _http_fetch_files(entry, dest, manifest.config)
            result = SourceSyncResult(source_id=source_id, ok=True, files_fetched=fetched,
                                       files_unchanged=unchanged, revision=new_revision)
        elif entry.acquisition_method == "wikipedia_api":
            fetched, unchanged, new_revision = _wikipedia_fetch_pages(entry, dest)
            result = SourceSyncResult(source_id=source_id, ok=True, files_fetched=fetched,
                                       files_unchanged=unchanged, revision=new_revision)
        else:
            return SourceSyncResult(source_id=source_id, ok=False,
                                     error=f"acquisition_method {entry.acquisition_method!r} has no local-file handler")
    except SourceAcquisitionError as exc:
        entry.sync_status = "sync_failed"
        entry.error = str(exc)
        manifest.save()
        return SourceSyncResult(source_id=source_id, ok=False, error=str(exc))

    entry.sync_status = "synced"
    entry.revision = result.revision
    entry.last_synced_at = time.time()
    entry.file_count = result.files_fetched + result.files_unchanged
    entry.error = None
    manifest.save()
    return result


def sync_all(manifest: SourceManifest) -> list[SourceSyncResult]:
    return [sync_source(e.source_id, manifest) for e in manifest.list()]


# ---------------------------------------------------------------------------
# Ingestion of synced raw files into the existing knowledge index
# ---------------------------------------------------------------------------

def _url_resolver(entry: SourceEntry, raw_root: Path):
    def resolve(local_path: Path) -> str:
        rel = local_path.relative_to(raw_root).as_posix()
        if entry.acquisition_method == "git_sparse_checkout" and entry.repo_url and "github.com" in entry.repo_url:
            return f"{entry.repo_url}/blob/{entry.revision or entry.branch}/{rel}"
        if entry.acquisition_method == "git_sparse_checkout" and entry.repo_url:
            return f"{entry.repo_url}/src/commit/{entry.revision or entry.branch}/{rel}"
        if entry.acquisition_method == "http_fetch" and entry.repo_url:
            return f"{entry.repo_url}/src/branch/{entry.branch}/{rel}"
        if entry.acquisition_method == "wikipedia_api":
            # local_path.stem is "Fillet_(mechanics)" (the fetched filename,
            # underscores for spaces) -- Wikipedia's own URL scheme uses the
            # same underscore convention directly, no further encoding needed
            # for the article titles this source is configured with.
            return f"{entry.canonical_url}wiki/{local_path.stem}"
        return entry.canonical_url
    return resolve


def ingest_source(source_id: str, manifest: SourceManifest, index: KnowledgeIndex) -> SourceIngestResult:
    """Runs the EXISTING ingestion.add_directory over this source's synced
    raw files -- the one and only ingestion code path in this project,
    reused exactly as-is, just with source_id/source_revision/url_for_path
    set so every resulting document/chunk carries real provenance."""
    entry = manifest.get(source_id)
    if entry is None:
        return SourceIngestResult(source_id=source_id, report=IngestionReport())
    if entry.sync_status != "synced":
        issue_report = IngestionReport()
        issue_report.failed.append(
            IngestionIssue(
                source_path=str(_raw_dir(manifest.config, source_id)),
                reason=f"source {source_id!r} has not been synced yet (sync_status={entry.sync_status!r})",
            )
        )
        return SourceIngestResult(source_id=source_id, report=issue_report)

    raw_root = _raw_dir(manifest.config, source_id)
    report = add_directory(
        index, raw_root, source_or_publisher=entry.title, license_notes=f"{entry.license} -- {entry.license_note}",
        config=manifest.config, recursive=True,
        source_id=source_id, source_revision=entry.revision, url_for_path=_url_resolver(entry, raw_root),
    )

    pruned = _prune_stale_documents(source_id, raw_root, index, manifest.config)

    entry.last_ingested_at = time.time()
    entry.ingest_status = "ingested" if not report.failed else (
        "partially_ingested" if report.added else "ingest_failed"
    )
    manifest.save()
    return SourceIngestResult(source_id=source_id, report=report, pruned_stale_documents=pruned)


def _prune_stale_documents(source_id: str, raw_root: Path, index: KnowledgeIndex, config: KnowledgeConfig) -> list[str]:
    """Section 3: "When a source is removed or a document disappears,
    avoid silently retaining stale chunks." A document tagged with this
    source_id is stale if its stored copy's content no longer matches any
    file CURRENTLY present under raw_root (i.e. it came from a file that
    was removed or changed upstream and the new version was re-ingested
    under a new document_id). Scoped strictly to source_id-tagged
    documents -- a user's own manually-imported documents (source_id=None)
    are never touched by this, satisfying the "never delete unrelated
    user-imported documents" guarantee directly via the data model, not
    just by convention."""
    from knowledge.ingestion import remove_document

    current_hashes = set()
    if raw_root.exists():
        import hashlib

        for p in raw_root.rglob("*"):
            if p.is_file():
                h = hashlib.sha256()
                h.update(p.read_bytes())
                current_hashes.add(h.hexdigest())

    pruned = []
    for record in list(index.documents.values()):
        if record.source_id != source_id:
            continue
        if record.document_id not in current_hashes:
            if remove_document(index, record.document_id, config=config):
                pruned.append(record.document_id)
    return pruned


def ingest_all(manifest: SourceManifest, index: KnowledgeIndex) -> list[SourceIngestResult]:
    return [ingest_source(e.source_id, manifest, index) for e in manifest.list() if e.sync_status == "synced"]


# ---------------------------------------------------------------------------
# Status reporting
# ---------------------------------------------------------------------------

def storage_usage(config: KnowledgeConfig) -> dict:
    def _dir_size(path: Path) -> int:
        if not path.exists():
            return 0
        return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())

    return {
        "raw_sources_bytes": _dir_size(config.store_dir / "sources" / "raw"),
        "documents_bytes": _dir_size(config.documents_dir),
        "index_bytes": config.index_path.stat().st_size if config.index_path.exists() else 0,
    }
