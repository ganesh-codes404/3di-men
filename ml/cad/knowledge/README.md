# knowledge/ — engineering reference retrieval

A local-first document ingestion + retrieval subsystem that grounds
`llm_decompose.py`'s prompt decomposition in real engineering reference
material the user supplies (CAD textbooks, mechanical design guides,
manufacturing references, your own notes) — **never** the geometry
generation itself, which stays entirely with this project's own
proprietary `program_model.py` + `sandbox.py` tool-composition pipeline, as
before.

## Why TF-IDF, not a neural embedding model

This subsystem's retrieval uses `scikit-learn`'s `TfidfVectorizer` +
brute-force sparse cosine similarity — the exact same pattern
`program_model.py` already uses in production for its own op-sequence
retrieval (`retrieve_nearest_op_sequence`). Two reasons, both concrete:

1. **Consistency.** The project already has one proven, in-production
   retrieval mechanism. A second, differently-shaped one (a neural
   embedding model + a vector database) for a conceptually similar job
   would be an unjustified architectural split.
2. **Hardware.** This project's own hardware (confirmed via direct
   benchmark this session: a 2015 dual-core 1.8GHz Intel CPU, no GPU
   offload previously configured anywhere) already runs local LLM
   generation at roughly 0.1–0.4 tokens/sec. A `sentence-transformers`/
   `torch` forward pass per chunk would add real, measurable cost on this
   machine for no retrieval-quality benefit at the scale this subsystem
   actually needs — a personal reference library of dozens to a few
   hundred documents, not a web-scale corpus.

The embedding backend sits behind a small interface
(`knowledge/embeddings.py`'s `EmbeddingBackend`) specifically so a neural
backend *can* be swapped in later without touching `index.py` or
`retrieval.py`, if a future corpus size or quality need actually justifies
it — not because one is needed today.

## Setup

No new heavy dependencies: `scikit-learn`, `PyMuPDF` (PDF text), `beautifulsoup4`+`lxml` (HTML), and `joblib` are already in `requirements.txt`. If you haven't installed them:

```bash
cd ml/cad
pip3 install -r requirements.txt
```

Automated acquisition (`source-sync`) additionally needs `git` on `PATH` for
the git-based sources (FreeCAD/CadQuery/build123d); the Blender manual is
fetched over plain HTTP and needs nothing extra.

Optional OCR fallback for scanned PDFs (off by default — see Limitations):

```bash
pip3 install pytesseract Pillow   # plus a system `tesseract` binary
export KNOWLEDGE_ENABLE_OCR=1
```

## Usage

### Manual ingestion (your own documents)

```bash
# Ingest one document or a whole directory (PDF, Markdown, reStructuredText, plain text, HTML)
python3 -m knowledge.cli add path/to/some-reference.pdf --source "Publisher Name" --license "personal use"
python3 -m knowledge.cli add-dir path/to/a/folder/of/references

# Inspect what's indexed
python3 -m knowledge.cli list
python3 -m knowledge.cli check          # integrity check

# Remove a document (and its indexed chunks) by id or id prefix
python3 -m knowledge.cli remove 5cfe5f04

# Rebuild the index's embedding vectors (e.g. after a config change)
python3 -m knowledge.cli rebuild

# Query it directly, independent of the LLM pipeline
python3 -m knowledge.cli query "fillet radius too large causes failure" --top-k 5
```

### Automated documentation acquisition (`knowledge/sources.py`)

A second, independent source of documents: a small, hand-verified registry
of official open CAD/mechanical-design documentation repositories (see
"Acquired documentation sources" below), synced with `git` sparse-checkouts
or direct HTTP fetches (never a full clone of a multi-hundred-MB repo), then
fed through the exact same `add_directory()` ingestion path manual
documents go through — there is only one ingestion code path in this
project.

```bash
# See every candidate source and its current status (no network call)
python3 -m knowledge.cli source-discover

# Check reachability + recorded license status (no download)
python3 -m knowledge.cli source-validate

# Download/update all approved sources (or one: source-sync freecad_docs)
python3 -m knowledge.cli source-sync

# Ingest the synced files into the index (prunes any chunks whose upstream
# file has since disappeared -- never touches a document YOU imported manually)
python3 -m knowledge.cli source-ingest

# Both of the above in one step, for routine re-syncing later
python3 -m knowledge.cli source-reindex

# Per-source sync/ingest status, document/chunk counts, storage usage
python3 -m knowledge.cli source-status
```

A source only ever syncs if its `license_status` is `"approved"` — a source
whose documentation license is unclear (see `occt_docs` below) is kept
visible in `source-discover`/`source-status` but `source-sync` refuses it
with a specific reason, never silently skips or silently proceeds.

**Offline after acquisition:** once synced + ingested, `source-ingest`,
`query`, and ordinary `/api/build` generation make zero network calls —
only `source-sync`/`source-validate` touch the network, and only when you
explicitly run them.

Once documents are ingested (by either path), `llm_decompose.py`'s `decompose_prompt()`
automatically retrieves relevant passages for each build prompt and adds
them to the model's user message (never the system prompt — see the
comment in `decompose_prompt` for why: the system prompt is already a
carefully budgeted ~3900-token block, and growing it with variable-length
retrieved content risks repeating the exact context-overflow bug this
project hit once before). `server.py`'s `/api/build` response includes a
`knowledge_context` field with the real text actually shown to the model
for that request, if any.

## Configuration

All environment-variable driven (see `knowledge/config.py`), matching
`llm_decompose.py`'s own `MODEL_NAME`/`MAX_CONTEXT` convention — nothing
requires editing Python to reconfigure:

| Variable | Default | Meaning |
|---|---|---|
| `KNOWLEDGE_ENABLED` | `1` | Set to `0` to disable the whole subsystem — `decompose_prompt` then behaves exactly as it did before this subsystem existed. |
| `KNOWLEDGE_STORE_DIR` | `ml/cad/knowledge_store/` | Where ingested documents and the index are stored. Outside source code, gitignored. |
| `KNOWLEDGE_CHUNK_CHARS` | `1000` | Target chunk size. |
| `KNOWLEDGE_CHUNK_OVERLAP_CHARS` | `150` | Overlap between adjacent chunks. |
| `KNOWLEDGE_TOP_K` | `5` | Max chunks returned per query. |
| `KNOWLEDGE_MIN_SCORE` | `0.05` | Minimum cosine similarity to return a chunk at all. |
| `KNOWLEDGE_MAX_CONTEXT_CHARS` | `2000` | Hard cap on total retrieved text injected into one prompt (~500 tokens). |
| `KNOWLEDGE_ENABLE_OCR` | `0` | Optional OCR fallback for scanned PDFs (requires `pytesseract`+`Pillow`+a system `tesseract` binary). |

## Acquired documentation sources

Verified directly against each source during this work (the repository's
own LICENSE file or GitHub/Gitea license API, read for real, not assumed
from reputation — see each entry's `license_note` in
`knowledge/sources.py`'s `DEFAULT_SOURCES` for exactly what was checked).

| Source | License | Status | Method |
|---|---|---|---|
| [FreeCAD Documentation Wiki](https://wiki.freecad.org/) | CC0-1.0 | approved | `git` sparse-checkout of a curated subset of `wiki/*.md` |
| [CadQuery Documentation](https://cadquery.readthedocs.io/) | Apache-2.0 | approved | `git` sparse-checkout of `doc/*.rst` |
| [build123d Documentation](https://build123d.readthedocs.io/) | Apache-2.0 | approved | `git` sparse-checkout of `docs/*.rst` |
| [Blender Manual](https://docs.blender.org/manual/en/latest/) | CC-BY-SA-4.0 | approved | direct HTTP fetch of individual `manual/*.rst` files (the repo is ~876MB, almost entirely screenshots — never cloned) |
| [OCCT Overview Docs](https://dev.opencascade.org/doc/overview/html/) | LGPL-2.1 (a *software* license; no separate docs license found) | **needs_review — not synced** | would be `git` sparse-checkout of `dox/*.md` if approved |

"Open educational engineering resources" (the mission's 6th candidate
category) has no concrete URL in its own description — left unconfigured
rather than inventing a source, per this subsystem's own "skip or flag any
source whose terms are unclear" rule applied to "unspecified" as a case of
"unclear."

To add a new source: add a `SourceEntry` to `DEFAULT_SOURCES` in
`knowledge/sources.py` with a real, checked `license`/`license_status`/
`license_note` — there's no separate config file to maintain, the registry
IS the config, consistent with this project's existing `TOOL_OPS`/
`KNOWLEDGE_DOMAINS`-style authored-constant conventions.

## Architecture

```
document_parser.py  -- PDF (PyMuPDF) / Markdown / reStructuredText / HTML
                        (bs4) / plain text -> a single format-agnostic
                        ParsedDocument
chunking.py          -- section-aware packing into size-bounded Chunks,
                        with closed-vocabulary concept-type/domain tagging
embeddings.py         -- TfidfEmbeddingBackend (swappable interface)
index.py              -- persistent (joblib) KnowledgeIndex: add/remove/
                        rebuild/integrity_check
ingestion.py          -- the one orchestration entry point: duplicate
                        detection (content-hash document_id), per-file
                        error reporting, never silent; threads optional
                        source_id/source_url/source_revision provenance
                        through to DocumentRecord/Chunk
sources.py            -- automated acquisition: a verified source registry
                        (DEFAULT_SOURCES), a persisted SourceManifest,
                        git-sparse-checkout/http-fetch acquisition, and
                        prune-aware re-ingestion -- calls ingestion.py's
                        add_directory(), never a second ingestion path
retrieval.py          -- query -> ranked, deduplicated, budget-limited
                        RetrievalResult list; cosine similarity via the
                        SAME sparse-matmul pattern program_model.py
                        already uses in production
evaluation.py         -- retrieval precision/hit-rate against a small
                        labeled query set
pipeline.py            -- the ONE integration point llm_decompose.py calls;
                        never raises, returns "" when there's nothing to add
cli.py                -- python -m knowledge.cli add/add-dir/list/remove/
                        rebuild/check/query/source-discover/source-validate/
                        source-sync/source-ingest/source-reindex/source-status
```

## Limitations (disclosed honestly, not silently worked around)

- **No OCR by default.** A scanned PDF with no text layer is reported as
  unprocessable (`DocumentParseError`) rather than silently indexed as
  near-empty content. Enable `KNOWLEDGE_ENABLE_OCR=1` and install
  `pytesseract`+`Pillow`+a system `tesseract` binary if you need it.
- **No workspace UI panel yet.** `server.py`'s `/api/build` response
  carries `knowledge_context` (the real retrieved text, if any), but
  `frontend/workspace.html` does not yet render a dedicated "knowledge
  sources" panel showing it — see `ml/cad/MISSION_STATUS.md` for the
  honest status of this and other not-yet-built mission items.
- **No structured engineering-rules layer or feedback-driven repair.**
  This subsystem is retrieval over documents only (section 8/13 of the
  originating mission spec). A second, explicit, validated-rules
  representation and a repair loop that retrieves targeted troubleshooting
  knowledge on a geometry failure are both future work.
- **Full-corpus re-embedding on every add/remove.** Fine at this
  subsystem's intended scale (a personal reference library); would need
  revisiting for a very large corpus.
- **PDF heading detection is conservative by design.** Page number is
  always a reliable anchor; a section heading is only set when a line
  clearly looks like one (numbered or capitalized, short, no trailing
  punctuation) — many PDFs will have `section=None` on most chunks, which
  is the honest outcome rather than a fabricated section name.
- **reStructuredText parsing is a pragmatic subset, not full docutils.**
  Inline markup (`` `code` ``, `*emphasis*`, `` `link <url>`_ ``) is left
  as-is rather than stripped; heading/admonition/code-block detection is
  tested against real acquired files (CadQuery/build123d/Blender) but will
  misparse unusual directive nesting a full reST parser would handle.
- **OCCT is deliberately not synced.** Its repository license (LGPL-2.1) is
  a software license; no separate documentation-content license was found
  during this work's verification pass, so `source-sync occt_docs` refuses
  with a specific error rather than guessing. Resolve by finding/obtaining
  an explicit docs-license statement, updating that `SourceEntry`'s
  `license_status` to `"approved"` in `knowledge/sources.py`, then running
  `source-sync occt_docs`.
- **"Open educational engineering resources" has no configured source.**
  The originating request named this category with no concrete URL —
  nothing was fabricated to fill the gap. Add a verified `SourceEntry` to
  `DEFAULT_SOURCES` if/when a specific, license-checked resource is chosen.
- **Acquired document quality varies by source.** FreeCAD's wiki-to-
  Markdown mirror carries some leftover MediaWiki template syntax (e.g.
  `{{Properties_Title|Base}}`) that this parser doesn't specifically strip
  — readable but not fully clean. Retrieval quality against the acquired
  corpus is also uneven in practice: these sources are CAD-*tool* API/
  operation documentation, not product-*design* guidance, so a prompt like
  "a water bottle with a screw cap" retrieves only loosely related passages
  (confirmed by direct testing against the real synced corpus), while a
  prompt closer to the tools' own vocabulary (fillets, booleans, hole
  patterns) retrieves well. This is reported honestly rather than claimed
  as uniformly high quality.
- **The baseline-vs-RAG benchmark comparison (section 7/12) was not
  executed with this real acquired corpus** — `scripts/benchmark_pipeline.py`
  is ready and now has real documentation to compare against, but a full
  run is dozens of real local-LLM calls, each confirmed to take on the
  order of tens of minutes on this project's hardware. See
  `ml/cad/MISSION_STATUS.md` for the full, honest accounting of what was
  and wasn't run.
