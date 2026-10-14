"""Engineering knowledge retrieval subsystem for 3di-men.

Ingests local CAD/mechanical-design reference documents (PDF, Markdown,
plain text, HTML), chunks them in a structure-aware way, indexes them with
the SAME lightweight TF-IDF + sparse-cosine retrieval pattern
program_model.py already uses in production (see program_model.py's
_top_k_neighbors/retrieve_nearest_op_sequence) -- not a second, inconsistent
retrieval paradigm -- and exposes a small, optional retrieval() call that
llm_decompose.py's prompt pipeline can use to ground its decomposition in
real reference material.

Deliberately NOT a neural-embedding/vector-database stack: this machine is
CPU-only with no GPU offload in use anywhere in the project (confirmed via
direct benchmark elsewhere in this project's history -- local LLM generation
runs at roughly 0.1-0.4 tokens/sec here), so a sentence-transformers/torch
forward pass per chunk would add real, measurable cost for no retrieval-
quality benefit at the scale this subsystem actually needs to operate at (a
personal reference library of dozens to low hundreds of documents, not a
web-scale corpus). TF-IDF + brute-force sparse cosine similarity (scikit-
learn, already a dependency) is fast enough at that scale and keeps this
subsystem's own retrieval behavior consistent with the rest of the project.

This entire subsystem is OPTIONAL and must degrade gracefully: an empty or
missing index, a missing optional dependency (e.g. PyMuPDF for PDF
ingestion), or a retrieval() call with nothing relevant to return must never
break llm_decompose.py's prompt pipeline, which must keep working exactly as
it did before this subsystem existed when there is nothing useful to add.

Module layout:
- schemas.py     -- Document/Chunk/RetrievalResult/IngestionReport dataclasses
- config.py       -- storage paths, chunk size/overlap, env-var-driven tunables
- document_parser.py -- format-specific text + structure extraction
- chunking.py     -- section-aware chunking with metadata + concept-type tagging
- embeddings.py   -- the swappable TF-IDF embedding backend
- index.py        -- persistent (joblib-backed) chunk index: add/remove/rebuild
- retrieval.py     -- query -> ranked, deduplicated, budget-limited chunks
- ingestion.py    -- orchestrates parse -> chunk -> embed -> index, with
                      duplicate detection and honest per-file error reporting
- evaluation.py   -- retrieval precision/recall against a small labeled query set
- cli.py          -- `python -m knowledge.cli add/list/remove/rebuild/query`
"""
