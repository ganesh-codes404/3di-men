"""query -> ranked, deduplicated, budget-limited chunks (section 7).

Semantic ranking is TF-IDF cosine similarity -- the same sparse matmul
pattern program_model.py's own _top_k_neighbors already uses in production
(bundle.train_text_vectors @ x.T), not a separately-invented scoring
method. "Hybrid" retrieval here means semantic ranking (TF-IDF already
captures a good deal of lexical/keyword overlap, since it IS a lexical
n-gram method, just weighted) plus metadata filters (domain, concept_type)
-- section 7 explicitly allows starting with semantic + filters and adding
a separate lexical signal "where evaluation shows it helps"; nothing in
this subsystem's own evaluation (see evaluation.py) has shown that yet, so
a second scoring pass is not added speculatively.

Query expansion is deliberately minimal (section 7: "must not become an
elaborate second LLM pipeline") -- domain detection on the QUERY reuses
chunking.py's own keyword-based _infer_domains rather than a new mechanism,
and is used only as an optional filter/boost the caller can opt into, not
an automatic rewrite of the query text.
"""

from __future__ import annotations

from knowledge.chunking import _infer_domains
from knowledge.config import KnowledgeConfig
from knowledge.index import KnowledgeIndex
from knowledge.schemas import ConceptType, RetrievalResult


def suggest_domains(query: str) -> tuple[str, ...]:
    """Deterministic query-side domain detection -- the "identify likely
    concepts and retrieval topics" step from section 7, kept intentionally
    cheap (a keyword lookup, not a second model call)."""
    return _infer_domains(query)


def _matched_terms(index: KnowledgeIndex, query_vec, chunk_vector_row) -> tuple[str, ...]:
    feature_names = index.embedding_backend.feature_names()
    if not feature_names:
        return ()
    query_nonzero = set(query_vec.indices)
    chunk_nonzero = set(chunk_vector_row.indices)
    overlap = query_nonzero & chunk_nonzero
    if not overlap:
        return ()
    return tuple(sorted(feature_names[i] for i in overlap))


def retrieve(
    index: KnowledgeIndex,
    query: str,
    *,
    top_k: int | None = None,
    min_score: float | None = None,
    max_context_chars: int | None = None,
    domains: tuple[str, ...] | None = None,
    concept_types: tuple[ConceptType, ...] | None = None,
    config: KnowledgeConfig | None = None,
) -> list[RetrievalResult]:
    """Never raises on an empty/unfit index -- returns []. This is the
    subsystem's core "graceful degradation" contract (section 3): a caller
    (llm_decompose.py's prompt pipeline) must be able to call this
    unconditionally and treat an empty result exactly like "nothing useful
    to add," never like a failure to special-case."""
    config = config or index.config
    if not query or not query.strip():
        return []
    if not index.chunks or index.chunk_vectors is None or not index.embedding_backend.is_fitted:
        return []

    top_k = config.default_top_k if top_k is None else top_k
    min_score = config.min_score if min_score is None else min_score
    max_context_chars = config.max_context_chars if max_context_chars is None else max_context_chars

    query_vec = index.embedding_backend.transform([query])
    sims = (index.chunk_vectors @ query_vec.T).toarray().ravel()

    candidate_indices = range(len(index.chunks))
    if domains:
        domain_set = set(domains)
        candidate_indices = [i for i in candidate_indices if domain_set & set(index.chunks[i].domains)]
    if concept_types:
        type_set = set(concept_types)
        candidate_indices = [i for i in candidate_indices if index.chunks[i].concept_type in type_set]

    scored = [(i, float(sims[i])) for i in candidate_indices if sims[i] >= min_score]
    scored.sort(key=lambda t: -t[1])

    results: list[RetrievalResult] = []
    seen_hashes: set[str] = set()
    total_chars = 0
    query_vec_row = query_vec.tocsr()[0]
    chunk_vectors_csr = index.chunk_vectors.tocsr()
    for i, score in scored:
        if len(results) >= top_k:
            break
        chunk = index.chunks[i]
        if chunk.content_hash in seen_hashes:
            continue  # section 7: "Deduplicate overlapping or redundant results"
        if results and total_chars + len(chunk.text) > max_context_chars:
            continue  # keep scanning -- a smaller, still-relevant chunk further down may still fit
        seen_hashes.add(chunk.content_hash)
        matched = _matched_terms(index, query_vec_row, chunk_vectors_csr[i])
        results.append(RetrievalResult(chunk=chunk, score=score, matched_terms=matched))
        total_chars += len(chunk.text)
    return results


def format_context_block(results: list[RetrievalResult]) -> str:
    """Renders retrieval results as a clearly-labeled, source-attributed
    text block for injection into a prompt (see knowledge/pipeline.py).
    Every passage is tagged with its real source so the model (and a human
    reviewing the workspace's knowledge panel later) can tell retrieved
    fact from the model's own reasoning (section 9's "clearly distinguish
    sourced facts... and uncertain inferences")."""
    if not results:
        return ""
    lines = ["Reference notes (retrieved from ingested documents -- may be incomplete or not directly applicable; use judgment):"]
    for r in results:
        c = r.chunk
        loc = f"p.{c.page_start}" if c.page_start else (c.section or "")
        source = f"{c.title or c.source_path}" + (f", {loc}" if loc else "")
        lines.append(f"- [{source}] {c.text.strip()}")
    return "\n".join(lines)
