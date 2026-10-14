"""Retrieval evaluation (sections 7 and 12's "Knowledge retrieval precision
on annotated queries" metric) -- a labeled query set, each with either the
exact chunk_ids expected back or a looser substring match against the
retrieved text (exact chunk_ids are brittle across re-ingestion since they
depend on content-hash document_ids; substring matching lets a test fixture
assert "the chunk about X was retrieved" without hardcoding an id).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from knowledge.index import KnowledgeIndex
from knowledge.retrieval import retrieve


@dataclass
class LabeledQuery:
    query: str
    expected_chunk_ids: tuple[str, ...] = ()
    expected_text_contains: tuple[str, ...] = ()
    top_k: int = 5


@dataclass
class QueryEvalResult:
    query: str
    retrieved_chunk_ids: tuple[str, ...]
    hit: bool  # at least one expected chunk (by id or substring) was retrieved
    precision: float  # fraction of retrieved chunks that were expected (only meaningful with ids)


@dataclass
class EvaluationSummary:
    results: list[QueryEvalResult] = field(default_factory=list)

    @property
    def hit_rate(self) -> float:
        if not self.results:
            return 0.0
        return sum(1 for r in self.results if r.hit) / len(self.results)

    @property
    def mean_precision(self) -> float:
        scored = [r.precision for r in self.results if r.retrieved_chunk_ids]
        return sum(scored) / len(scored) if scored else 0.0


def _is_hit(chunk, labeled: LabeledQuery) -> bool:
    if labeled.expected_chunk_ids and chunk.chunk_id in labeled.expected_chunk_ids:
        return True
    if labeled.expected_text_contains:
        lowered = chunk.text.lower()
        return any(s.lower() in lowered for s in labeled.expected_text_contains)
    return False


def evaluate_retrieval(index: KnowledgeIndex, labeled_queries: list[LabeledQuery]) -> EvaluationSummary:
    summary = EvaluationSummary()
    for lq in labeled_queries:
        results = retrieve(index, lq.query, top_k=lq.top_k)
        hits = [r for r in results if _is_hit(r.chunk, lq)]
        precision = len(hits) / len(results) if results else 0.0
        summary.results.append(
            QueryEvalResult(
                query=lq.query,
                retrieved_chunk_ids=tuple(r.chunk.chunk_id for r in results),
                hit=len(hits) > 0,
                precision=precision,
            )
        )
    return summary
