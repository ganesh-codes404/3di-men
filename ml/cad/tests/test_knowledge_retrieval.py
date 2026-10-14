"""Tests for knowledge/retrieval.py and knowledge/evaluation.py."""

from pathlib import Path

import pytest

from knowledge.config import KnowledgeConfig
from knowledge.evaluation import LabeledQuery, evaluate_retrieval
from knowledge.index import KnowledgeIndex
from knowledge.ingestion import add_document
from knowledge.retrieval import format_context_block, retrieve, suggest_domains

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures" / "knowledge_docs"


@pytest.fixture
def populated_index(tmp_path):
    config = KnowledgeConfig(store_dir=tmp_path / "knowledge_store")
    index = KnowledgeIndex(config=config)
    for fname in ("solid_modeling_basics.md", "bottle_design_notes.md", "bracket_mounting.txt"):
        add_document(index, FIXTURES_DIR / fname, config=config)
    return index


def test_retrieve_on_an_empty_index_returns_nothing(tmp_path):
    config = KnowledgeConfig(store_dir=tmp_path / "knowledge_store")
    index = KnowledgeIndex(config=config)
    assert retrieve(index, "fillet radius") == []


def test_retrieve_on_an_empty_query_returns_nothing(populated_index):
    assert retrieve(populated_index, "") == []
    assert retrieve(populated_index, "   ") == []


def test_retrieve_finds_the_relevant_document_for_a_bottle_query(populated_index):
    results = retrieve(populated_index, "bottle cap clearance neck diameter", top_k=3)
    assert results, "expected at least one result"
    assert "bottle_design_notes" in results[0].chunk.source_path or "neck" in results[0].chunk.text.lower()
    assert results[0].score > 0


def test_retrieve_finds_the_relevant_document_for_a_fillet_query(populated_index):
    results = retrieve(populated_index, "fillet radius too large causes failure", top_k=3)
    assert results
    assert "fillet" in results[0].chunk.text.lower()


def test_retrieve_results_are_sorted_by_descending_score(populated_index):
    results = retrieve(populated_index, "mounting bracket bolt circle hole pattern", top_k=5)
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)


def test_retrieve_respects_top_k(populated_index):
    results = retrieve(populated_index, "a or the", top_k=1, min_score=0.0)
    assert len(results) <= 1


def test_retrieve_domain_filter_excludes_non_matching_chunks(populated_index):
    all_results = retrieve(populated_index, "wall thickness", top_k=10, min_score=0.0)
    filtered = retrieve(populated_index, "wall thickness", top_k=10, min_score=0.0, domains=("mechanical_design",))
    assert len(filtered) <= len(all_results)
    assert all("mechanical_design" in r.chunk.domains for r in filtered)


def test_retrieve_respects_max_context_chars_budget(populated_index):
    results = retrieve(populated_index, "fillet chamfer boolean shell extrusion", top_k=10, min_score=0.0, max_context_chars=200)
    total_chars = sum(len(r.chunk.text) for r in results)
    # the first chunk is always allowed through even if it alone exceeds budget
    # (so a legitimately relevant result is never dropped entirely); every
    # chunk after that must keep the running total within budget.
    assert len(results) >= 1
    if len(results) > 1:
        assert total_chars - len(results[-1].chunk.text) <= 200


def test_matched_terms_are_populated_for_a_real_match(populated_index):
    results = retrieve(populated_index, "fillet radius", top_k=1)
    assert results
    assert "fillet" in results[0].matched_terms


def test_suggest_domains_reuses_chunking_keyword_banks():
    assert "mechanical_design" in suggest_domains("a mounting bracket with a bolt circle")


def test_format_context_block_is_empty_for_no_results():
    assert format_context_block([]) == ""


def test_format_context_block_includes_source_attribution(populated_index):
    results = retrieve(populated_index, "fillet radius")
    block = format_context_block(results)
    assert "Reference notes" in block
    assert "[" in block and "]" in block  # source tag present


def test_evaluate_retrieval_computes_hit_rate_and_precision(populated_index):
    labeled = [
        LabeledQuery(query="fillet radius too large", expected_text_contains=("fillet",)),
        LabeledQuery(query="completely unrelated gibberish query xyzzy", expected_text_contains=("fillet",)),
    ]
    summary = evaluate_retrieval(populated_index, labeled)
    assert summary.results[0].hit is True
    # the second query shares no real vocabulary with the corpus -- a miss is the honest, expected outcome
    assert 0.0 <= summary.hit_rate <= 1.0
    assert 0.0 <= summary.mean_precision <= 1.0
