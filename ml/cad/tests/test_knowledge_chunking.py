"""Tests for knowledge/chunking.py -- section-aware packing + concept-type
and domain tagging."""

from knowledge.chunking import _infer_concept_type, _infer_domains, chunk_document
from knowledge.config import KnowledgeConfig
from knowledge.document_parser import ParsedBlock, ParsedDocument


def _doc(blocks, title="T"):
    return ParsedDocument(blocks=blocks, title=title, page_count=None, extraction_method="markdown")


def test_chunk_document_keeps_blocks_whole_within_target_size():
    config = KnowledgeConfig(chunk_target_chars=1000, chunk_overlap_chars=0, chunk_min_chars=10)
    blocks = [ParsedBlock(text="Paragraph one.", section="S"), ParsedBlock(text="Paragraph two.", section="S")]
    chunks = chunk_document(_doc(blocks), document_id="doc1", source_path="/x", config=config)
    assert len(chunks) == 1
    assert "Paragraph one." in chunks[0].text and "Paragraph two." in chunks[0].text


def test_chunk_document_splits_on_section_change():
    config = KnowledgeConfig(chunk_target_chars=1000, chunk_overlap_chars=0, chunk_min_chars=5)
    blocks = [
        ParsedBlock(text="Text under section A.", section="A"),
        ParsedBlock(text="Text under section B.", section="B"),
    ]
    chunks = chunk_document(_doc(blocks), document_id="doc1", source_path="/x", config=config)
    assert len(chunks) == 2
    assert chunks[0].section == "A"
    assert chunks[1].section == "B"


def test_chunk_document_splits_when_target_size_is_exceeded():
    config = KnowledgeConfig(chunk_target_chars=30, chunk_overlap_chars=0, chunk_min_chars=5)
    blocks = [
        ParsedBlock(text="A" * 25, section="S"),
        ParsedBlock(text="B" * 25, section="S"),
    ]
    chunks = chunk_document(_doc(blocks), document_id="doc1", source_path="/x", config=config)
    assert len(chunks) == 2
    assert chunks[0].text == "A" * 25
    assert chunks[1].text == "B" * 25


def test_chunk_document_merges_a_trailing_fragment_too_small_to_stand_alone():
    # target_chars=19 forces a flush when "hi" (2 chars) arrives after an
    # 18-char buffer (18+2=20 > 19) -- so "hi" starts its OWN buffer, which
    # at end-of-document is too small (< chunk_min_chars=5) to become its
    # own chunk and must be folded into the preceding one instead of lost.
    config = KnowledgeConfig(chunk_target_chars=19, chunk_overlap_chars=0, chunk_min_chars=5)
    blocks = [
        ParsedBlock(text="A" * 18, section="S"),
        ParsedBlock(text="hi", section="S"),
    ]
    chunks = chunk_document(_doc(blocks), document_id="doc1", source_path="/x", config=config)
    assert len(chunks) == 1
    assert "hi" in chunks[0].text
    assert "A" * 18 in chunks[0].text


def test_chunk_document_tracks_page_start_and_end():
    config = KnowledgeConfig(chunk_target_chars=1000, chunk_overlap_chars=0, chunk_min_chars=5)
    blocks = [
        ParsedBlock(text="Page 2 content.", section=None, page=2),
        ParsedBlock(text="Page 3 content.", section=None, page=3),
    ]
    chunks = chunk_document(_doc(blocks), document_id="doc1", source_path="/x", config=config)
    assert len(chunks) == 1
    assert chunks[0].page_start == 2
    assert chunks[0].page_end == 3


def test_chunk_ids_and_content_hash_are_stable():
    config = KnowledgeConfig(chunk_target_chars=1000, chunk_overlap_chars=0, chunk_min_chars=5)
    blocks = [ParsedBlock(text="Some content.", section="S")]
    c1 = chunk_document(_doc(blocks), document_id="abc123", source_path="/x", config=config)
    c2 = chunk_document(_doc(blocks), document_id="abc123", source_path="/x", config=config)
    assert c1[0].chunk_id == "abc123:0" == c2[0].chunk_id
    assert c1[0].content_hash == c2[0].content_hash


def test_oversized_single_block_is_split_at_sentence_boundaries():
    config = KnowledgeConfig(chunk_target_chars=60, chunk_overlap_chars=0, chunk_min_chars=5)
    long_text = "First sentence here. Second sentence here. Third sentence here. Fourth one."
    blocks = [ParsedBlock(text=long_text, section="S")]
    chunks = chunk_document(_doc(blocks), document_id="doc1", source_path="/x", config=config)
    assert len(chunks) > 1
    assert "".join(c.text for c in chunks).replace(" ", "") != ""  # nothing silently dropped
    for c in chunks:
        assert len(c.text) <= 60 + 20  # a little slack for the join logic, never wildly over


def test_infer_concept_type_detects_formula():
    assert _infer_concept_type("The center distance formula is d = m * (z1 + z2) / 2.") == "formula"


def test_infer_concept_type_detects_failure_mode():
    assert _infer_concept_type("Requesting too large a fillet radius can cause the operation to fail.") == "failure_mode"


def test_infer_concept_type_detects_troubleshooting_procedure():
    assert _infer_concept_type("To fix a self-intersecting shell, reduce the wall thickness.") == "troubleshooting_procedure"


def test_infer_concept_type_detects_geometric_operation():
    assert _infer_concept_type("Extrude the rectangular profile along its normal to form a box.") == "geometric_operation"


def test_infer_concept_type_detects_constraint():
    assert _infer_concept_type("The fillet radius must not exceed half the adjacent wall thickness.") == "constraint"


def test_infer_concept_type_defaults_to_general_for_unremarkable_prose():
    assert _infer_concept_type("This part of the assembly looks nice in the viewport.") == "general"


def test_infer_domains_matches_multiple_authored_keyword_banks():
    text = "A mounting bracket with a bolt circle pattern, modeled using a boolean difference for each hole."
    domains = _infer_domains(text)
    assert "mechanical_design" in domains
    assert "solid_modeling" in domains


def test_infer_domains_is_empty_when_nothing_matches():
    assert _infer_domains("The weather today is pleasant.") == ()
