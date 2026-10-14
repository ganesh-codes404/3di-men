"""Tests for knowledge/index.py (persistence, integrity) and
knowledge/ingestion.py (parse->chunk->index orchestration, duplicate
detection, per-file error reporting)."""

from pathlib import Path

import pytest

from knowledge.config import KnowledgeConfig
from knowledge.document_parser import DocumentParseError
from knowledge.index import KnowledgeIndex
from knowledge.ingestion import add_directory, add_document
from knowledge.ingestion import remove_document as do_remove

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures" / "knowledge_docs"


@pytest.fixture
def config(tmp_path):
    return KnowledgeConfig(store_dir=tmp_path / "knowledge_store")


def test_add_document_indexes_chunks_and_copies_the_file(config):
    index = KnowledgeIndex(config=config)
    record = add_document(index, FIXTURES_DIR / "solid_modeling_basics.md", config=config)

    assert record.chunk_count > 0
    assert index.has_document(record.document_id)
    assert len(index.chunks_for_document(record.document_id)) == record.chunk_count
    assert index.chunk_vectors is not None
    assert index.chunk_vectors.shape[0] == len(index.chunks)

    stored = list(config.documents_dir.glob(f"{record.document_id}.*"))
    assert len(stored) == 1


def test_add_document_rejects_an_exact_duplicate(config):
    index = KnowledgeIndex(config=config)
    add_document(index, FIXTURES_DIR / "solid_modeling_basics.md", config=config)
    with pytest.raises(ValueError, match="duplicate"):
        add_document(index, FIXTURES_DIR / "solid_modeling_basics.md", config=config)


def test_add_document_raises_on_unsupported_or_unprocessable_file(tmp_path, config):
    index = KnowledgeIndex(config=config)
    bad = tmp_path / "notes.xyz"
    bad.write_text("whatever")
    with pytest.raises(DocumentParseError):
        add_document(index, bad, config=config)

    empty = tmp_path / "empty.txt"
    empty.write_text("")
    with pytest.raises(DocumentParseError):
        add_document(index, empty, config=config)


def test_add_directory_reports_added_duplicates_and_failures(tmp_path, config):
    src_dir = tmp_path / "docs"
    src_dir.mkdir()
    (src_dir / "a.md").write_text("# A\n\nSome real content about extrusion and lofting.")
    (src_dir / "b.txt").write_text("")  # will fail to parse (empty)
    (src_dir / "ignored.xyz").write_text("not a supported format at all")  # not even attempted

    index = KnowledgeIndex(config=config)
    report = add_directory(index, src_dir, config=config)

    assert len(report.added) == 1
    assert report.added[0].filename == "a.md"
    assert len(report.failed) == 1
    assert report.failed[0].source_path.endswith("b.txt")
    assert not report.skipped_duplicates

    # re-running the same ingestion reports the previously-added file as a duplicate, not a failure
    report2 = add_directory(index, src_dir, config=config)
    assert not report2.added
    assert any(p.endswith("a.md") for p in report2.skipped_duplicates)


def test_remove_document_deletes_chunks_and_stored_copy(config):
    index = KnowledgeIndex(config=config)
    record = add_document(index, FIXTURES_DIR / "bracket_mounting.txt", config=config)
    assert index.has_document(record.document_id)

    removed = do_remove(index, record.document_id, config=config)
    assert removed is True
    assert not index.has_document(record.document_id)
    assert index.chunks_for_document(record.document_id) == []
    assert list(config.documents_dir.glob(f"{record.document_id}.*")) == []


def test_remove_document_is_a_safe_no_op_for_an_unknown_id(config):
    index = KnowledgeIndex(config=config)
    assert do_remove(index, "not-a-real-id", config=config) is False


def test_index_integrity_check_is_clean_after_normal_use(config):
    index = KnowledgeIndex(config=config)
    add_document(index, FIXTURES_DIR / "solid_modeling_basics.md", config=config)
    add_document(index, FIXTURES_DIR / "bottle_design_notes.md", config=config)
    assert index.integrity_check() == []


def test_index_integrity_check_detects_a_chunk_count_mismatch(config):
    index = KnowledgeIndex(config=config)
    record = add_document(index, FIXTURES_DIR / "solid_modeling_basics.md", config=config)
    index.documents[record.document_id].chunk_count = 999  # corrupt it directly
    issues = index.integrity_check()
    assert any("chunk_count" in issue for issue in issues)


def test_index_rebuild_refits_vectors_without_losing_chunks(config):
    index = KnowledgeIndex(config=config)
    add_document(index, FIXTURES_DIR / "solid_modeling_basics.md", config=config)
    n_chunks_before = len(index.chunks)
    index.rebuild()
    assert len(index.chunks) == n_chunks_before
    assert index.chunk_vectors.shape[0] == n_chunks_before


def test_index_save_and_load_round_trip(config):
    index = KnowledgeIndex(config=config)
    record = add_document(index, FIXTURES_DIR / "solid_modeling_basics.md", config=config)
    index.save()

    loaded = KnowledgeIndex.load(config=config)
    assert loaded.has_document(record.document_id)
    assert len(loaded.chunks) == len(index.chunks)
    assert loaded.integrity_check() == []


def test_index_load_with_no_saved_file_returns_an_empty_valid_index(config):
    index = KnowledgeIndex.load(config=config)
    assert index.documents == {}
    assert index.chunks == []
    assert index.integrity_check() == []
