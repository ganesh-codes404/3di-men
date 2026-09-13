import pytest

from threedimen.dataset import DatasetManifest, DatasetStore
from threedimen.geometry.ingest import ingest_mesh
from threedimen.quality import assess_quality
from threedimen.teacher import MockTeacher


def _make_sample(store: DatasetStore, prompt: str, category: str = "bracket"):
    artifact = MockTeacher().generate(prompt)
    result = ingest_mesh(artifact.mesh_path, strict=True)
    report = assess_quality(result.surface)
    return store.add_sample(artifact, result, category=category, quality=report.to_dict())


def test_add_sample_writes_expected_files(tmp_path):
    store = DatasetStore(tmp_path)
    sample = _make_sample(store, "a bracket")

    assert (store.root / sample.raw_mesh_path).exists()
    assert (store.root / sample.processed_mesh_path).exists()
    assert (store.root / f"{sample.surface_representation_path}.json").exists()
    assert (store.root / f"{sample.surface_representation_path}.npz").exists()


def test_add_sample_persists_across_store_instances(tmp_path):
    store1 = DatasetStore(tmp_path)
    sample = _make_sample(store1, "a pump")

    store2 = DatasetStore(tmp_path)
    manifest = store2.load_manifest()
    assert len(manifest.samples) == 1
    assert manifest.samples[0].id == sample.id
    assert manifest.samples[0].prompt == "a pump"


def test_stats_counts_by_category_and_source(tmp_path):
    store = DatasetStore(tmp_path)
    _make_sample(store, "a bracket one", category="bracket")
    _make_sample(store, "a bracket two", category="bracket")
    _make_sample(store, "a pump", category="pump")

    stats = store.stats()
    assert stats["sample_count"] == 3
    assert stats["by_category"]["bracket"] == 2
    assert stats["by_category"]["pump"] == 1
    assert stats["by_source_model"]["mock"] == 3
    assert stats["quality_accepted"] + stats["quality_rejected"] == 3


def test_validate_detects_missing_file(tmp_path):
    store = DatasetStore(tmp_path)
    sample = _make_sample(store, "a bracket")
    (store.root / sample.processed_mesh_path).unlink()

    problems = store.validate()
    assert any(sample.id in p and "processed_mesh_path" in p for p in problems)


def test_validate_clean_dataset_has_no_problems(tmp_path):
    store = DatasetStore(tmp_path)
    _make_sample(store, "a bracket")
    assert store.validate() == []


def test_manifest_rejects_incompatible_major_version():
    with pytest.raises(ValueError):
        DatasetManifest.from_dict({"dataset_version": "99.0.0", "samples": []})


def test_manifest_accepts_same_major_version():
    manifest = DatasetManifest.from_dict({"dataset_version": "0.1.5", "samples": []})
    assert manifest.dataset_version == "0.1.5"


def test_add_sample_rejects_invalid_surface(tmp_path):
    from threedimen.geometry.ingest import IngestResult, ValidationReport
    from threedimen.geometry.representation import SurfaceRepresentation
    from threedimen.teacher.base import MeshArtifact

    store = DatasetStore(tmp_path)
    artifact = MockTeacher().generate("whatever")
    bad_result = IngestResult(
        surface=SurfaceRepresentation(patches=[]),
        validation=ValidationReport(ok=False, errors=["fake failure"]),
        repairs_applied=[],
        warnings=[],
        source_path=None,
    )
    with pytest.raises(ValueError):
        store.add_sample(artifact, bad_result)
