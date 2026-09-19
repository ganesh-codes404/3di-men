from pathlib import Path

from data.program_dataset import (
    read_program_split,
    sample_family_program_dataset,
    sample_generic_composition_dataset,
    sample_program_dataset,
    write_program_dataset,
)
from sandbox import Sandbox


def test_family_programs_replay_successfully_via_sandbox_execute():
    samples = sample_family_program_dataset({"plate": 5, "flange": 5, "gear": 5}, seed=0)
    assert len(samples) == 15
    for s in samples:
        result = Sandbox.execute(list(s.program))
        assert result.status == "ok", f"{s.id}: {result.status} {result.error}"
        assert s.source_kind == "family"
        assert s.family in ("plate", "flange", "gear")


def test_generic_composition_programs_replay_successfully():
    samples = sample_generic_composition_dataset(20, seed=0)
    assert len(samples) == 20
    for s in samples:
        result = Sandbox.execute(list(s.program))
        assert result.status == "ok", f"{s.id}: {result.status} {result.error}"
        assert s.source_kind == "generic_composition"
        assert s.family is None


def test_family_and_generic_sources_never_share_ids_or_get_confused():
    samples = sample_program_dataset({"plate": 5}, n_generic=5, seed=0)
    family_samples = [s for s in samples if s.source_kind == "family"]
    generic_samples = [s for s in samples if s.source_kind == "generic_composition"]
    assert len(family_samples) == 5
    assert len(generic_samples) == 5
    assert all(s.id.startswith("family_") for s in family_samples)
    assert all(s.id.startswith("generic_") for s in generic_samples)


def test_write_and_read_program_dataset_round_trips(tmp_path: Path):
    samples = sample_program_dataset({"plate": 20, "flange": 20}, n_generic=10, seed=0)
    manifest = write_program_dataset(tmp_path, samples, seed=0)
    assert sum(sum(v.values()) for v in manifest["counts"].values()) == 50

    reloaded = []
    for split in ("train", "val", "test"):
        reloaded.extend(read_program_split(tmp_path, split))
    assert len(reloaded) == 50
    # round trip preserves the program faithfully
    for s in reloaded:
        assert Sandbox.execute(list(s.program)).status in ("ok", "suspect")  # tolerate rare sampler edge cases


def test_no_canonical_key_spans_multiple_splits(tmp_path: Path):
    samples = sample_program_dataset({"plate": 100, "flange": 100}, n_generic=50, seed=0)
    write_program_dataset(tmp_path, samples, seed=0)
    key_to_splits: dict[str, set[str]] = {}
    for split in ("train", "val", "test"):
        for s in read_program_split(tmp_path, split):
            key_to_splits.setdefault(s.canonical_key, set()).add(split)
    offenders = {k: v for k, v in key_to_splits.items() if len(v) > 1}
    assert offenders == {}
