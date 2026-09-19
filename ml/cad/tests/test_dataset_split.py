from pathlib import Path

from data.dataset_split import assign_split
from data.prompt_dataset import sample_prompt_dataset
from data.prompt_store import read_split, write_dataset


def test_assign_split_is_pure_function_of_inputs():
    key = "some-canonical-key"
    a = assign_split(key, seed=0)
    b = assign_split(key, seed=0)
    assert a == b
    assert assign_split(key, seed=0) in ("train", "val", "test")


def test_no_canonical_key_spans_multiple_splits(tmp_path: Path):
    samples = sample_prompt_dataset({"plate": 200, "flange": 200}, seed=0)
    write_dataset(tmp_path, samples, seed=0)

    key_to_splits: dict[str, set[str]] = {}
    for split in ("train", "val", "test"):
        for s in read_split(tmp_path, split):
            key_to_splits.setdefault(s.canonical_key, set()).add(split)

    offenders = {k: v for k, v in key_to_splits.items() if len(v) > 1}
    assert offenders == {}, f"{len(offenders)} canonical_key(s) span multiple splits -- leakage"


def test_manifest_reports_row_counts_and_distinct_object_counts(tmp_path: Path):
    samples = sample_prompt_dataset({"plate": 200, "flange": 200}, seed=0)
    manifest = write_dataset(tmp_path, samples, seed=0)

    total_rows = sum(sum(fam_counts.values()) for fam_counts in manifest.counts.values())
    assert total_rows == 400

    # discrete-grid duplication is expected (small sampler grids) -- this
    # asserts the manifest actually SURFACES that, rather than hiding it
    # behind a row count that looks bigger than the real object diversity
    total_unique = sum(sum(fam_counts.values()) for fam_counts in manifest.unique_object_counts.values())
    assert total_unique <= total_rows
