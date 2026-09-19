"""Storage for the synthetic prompt dataset: one manifest.json +
train/val/test .jsonl files. No per-sample directories -- unlike
threedimen's DatasetStore (which exists specifically to keep bulky mesh
files out of its manifest), this data is small structured JSON with no
binary payload, so per-sample dirs would be pure overhead.

Atomic writes (tmp file + rename) follow the same convention as
threedimen/dataset/store.py, for the same reason: a crash mid-write should
never leave a half-written file that looks valid.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from data.dataset_split import Split, assign_split
from data.prompt_dataset import PromptSample

_SPLITS: tuple[Split, ...] = ("train", "val", "test")


@dataclass
class DatasetManifest:
    version: str
    seed: int
    counts: dict[str, dict[str, int]]  # split -> family -> row count
    unique_object_counts: dict[str, dict[str, int]]  # split -> family -> DISTINCT canonical_key count

    def to_dict(self) -> dict:
        return {
            "version": self.version,
            "seed": self.seed,
            "counts": self.counts,
            "unique_object_counts": self.unique_object_counts,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "DatasetManifest":
        return cls(version=d["version"], seed=d["seed"], counts=d["counts"], unique_object_counts=d["unique_object_counts"])


def _atomic_write(path: Path, content: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(content)
    tmp.replace(path)


def write_dataset(root: Path, samples: list[PromptSample], seed: int, version: str = "1.0.0") -> DatasetManifest:
    root.mkdir(parents=True, exist_ok=True)
    by_split: dict[Split, list[PromptSample]] = {s: [] for s in _SPLITS}
    for sample in samples:
        by_split[assign_split(sample.canonical_key, seed=seed)].append(sample)

    counts: dict[str, dict[str, int]] = {}
    unique_counts: dict[str, dict[str, int]] = {}
    for split in _SPLITS:
        rows = by_split[split]
        content = "\n".join(json.dumps(s.to_dict()) for s in rows)
        _atomic_write(root / f"{split}.jsonl", content + ("\n" if content else ""))

        split_counts: dict[str, int] = {}
        split_unique_keys: dict[str, set[str]] = {}
        for s in rows:
            split_counts[s.part_family] = split_counts.get(s.part_family, 0) + 1
            split_unique_keys.setdefault(s.part_family, set()).add(s.canonical_key)
        counts[split] = split_counts
        unique_counts[split] = {fam: len(keys) for fam, keys in split_unique_keys.items()}

    manifest = DatasetManifest(version=version, seed=seed, counts=counts, unique_object_counts=unique_counts)
    _atomic_write(root / "manifest.json", json.dumps(manifest.to_dict(), indent=2))
    return manifest


def read_split(root: Path, split: Split) -> list[PromptSample]:
    path = root / f"{split}.jsonl"
    if not path.exists():
        return []
    lines = path.read_text().splitlines()
    return [PromptSample.from_dict(json.loads(line)) for line in lines if line.strip()]


def read_manifest(root: Path) -> DatasetManifest:
    return DatasetManifest.from_dict(json.loads((root / "manifest.json").read_text()))
