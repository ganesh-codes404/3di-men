"""Loads a DatasetStore's samples into (prompt, point_cloud) training pairs,
reusing Part 7's PointSampleEncoder rather than re-implementing sampling."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..dataset import DatasetStore
from ..geometry.representation import SurfaceRepresentation
from ..ml.encoder import PointSampleEncoder


@dataclass
class TrainingPair:
    prompt: str
    category: str | None
    points: np.ndarray  # (num_points, 3) -- already centered/scale-normalized by ingest


def load_training_pairs(root: str | Path, num_points: int = 512, seed: int = 0) -> list[TrainingPair]:
    store = DatasetStore(root)
    manifest = store.load_manifest()
    encoder = PointSampleEncoder(num_points=num_points, seed=seed)

    pairs = []
    for sample in manifest.samples:
        surface = SurfaceRepresentation.load(store.root / sample.surface_representation_path)
        encoded = encoder.encode(surface)
        pairs.append(
            TrainingPair(prompt=sample.prompt, category=sample.metadata.category, points=encoded.points)
        )
    return pairs
