"""Deterministic, network-free teacher for tests and local development.

Returns a simple procedural primitive chosen deterministically from a hash of
the prompt, so the rest of the pipeline (ingestion, dataset writing, quality
filtering) is fully testable without any real generative model, GPU, or
network access."""

from __future__ import annotations

import hashlib
import tempfile
import time
from pathlib import Path

import trimesh

from .base import MeshArtifact, TeacherModel

_SHAPES = ["box", "sphere", "cylinder", "capsule"]


class MockTeacher(TeacherModel):
    name = "mock"

    def __init__(self, cache_dir: str | Path | None = None):
        self.cache_dir = Path(cache_dir) if cache_dir else Path(tempfile.gettempdir()) / "threedimen_mock_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def generate(self, prompt: str) -> MeshArtifact:
        t0 = time.perf_counter()
        seed = int(hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:8], 16)
        shape = _SHAPES[seed % len(_SHAPES)]
        size = 0.5 + (seed % 100) / 100.0

        if shape == "box":
            mesh = trimesh.creation.box(extents=[size, size * 0.8, size * 1.2])
        elif shape == "sphere":
            mesh = trimesh.creation.icosphere(radius=size, subdivisions=2)
        elif shape == "cylinder":
            mesh = trimesh.creation.cylinder(radius=size * 0.5, height=size * 1.5, sections=24)
        else:
            mesh = trimesh.creation.capsule(radius=size * 0.4, height=size)

        mesh_path = self.cache_dir / f"{seed:08x}.stl"
        mesh.export(mesh_path)

        return MeshArtifact(
            mesh_path=mesh_path,
            prompt=prompt,
            source_model=self.name,
            generation_time_seconds=time.perf_counter() - t0,
            metadata={"shape": shape, "deterministic_seed": seed},
        )
