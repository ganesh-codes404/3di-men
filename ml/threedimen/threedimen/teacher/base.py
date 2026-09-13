"""TeacherModel interface (Part 3).

Hunyuan3D is one implementation, not a hardcoded assumption. Swapping to
another text-to-3D model, an image-to-3D model, a local model, or a future
proprietary teacher means implementing this same interface -- nothing in
geometry/ingest.py or dataset/ ever imports a specific teacher by name."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class MeshArtifact:
    """What a teacher hands back: a raw mesh file plus provenance. Kept as a
    file path (not an in-memory mesh) so large outputs don't have to live in
    memory across the generate -> ingest -> dataset-write pipeline."""

    mesh_path: Path
    prompt: str
    source_model: str
    generation_time_seconds: float
    metadata: dict[str, Any] = field(default_factory=dict)


class TeacherModel(ABC):
    """generate(prompt) -> MeshArtifact."""

    name: str

    @abstractmethod
    def generate(self, prompt: str) -> MeshArtifact: ...
