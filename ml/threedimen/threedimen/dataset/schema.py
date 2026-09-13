"""Part 4: dataset sample schema. Versioned and reproducible -- every record
carries the exact pipeline/representation versions it was produced with, so
a schema change later never silently reinterprets old samples."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from .. import DATASET_VERSION, PIPELINE_VERSION, REPRESENTATION_VERSION


@dataclass
class SampleMetadata:
    category: str | None
    bounding_box: dict
    vertex_count: int
    face_count: int
    normalization: dict
    generation_source: str
    generation_time_seconds: float
    pipeline_version: str = PIPELINE_VERSION
    representation_version: str = REPRESENTATION_VERSION
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "SampleMetadata":
        return cls(**d)


@dataclass
class DatasetSample:
    """One training example. Only ever stores *references* to files under
    the dataset root -- raw_mesh_path is exactly what the teacher produced,
    processed_mesh_path is a quick-viewing export, surface_representation_path
    is the authoritative geometry (see geometry/representation.py)."""

    id: str
    prompt: str
    source_model: str
    raw_mesh_path: str
    processed_mesh_path: str
    surface_representation_path: str
    metadata: SampleMetadata
    quality: dict | None = None

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "prompt": self.prompt,
            "source_model": self.source_model,
            "raw_mesh_path": self.raw_mesh_path,
            "processed_mesh_path": self.processed_mesh_path,
            "surface_representation_path": self.surface_representation_path,
            "metadata": self.metadata.to_dict(),
            "quality": self.quality,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "DatasetSample":
        return cls(
            id=d["id"],
            prompt=d["prompt"],
            source_model=d["source_model"],
            raw_mesh_path=d["raw_mesh_path"],
            processed_mesh_path=d["processed_mesh_path"],
            surface_representation_path=d["surface_representation_path"],
            metadata=SampleMetadata.from_dict(d["metadata"]),
            quality=d.get("quality"),
        )


@dataclass
class DatasetManifest:
    dataset_version: str = DATASET_VERSION
    samples: list[DatasetSample] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"dataset_version": self.dataset_version, "samples": [s.to_dict() for s in self.samples]}

    @classmethod
    def from_dict(cls, d: dict) -> "DatasetManifest":
        version = d.get("dataset_version", DATASET_VERSION)
        if version.split(".")[0] != DATASET_VERSION.split(".")[0]:
            raise ValueError(
                f"dataset manifest version {version} has a different major version than this "
                f"code's DATASET_VERSION {DATASET_VERSION} -- schema may be incompatible"
            )
        return cls(
            dataset_version=version,
            samples=[DatasetSample.from_dict(s) for s in d.get("samples", [])],
        )
