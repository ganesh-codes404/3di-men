"""The 3di-men surface representation (Part 1 of the architecture).

Design decision (documented per the project's "explain tradeoffs before
locking one in" principle): three ways to model "a 3D surface" were
considered before writing this file.

  (a) Wrap a single polygon mesh (vertices/faces/normals) directly as "the"
      representation. Simplest, but a training target and a dataset format
      built around "it's just a mesh" can't grow into parametric surfaces,
      NURBS, or B-rep later without a breaking rewrite -- exactly what we
      were told not to paint ourselves into.
  (b) A layered model: an abstract SurfacePatch (a self-contained piece of
      surface, in whatever representation it's stored in) plus a top-level
      SurfaceRepresentation that composes one or more patches, with shared
      transform/material/semantic-part/bounding-box context living at the
      composition level, not duplicated per patch. Today exactly one
      concrete SurfacePatch exists -- MeshPatch, a plain triangle mesh --
      but the abstract contract (to_mesh_data/bounding_box/serialize) is
      what a future NurbsPatch or BRepFace would also implement. Adding one
      is a new subclass, not a redesign.
  (c) Build real parametric/NURBS/B-rep support now. Explicitly out of scope
      for this milestone -- would be significant unused machinery for a
      pipeline that currently only ever produces triangle meshes (Hunyuan3D
      output, procedural CAD parts).

(b) is what's implemented below. Every SurfacePatch must be able to
tessellate itself into a plain (vertices, faces, normals) mesh via
to_mesh_data() -- that's the common denominator the rest of the system
(export, evaluation, ML encoders) is allowed to depend on, so none of that
downstream code needs to know or care how many patch kinds exist.

Serialization is split deliberately: small/structural data (patch kinds,
counts, transform, metadata) goes to JSON; bulk numeric arrays (vertex/face
data) go to a companion .npz. This keeps the JSON side human-diffable and
avoids re-embedding large arrays inside a text format built for neither.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

import numpy as np

from .. import REPRESENTATION_VERSION


@dataclass
class BoundingBox:
    """Axis-aligned bounding box, in whatever units/frame the containing
    SurfaceRepresentation's patches are expressed in (see Transform)."""

    min: np.ndarray
    max: np.ndarray

    def __post_init__(self) -> None:
        self.min = np.asarray(self.min, dtype=np.float64).reshape(3)
        self.max = np.asarray(self.max, dtype=np.float64).reshape(3)

    @classmethod
    def from_points(cls, points: np.ndarray) -> "BoundingBox":
        points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        if len(points) == 0:
            return cls(np.zeros(3), np.zeros(3))
        return cls(points.min(axis=0), points.max(axis=0))

    @classmethod
    def union(cls, boxes: list["BoundingBox"]) -> "BoundingBox":
        if not boxes:
            return cls(np.zeros(3), np.zeros(3))
        mins = np.stack([b.min for b in boxes])
        maxs = np.stack([b.max for b in boxes])
        return cls(mins.min(axis=0), maxs.max(axis=0))

    @property
    def size(self) -> np.ndarray:
        return self.max - self.min

    @property
    def center(self) -> np.ndarray:
        return (self.max + self.min) / 2

    @property
    def diagonal(self) -> float:
        return float(np.linalg.norm(self.size))

    def to_dict(self) -> dict:
        return {"min": self.min.tolist(), "max": self.max.tolist()}

    @classmethod
    def from_dict(cls, d: dict) -> "BoundingBox":
        return cls(np.array(d["min"]), np.array(d["max"]))


@dataclass
class Transform:
    """A single affine transform applied to every patch in a
    SurfaceRepresentation at render/export time. Patches themselves are
    stored untransformed (in their own local frame) so the same patch data
    could in principle be reused under a different placement."""

    matrix: np.ndarray  # 4x4 homogeneous

    def __post_init__(self) -> None:
        self.matrix = np.asarray(self.matrix, dtype=np.float64).reshape(4, 4)

    @classmethod
    def identity(cls) -> "Transform":
        return cls(np.eye(4))

    @property
    def is_identity(self) -> bool:
        return bool(np.allclose(self.matrix, np.eye(4)))

    def apply(self, points: np.ndarray) -> np.ndarray:
        points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        homogeneous = np.hstack([points, np.ones((len(points), 1))])
        return (homogeneous @ self.matrix.T)[:, :3]

    def to_list(self) -> list:
        return self.matrix.tolist()

    @classmethod
    def from_list(cls, data: list) -> "Transform":
        return cls(np.asarray(data, dtype=np.float64))


@dataclass
class MaterialInfo:
    """Metadata-only material description. No texture/shader pipeline here --
    just enough to round-trip what a source glTF/GLB commonly carries."""

    name: str = "default"
    base_color: tuple[float, float, float, float] | None = None  # linear RGBA, 0-1
    metallic: float | None = None
    roughness: float | None = None

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "base_color": list(self.base_color) if self.base_color else None,
            "metallic": self.metallic,
            "roughness": self.roughness,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "MaterialInfo":
        bc = d.get("base_color")
        return cls(
            name=d.get("name", "default"),
            base_color=tuple(bc) if bc else None,
            metallic=d.get("metallic"),
            roughness=d.get("roughness"),
        )


@dataclass
class SemanticPart:
    """A named region of a patch (e.g. "handle", "bolt_hole_1"). Optional --
    empty for anything ingested without segmentation. face_indices=None means
    the whole patch is this part."""

    label: str
    patch_index: int
    face_indices: np.ndarray | None = None

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "patch_index": self.patch_index,
            "face_indices": None if self.face_indices is None else self.face_indices.tolist(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "SemanticPart":
        fi = d.get("face_indices")
        return cls(
            label=d["label"],
            patch_index=d["patch_index"],
            face_indices=None if fi is None else np.asarray(fi, dtype=np.int64),
        )


class SurfacePatch(ABC):
    """Abstract contract every surface-patch kind must satisfy. MeshPatch is
    the only implementation today; a future NurbsPatch/BRepFace would
    implement the same five methods and register itself via `kind` --
    nothing else in this file or in geometry/ingest.py, dataset/, or the ML
    interfaces needs to change."""

    kind: ClassVar[str]
    _registry: ClassVar[dict[str, type["SurfacePatch"]]] = {}

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if "kind" in cls.__dict__:
            SurfacePatch._registry[cls.kind] = cls

    @abstractmethod
    def to_mesh_data(self) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        """Tessellate to (vertices Nx3, faces Mx3, normals Nx3|None). This is
        the one operation every downstream consumer (export, evaluation,
        SurfaceEncoder) is allowed to rely on regardless of patch kind."""

    @abstractmethod
    def bounding_box(self) -> BoundingBox: ...

    @abstractmethod
    def _array_data(self) -> dict[str, np.ndarray]:
        """Named numpy arrays this patch owns, packed into the parent
        SurfaceRepresentation's .npz on save."""

    @abstractmethod
    def _to_meta(self) -> dict:
        """JSON-safe scalar/small metadata (no large arrays)."""

    @classmethod
    @abstractmethod
    def _from_meta_and_arrays(cls, meta: dict, arrays: dict[str, np.ndarray]) -> "SurfacePatch": ...

    def to_dict(self) -> dict:
        return {"kind": self.kind, "meta": self._to_meta()}

    def array_items(self, prefix: str) -> dict[str, np.ndarray]:
        return {f"{prefix}:{key}": value for key, value in self._array_data().items()}

    @classmethod
    def from_dict(cls, d: dict, prefix: str, arrays: dict[str, np.ndarray]) -> "SurfacePatch":
        subclass = cls._registry.get(d["kind"])
        if subclass is None:
            raise ValueError(
                f"unknown surface patch kind {d['kind']!r} -- known kinds: {sorted(cls._registry)}"
            )
        own_prefix = prefix + ":"
        own_arrays = {k[len(own_prefix):]: v for k, v in arrays.items() if k.startswith(own_prefix)}
        return subclass._from_meta_and_arrays(d["meta"], own_arrays)


class MeshPatch(SurfacePatch):
    """A plain triangle-mesh patch. Every mesh ingested today (from Hunyuan3D,
    from ml/cad's procedural generator, from any standard file format) becomes
    exactly one of these."""

    kind: ClassVar[str] = "mesh"

    def __init__(self, vertices: np.ndarray, faces: np.ndarray, normals: np.ndarray | None = None):
        self.vertices = np.asarray(vertices, dtype=np.float64).reshape(-1, 3)
        self.faces = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
        self.normals = None if normals is None else np.asarray(normals, dtype=np.float64).reshape(-1, 3)

    def to_mesh_data(self) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        return self.vertices, self.faces, self.normals

    def bounding_box(self) -> BoundingBox:
        return BoundingBox.from_points(self.vertices)

    def _array_data(self) -> dict[str, np.ndarray]:
        data = {"vertices": self.vertices, "faces": self.faces}
        if self.normals is not None:
            data["normals"] = self.normals
        return data

    def _to_meta(self) -> dict:
        return {
            "vertex_count": int(len(self.vertices)),
            "face_count": int(len(self.faces)),
            "has_normals": self.normals is not None,
        }

    @classmethod
    def _from_meta_and_arrays(cls, meta: dict, arrays: dict[str, np.ndarray]) -> "MeshPatch":
        return cls(arrays["vertices"], arrays["faces"], arrays.get("normals"))

    @classmethod
    def from_trimesh(cls, mesh: "trimesh.Trimesh") -> "MeshPatch":  # noqa: F821
        normals = mesh.vertex_normals if len(mesh.vertex_normals) == len(mesh.vertices) else None
        return cls(mesh.vertices, mesh.faces, normals)


@dataclass
class SurfaceRepresentation:
    """The top-level 3di-men geometry object. Composes one or more
    SurfacePatch instances plus the context shared across all of them
    (transform, materials, semantic parts, free-form metadata)."""

    patches: list[SurfacePatch]
    version: str = REPRESENTATION_VERSION
    parts: list[SemanticPart] = field(default_factory=list)
    transform: Transform = field(default_factory=Transform.identity)
    materials: list[MaterialInfo] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def bounding_box(self) -> BoundingBox:
        return BoundingBox.union([p.bounding_box() for p in self.patches])

    def to_trimesh(self) -> "trimesh.Trimesh":  # noqa: F821
        import trimesh

        meshes = []
        for patch in self.patches:
            v, f, n = patch.to_mesh_data()
            meshes.append(trimesh.Trimesh(vertices=v, faces=f, vertex_normals=n, process=False))
        merged = meshes[0] if len(meshes) == 1 else trimesh.util.concatenate(meshes)
        if not self.transform.is_identity:
            merged.apply_transform(self.transform.matrix)
        return merged

    @classmethod
    def from_trimesh(cls, mesh: "trimesh.Trimesh", **metadata: Any) -> "SurfaceRepresentation":  # noqa: F821
        return cls(patches=[MeshPatch.from_trimesh(mesh)], metadata=dict(metadata))

    def summary_dict(self) -> dict:
        """Small JSON-safe summary -- what dataset metadata (Part 4) embeds,
        as opposed to the full save()/load() round-trip format."""
        vertex_count = sum(len(p.to_mesh_data()[0]) for p in self.patches)
        face_count = sum(len(p.to_mesh_data()[1]) for p in self.patches)
        return {
            "representation_version": self.version,
            "patch_count": len(self.patches),
            "patch_kinds": sorted({p.kind for p in self.patches}),
            "vertex_count": vertex_count,
            "face_count": face_count,
            "part_count": len(self.parts),
            "bounding_box": self.bounding_box.to_dict(),
        }

    def to_dict(self) -> dict:
        return {
            "version": self.version,
            "patches": [p.to_dict() for p in self.patches],
            "parts": [sp.to_dict() for sp in self.parts],
            "transform": self.transform.to_list(),
            "materials": [m.to_dict() for m in self.materials],
            "metadata": self.metadata,
        }

    def _all_arrays(self) -> dict[str, np.ndarray]:
        arrays: dict[str, np.ndarray] = {}
        for i, patch in enumerate(self.patches):
            arrays.update(patch.array_items(f"patch{i}"))
        return arrays

    def save(self, path: str | Path) -> Path:
        """Writes <path>.json (structure/metadata) + <path>.npz (arrays).
        Returns the .json path. Caller passes a path with or without suffix;
        both are normalized to .json/.npz."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        stem = path.with_suffix("")
        json_path = stem.with_suffix(".json")
        npz_path = stem.with_suffix(".npz")
        json_path.write_text(json.dumps(self.to_dict(), indent=2))
        arrays = self._all_arrays()
        if arrays:
            np.savez_compressed(npz_path, **arrays)
        elif npz_path.exists():
            npz_path.unlink()
        return json_path

    @classmethod
    def load(cls, path: str | Path) -> "SurfaceRepresentation":
        path = Path(path)
        stem = path.with_suffix("")
        json_path = stem.with_suffix(".json")
        npz_path = stem.with_suffix(".npz")
        if not json_path.exists():
            raise FileNotFoundError(f"no surface representation at {json_path}")
        data = json.loads(json_path.read_text())
        arrays = dict(np.load(npz_path)) if npz_path.exists() else {}
        patches = [
            SurfacePatch.from_dict(pd, f"patch{i}", arrays) for i, pd in enumerate(data["patches"])
        ]
        return cls(
            patches=patches,
            version=data.get("version", REPRESENTATION_VERSION),
            parts=[SemanticPart.from_dict(d) for d in data.get("parts", [])],
            transform=Transform.from_list(data["transform"]) if "transform" in data else Transform.identity(),
            materials=[MaterialInfo.from_dict(d) for d in data.get("materials", [])],
            metadata=data.get("metadata", {}),
        )
