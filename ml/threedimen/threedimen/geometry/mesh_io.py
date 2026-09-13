"""Loading/exporting standard mesh formats (Part 2). Deliberately thin --
trimesh already does this well; this module just pins the supported format
list and turns trimesh's exceptions into clear, 3di-men-specific errors."""

from __future__ import annotations

from pathlib import Path

import trimesh

from .errors import MeshLoadError

SUPPORTED_FORMATS = {".glb", ".gltf", ".obj", ".ply", ".stl"}


def load_mesh(path: str | Path) -> trimesh.Trimesh:
    path = Path(path)
    if not path.exists():
        raise MeshLoadError(f"no such file: {path}")
    if path.suffix.lower() not in SUPPORTED_FORMATS:
        raise MeshLoadError(
            f"unsupported mesh format {path.suffix!r} for {path.name} -- "
            f"supported: {sorted(SUPPORTED_FORMATS)}"
        )
    try:
        loaded = trimesh.load(path, force="mesh")
    except Exception as exc:  # trimesh raises a mix of exception types depending on format
        raise MeshLoadError(f"failed to load {path}: {exc}") from exc
    if not isinstance(loaded, trimesh.Trimesh):
        raise MeshLoadError(
            f"{path} did not load as a single mesh (got {type(loaded).__name__}) -- "
            "scenes with multiple disjoint objects aren't supported yet"
        )
    return loaded


def export_mesh(mesh_or_surface, path: str | Path) -> Path:
    """Accepts either a trimesh.Trimesh or anything with a .to_trimesh()
    method (i.e. a SurfaceRepresentation)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    mesh = mesh_or_surface.to_trimesh() if hasattr(mesh_or_surface, "to_trimesh") else mesh_or_surface
    if path.suffix.lower() not in SUPPORTED_FORMATS:
        raise MeshLoadError(
            f"unsupported export format {path.suffix!r} -- supported: {sorted(SUPPORTED_FORMATS)}"
        )
    mesh.export(path)
    return path
