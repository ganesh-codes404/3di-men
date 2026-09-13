"""Part 2: mesh ingestion pipeline.

load -> validate -> normalize coordinate system -> repair (optional) ->
recenter + normalize scale -> convert to SurfaceRepresentation.

Nothing here silently changes geometry: every repair or normalization step
is appended to IngestResult.repairs_applied with what it did and why, and
normalization is fully invertible via the recorded parameters in
surface.metadata["normalization"] (original_vertices = stored_vertices /
scale_factor + original_center).

Coordinate convention: 3di-men's internal convention is Z-up. glTF/GLB is
Y-up per spec, so those two formats get rotated on ingest; OBJ/PLY/STL carry
no mandated up-axis convention and are passed through unrotated rather than
guessed at.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import trimesh

from .errors import MeshValidationError
from .mesh_io import load_mesh
from .representation import SurfaceRepresentation

_GLTF_Y_UP_TO_Z_UP = trimesh.transformations.rotation_matrix(-np.pi / 2, [1, 0, 0])
_GLTF_FORMATS = {".glb", ".gltf"}


@dataclass
class ValidationReport:
    ok: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"ok": self.ok, "errors": self.errors, "warnings": self.warnings}


@dataclass
class IngestResult:
    surface: SurfaceRepresentation
    validation: ValidationReport
    repairs_applied: list[str]
    warnings: list[str]
    source_path: Path | None


def _degenerate_area_threshold(mesh: trimesh.Trimesh) -> float:
    """A face-area cutoff scaled to the mesh's own size, not a fixed
    absolute value. A fixed epsilon like 1e-12 misclassifies every face of a
    legitimately tiny (but internally consistent) mesh as degenerate --
    scaling by the bounding-box diagonal keeps the same *relative*
    sensitivity regardless of whether the mesh is in mm, meters, or a
    unit-cube-normalized frame."""
    diagonal = float(np.linalg.norm(mesh.bounds[1] - mesh.bounds[0])) if mesh.bounds is not None else 0.0
    return max((diagonal * 1e-6) ** 2, 1e-18)


def _validate_raw(mesh: trimesh.Trimesh) -> ValidationReport:
    errors: list[str] = []
    warnings: list[str] = []

    if len(mesh.vertices) == 0:
        errors.append("mesh has zero vertices")
    if len(mesh.faces) == 0:
        errors.append("mesh has zero faces")
    if len(mesh.vertices) and not np.all(np.isfinite(mesh.vertices)):
        errors.append("mesh contains non-finite (NaN/Inf) vertex coordinates")
    if len(mesh.faces) and mesh.faces.size and mesh.faces.max() >= len(mesh.vertices):
        errors.append("face indices reference out-of-range vertices")
    if errors:
        return ValidationReport(ok=False, errors=errors, warnings=warnings)

    area_eps = _degenerate_area_threshold(mesh)
    degenerate = int(np.sum(mesh.area_faces < area_eps))
    if degenerate:
        warnings.append(f"{degenerate} degenerate (near-zero-area) face(s)")
    if not mesh.is_watertight:
        warnings.append("mesh is not watertight")
    if not mesh.is_winding_consistent:
        warnings.append("mesh has inconsistent face winding")

    diagonal = float(np.linalg.norm(mesh.bounds[1] - mesh.bounds[0]))
    if diagonal < 1e-9:
        errors.append("mesh bounding box has ~zero size (degenerate/collapsed geometry)")

    return ValidationReport(ok=len(errors) == 0, errors=errors, warnings=warnings)


def ingest_mesh(
    source: str | Path | trimesh.Trimesh,
    *,
    normalize_scale: bool = True,
    target_diagonal: float = 1.0,
    repair: bool = True,
    strict: bool = False,
) -> IngestResult:
    """Run the full ingestion pipeline.

    strict=True raises MeshValidationError on any validation error instead of
    returning a failed IngestResult -- use strict mode where a bad mesh
    should stop a pipeline (e.g. dataset ingestion); use strict=False for a
    `3dimen mesh validate` style command that should report problems rather
    than raise.
    """
    source_path = Path(source) if isinstance(source, (str, Path)) else None
    mesh = load_mesh(source_path) if source_path is not None else source
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError(f"expected a trimesh.Trimesh or a file path, got {type(mesh)}")

    validation = _validate_raw(mesh)
    if not validation.ok:
        if strict:
            raise MeshValidationError("; ".join(validation.errors))
        return IngestResult(
            surface=SurfaceRepresentation(patches=[]),
            validation=validation,
            repairs_applied=[],
            warnings=validation.warnings,
            source_path=source_path,
        )

    repairs: list[str] = []
    mesh = mesh.copy()

    axis_converted = False
    if source_path is not None and source_path.suffix.lower() in _GLTF_FORMATS:
        mesh.apply_transform(_GLTF_Y_UP_TO_Z_UP)
        axis_converted = True
        repairs.append("rotated from glTF Y-up to 3di-men's internal Z-up convention")

    if repair:
        area_eps = _degenerate_area_threshold(mesh)
        degenerate_mask = mesh.area_faces < area_eps
        degenerate_before = int(np.sum(degenerate_mask))
        if 0 < degenerate_before < len(mesh.faces):
            mesh.update_faces(~degenerate_mask)
            mesh.remove_unreferenced_vertices()
            repairs.append(f"removed {degenerate_before} degenerate face(s)")
        elif degenerate_before >= len(mesh.faces) > 0:
            # Every face falls under the (scale-relative) threshold -- this
            # almost always means the mesh is just very small overall, not
            # that it's actually broken. Don't strip it down to nothing;
            # leave it for validation/quality checks downstream to judge.
            pass
        if len(mesh.faces) and not mesh.is_winding_consistent:
            mesh.fix_normals()
            repairs.append("fixed inconsistent face winding / normals")

    if len(mesh.faces) == 0 or mesh.bounds is None:
        post_repair_error = "mesh has no faces remaining after repair -- geometry is entirely degenerate"
        if strict:
            raise MeshValidationError(post_repair_error)
        return IngestResult(
            surface=SurfaceRepresentation(patches=[]),
            validation=ValidationReport(ok=False, errors=[post_repair_error], warnings=validation.warnings),
            repairs_applied=repairs,
            warnings=validation.warnings,
            source_path=source_path,
        )

    original_bounds = mesh.bounds.copy()
    original_center = (original_bounds[0] + original_bounds[1]) / 2
    original_diagonal = float(np.linalg.norm(original_bounds[1] - original_bounds[0]))

    # Recentering always happens (non-destructive repositioning). Scale
    # normalization is optional and, when applied, fully invertible via the
    # recorded scale_factor/original_center below.
    mesh.apply_translation(-original_center)
    scale_factor = 1.0
    if normalize_scale and original_diagonal > 1e-9:
        scale_factor = target_diagonal / original_diagonal
        mesh.apply_scale(scale_factor)
        repairs.append(f"rescaled by {scale_factor:.6g} to target bounding-box diagonal {target_diagonal}")

    surface = SurfaceRepresentation.from_trimesh(
        mesh,
        source_format=source_path.suffix.lower() if source_path else None,
        axis_converted=axis_converted,
        normalization={
            "recentered": True,
            "scale_normalized": normalize_scale,
            "scale_factor": scale_factor,
            "target_diagonal": target_diagonal,
            "original_center": original_center.tolist(),
            "original_bounding_box": {"min": original_bounds[0].tolist(), "max": original_bounds[1].tolist()},
            "original_diagonal": original_diagonal,
        },
    )
    return IngestResult(
        surface=surface,
        validation=validation,
        repairs_applied=repairs,
        warnings=validation.warnings,
        source_path=source_path,
    )
