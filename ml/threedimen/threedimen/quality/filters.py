"""Part 6: quality filtering.

Computed from a SurfaceRepresentation (via its merged trimesh), never from a
specific teacher, so it applies identically to Hunyuan3D output, mock output,
or any future teacher. Deliberately simple duplicate detection (a rounded
bbox-aspect/vertex/face/volume key, not a real perceptual hash) -- per the
brief, real CLIP/geometry-quality scoring is future work, not this pass.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..geometry.representation import SurfaceRepresentation


@dataclass
class QualityThresholds:
    min_vertex_count: int = 8
    max_vertex_count: int = 2_000_000
    min_face_count: int = 4
    max_degenerate_face_fraction: float = 0.05
    max_disconnected_components: int = 8
    min_bbox_diagonal: float = 1e-6
    max_bbox_diagonal: float = 1e6
    # Hunyuan3D output is frequently non-watertight (small marching-cubes
    # cracks) without being a bad sample -- don't reject on this alone unless
    # the caller specifically wants watertight-only data.
    require_watertight: bool = False


@dataclass
class QualityReport:
    valid: bool
    reasons: list[str] = field(default_factory=list)
    watertight: bool = False
    manifold: bool = False
    vertex_count: int = 0
    face_count: int = 0
    degenerate_face_count: int = 0
    disconnected_components: int = 0
    non_manifold_edge_count: int = 0
    bbox_sane: bool = True
    duplicate_key: str | None = None

    def to_dict(self) -> dict:
        return {
            "valid": self.valid,
            "reasons": self.reasons,
            "watertight": self.watertight,
            "manifold": self.manifold,
            "vertex_count": self.vertex_count,
            "face_count": self.face_count,
            "degenerate_face_count": self.degenerate_face_count,
            "disconnected_components": self.disconnected_components,
            "non_manifold_edge_count": self.non_manifold_edge_count,
            "bbox_sane": self.bbox_sane,
            "duplicate_key": self.duplicate_key,
        }


def _duplicate_key(mesh, precision: int = 2) -> str:
    bb = mesh.bounds
    size = bb[1] - bb[0]
    longest = max(float(size.max()), 1e-9)
    aspect = tuple(round(float(s / longest), precision) for s in size)
    try:
        volume = round(float(mesh.volume) if mesh.is_watertight else float(mesh.convex_hull.volume), 3)
    except Exception:
        volume = -1.0
    return f"{aspect}|{len(mesh.vertices)}|{len(mesh.faces)}|{volume}"


def assess_quality(
    surface: SurfaceRepresentation,
    *,
    thresholds: QualityThresholds | None = None,
    existing_duplicate_keys: "set[str] | None" = None,
) -> QualityReport:
    thresholds = thresholds or QualityThresholds()
    mesh = surface.to_trimesh()
    reasons: list[str] = []

    vertex_count = len(mesh.vertices)
    face_count = len(mesh.faces)
    degenerate = int(np.sum(mesh.area_faces < 1e-12)) if face_count else 0
    watertight = bool(mesh.is_watertight)
    manifold = bool(watertight and mesh.is_winding_consistent)
    disconnected = len(mesh.split(only_watertight=False)) if face_count else 0

    non_manifold_edges = 0
    if face_count:
        _, counts = np.unique(mesh.edges_sorted, axis=0, return_counts=True)
        non_manifold_edges = int(np.sum(counts > 2))

    bbox = surface.bounding_box
    bbox_sane = thresholds.min_bbox_diagonal <= bbox.diagonal <= thresholds.max_bbox_diagonal

    if vertex_count < thresholds.min_vertex_count:
        reasons.append(f"too few vertices ({vertex_count} < {thresholds.min_vertex_count})")
    if vertex_count > thresholds.max_vertex_count:
        reasons.append(f"too many vertices ({vertex_count} > {thresholds.max_vertex_count})")
    if face_count < thresholds.min_face_count:
        reasons.append(f"too few faces ({face_count} < {thresholds.min_face_count})")
    if face_count and degenerate / face_count > thresholds.max_degenerate_face_fraction:
        reasons.append(f"too many degenerate faces ({degenerate}/{face_count})")
    if disconnected > thresholds.max_disconnected_components:
        reasons.append(
            f"too many disconnected components ({disconnected} > {thresholds.max_disconnected_components})"
        )
    if not bbox_sane:
        reasons.append(f"bounding box diagonal {bbox.diagonal:.6g} outside sane range")
    if thresholds.require_watertight and not watertight:
        reasons.append("mesh is not watertight")

    duplicate_key = _duplicate_key(mesh)
    if existing_duplicate_keys is not None and duplicate_key in existing_duplicate_keys:
        reasons.append(f"near-duplicate of an existing sample (key {duplicate_key})")

    return QualityReport(
        valid=len(reasons) == 0,
        reasons=reasons,
        watertight=watertight,
        manifold=manifold,
        vertex_count=vertex_count,
        face_count=face_count,
        degenerate_face_count=degenerate,
        disconnected_components=disconnected,
        non_manifold_edge_count=non_manifold_edges,
        bbox_sane=bbox_sane,
        duplicate_key=duplicate_key,
    )


def is_acceptable(report: QualityReport) -> bool:
    return report.valid
