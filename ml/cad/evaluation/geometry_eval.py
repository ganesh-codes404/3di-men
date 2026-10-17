"""CAD-aware geometry evaluation (docs/cad-finetuning-technical-report.md, Section 6).

Pure CPU (trimesh + numpy + scipy). Deliberately has no dependency on Hunyuan3D or
any generative model, so it can be validated against known-ground-truth procedural
parts before it is ever pointed at a model's output.

Two families of checks:
  - Geometry validity: is this a sane mesh at all (watertight, manifold, no
    degenerate faces)?
  - CAD characteristics: does it look like a machined/CAD part (planar faces,
    cylindrical bores, mirror symmetry, circular holes)?
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import trimesh


@dataclass
class GeometryReport:
    watertight: bool
    manifold: bool
    euler_number: int
    face_count: int
    vertex_count: int
    degenerate_face_count: int
    bounding_box_mm: tuple[float, float, float]

    planar_face_fraction: float  # fraction of surface area that lies on a near-planar patch
    cylindrical_patch_count: int
    detected_hole_count: int
    hole_circularity: list[float] = field(default_factory=list)  # 1.0 = perfect circle
    hole_diameters_mm: list[float] = field(default_factory=list)  # row-aligned with hole_circularity --
        # confirmed real gap this closes: _classify_wall_component already fits a circle to every
        # detected hole and computes its real radius, but that radius was discarded the moment
        # circularity was read off it -- nothing downstream could independently verify a claimed
        # bore/hole diameter (e.g. "a 12mm bore") against what was actually built, only that SOME
        # hole exists at all (detected_hole_count) and looks round (hole_circularity). See
        # spec_validation.validate_hole_diameters_against_description, which this feeds.
    mirror_symmetry_score: float = 0.0  # 1.0 = perfect mirror symmetry about best-fit plane

    def as_dict(self) -> dict:
        return {
            "watertight": self.watertight,
            "manifold": self.manifold,
            "euler_number": self.euler_number,
            "face_count": self.face_count,
            "vertex_count": self.vertex_count,
            "degenerate_face_count": self.degenerate_face_count,
            "bounding_box_mm": list(self.bounding_box_mm),
            "planar_face_fraction": self.planar_face_fraction,
            "cylindrical_patch_count": self.cylindrical_patch_count,
            "detected_hole_count": self.detected_hole_count,
            "hole_circularity": self.hole_circularity,
            "hole_diameters_mm": self.hole_diameters_mm,
            "mirror_symmetry_score": self.mirror_symmetry_score,
        }


def _degenerate_face_count(mesh: trimesh.Trimesh, area_eps: float = 1e-8) -> int:
    return int(np.sum(mesh.area_faces < area_eps))


def _planar_face_fraction(mesh: trimesh.Trimesh, angle_tol_deg: float = 3.0) -> float:
    """Group adjacent faces with near-identical normals; report the area share of
    the largest planar patches relative to total surface area."""
    try:
        groups = mesh.facets  # trimesh's coplanar face grouping
    except Exception:
        return 0.0
    if len(groups) == 0:
        return 0.0
    planar_area = sum(mesh.area_faces[g].sum() for g in groups)
    return float(planar_area / max(mesh.area, 1e-9))


def _fit_circle_xy(xy: np.ndarray) -> tuple[float, float, float, float]:
    """Algebraic (Kasa) circle fit to 2D points. Returns (cx, cy, radius, residual),
    where residual is the std-dev of per-point radius error, normalized by radius
    (0 = perfect circle)."""
    x, y = xy[:, 0], xy[:, 1]
    A = np.column_stack([x, y, np.ones_like(x)])
    b = x**2 + y**2
    coeffs, *_ = np.linalg.lstsq(A, b, rcond=None)
    cx, cy = coeffs[0] / 2, coeffs[1] / 2
    r2 = coeffs[2] + cx**2 + cy**2
    r = float(np.sqrt(max(r2, 0.0)))
    if r < 1e-9:
        return cx, cy, r, 1.0
    per_point_r = np.linalg.norm(xy - [cx, cy], axis=1)
    residual = float(np.std(per_point_r) / r)
    return float(cx), float(cy), r, residual


def _wall_components(mesh: trimesh.Trimesh, z_normal_tol: float = 0.35):
    """Connected components of faces whose normal is roughly horizontal (i.e.
    candidate cylindrical/fillet side-wall faces, as opposed to the top/bottom
    planar faces of an extruded part). Returns a list of face-index arrays.
    """
    normals = mesh.face_normals
    wall_mask = np.abs(normals[:, 2]) < z_normal_tol
    if not wall_mask.any():
        return []

    face_idx = np.where(wall_mask)[0]
    remap = {f: i for i, f in enumerate(face_idx)}
    adjacency = mesh.face_adjacency
    edges = [
        (remap[a], remap[b]) for a, b in adjacency if wall_mask[a] and wall_mask[b]
    ]
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    n = len(face_idx)
    if edges:
        rows, cols = zip(*edges)
        graph = coo_matrix((np.ones(len(edges)), (rows, cols)), shape=(n, n))
    else:
        graph = coo_matrix((n, n))
    _, labels = connected_components(graph, directed=False)

    components = []
    for label in np.unique(labels):
        components.append(face_idx[labels == label])
    return components


def _classify_wall_component(mesh: trimesh.Trimesh, faces: np.ndarray, radius_residual_tol: float = 0.08):
    """Fit a circle (in XY) to a candidate side-wall patch and classify it.

    Returns a dict with is_circular, is_full_loop, is_concave, radius, circularity,
    or None if the patch has too few faces to fit reliably.
    """
    if len(faces) < 6:
        return None
    tri_centroids = mesh.triangles_center[faces]
    cx, cy, r, residual = _fit_circle_xy(tri_centroids[:, :2])
    if r < 1e-6 or residual > radius_residual_tol:
        return None

    angles = np.arctan2(tri_centroids[:, 1] - cy, tri_centroids[:, 0] - cx)
    sorted_angles = np.sort(angles)
    gaps = np.diff(np.concatenate([sorted_angles, [sorted_angles[0] + 2 * np.pi]]))
    max_gap_deg = np.degrees(gaps.max())
    is_full_loop = max_gap_deg < 60.0  # covers at least ~300 degrees of the circle

    normals_xy = mesh.face_normals[faces][:, :2]
    normals_xy = normals_xy / np.clip(np.linalg.norm(normals_xy, axis=1, keepdims=True), 1e-9, None)
    outward = tri_centroids[:, :2] - [cx, cy]
    outward = outward / np.clip(np.linalg.norm(outward, axis=1, keepdims=True), 1e-9, None)
    concavity = float(np.mean(np.sum(normals_xy * outward, axis=1)))  # -1 concave (hole) .. +1 convex (boss/fillet)

    return {
        "center": (cx, cy),
        "radius": r,
        "circularity": float(max(0.0, 1.0 - residual)),
        "is_full_loop": is_full_loop,
        "is_concave": concavity < -0.5,
        "is_convex": concavity > 0.5,
    }


def _detect_holes_and_cylinders(mesh: trimesh.Trimesh) -> dict:
    """Classify vertical side-wall patches into through-holes (concave, full 360
    degree loop, circular in cross-section) vs. general cylindrical patches
    (circular in cross-section, convex or concave, full loop or partial arc like
    a fillet).

    Limitation: this only understands vertical (Z-axis) cylindrical features on
    largely axis-aligned, watertight, extruded parts -- it is the right amount of
    machinery for the v1 procedural dataset (Section 8 of the technical report),
    not a general-purpose feature recognizer for arbitrary CAD/mesh orientations.
    """
    hole_circularity = []
    hole_diameters_mm = []
    cylindrical_patches = 0
    for faces in _wall_components(mesh):
        result = _classify_wall_component(mesh, faces)
        if result is None:
            continue
        cylindrical_patches += 1
        if result["is_concave"] and result["is_full_loop"]:
            hole_circularity.append(result["circularity"])
            hole_diameters_mm.append(result["radius"] * 2.0)
    return {
        "hole_count": len(hole_circularity),
        "hole_circularity": hole_circularity,
        "hole_diameters_mm": hole_diameters_mm,
        "cylindrical_patch_count": cylindrical_patches,
    }


def _mirror_symmetry_score(mesh: trimesh.Trimesh, samples: int = 2000) -> float:
    """Reflect the mesh about the plane x=centroid_x (its intended symmetry plane
    for axis-aligned procedural parts) and measure how well the reflection matches
    the original via nearest-surface distance, normalized by the bounding box
    diagonal. 1.0 = perfect mirror symmetry, 0.0 = no resemblance.
    """
    centroid = mesh.bounds.mean(axis=0)
    mirrored = mesh.copy()
    mirrored.vertices[:, 0] = 2 * centroid[0] - mirrored.vertices[:, 0]
    mirrored.invert()  # flip winding after reflection

    pts, _ = trimesh.sample.sample_surface(mesh, samples)
    closest, dist, _ = mirrored.nearest.on_surface(pts)
    diag = np.linalg.norm(mesh.bounds[1] - mesh.bounds[0])
    mean_err = dist.mean() / max(diag, 1e-9)
    return float(max(0.0, 1.0 - mean_err * 10))  # empirically-scaled, not a calibrated metric


def evaluate_mesh(mesh: trimesh.Trimesh) -> GeometryReport:
    features = _detect_holes_and_cylinders(mesh)
    bb = mesh.bounds[1] - mesh.bounds[0]
    return GeometryReport(
        watertight=bool(mesh.is_watertight),
        manifold=bool(mesh.is_watertight and mesh.is_winding_consistent),
        euler_number=int(mesh.euler_number),
        face_count=int(len(mesh.faces)),
        vertex_count=int(len(mesh.vertices)),
        degenerate_face_count=_degenerate_face_count(mesh),
        bounding_box_mm=(float(bb[0]), float(bb[1]), float(bb[2])),
        planar_face_fraction=_planar_face_fraction(mesh),
        cylindrical_patch_count=features["cylindrical_patch_count"],
        detected_hole_count=features["hole_count"],
        hole_circularity=features["hole_circularity"],
        hole_diameters_mm=features["hole_diameters_mm"],
        mirror_symmetry_score=_mirror_symmetry_score(mesh),
    )
