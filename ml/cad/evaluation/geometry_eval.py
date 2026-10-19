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
    hole_axes: list[str] = field(default_factory=list)  # row-aligned with hole_circularity/hole_diameters_mm --
        # "x"/"y"/"z", whichever world axis the hole's own axis runs along. Confirmed real gap this
        # closes: _detect_holes_and_cylinders used to only ever test the Z axis at all (see that
        # function's own prior docstring, and data/procedural_generator.py's BracketParams, whose two
        # arms drill horizontally -- Y-axis and X-axis respectively -- and were INVISIBLE to the old
        # detector, reading detected_hole_count=0 for every bracket regardless of how correctly it was
        # actually drilled). Still axis-ALIGNED only (x/y/z, never an arbitrary tilted direction) --
        # every hole this project's own builders ever produce runs along one of the three world axes
        # (every primitive/relationship in this codebase is built or rotated in 90-degree increments,
        # see assembly_plan._ALIGN_LOCAL_Z_TO), so a true arbitrary-direction detector would be solving
        # a problem that doesn't actually occur here.
    hole_through: list[bool] = field(default_factory=list)  # row-aligned with hole_circularity -- True when
        # the hole's own cylindrical surface spans (within a tolerance) the mesh's FULL extent along its
        # axis, i.e. a genuine through-hole rather than a blind bore stopping partway. A geometric
        # heuristic (coverage of the component's own extent against the mesh's extent along the same
        # axis), not a feature-tree-derived fact (this project only ever has meshes to work from) -- see
        # this module's own _classify_wall_component docstring for exactly how it's computed.
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
            "hole_axes": self.hole_axes,
            "hole_through": self.hole_through,
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


_AXIS_NAMES = ("x", "y", "z")


def _perp_indices(axis: int) -> tuple[int, int]:
    """The two coordinate indices OTHER than `axis` -- the plane a hole
    running along `axis` is circular in, generalizing the old hardcoded
    [:, :2] (XY) slicing to work for any of the 3 world axes."""
    return tuple(i for i in range(3) if i != axis)  # type: ignore[return-value]


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


def _wall_components(mesh: trimesh.Trimesh, axis: int = 2, normal_tol: float = 0.35, exclude: np.ndarray | None = None):
    """Connected components of faces whose normal runs roughly PERPENDICULAR
    to `axis` (i.e. candidate cylindrical/fillet side-wall faces for a hole
    or boss running along `axis`, as opposed to the end-cap faces at either
    end of it). Returns a list of face-index arrays.

    `exclude` (a boolean mask over ALL mesh faces, True = already claimed by
    an earlier axis pass) lets _detect_holes_and_cylinders check multiple
    axes without a face being double-counted into two different components
    -- see that function's own docstring for why this is cheap insurance
    rather than a correctness requirement (a real hole's own wall faces
    reliably fail every OTHER axis's circularity/full-loop test already,
    see that function's docstring, but excluding claimed faces up front is
    simpler to reason about than relying on that alone)."""
    normals = mesh.face_normals
    wall_mask = np.abs(normals[:, axis]) < normal_tol
    if exclude is not None:
        wall_mask = wall_mask & ~exclude
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


def _classify_wall_component(mesh: trimesh.Trimesh, faces: np.ndarray, axis: int = 2, radius_residual_tol: float = 0.08):
    """Fit a circle, in the plane perpendicular to `axis`, to a candidate
    side-wall patch and classify it.

    Returns a dict with is_circular, is_full_loop, is_concave, radius,
    circularity, axis, is_through, or None if the patch has too few faces
    to fit reliably.

    `is_through` is a geometric heuristic, not a feature-tree fact (this
    project only ever has a mesh to work from): True when the component's
    own extent along `axis` covers at least 90% of the MESH's full extent
    along that same axis -- a genuine through-hole's cylindrical wall runs
    the full thickness of the part it passes through, while a blind bore's
    wall stops well short of the far side."""
    if len(faces) < 6:
        return None
    u_idx, v_idx = _perp_indices(axis)
    tri_centroids = mesh.triangles_center[faces]
    uv = tri_centroids[:, [u_idx, v_idx]]
    cx, cy, r, residual = _fit_circle_xy(uv)
    if r < 1e-6 or residual > radius_residual_tol:
        return None

    angles = np.arctan2(uv[:, 1] - cy, uv[:, 0] - cx)
    sorted_angles = np.sort(angles)
    gaps = np.diff(np.concatenate([sorted_angles, [sorted_angles[0] + 2 * np.pi]]))
    max_gap_deg = np.degrees(gaps.max())
    is_full_loop = max_gap_deg < 60.0  # covers at least ~300 degrees of the circle

    normals_uv = mesh.face_normals[faces][:, [u_idx, v_idx]]
    normals_uv = normals_uv / np.clip(np.linalg.norm(normals_uv, axis=1, keepdims=True), 1e-9, None)
    outward = uv - [cx, cy]
    outward = outward / np.clip(np.linalg.norm(outward, axis=1, keepdims=True), 1e-9, None)
    concavity = float(np.mean(np.sum(normals_uv * outward, axis=1)))  # -1 concave (hole) .. +1 convex (boss/fillet)

    # VERTEX extent, not triangle-centroid extent: a cylindrical wall with
    # no height subdivision (the normal case -- a straight bore has no
    # reason to be cut into height bands) is represented by just 2
    # triangles per circumferential section, each spanning the FULL
    # height in one long thin triangle. That triangle's CENTROID then
    # lands 1/3 of the way in from each end (not at the true endpoint),
    # so using triangle_center here (confirmed directly: a 40mm-thick
    # box with a full through-hole) made every through-hole's own
    # measured span come out as 1/3 of its real length, misclassifying
    # it as a blind hole. Vertices give the real geometric extent
    # regardless of how coarsely the wall happens to be triangulated.
    wall_vertex_ids = np.unique(mesh.faces[faces].ravel())
    along_axis = mesh.vertices[wall_vertex_ids, axis]
    component_span = float(along_axis.max() - along_axis.min())
    mesh_extent = float(mesh.bounds[1][axis] - mesh.bounds[0][axis])
    is_through = mesh_extent > 1e-6 and (component_span / mesh_extent) >= 0.9

    return {
        "center": (cx, cy),
        "radius": r,
        "circularity": float(max(0.0, 1.0 - residual)),
        "is_full_loop": is_full_loop,
        "is_concave": concavity < -0.5,
        "is_convex": concavity > 0.5,
        "axis": _AXIS_NAMES[axis],
        "is_through": is_through,
    }


def _detect_holes_and_cylinders(mesh: trimesh.Trimesh) -> dict:
    """Classify side-wall patches into through/blind holes (concave, full
    360 degree loop, circular in cross-section) vs. general cylindrical
    patches (circular in cross-section, convex or concave, full loop or
    partial arc like a fillet).

    Tries all 3 world axes (not just Z) -- confirmed real gap this closes:
    the old Z-only version read detected_hole_count=0 for any part whose
    holes run horizontally (e.g. data/procedural_generator.py's
    BracketParams, whose two arms drill along Y and X respectively,
    documented there as a known, deferred limitation until now). Still
    axis-ALIGNED only, never an arbitrary tilted direction -- every hole
    this project's builders can actually produce runs along one of the
    three world axes (see assembly_plan._ALIGN_LOCAL_Z_TO), so detecting
    anything beyond that would be solving a problem that doesn't occur
    here, not a real gap.

    Each face is attributed to at most ONE axis's wall component (`exclude`
    threaded through _wall_components across passes) -- a genuine hole's
    own wall reliably fails the circularity/full-loop test under every
    OTHER axis already (its face normals are mostly NOT perpendicular to a
    different axis), but excluding already-claimed faces up front is
    simpler to reason about than relying on that alone, and guarantees no
    double-counted hole across axes."""
    hole_circularity = []
    hole_diameters_mm = []
    hole_axes = []
    hole_through = []
    cylindrical_patches = 0
    claimed = np.zeros(len(mesh.faces), dtype=bool)
    for axis in range(3):
        for faces in _wall_components(mesh, axis=axis, exclude=claimed):
            result = _classify_wall_component(mesh, faces, axis=axis)
            if result is None:
                continue
            claimed[faces] = True
            cylindrical_patches += 1
            if result["is_concave"] and result["is_full_loop"]:
                hole_circularity.append(result["circularity"])
                hole_diameters_mm.append(result["radius"] * 2.0)
                hole_axes.append(result["axis"])
                hole_through.append(result["is_through"])
    return {
        "hole_count": len(hole_circularity),
        "hole_circularity": hole_circularity,
        "hole_diameters_mm": hole_diameters_mm,
        "hole_axes": hole_axes,
        "hole_through": hole_through,
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
        hole_axes=features["hole_axes"],
        hole_through=features["hole_through"],
        mirror_symmetry_score=_mirror_symmetry_score(mesh),
    )
