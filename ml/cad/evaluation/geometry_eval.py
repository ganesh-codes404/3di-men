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
    hole_positions_mm: list[tuple[float, float, float]] = field(default_factory=list)  # row-aligned with
        # hole_circularity -- each detected hole's own real, measured 3D center (the circle-fit's own
        # (cx, cy) in the plane perpendicular to the hole's axis, plus that axis's own midpoint-along-run
        # coordinate). Confirmed real gap this closes: _classify_wall_component already fits this circle
        # and computes its center for every detected hole, but the center was discarded the moment
        # circularity/radius were read off it -- nothing downstream could compare two parts' own hole
        # LAYOUTS (e.g. a motor's mounting-hole pattern against its bracket's), only their counts and
        # diameters. See assembly_compatibility.py, which this feeds.
    detected_slot_count: int = 0  # the drill_slot "stadium" cross-section (two rounded ends, straight
        # sides) -- confirmed real gap this closes: a slot's own wall loop has a HIGH circle-fit
        # residual by construction (it isn't a circle), so without a dedicated fallback it was
        # discarded by the exact same check that rejects genuine noise, indistinguishable from "no
        # feature detected at all." See _fit_elongated_loop_uv's own docstring.
    slot_dimensions_mm: list[tuple[float, float]] = field(default_factory=list)  # row-aligned with
        # detected_slot_count -- (length, width) of each slot's own oriented bounding footprint (a PCA
        # fit, not a precisely-fitted stadium radius/length pair -- see _fit_elongated_loop_uv).
    slot_axes: list[str] = field(default_factory=list)  # row-aligned with slot_dimensions_mm
    slot_through: list[bool] = field(default_factory=list)  # row-aligned with slot_dimensions_mm
    slot_positions_mm: list[tuple[float, float, float]] = field(default_factory=list)  # row-aligned with slot_dimensions_mm
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
            "hole_positions_mm": [list(p) for p in self.hole_positions_mm],
            "detected_slot_count": self.detected_slot_count,
            "slot_dimensions_mm": [list(d) for d in self.slot_dimensions_mm],
            "slot_axes": self.slot_axes,
            "slot_through": self.slot_through,
            "slot_positions_mm": [list(p) for p in self.slot_positions_mm],
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


def _fit_elongated_loop_uv(uv: np.ndarray, normals_uv: np.ndarray) -> "dict | None":
    """Fallback for a wall loop a circle fit rejects -- the drill_slot
    "stadium" cross-section (two rounded ends `slot_length` apart,
    straight parallel sides, see sandbox.py's own drill_slot docstring)
    has a HIGH circle-fit residual by construction (it is not a circle),
    so without this fallback every slot's own wall is silently discarded
    by _classify_wall_component's existing `residual > radius_residual_
    tol` check, the same way a genuinely noisy/irregular patch is --
    confirmed real gap this closes: nothing in this module could tell a
    requested slot apart from "no feature detected at all."

    PCA-based, not a stadium-specific model fit: the major/minor
    eigenvectors of the loop's own 2D point scatter give an oriented
    bounding footprint (length along the elongation axis, width across
    it) that is honest about what it measures -- an overall footprint,
    not a precisely fitted stadium radius/length pair. Returns None when
    the loop isn't actually elongated (major/minor ratio too close to
    1 -- a near-circular shape the circle fit should have caught, or
    already did) or doesn't close into a full loop, so a stray noisy
    patch is never forced into a slot classification.

    `normals_uv` (each face's own normal, projected into the same uv
    plane) feeds a CRITICAL guard confirmed necessary by a real,
    reproduced regression: _wall_components' per-axis normal-direction
    mask also admits a part's own FLAT side walls and end caps whenever
    their normals happen to have a small component along the axis being
    queried (e.g. querying axis=1 admits a box's +-X and +-Z faces,
    whose normals have zero Y-component) -- when those merge into one
    connected component, the resulting blob can coincidentally be
    elongated, residual-low against the stadium model, AND close at both
    ends, passing every check above even though it is not a tubular
    feature wall at all (confirmed directly: a plain box's own 4 flat
    faces merged this way, falsely accepted as a "slot," then stole real
    hole/slot wall faces via the shared `claimed` mask before the
    correct axis's own pass ever ran). What a genuine tubular wall has
    that a handful of flat planar faces fundamentally cannot: face
    normals that sweep CONTINUOUSLY in direction around the loop
    (confirmed directly: a real slot's own wall has an 11-degree max gap
    between consecutive normal directions; the false flat-faced blob has
    a 90-degree gap, sitting at exactly 4 discrete directions with
    nothing in between)."""
    norm_mag = np.linalg.norm(normals_uv, axis=1)
    valid = norm_mag > 1e-6
    if valid.sum() < 6:
        return None
    normal_angles = np.arctan2(normals_uv[valid, 1], normals_uv[valid, 0])
    sorted_normal_angles = np.sort(normal_angles)
    normal_gaps = np.diff(np.concatenate([sorted_normal_angles, [sorted_normal_angles[0] + 2 * np.pi]]))
    if np.degrees(normal_gaps.max()) > 45.0:
        return None  # normals cluster at a few discrete directions -- flat faces, not a curved tubular wall

    centroid = uv.mean(axis=0)
    centered = uv - centroid
    cov = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    major_axis, minor_axis = eigvecs[:, order[0]], eigvecs[:, order[1]]
    major_proj = centered @ major_axis
    minor_proj = centered @ minor_axis
    length = float(major_proj.max() - major_proj.min())
    width = float(minor_proj.max() - minor_proj.min())
    if width < 1e-6 or length / width < 1.3:
        return None

    # Residual check: the proper generalization of the circle fit's own
    # residual (std-dev of per-point radius error) to a stadium boundary
    # -- distance from each point to the nearest point on the stadium's
    # OWN boundary curve (the two straight sides plus two rounded ends of
    # radius=width/2, half-span=(length-width)/2 along the major axis).
    # Confirmed necessary, not just thorough: an elongation+closure check
    # alone (tried first) is satisfied by more than just a genuine
    # stadium wall -- a mis-merged blob combining a box's own flat end
    # cap with its side walls (a separate, PRE-EXISTING characteristic of
    # _wall_components' per-axis normal-direction filter, which also
    # happens to admit a flat cap's own near-zero-component normal under
    # an unrelated axis's mask) can be elongated and happen to sweep both
    # ends too, live-reproduced to otherwise false-positive as a slot and
    # steal real slot-wall faces via the `claimed` mask before the
    # correct axis's own pass ever runs. That blob's own points scatter
    # across the box's entire face, nowhere near a thin boundary curve,
    # which this residual check correctly rejects where the cruder
    # heuristics above did not.
    half_span = max((length - width) / 2.0, 0.0)
    radius = width / 2.0
    abs_major = np.abs(major_proj)
    on_straight = abs_major <= half_span
    dist = np.empty_like(abs_major)
    dist[on_straight] = np.abs(np.abs(minor_proj[on_straight]) - radius)
    excess = abs_major[~on_straight] - half_span
    dist[~on_straight] = np.abs(np.sqrt(excess**2 + minor_proj[~on_straight]**2) - radius)
    residual = float(np.std(dist) / radius) if radius > 1e-9 else 1.0
    if residual > 0.12:
        return None

    # Closure check: a low stadium-boundary residual alone does not rule
    # out a PARTIAL arc (e.g. just one straight side, no end caps) --
    # its own points would still lie close to the model's boundary
    # curve without the loop actually closing. The two rounded ends are
    # what close a genuine stadium; confirm both are present by
    # requiring points near each extreme of the major axis to sweep
    # close to the shape's own full width (which only the curved end
    # caps, not a one-sided strip, can produce).
    end_fraction = 0.15
    near_min = major_proj <= (major_proj.min() + length * end_fraction)
    near_max = major_proj >= (major_proj.max() - length * end_fraction)
    for end_mask in (near_min, near_max):
        if not end_mask.any():
            return None
        end_width = float(minor_proj[end_mask].max() - minor_proj[end_mask].min())
        if end_width < width * 0.6:
            return None
    return {"center_uv": (float(centroid[0]), float(centroid[1])), "length": length, "width": width}


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
    raw_normals_uv = mesh.face_normals[faces][:, [u_idx, v_idx]]
    cx, cy, r, residual = _fit_circle_xy(uv)
    shape = "circle"
    slot_length = slot_width = None
    if r < 1e-6 or residual > radius_residual_tol:
        elongated = _fit_elongated_loop_uv(uv, raw_normals_uv)
        if elongated is None:
            return None
        shape = "slot"
        cx, cy = elongated["center_uv"]
        slot_length, slot_width = elongated["length"], elongated["width"]
        r = slot_width / 2.0  # the slot's own half-width -- used below exactly like a hole's radius for is_through/concavity, never for a reported "diameter"

    if shape == "slot":
        # _fit_elongated_loop_uv already did its OWN loop-closure check
        # (both ends sweep the full width) -- the angular-gap-from-
        # centroid method below is specific to roughly-isotropic
        # (circular) point clouds and, confirmed directly, incorrectly
        # rejects a genuine elongated stadium wall (an 85-degree max gap
        # on a real drill_slot build, well past the circular check's own
        # 60-degree cutoff) for the same reason a slot needed its own
        # fallback fit in the first place -- recomputing it here would
        # just silently undo that fallback's own already-correct answer.
        is_full_loop = True
    else:
        angles = np.arctan2(uv[:, 1] - cy, uv[:, 0] - cx)
        sorted_angles = np.sort(angles)
        gaps = np.diff(np.concatenate([sorted_angles, [sorted_angles[0] + 2 * np.pi]]))
        max_gap_deg = np.degrees(gaps.max())
        is_full_loop = max_gap_deg < 60.0  # covers at least ~300 degrees of the circle

    normals_uv = raw_normals_uv / np.clip(np.linalg.norm(raw_normals_uv, axis=1, keepdims=True), 1e-9, None)
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

    # Full 3D world-space center -- (cx, cy) alone is only the IN-PLANE
    # position (the u/v coordinates perpendicular to `axis`); the third
    # coordinate (along `axis` itself) is this wall's own midpoint along
    # its run, reusing `along_axis` (already computed above for the
    # is_through check) rather than a second vertex pass. Confirmed real
    # gap this closes: nothing anywhere previously exposed WHERE a
    # detected hole actually sits -- only its count/diameter/axis/
    # through-ness -- so comparing two parts' own hole LAYOUTS (e.g. a
    # motor's mounting pattern against its bracket's) had no position
    # data to compare at all.
    center_3d = [0.0, 0.0, 0.0]
    center_3d[u_idx], center_3d[v_idx] = cx, cy
    center_3d[axis] = float(along_axis.mean())

    return {
        "shape": shape,
        "center": (cx, cy),
        "center_3d": tuple(center_3d),
        "radius": r,
        "circularity": float(max(0.0, 1.0 - residual)) if shape == "circle" else None,
        "slot_length_mm": slot_length,
        "slot_width_mm": slot_width,
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
    hole_positions_mm = []
    slot_dimensions_mm = []
    slot_axes = []
    slot_through = []
    slot_positions_mm = []
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
                if result["shape"] == "slot":
                    slot_dimensions_mm.append((result["slot_length_mm"], result["slot_width_mm"]))
                    slot_axes.append(result["axis"])
                    slot_through.append(result["is_through"])
                    slot_positions_mm.append(result["center_3d"])
                else:
                    hole_circularity.append(result["circularity"])
                    hole_diameters_mm.append(result["radius"] * 2.0)
                    hole_axes.append(result["axis"])
                    hole_through.append(result["is_through"])
                    hole_positions_mm.append(result["center_3d"])
    return {
        "hole_count": len(hole_circularity),
        "hole_circularity": hole_circularity,
        "hole_diameters_mm": hole_diameters_mm,
        "hole_axes": hole_axes,
        "hole_through": hole_through,
        "hole_positions_mm": hole_positions_mm,
        "slot_count": len(slot_dimensions_mm),
        "slot_dimensions_mm": slot_dimensions_mm,
        "slot_axes": slot_axes,
        "slot_through": slot_through,
        "slot_positions_mm": slot_positions_mm,
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
        hole_positions_mm=features["hole_positions_mm"],
        detected_slot_count=features["slot_count"],
        slot_dimensions_mm=features["slot_dimensions_mm"],
        slot_axes=features["slot_axes"],
        slot_through=features["slot_through"],
        slot_positions_mm=features["slot_positions_mm"],
        mirror_symmetry_score=_mirror_symmetry_score(mesh),
    )
