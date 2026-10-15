"""Post-CSG triangulation quality repair, shared by both geometry paths
(data/procedural_generator.py's hand-coded builders and sandbox.py's tool
calls) -- deliberately ONE shared module rather than duplicated, unlike the
small per-family polygon helpers those two paths intentionally keep
separate: this is a generic mesh-processing utility with no family-specific
semantics, so duplicating it would only risk the two copies drifting apart
on a correctness-sensitive algorithm.

Root cause this fixes: `trimesh.Trimesh.difference()`/`.union()` (backed by
manifold3d) fully RE-TRIANGULATES the entire result mesh, not just the
region near the cut -- it does not preserve the quality triangulation that
`engine="triangle"` produced for the pre-boolean mesh. The replacement
triangulation manifold3d produces on a flat face is frequently a fan of
long, thin sliver triangles spanning the WHOLE face, not just near the new
hole. Confirmed empirically across every family that drills holes (plate:
59/3968 faces under 1% of max area; bracket: 140/920 -- proportionally far
worse because the bracket's flat arm faces are simpler/larger relative to
its low overall triangle count), so this is systemic, not a one-family bug.

Fix: after a CSG op, find each maximal coplanar "facet" (trimesh's own
grouping, which was reliable in testing), re-triangulate JUST that facet's
boundary polygon (outer ring plus hole rings, e.g. the ring left by a
drilled hole) with the same `triangle`-engine quality constraint the
original builders use, and splice the result back in. Curved surfaces
(cylinder walls, gear teeth, involute profiles) are untouched -- trimesh
only groups genuinely coplanar adjacent faces into a facet, so this never
touches non-flat geometry. Boundary EDGES, however, are derived directly
from the facet's own face windings (_facet_boundary_edges), not from
`Trimesh.facets_boundary` -- that property was observed to report a
branching vertex (two outgoing boundary edges) on a real post-boolean
facet even though the facet's own faces gave an unambiguous simple
boundary; deriving it locally sidesteps whatever that quirk is instead of
working around it blindly.

Safety posture: every facet is processed independently inside a try/except.
A facet that fails to re-triangulate (a malformed boundary, an edge case in
loop-ordering) is left with its ORIGINAL triangulation rather than raising
or silently producing wrong geometry -- worst case behavior is unchanged
from before this module existed, never worse.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import trimesh

_MIN_FACES_TO_REMESH = 3  # a facet already at or below this many triangles has nothing to gain from re-triangulation


def _plane_basis(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Right-handed orthonormal (u, v) with u x v == normal, so a later
    outward-normal check can just compare signs, never guess."""
    normal = normal / np.linalg.norm(normal)
    arbitrary = np.array([1.0, 0.0, 0.0]) if abs(normal[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(normal, arbitrary)
    u = u / np.linalg.norm(u)
    v = np.cross(normal, u)
    return u, v


def _facet_boundary_edges(mesh: trimesh.Trimesh, face_indices: np.ndarray) -> np.ndarray:
    """Derives boundary edges directly from the facet's own faces rather
    than trusting `Trimesh.facets_boundary` -- empirically, that property
    can report a vertex with two outgoing boundary edges even when the
    facet's own face windings give a perfectly simple, unambiguous boundary
    (observed on a real post-boolean facet at a slab corner vertex; the
    facet grouping itself was correct, only trimesh's boundary-edge
    extraction was not). A directed edge (a, b) from any face in the group
    is a boundary edge iff its reverse (b, a) does not also belong to some
    other face in the same group -- shared internal edges cancel out by
    construction, so this can only branch if the underlying geometry is
    genuinely non-manifold, not from grouping noise."""
    edge_count: dict[tuple[int, int], int] = {}
    for f in mesh.faces[face_indices]:
        for a, b in ((f[0], f[1]), (f[1], f[2]), (f[2], f[0])):
            edge_count[(int(a), int(b))] = edge_count.get((int(a), int(b)), 0) + 1
    boundary = [(a, b) for (a, b) in edge_count if (b, a) not in edge_count]
    return np.array(boundary, dtype=np.int64) if boundary else np.zeros((0, 2), dtype=np.int64)


def _order_boundary_loops(edges: np.ndarray) -> list[list[int]]:
    """`edges` is trimesh's facets_boundary output: directed (n, 2) vertex
    index pairs that trace the facet's boundary (outer ring plus any hole
    rings) with the facet consistently on one side. Rebuilds each closed
    loop by following successor pointers. Raises ValueError (caught by the
    caller) rather than guessing on any malformed/branching input -- a
    facet whose boundary doesn't cleanly decompose into simple loops is
    left untouched, not patched into something potentially wrong."""
    successor: dict[int, int] = {}
    for a, b in edges:
        a, b = int(a), int(b)
        if a in successor:
            raise ValueError(f"vertex {a} has more than one outgoing boundary edge")
        successor[a] = b

    loops: list[list[int]] = []
    unvisited = set(successor.keys())
    while unvisited:
        start = next(iter(unvisited))
        loop = [start]
        unvisited.discard(start)
        current = successor[start]
        while current != start:
            if current not in unvisited:
                raise ValueError("boundary edges did not close into simple loops")
            loop.append(current)
            unvisited.discard(current)
            current = successor[current]
        loops.append(loop)
    return loops


def _shoelace_area(points_2d: np.ndarray) -> float:
    x, y = points_2d[:, 0], points_2d[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _simplify_collinear_indices(points_3d: np.ndarray, indices: list[int], angle_tol: float = 1e-6) -> list[int]:
    """Drops any boundary vertex whose two neighbors already lie on (near
    enough) a straight line through it -- the manifold3d re-triangulation
    that made this facet's boundary irregular in the first place packs many
    such redundant points along what is really one straight rectangle edge.

    Collinearity is tested in the mesh's own 3D coordinates, NOT after
    projecting into a facet's local 2D (u, v) basis -- a shared physical
    edge is genuinely straight in 3D regardless of which of its two
    (non-coplanar, by definition of "facet") neighboring facets is looking
    at it, so testing in 3D guarantees both facets decide to drop the exact
    same redundant points from their shared edge. A per-facet 2D projection
    does NOT give this guarantee: floating-point noise from two different
    projection bases can make the same physical point look "collinear
    enough to drop" on one side and not the other, which was tried first
    and produced mismatched (non-watertight) seams between facets."""
    n = len(indices)
    if n < 4:
        return indices
    keep = []
    for i in range(n):
        prev_p, cur_p, next_p = points_3d[indices[i - 1]], points_3d[indices[i]], points_3d[indices[(i + 1) % n]]
        a, b = cur_p - prev_p, next_p - cur_p
        cross_norm = np.linalg.norm(np.cross(a, b))
        scale = (np.linalg.norm(a) * np.linalg.norm(b)) or 1.0
        if cross_norm / scale > angle_tol:
            keep.append(indices[i])
    if len(keep) < 3:
        return indices  # simplification degenerated the loop -- bail, caller falls back safely
    return keep


def _remesh_one_facet(mesh: trimesh.Trimesh, face_indices: np.ndarray, normal: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    """Returns (new_vertices, new_faces) in mesh's global vertex-index space
    (new_faces reference newly-appended vertices only), or None if this
    facet should be left as-is."""
    from shapely.geometry import Polygon

    boundary_edges = _facet_boundary_edges(mesh, face_indices)
    if len(boundary_edges) < 3:
        return None
    loops = _order_boundary_loops(boundary_edges)
    if not loops:
        return None

    u, v = _plane_basis(normal)
    origin = mesh.vertices[loops[0][0]]

    def to_2d(vertex_idx: int) -> np.ndarray:
        p = mesh.vertices[vertex_idx] - origin
        return np.array([float(p @ u), float(p @ v)])

    loops_2d = [np.array([to_2d(i) for i in loop]) for loop in loops]
    areas = [abs(_shoelace_area(loop)) for loop in loops_2d]
    outer_idx = int(np.argmax(areas))
    outer = loops_2d[outer_idx]
    holes = [loop for i, loop in enumerate(loops_2d) if i != outer_idx]

    polygon = Polygon(outer, holes)
    if not polygon.is_valid or polygon.area <= 1e-9:
        return None

    target_area = max(1.0, polygon.area / 300)
    # "Y" prohibits inserting new Steiner points ON the boundary -- without
    # it, a quality-constrained retriangulation can subdivide a shared edge
    # differently than the ADJACENT facet's own (independent) retriangulation
    # does, leaving a non-conforming seam (mismatched edges -> a non-manifold
    # gap) between two facets that used to share that edge cleanly. Interior
    # Steiner points (where the actual sliver-triangle fix comes from) are
    # still allowed.
    verts_2d, faces_2d = trimesh.creation.triangulate_polygon(polygon, engine="triangle", triangle_args=f"pq30a{target_area:.3f}Y")
    if len(faces_2d) == 0:
        return None

    verts_3d = origin + verts_2d[:, 0:1] * u + verts_2d[:, 1:2] * v

    # triangulate_polygon's winding isn't guaranteed to match the facet's
    # original outward normal -- check one face and flip all of them together
    # if it doesn't, rather than leaving inverted normals on this patch.
    p0, p1, p2 = verts_3d[faces_2d[0]]
    computed_normal = np.cross(p1 - p0, p2 - p0)
    if np.dot(computed_normal, normal) < 0:
        faces_2d = faces_2d[:, [0, 2, 1]]

    return verts_3d, faces_2d


@dataclass
class SelectedFace:
    """One flat facet of a mesh, described in its own 2D plane coordinates
    (outer boundary ring + any hole rings, e.g. an existing drilled hole in
    that face) plus enough to map new 2D geometry back into world space --
    the shared building block for face-based tools (extrude_face,
    inset_pocket in sandbox.py) that need to build NEW geometry relative to
    an EXISTING flat face, not just re-triangulate one in place."""

    outer: np.ndarray  # (N, 2)
    holes: list[np.ndarray]
    origin: np.ndarray  # (3,) a point on the facet's plane
    normal: np.ndarray  # (3,) unit outward normal
    u: np.ndarray  # (3,) unit in-plane basis vector
    v: np.ndarray  # (3,) unit in-plane basis vector, with u x v == normal

    def to_3d(self, points_2d: np.ndarray) -> np.ndarray:
        return self.origin + points_2d[:, 0:1] * self.u + points_2d[:, 1:2] * self.v


def find_facet_by_direction(mesh: trimesh.Trimesh, direction: np.ndarray, min_alignment: float = 0.5) -> SelectedFace | None:
    """Finds the flat facet whose normal is MOST ALIGNED with `direction`
    (need not be unit-length) and returns it as a SelectedFace. This is the
    only "face selection" this project supports -- by which axis-ish
    direction a face points, a bounded/categorical choice a model can
    predict, NOT an arbitrary face index (which wouldn't be a stable,
    learnable argument the way everything else in sandbox.py's ARG_SCHEMA
    is). Returns None if no facet faces `direction` within `min_alignment`
    (cosine similarity) -- e.g. asking a cylinder's curved side for a face
    "up" finds nothing, correctly, since there is no flat face there.

    Deliberately uses a permissive minimum facet size (any facet trimesh
    reports at all, down to its own 2-triangle minimum) -- unlike
    remesh_flat_facets's _MIN_FACES_TO_REMESH, which asks "is this facet
    worth re-triangulating," this asks "does a flat face exist here at
    all," and even a freshly-created trimesh.creation.box()'s faces (2
    triangles each) must be selectable."""
    direction = np.asarray(direction, dtype=float)
    direction = direction / np.linalg.norm(direction)

    best_face_indices = None
    best_normal = None
    best_score = min_alignment
    for face_indices in mesh.facets:
        mean_normal = mesh.face_normals[face_indices].mean(axis=0)
        norm_len = np.linalg.norm(mean_normal)
        if norm_len < 0.99:  # not genuinely flat (member faces don't agree) -- see remesh_flat_facets's same check
            continue
        normal = mean_normal / norm_len
        score = float(np.dot(normal, direction))
        if score > best_score:
            best_score = score
            best_face_indices = face_indices
            best_normal = normal
    if best_face_indices is None:
        return None

    boundary_edges = _facet_boundary_edges(mesh, best_face_indices)
    if len(boundary_edges) < 3:
        return None
    try:
        loops = _order_boundary_loops(boundary_edges)
    except ValueError:
        return None
    if not loops:
        return None

    u, v = _plane_basis(best_normal)
    origin = mesh.vertices[loops[0][0]]

    def to_2d(vertex_idx: int) -> np.ndarray:
        p = mesh.vertices[vertex_idx] - origin
        return np.array([float(p @ u), float(p @ v)])

    loops_2d = [np.array([to_2d(i) for i in loop]) for loop in loops]
    areas = [abs(_shoelace_area(loop)) for loop in loops_2d]
    outer_idx = int(np.argmax(areas))
    outer = loops_2d[outer_idx]
    holes = [loop for i, loop in enumerate(loops_2d) if i != outer_idx]
    return SelectedFace(outer=outer, holes=holes, origin=origin, normal=best_normal, u=u, v=v)


def remesh_flat_facets(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Re-triangulates every sufficiently-large flat facet of `mesh` with a
    quality constraint, leaving curved regions and small/simple facets
    untouched. Safe to call on any mesh, including one with no flat facets
    worth improving (returns it unchanged in that case)."""
    if len(mesh.faces) == 0:
        return mesh

    facets = mesh.facets
    if len(facets) == 0:
        return mesh

    keep_mask = np.ones(len(mesh.faces), dtype=bool)
    extra_vertices: list[np.ndarray] = []
    extra_faces: list[np.ndarray] = []
    next_vertex_id = len(mesh.vertices)
    changed = False

    for face_indices in facets:
        if len(face_indices) < _MIN_FACES_TO_REMESH:
            continue
        # derive the facet's normal from its own member faces rather than
        # trusting Trimesh.facets_normal -- a group whose member faces are
        # NOT actually closely aligned (trimesh's curvature-based grouping
        # test can pass on near-degenerate slivers) averages to a normal
        # with norm well below 1; treat that as "not really flat, leave it
        # alone" instead of dividing by a near-zero vector downstream.
        mean_normal = mesh.face_normals[face_indices].mean(axis=0)
        norm_len = np.linalg.norm(mean_normal)
        if norm_len < 0.99:
            continue
        normal = mean_normal / norm_len
        try:
            result = _remesh_one_facet(mesh, face_indices, normal)
        except Exception:
            result = None  # leave this facet's original triangulation alone -- see module docstring's safety posture
        if result is None:
            continue

        new_vertices, new_faces = result
        extra_vertices.append(new_vertices)
        extra_faces.append(new_faces + next_vertex_id)
        next_vertex_id += len(new_vertices)
        keep_mask[face_indices] = False
        changed = True

    if not changed:
        return mesh

    all_vertices = np.vstack([mesh.vertices] + extra_vertices)
    all_faces = np.vstack([mesh.faces[keep_mask]] + extra_faces)
    candidate = trimesh.Trimesh(vertices=all_vertices, faces=all_faces, process=True)

    # Final safety net, not just an optimization: splicing independently
    # re-triangulated facets back together can leave a non-conforming seam
    # (see the module docstring's "Y" flag note) on meshes this function
    # hasn't been tested against. A mesh that WAS watertight and became
    # not-watertight, or whose volume moved by more than a tiny numerical
    # tolerance, is treated as this pass having failed on this input --
    # fall back to the original mesh (uglier triangulation, still correct)
    # rather than ever hand back a result that is geometrically worse than
    # what came in.
    was_watertight = mesh.is_watertight
    if was_watertight and not candidate.is_watertight:
        return mesh
    if was_watertight and mesh.volume > 0:
        volume_drift = abs(candidate.volume - mesh.volume) / mesh.volume
        if volume_drift > 1e-4:
            return mesh
    return candidate


def measure_wall_thickness_deviation(
    mesh: trimesh.Trimesh, requested_thickness: float, n_samples: int = 250,
) -> float | None:
    """Ray-cast wall-thickness measurement for sandbox.py's shell() --
    built because shell() is a disclosed APPROXIMATION (an axis-scaled
    copy-and-subtract, not a true constant-distance offset surface), and
    the mission requirement this serves is "measure approximation error
    instead of assuming a visually smooth mesh is accurate... fail
    explicitly rather than presenting an approximation as an exact shell."

    Method: sample points (area-weighted, via Trimesh.sample) on the
    shelled mesh's own surface, cast a ray inward along each sample's
    outward face normal, and take the distance between the first two hit
    points along that ray as the local wall thickness at that point.

    Returns the 90th-percentile deviation fraction from `requested_thickness`
    (None if too few usable samples to measure anything) -- the 90th
    percentile, not the mean or median, is the deliberate choice: confirmed
    directly (not assumed) that shell()'s own documented worst case -- "a
    gear's teeth would shrink proportionally, not stay a constant
    thickness" -- affects only a SMALL FRACTION of total surface area (the
    teeth), so an area-weighted median or mean is dominated by the
    accurately-shelled flat faces and stays near 0% deviation even on a
    badly-approximated gear shell (confirmed: median showed 0.0 deviation
    on a real shelled gear profile in direct testing). The 90th percentile
    of per-sample deviation correctly surfaces that defect (confirmed:
    0.72 on the same gear, vs. 0.0 on a plain box) while still being
    robust to the OTHER real artifact this method has -- a straight-inward
    ray from a point near an edge/corner can graze past the true nearby
    wall and hit a much farther one, producing an implausibly large
    single-sample "thickness." Those are filtered out explicitly (any
    sample outside [0.1x, 5x] of the requested thickness) before computing
    the percentile, since they are ray-geometry artifacts, not real
    thin/thick walls -- confirmed necessary, not hypothetical: an
    unfiltered run on a plain shelled box returned a raw mean of ~8.9x the
    requested thickness, purely from this effect, despite a near-perfect
    median of exactly the requested value."""
    if not mesh.is_watertight or len(mesh.faces) == 0 or requested_thickness <= 0:
        return None

    # Fixed seed: confirmed necessary, not defensive -- this function gates
    # a hard pass/fail decision in shell(), so its own result must be
    # deterministic given the same mesh and inputs. Without a fixed seed,
    # Trimesh.sample()'s random point placement made the SAME shape measure
    # anywhere from 36% to 40%+ deviation across repeated calls for a shape
    # sitting near the threshold -- a shell() call that sometimes succeeds
    # and sometimes raises for identical inputs would be its own new bug.
    points, face_idx = mesh.sample(n_samples, return_index=True, seed=0)
    normals = mesh.face_normals[face_idx]
    origins = points + normals * min(0.5, requested_thickness * 0.1)
    directions = -normals

    locations, index_ray, _ = mesh.ray.intersects_location(ray_origins=origins, ray_directions=directions)
    if len(locations) == 0:
        return None

    thicknesses = []
    for i in range(n_samples):
        hits = locations[index_ray == i]
        if len(hits) < 2:
            continue
        dists = np.linalg.norm(hits - origins[i], axis=1)
        dists.sort()
        thicknesses.append(dists[1] - dists[0])

    if not thicknesses:
        return None
    thicknesses = np.array(thicknesses)
    plausible = thicknesses[(thicknesses < 5 * requested_thickness) & (thicknesses > 0.1 * requested_thickness)]
    if len(plausible) < max(5, n_samples // 10):
        return None  # too few usable samples to trust a percentile from

    deviation = np.abs(plausible - requested_thickness) / requested_thickness
    return float(np.percentile(deviation, 90))


def mesh_topology_stats(mesh: trimesh.Trimesh) -> dict:
    """Plain-dict, JSON-ready summary of a mesh's polygon topology -- the
    numbers this project's build tools have always computed internally to
    judge geometry quality (watertightness, winding, volume/area) but never
    surfaced anywhere a person could see them. Exists so the frontend can
    display, per built part and for the combined assembly, exactly what a
    modeler means by "check the topology": vertex/face/edge counts, the
    Euler characteristic (2 for a single closed genus-0 solid -- the
    at-a-glance sanity number), and whether the mesh is actually a valid
    manifold volume.

    Deliberately reads only existing trimesh-computed properties (no new
    algorithm) so it can be called on any mesh this project produces,
    including ones remesh_flat_facets() has already processed, with no risk
    of disagreeing with what the rest of the pipeline already decided."""
    bounds = mesh.bounds
    extents = bounds[1] - bounds[0]
    is_volume = bool(mesh.is_volume)
    return {
        "vertex_count": int(len(mesh.vertices)),
        "face_count": int(len(mesh.faces)),
        "edge_count": int(len(mesh.edges_unique)),
        "euler_number": int(mesh.euler_number),
        "is_watertight": bool(mesh.is_watertight),
        "is_winding_consistent": bool(mesh.is_winding_consistent),
        "is_volume": is_volume,
        "volume_mm3": float(mesh.volume) if is_volume else None,
        "surface_area_mm2": float(mesh.area),
        "bounding_box_mm": [float(extents[0]), float(extents[1]), float(extents[2])],
    }
