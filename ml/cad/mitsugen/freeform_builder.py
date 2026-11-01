"""Stage D counterpart for free-form objects: constructs a real CSG solid
from a FreeformSpec's primitives + boolean operations, via trimesh's own
primitive creators and boolean engine (the same "manifold" engine
mitsugen/builder.py's apply_hollowing already uses) -- never a stand-in
for an unbuilt shape, and never a call into 3di-men's own sandbox.py
tool-composing engine (see mitsugen/__init__.py)."""

from __future__ import annotations

import numpy as np
import trimesh

from mitsugen.builder import BuildResult
from mitsugen.freeform import BooleanStep, FreeformSpec, PrimitiveSpec

_AXIS_VECTORS = {"x": np.array([1.0, 0.0, 0.0]), "y": np.array([0.0, 1.0, 0.0]), "z": np.array([0.0, 0.0, 1.0])}
# The two axes spanning the plane PERPENDICULAR to each pattern axis --
# circular_pattern's own copies are offset within this plane, same
# convention sandbox.py's own circular_pattern tool uses for axis="z"
# (offset in x/y), generalized here to all three axes.
_PERPENDICULAR_AXES = {
    "z": (np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])),
    "x": (np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0])),
    "y": (np.array([0.0, 0.0, 1.0]), np.array([1.0, 0.0, 0.0])),
}


def _euler_matrix_deg(angles_deg: tuple[float, float, float]) -> np.ndarray:
    """X-then-Y-then-Z fixed-order Euler rotation, in degrees -- the same
    simple convention documented in this module's own primitive schema,
    not a general quaternion."""
    rx, ry, rz = np.radians(angles_deg)
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)
    mx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    my = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    mz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return mz @ my @ mx


def _build_primitive_mesh(p: PrimitiveSpec) -> "trimesh.Trimesh | None":
    """None if this primitive's own fields don't support construction --
    disclosed by the caller, never guessed at."""
    if p.kind == "box":
        if p.size_mm is None:
            return None
        mesh = trimesh.creation.box(extents=list(p.size_mm))
    elif p.kind == "sphere":
        if not p.radius_mm or p.radius_mm <= 0:
            return None
        mesh = trimesh.creation.icosphere(subdivisions=3, radius=p.radius_mm)
    elif p.kind == "cylinder":
        if not p.radius_mm or not p.height_mm or p.radius_mm <= 0 or p.height_mm <= 0:
            return None
        mesh = trimesh.creation.cylinder(radius=p.radius_mm, height=p.height_mm, sections=32)
    elif p.kind == "cone":
        if not p.radius_mm or not p.height_mm or p.radius_mm <= 0 or p.height_mm <= 0:
            return None
        mesh = trimesh.creation.cone(radius=p.radius_mm, height=p.height_mm, sections=32)
        mesh.apply_translation([0.0, 0.0, -p.height_mm / 2.0])  # trimesh's cone spans z=[0, height] -- recenter so position_mm means the same "own center" thing for every primitive kind
    elif p.kind == "torus":
        if not p.major_radius_mm or not p.minor_radius_mm or p.major_radius_mm <= 0 or p.minor_radius_mm <= 0:
            return None
        mesh = trimesh.creation.torus(major_radius=p.major_radius_mm, minor_radius=p.minor_radius_mm, major_sections=32, minor_sections=24)
    else:
        return None

    transform = np.eye(4)
    transform[:3, :3] = _euler_matrix_deg(p.rotation_deg)
    transform[:3, 3] = p.position_mm
    mesh.apply_transform(transform)
    return mesh


def _aabb_overlap(a: "trimesh.Trimesh", b: "trimesh.Trimesh") -> bool:
    a_min, a_max = a.bounds
    b_min, b_max = b.bounds
    return bool(np.all(a_min <= b_max) and np.all(b_min <= a_max))


def _overlaps_anchor(moving: "trimesh.Trimesh", anchor: "trimesh.Trimesh") -> bool:
    """True if `moving` genuinely shares volume with `anchor` --
    resolved with a real boolean intersection, not an AABB-only check,
    since AABB has two distinct, confirmed-real blind spots here: (1) a
    coarse primitive's own bounding box can overlap another's while
    neither actual VERTEX lies inside the other (two 10mm boxes
    overlapping by half their width share no vertex, only the
    intersection region itself -- so a per-vertex containment/distance
    check alone also misses this), and (2) once `anchor` is a compound
    of several already-unioned pieces (e.g. a car body with two wheels
    already attached), its own OVERALL bounding box balloons to cover
    empty space between them, which a THIRD, still-separate piece can
    fall inside of without touching any real material at all (confirmed
    live: exactly this made a toy car's 3rd/4th wheel wrongly read as
    "already touching" under an earlier, AABB-only version of this
    check, leaving them genuinely disconnected).

    The cheap AABB check is kept as a fast, always-CORRECT reject when
    the boxes don't even overlap (true non-overlap there guarantees true
    non-overlap in reality) -- only when AABBs DO overlap does this fall
    through to the unambiguous, if costlier, real intersection-volume
    test, which has no blind spot for compound or non-convex shapes."""
    if not _aabb_overlap(moving, anchor):
        return False
    try:
        intersection = moving.intersection(anchor, engine="manifold")
    except Exception:
        return False
    return bool(intersection is not None and not intersection.is_empty and intersection.volume > 1e-9)


def _nudge_into_contact(moving: "trimesh.Trimesh", anchor: "trimesh.Trimesh") -> "trimesh.Trimesh":
    """Translates `moving` the minimum distance needed to genuinely
    overlap `anchor` by a small margin, preserving `moving`'s own size,
    shape, and orientation -- a bounded correction for a position that's
    close but not QUITE touching, never a reshaping.

    Confirmed real root cause this exists to fix (live-reproduced): a
    local LLM's own numeric position estimate for a secondary primitive
    (a wheel, a handle, a leg) is close to right but not precise enough
    to guarantee actual geometric contact with the main body -- the
    resulting boolean "union" is still a perfectly valid, individually
    watertight mesh, just left as several DISCONNECTED bodies (confirmed
    live: a real "toy car" decomposition produced body_count=4, not 1 --
    trimesh's own `is_watertight` is a per-component property and does
    not catch this at all). No amount of prompt wording can guarantee a
    small quantized model's arithmetic is exact every time, so this is a
    deterministic, bounded geometric repair applied to every union/
    difference/intersection/pattern target in build_freeform_solid below,
    not a prompt-only fix.

    Refuses (returns `moving` unchanged) when the gap exceeds a bound
    scaled to `moving`'s own size (twice its largest extent) -- nudging
    a primitive across an large, open-ended distance would silently
    invent a position the LLM never intended, rather than correct a near
    miss; the existing watertight/body_count check downstream still
    catches and discloses that case rather than this function papering
    over it."""
    if _overlaps_anchor(moving, anchor):
        return moving

    closest_on_anchor, dist, _ = anchor.nearest.on_surface(moving.vertices)
    idx = int(np.argmin(dist))
    gap = float(dist[idx])
    if gap <= 0:
        return moving

    max_extent = float(np.max(moving.extents)) if len(moving.vertices) else 0.0
    max_gap_mm = max(max_extent * 2.0, 5.0)  # a floor for degenerate/near-zero-size primitives
    if gap > max_gap_mm:
        return moving

    direction = closest_on_anchor[idx] - moving.vertices[idx]
    norm = float(np.linalg.norm(direction))
    if norm < 1e-9:
        return moving
    direction = direction / norm

    overlap_margin_mm = max(1.0, 0.05 * max_extent)  # guarantees real volume overlap, not mere tangency
    moved = moving.copy()
    moved.apply_translation(direction * (gap + overlap_margin_mm))
    return moved


def _pattern_mesh(target_mesh: "trimesh.Trimesh", step: BooleanStep) -> "trimesh.Trimesh":
    """Builds the single, already-unioned mesh of every copy a pattern
    operation describes, offset from the target's own already-
    positioned mesh -- never reorients a copy, same translate-only
    convention sandbox.py's own linear_pattern/circular_pattern tools
    use (fresh, independent geometry here, not a call into them)."""
    count = max(1, step.count or 1)
    copies = []
    if step.op == "linear_pattern":
        axis_vec = _AXIS_VECTORS[step.axis]
        for i in range(count):
            offset = (i - (count - 1) / 2) * step.spacing_mm
            copy = target_mesh.copy()
            copy.apply_translation(offset * axis_vec)
            copies.append(copy)
    else:  # circular_pattern
        u_axis, v_axis = _PERPENDICULAR_AXES[step.axis]
        for i in range(count):
            angle = 2 * np.pi * i / count
            offset = step.pattern_radius_mm * (np.cos(angle) * u_axis + np.sin(angle) * v_axis)
            copy = target_mesh.copy()
            copy.apply_translation(offset)
            copies.append(copy)

    result = copies[0]
    for copy in copies[1:]:
        result = result.union(copy, engine="manifold")
    return result


def build_freeform_solid(spec: FreeformSpec) -> BuildResult:
    """Never raises -- any construction failure (an unbuildable primitive,
    a failed boolean op, a non-watertight or empty result) comes back as
    a typed BuildResult(mesh=None, error=...), matching
    mitsugen/builder.py's own discipline for the radial-sweep path."""
    if not spec.primitives:
        return BuildResult(mesh=None, profiles=[], hollow_applied=False, error="no primitives to build")

    meshes_by_id: dict[str, trimesh.Trimesh] = {}
    unbuildable: list[str] = []
    for p in spec.primitives:
        mesh = _build_primitive_mesh(p)
        if mesh is None:
            unbuildable.append(p.id)
            continue
        meshes_by_id[p.id] = mesh

    if unbuildable:
        # A primitive the spec called for but couldn't actually build is a
        # requested feature this would otherwise go missing -- fail loudly
        # rather than silently export a solid that's missing a wheel/handle/
        # whatever that primitive was supposed to be, matching the "never a
        # partial/guessed mesh on failure" discipline builder.py documents.
        return BuildResult(mesh=None, profiles=[], hollow_applied=False, error=f"primitive(s) could not be built from their own fields: {', '.join(unbuildable)}")

    base_id = spec.primitives[0].id
    result = meshes_by_id[base_id]
    for step in spec.operations:
        # Every target gets a bounded nudge toward actual contact with the
        # running result FIRST (see _nudge_into_contact's own docstring) --
        # for a pattern op this corrects the single declared primitive
        # BEFORE it gets copied around the pattern's axis/circle, so every
        # copy inherits the same correction (the touching problem is
        # almost always "slightly off along one consistent direction,"
        # which a pre-pattern nudge fixes for all copies at once).
        target = _nudge_into_contact(meshes_by_id[step.target_id], result)
        try:
            if step.op in ("linear_pattern", "circular_pattern"):
                result = result.union(_pattern_mesh(target, step), engine="manifold")
            elif step.op == "union":
                result = result.union(target, engine="manifold")
            elif step.op == "difference":
                result = result.difference(target, engine="manifold")
            elif step.op == "intersection":
                result = result.intersection(target, engine="manifold")
            else:
                return BuildResult(mesh=None, profiles=[], hollow_applied=False, error=f"unknown operation {step.op!r}")
        except Exception as exc:
            return BuildResult(mesh=None, profiles=[], hollow_applied=False, error=f"{step.op} with '{step.target_id}' failed: {exc}")

    if result is None or result.is_empty:
        return BuildResult(mesh=None, profiles=[], hollow_applied=False, error="combined primitive geometry is empty")
    if not result.is_watertight:
        result.fix_normals()
    if not result.is_watertight:
        return BuildResult(mesh=None, profiles=[], hollow_applied=False, error="combined primitive geometry is not watertight -- rejected rather than exported as-is")

    return BuildResult(mesh=result, profiles=[], hollow_applied=False, error=None)
