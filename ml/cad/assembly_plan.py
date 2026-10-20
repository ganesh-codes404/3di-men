"""Deterministic relationship resolution for multi-part assemblies.

Replaces raw LLM-estimated position_mm (for parts that relate to another
part) with a small set of SPATIAL RELATIONSHIP types -- attached_to_face,
concentric_with, symmetric_about -- resolved here in plain Python from
each referenced part's OWN ACTUAL built dimensions, never from LLM
arithmetic. See llm_decompose.py's ActionablePoint.relationship for how a
decomposed part declares one, and parallel_builder.py's dependency-ordered
build for how this plugs into the rest of the pipeline.

Directly motivated by two confirmed, real bugs (not hypothetical), both
found live-testing a multi-part prompt this session: an LLM decomposition
placed a domed lid at z = (box height) instead of z = (box height / 2) --
the box's own top SURFACE, since every primitive in this project centers
on its own given point -- and placed two different face-mounted features
(a dome on the top face, a plate on the front face) at the IDENTICAL
position, because only one of the two required axis offsets actually got
computed. Both are exactly the class of precise, derived-coordinate
arithmetic this project already keeps out of the LLM's hands for gear
mesh spacing (see llm_decompose._correct_gear_mesh_spacing's own
docstring for the same principle, confirmed the same way) -- this module
generalizes that principle to the handful of spatial relationships a real
assembly actually needs, instead of writing a bespoke corrector per case.

Deliberately NOT a general constraint solver: three relationship kinds,
each resolved by one direct formula from the target's own already-known
geometry. See program_model.py's own module docstring for why this
project consistently prefers a small set of reliable deterministic
operations over an attempt at full generality.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import trimesh

RelationshipKind = Literal["attached_to_face", "concentric_with", "symmetric_about", "parallel_to"]
FaceDirection = Literal["+x", "-x", "+y", "-y", "+z", "-z"]
Axis = Literal["x", "y", "z"]

_FACE_DIRECTION_VECTORS: dict[str, list[int]] = {
    "+x": [1, 0, 0], "-x": [-1, 0, 0], "+y": [0, 1, 0], "-y": [0, -1, 0], "+z": [0, 0, 1], "-z": [0, 0, -1],
}
_AXIS_INDEX = {"x": 0, "y": 1, "z": 2}


@dataclass
class Relationship:
    kind: RelationshipKind
    target_id: str
    # attached_to_face only: which face of the target, and an in-plane
    # offset from that face's own center along its own u/v axes (the SAME
    # face+offset_u+offset_v convention sandbox.py's counterbore_hole/
    # drill_slot/add_boss_with_bore already use -- reused deliberately,
    # not reinvented, so this is a second application of an already-tested
    # convention, not new untested math).
    face: FaceDirection | None = None
    offset_u: float = 0.0
    offset_v: float = 0.0
    standoff: float = 0.0  # extra gap beyond flush-against-the-face, 0 = touching
    # concentric_with / parallel_to: the shared coaxial/running axis -- the
    # OTHER two coordinates are copied from the target exactly; this axis's
    # own coordinate is target's + axis_offset (0 = exactly level with
    # target).
    axis: Axis | None = None
    axis_offset: float = 0.0
    # parallel_to only: a SECOND shaft/part running the same direction as
    # the target (same `axis`, same rotation) but offset sideways by a free
    # amount along one of the OTHER two world axes -- e.g. a second gear's
    # shaft, parallel to the first, mounted independently rather than
    # sharing its bore. Confirmed real gap this closes: niether
    # concentric_with (perfectly coaxial, zero lateral offset) nor plain
    # position_mm (a guess with no access to the target's REAL resolved
    # position, which can be anywhere once its own relationship chain
    # resolves -- e.g. elevated well above a base once a boss/face
    # attachment is involved) can express "same level and orientation as X,
    # just over sideways by N mm" -- see resolve_position's own parallel_to
    # branch and llm_decompose._correct_gear_mesh_spacing, which corrects
    # lateral_offset directly from real gear math instead of guessing.
    lateral_axis: Axis | None = None
    lateral_offset: float = 0.0


class RelationshipError(Exception):
    """A structurally invalid plan -- missing reference, self-reference, or
    a dependency cycle. Raised BEFORE any geometry is built, matching this
    project's standing rule: a malformed plan is rejected with a clear,
    typed reason, never silently guessed at or built incorrectly."""


def validate_relationships(ids: list[str], relationships: dict[str, Relationship | None]) -> None:
    id_set = set(ids)
    for pid, rel in relationships.items():
        if rel is None:
            continue
        if rel.target_id == pid:
            raise RelationshipError(f"part {pid!r} references itself")
        if rel.target_id not in id_set:
            raise RelationshipError(f"part {pid!r} references unknown part {rel.target_id!r}")

    WHITE, GRAY, BLACK = 0, 1, 2
    color = {pid: WHITE for pid in ids}

    def visit(pid: str, trail: list[str]) -> None:
        color[pid] = GRAY
        rel = relationships.get(pid)
        if rel is not None:
            if color[rel.target_id] == GRAY:
                raise RelationshipError("dependency cycle: " + " -> ".join(trail + [rel.target_id]))
            if color[rel.target_id] == WHITE:
                visit(rel.target_id, trail + [rel.target_id])
        color[pid] = BLACK

    for pid in ids:
        if color[pid] == WHITE:
            visit(pid, [pid])


def order_by_dependency(ids: list[str], relationships: dict[str, Relationship | None]) -> list[list[str]]:
    """Groups ids into dependency WAVES: wave 0 has no relationship, wave 1
    depends only on wave-0 parts, etc. Parts within one wave don't depend
    on each other and can be built in parallel; waves must be built in
    order. Call validate_relationships() first -- this assumes a cycle-
    free, fully-referenced graph and will recurse forever on a cycle."""
    depth: dict[str, int] = {}

    def compute_depth(pid: str) -> int:
        if pid in depth:
            return depth[pid]
        rel = relationships.get(pid)
        depth[pid] = 0 if rel is None else 1 + compute_depth(rel.target_id)
        return depth[pid]

    for pid in ids:
        compute_depth(pid)

    waves: list[list[str]] = [[] for _ in range(max(depth.values(), default=-1) + 1)]
    for pid in ids:
        waves[depth[pid]].append(pid)
    return waves


def resolve_position(relationship: Relationship, target_mesh: trimesh.Trimesh, own_mesh: trimesh.Trimesh) -> tuple[float, float, float]:
    """World-space position for a part related to `target_mesh` (already
    built AND already placed at its own final world position) via
    `relationship`, using `own_mesh`'s OWN actual bounding box (built at
    its local origin, per program_model.py's convention: every tool's
    geometry is predicted/built at its own origin first, placed after) to
    know how far to offset so the two parts actually meet rather than
    overlapping or floating apart -- the one piece of geometry arithmetic
    this project does not ask the LLM to do.

    Deliberately assumes `own_mesh`'s local origin IS its own placement-
    relevant anchor point (true for box/cylinder/sphere/gear -- centered
    at their own local origin by construction -- AND true by explicit
    design for add_dome, whose own docstring states "center is the FLAT
    BASE, not the bounding-box center"). A shape built with NO meaningful
    local-origin convention at all (confirmed real gap: add_extrusion's
    "l_shape" profile, whose own vertices just happen to start at a corner
    for an unrelated caller's hole-math convenience, see
    data/program_builders.py) cannot be distinguished from a deliberately
    base-anchored one by geometry alone -- there is no reliable way to
    "auto-correct" for the former without breaking the latter. That
    narrower gap is fixed at its own source instead where it's
    unambiguous: sandbox.py's "triangle" profile centers its own geometry
    like every other true primitive, rather than special-casing this
    shared resolver for one profile_kind's otherwise-ambiguous footprint.

    Two confirmed, REPRODUCED (not hypothetical) bugs in how the normal-
    direction offset used to be computed, both found by live-testing this
    exact function plus resolve_rotation_deg's rotation together (the way
    parallel_builder.py actually uses them, which the module's own prior
    test suite never exercised in one place):

    1. For any SIDE face (+-x/+-y), resolve_rotation_deg always rotates
       `own_mesh` so its own local Z axis ends up aligned with the face's
       outward normal (see that function's docstring). The old formula
       computed the offset by dotting `own_mesh`'s UNROTATED x/y/z extents
       against the face's world-space normal -- correct only for +z
       (where no rotation happens), wrong for every other face on any
       part whose local Z extent (its height/thickness) differs from its
       X/Y extent (its diameter/width). Reproduced: a boss (radius 10mm,
       height 10mm -- NOT anisotropic enough to show it) vs. a true
       anisotropic case (e.g. llm_decompose.py's own Example 5 boss,
       radius 5mm/height 30mm, attached to a housing's +y face) embeds the
       boss 10mm into the housing instead of landing flush on its surface,
       because the old formula used the boss's DIAMETER as the offset
       instead of its HEIGHT.
    2. Even on +z, the old formula silently assumed every shape is
       CENTER-anchored -- wrong for add_dome's explicitly base-anchored
       convention. Reproduced: a dome attached to any face landed with a
       gap equal to HALF its own height instead of flush (this module's
       own prior docstring claimed the opposite -- that case was never
       actually verified against the real resolve_rotation_deg-rotated
       placement before now).

    Fixed with one unified measurement instead of two special cases:
    `own_mesh.bounds[0][2]` -- `own_mesh`'s own local-Z MINIMUM, read
    before any rotation -- is exactly how far the part extends "behind"
    its own origin along the axis resolve_rotation_deg is about to align
    with the face normal, for EVERY local-origin convention this project
    uses: -own_extent/2 for a symmetric box/cylinder/gear (so the offset
    below becomes +own_extent/2, pushing the CENTER out so the back face
    lands flush, exactly the old formula's own +z-only behavior), and
    exactly 0 for add_dome's base-anchored convention (so the offset
    becomes 0 -- the origin already IS the contact point)."""
    target_bounds = target_mesh.bounds
    target_center = target_bounds.mean(axis=0)

    if relationship.kind == "concentric_with":
        idx = _AXIS_INDEX[relationship.axis or "z"]
        pos = list(target_center)
        pos[idx] = float(target_center[idx] + relationship.axis_offset)
        return (float(pos[0]), float(pos[1]), float(pos[2]))

    if relationship.kind == "parallel_to":
        # Same as concentric_with (copy the target's position on every
        # axis, then offset along the shared running `axis` by
        # axis_offset) PLUS a free lateral slide along a SECOND,
        # independent axis -- e.g. a second gear's shaft, running the
        # same direction as the first and level with it, just moved
        # sideways by lateral_offset. Confirmed real bug this fixes: a
        # plain position_mm guess for this exact case has no way to know
        # where the FIRST shaft's chain actually resolves to in world
        # space (which can be well above a base once a boss/face
        # attachment is involved) -- it can only guess at an absolute
        # coordinate, which is systematically wrong on whichever axis the
        # resolved chain actually moved along (confirmed live: "gear_B
        # excluded after a 52% overlap with the base," because the
        # guessed shaft_B sat near the base's own level instead of at
        # shaft_A's real, elevated one). This reads shaft_A's ACTUAL
        # resolved position directly off its already-built, already-
        # placed target_mesh, so the level always matches exactly, by
        # construction, not by guessing.
        idx = _AXIS_INDEX[relationship.axis or "z"]
        pos = list(target_center)
        pos[idx] = float(target_center[idx] + relationship.axis_offset)
        if relationship.lateral_axis:
            lidx = _AXIS_INDEX[relationship.lateral_axis]
            pos[lidx] = float(target_center[lidx] + relationship.lateral_offset)
        return (float(pos[0]), float(pos[1]), float(pos[2]))

    if relationship.kind == "symmetric_about":
        idx = _AXIS_INDEX[relationship.axis or "x"]
        # This formula mirrors the TARGET's own position across that axis's
        # plane through the origin -- correct for its intended use ("a
        # matched pair of standoffs/tabs on opposite sides," per
        # llm_decompose.py's own worked example: the target IS the other
        # twin, itself already offset from the origin, e.g. at x=25). It is
        # a confirmed real bug when the target sits ON the mirror plane
        # (target_center[idx] == 0) -- negating zero is zero, so the
        # "mirrored" part lands exactly on top of its target instead of
        # beside it (reproduced live: "a wheel symmetric_about frame," with
        # the frame centered at the origin -- this project's own default
        # convention for a single main part -- collapses to the frame's own
        # center). That's not a legitimate mirrored pair; it's a part that
        # needs a fixed offset from a CENTRAL body, which symmetric_about
        # was never designed to express -- fail loudly instead of silently
        # placing it flush against its target.
        if abs(target_center[idx]) < 1e-6:
            raise RelationshipError(
                f"symmetric_about target {relationship.target_id!r} sits exactly on the {relationship.axis or 'x'}=0 "
                "mirror plane -- mirroring it produces the identical position, not an offset part. "
                "symmetric_about is for a matched PAIR already offset from that plane (e.g. two tabs on either "
                "side of a body), not for positioning a single part a fixed distance from a centered one -- use "
                "plain position_mm for that instead."
            )
        pos = list(target_center)
        pos[idx] = float(-target_center[idx])
        return (float(pos[0]), float(pos[1]), float(pos[2]))

    if relationship.kind == "attached_to_face":
        from mesh_quality import find_facet_by_direction

        face_dir = relationship.face or "+z"
        direction = np.array(_FACE_DIRECTION_VECTORS[face_dir], dtype=float)
        face = find_facet_by_direction(target_mesh, direction)
        if face is None:
            raise RelationshipError(f"target has no flat face pointing {face_dir!r} for attached_to_face")

        face_u_min, face_u_max = float(face.outer[:, 0].min()), float(face.outer[:, 0].max())
        face_v_min, face_v_max = float(face.outer[:, 1].min()), float(face.outer[:, 1].max())
        face_center_u = (face_u_min + face_u_max) / 2.0
        face_center_v = (face_v_min + face_v_max) / 2.0
        u, v = face_center_u + relationship.offset_u, face_center_v + relationship.offset_v
        # Confirmed real failure mode this guards against: "a gusset at
        # X=-45mm on a 60mm-wide plate" -- an offset_u/offset_v large
        # enough to place the attachment point past the target's own face
        # boundary runs the attached part off the edge of its target
        # instead of onto it. Checked against the face's OWN actual u/v
        # extent (not a guessed/hardcoded size), so this scales correctly
        # with whatever the target's real built dimensions turned out to
        # be. A typed rejection here, not a silent out-of-bounds placement
        # that passes every per-part geometry check while still being
        # wrong -- exactly the class of failure section 9 of this
        # project's own repair-mission briefs keeps describing.
        if not (face_u_min <= u <= face_u_max) or not (face_v_min <= v <= face_v_max):
            raise RelationshipError(
                f"attached_to_face offset_u={relationship.offset_u:g}/offset_v={relationship.offset_v:g} places "
                f"the attachment point at u={u:.1f}, v={v:.1f}mm, outside target {relationship.target_id!r}'s own "
                f"{face_dir!r} face (which spans u=[{face_u_min:.1f}, {face_u_max:.1f}], "
                f"v=[{face_v_min:.1f}, {face_v_max:.1f}]mm) -- this would run the attached part off the edge of its "
                "target instead of onto it"
            )
        attach_point = face.origin + u * face.u + v * face.v

        unit_normal = face.normal / np.linalg.norm(face.normal)
        # See this function's own docstring for why own_mesh.bounds[0][2]
        # (local Z minimum, read before rotation) is the single correct
        # measurement for every local-origin convention this project uses,
        # not a dot-product against the unrotated x/y/z extents.
        offset_along_normal = -float(own_mesh.bounds[0][2]) + relationship.standoff
        world_pos = attach_point + unit_normal * offset_along_normal
        return (float(world_pos[0]), float(world_pos[1]), float(world_pos[2]))

    raise RelationshipError(f"unsupported relationship kind {relationship.kind!r}")


# Euler degrees (X then Y then Z, see parallel_builder.PlacementTransform.rotation)
# that rotate a part's own local +Z axis to point along each world direction.
# Every axis/face value in this project is one of these 6 axis-aligned
# directions (never an arbitrary vector -- confirmed by RelationshipKind's
# own Axis/FaceDirection types), so a small fixed lookup is exact, not an
# approximation; no general vector-alignment math is needed.
_ALIGN_LOCAL_Z_TO: dict[str, tuple[float, float, float]] = {
    "+z": (0.0, 0.0, 0.0), "-z": (180.0, 0.0, 0.0),
    "+x": (0.0, 90.0, 0.0), "-x": (0.0, -90.0, 0.0),
    "+y": (-90.0, 0.0, 0.0), "-y": (90.0, 0.0, 0.0),
}


def resolve_rotation_deg(relationship: Relationship) -> tuple[float, float, float]:
    """World-frame Euler rotation (X, then Y, then Z -- see
    parallel_builder.PlacementTransform.rotation) to apply to a part's own
    mesh, about its own local origin, BEFORE resolve_position's translation.

    Confirmed real bug this fixes: every rotationally-symmetric primitive in
    sandbox.py (cylinder, capsule, torus, dome, ring, tube) and every gear
    defaults to (or always builds along) its own local Z axis. resolve_position
    alone only computes WHERE a part's center should land -- it never
    reorients the part itself, so a shaft built along Z stays pointing along
    Z even when concentric_with axis="y" correctly centers it on the y-axis
    line through its target boss. The shaft's CENTER ends up right; the
    shaft itself is still standing up instead of running through the
    housing -- visually "floating beside" the thing it's supposed to pass
    through, confirmed live by reproducing exactly this with real meshes.

    - concentric_with / parallel_to: align local Z with the shared/running
      axis. Sign doesn't matter for any rotationally-symmetric part (a
      cylinder/gear look identical pointing +y or -y), so "x"/"y"/"z" map
      to their + direction.
    - attached_to_face: align local Z with the face's own OUTWARD normal
      (sign matters here -- a boss must rise AWAY from its face, not into
      the part it's attached to).
    - symmetric_about: no rotation (out of scope for this fix -- see
      resolve_position's own symmetric_about handling for the position-side
      bug this project fixes separately).
    """
    if relationship.kind in ("concentric_with", "parallel_to"):
        axis = relationship.axis or "z"
        return _ALIGN_LOCAL_Z_TO[f"+{axis}"]
    if relationship.kind == "attached_to_face":
        face_dir = relationship.face or "+z"
        return _ALIGN_LOCAL_Z_TO[face_dir]
    return (0.0, 0.0, 0.0)


def separate_coincident_face_attachments(
    members: list[tuple[str, Relationship, trimesh.Trimesh]], target_mesh: trimesh.Trimesh,
) -> dict[str, Relationship]:
    """Confirmed real gaps (mission-brief audit, plus a live-reported
    regression after this function first shipped), in THREE separate
    regressions: "the request for gussets on opposite sides produces
    identical instances in the same position" (gusset_right 100%
    overlapping gusset_left on one run); a SECOND real run still
    overlapped 82% with genuinely different (but not different ENOUGH)
    offsets on a DIFFERENT attachment face, after an earlier version of
    this function only caught byte-IDENTICAL offsets; and a THIRD,
    reported live (not a unit-test scenario): a prompt producing THREE OR
    MORE gussets on the same face (e.g. "a gusset at each corner") hit
    this function's own original `len(members) != 2: return {}` scoping
    guard and did nothing at all, leaving every one of them overlapping
    for parallel_builder's downstream collision-exclusion to silently
    drop all but one of -- which looks, from the user's side, exactly
    like "gusset issues": fewer gussets came out than were asked for, no
    explanation given. All three trace to the same root cause: offset_u
    vs offset_v's correspondence to a "safe, wide" in-plane direction
    depends entirely on the specific face's own normal (_plane_basis),
    which neither the LLM nor a fixed worked example can reliably get
    right by picking a field name alone -- confirmed directly: a
    150x100x10 base's +x face spans only 10mm in u but 100mm in v, the
    OPPOSITE of what a +y face gives. Patching one worked example (as a
    prior round of this project did) only fixes that one face.

    So this checks for an ACTUAL overlap using each member's own real
    projected extent against their ACTUAL given offsets (not requiring
    them to be identical), for ANY NUMBER of members sharing the same
    target+face (not just exactly 2), and resolves it the same way
    regardless of which face, which offsets, or how many members were
    originally given -- the same "correct from real geometry the moment
    it's available, never invent a number" discipline resolve_position
    itself follows, and the geometric analogue of
    llm_decompose._correct_gear_mesh_spacing (after-the-fact correction
    from real numbers, just for a face-attached GROUP's in-plane offsets
    instead of a meshing gear pair's center distance).

    A group that's already sufficiently separated (no two members
    overlap) is left alone -- not this function's job -- and a group
    that doesn't fit evenly spaced on EITHER axis returns {} rather than
    guessing at a squeeze; the existing collision-exclusion safety net
    still catches that case downstream, same as before this function
    existed."""
    if len(members) < 2:
        return {}
    target_id0, face0 = members[0][1].target_id, members[0][1].face
    if any(rel.target_id != target_id0 or rel.face != face0 for _id, rel, _mesh in members[1:]):
        return {}  # not actually one group sharing the same attachment face -- not this function's job

    from mesh_quality import find_facet_by_direction

    face_dir = face0 or "+z"
    direction = np.array(_FACE_DIRECTION_VECTORS[face_dir], dtype=float)
    face = find_facet_by_direction(target_mesh, direction)
    if face is None:
        return {}
    face_u_min, face_u_max = float(face.outer[:, 0].min()), float(face.outer[:, 0].max())
    face_v_min, face_v_max = float(face.outer[:, 1].min()), float(face.outer[:, 1].max())
    face_center_u, face_center_v = (face_u_min + face_u_max) / 2.0, (face_v_min + face_v_max) / 2.0
    face_u_span, face_v_span = face_u_max - face_u_min, face_v_max - face_v_min

    def _projected_extent(mesh: trimesh.Trimesh, basis: np.ndarray) -> float:
        proj = mesh.vertices @ basis
        return float(proj.max() - proj.min())

    # Confirmed real bug this fixes (found live-testing a motor-mount-style
    # assembly, not hypothetical): own_u/own_v used to project each member's
    # OWN LOCAL mesh -- still in its as-built, UNROTATED orientation at this
    # point in the pipeline -- directly onto the face's WORLD-FRAME u/v
    # vectors. But resolve_rotation_deg is about to rotate this exact mesh
    # (aligning its local Z with the face's outward normal) before it's ever
    # placed, and that rotation changes which of the mesh's own local axes
    # ends up spanning u vs v. Reproduced directly: a 30x30x4mm triangular
    # gusset attached to a box's "-y" face (whose own u axis happens to be
    # world Z) measured own_u as 4mm -- the gusset's pre-rotation THICKNESS,
    # since world Z also happens to be the unrotated mesh's own local Z --
    # instead of the ~30mm it actually occupies once correctly rotated
    # upright against that face. Two gussets offset only 10mm apart along u
    # were then wrongly judged "already separated enough" (10mm > the wrong
    # 4mm) and left untouched, yet still overlapped by 67% in the real,
    # rotated, placed geometry -- confirmed by reproducing the full build
    # pipeline, not just this function in isolation. Rotating a COPY of each
    # mesh the same way resolve_rotation_deg/PlacementTransform.rotation
    # will (X then Y then Z, composed identically) before projecting fixes
    # this for every face, not just "-y" -- the bug is structural (measuring
    # pre-rotation geometry against a post-rotation frame), not specific to
    # one face's particular u/v mapping.
    rot_rx, rot_ry, rot_rz = (np.radians(d) for d in resolve_rotation_deg(members[0][1]))
    rot_matrix = (
        trimesh.transformations.rotation_matrix(rot_rz, (0.0, 0.0, 1.0))
        @ trimesh.transformations.rotation_matrix(rot_ry, (0.0, 1.0, 0.0))
        @ trimesh.transformations.rotation_matrix(rot_rx, (1.0, 0.0, 0.0))
    )
    rotated_meshes = []
    for _id, _rel, mesh in members:
        rotated = mesh.copy()
        rotated.apply_transform(rot_matrix)
        rotated_meshes.append(rotated)

    # PER-MEMBER sizes, not a single group-wide max. Confirmed real bug
    # the group-wide-max version of this had (live-reported "gusset
    # issues" that persisted after the N-member generalization above):
    # a structural "side" plate (60mm) sharing a face with two much
    # smaller gussets (30mm) used max(60, 30, 30)=60mm as the spacing
    # unit for ALL THREE, demanding 3*60+2*2=184mm of room on an axis
    # that only had 150mm -- rejected as "doesn't fit" even though the
    # three actually DO fit fine once each one only needs its own real
    # size, not the biggest member's size. Reproduced directly: this
    # exact (side, gusset_top_left, gusset_top_right) trio, real
    # relationships from a live decomposition, confirmed un-fittable
    # under the old uniform sizing and fittable (124mm needed vs 150mm
    # available) under this one.
    own_u_list = [_projected_extent(m, face.u) for m in rotated_meshes]
    own_v_list = [_projected_extent(m, face.v) for m in rotated_meshes]

    # Each member's ACTUAL attach-plane coordinate from its own given
    # offset -- not assumed identical. A pairwise 2D AABB overlap test in
    # the face's own u/v plane, across every pair in the group (not just
    # the first two), using EACH PAIR's own combined half-extents (not a
    # shared threshold): this is what the eventual world-space collision
    # check would also find, just computed here BEFORE placement instead
    # of after.
    positions = [(face_center_u + rel.offset_u, face_center_v + rel.offset_v) for _id, rel, _mesh in members]
    n = len(members)
    any_overlap = any(
        abs(positions[i][0] - positions[j][0]) < (own_u_list[i] + own_u_list[j]) / 2.0
        and abs(positions[i][1] - positions[j][1]) < (own_v_list[i] + own_v_list[j]) / 2.0
        for i in range(n) for j in range(i + 1, n)
    )
    if not any_overlap:
        return {}  # already sufficiently separated -- not this function's job

    gap = 2.0  # small clearance beyond touching -- visibly separate parts, not sharing an edge
    total_u = sum(own_u_list) + (n - 1) * gap
    total_v = sum(own_v_list) + (n - 1) * gap
    u_fits = total_u <= face_u_span
    v_fits = total_v <= face_v_span
    if not u_fits and not v_fits:
        return {}  # the whole group doesn't fit on either axis even with per-member sizing -- leave as-is; the collision-exclusion safety net still catches this

    # Re-centered on the face's own midpoint, sequentially packed using
    # each member's OWN real size (never a shared one), not a delta from
    # whichever (confirmed unreliable) offsets were originally given --
    # a clean, valid layout beats trying to preserve a guess that caused
    # the overlap in the first place. Ordered along the chosen axis by
    # each member's ORIGINALLY GIVEN offset, preserving whatever
    # left/right (or top/bottom) intent the relationships already
    # expressed, rather than an arbitrary id order. Reduces to the
    # original half/-half result exactly when every member is the same
    # size.
    use_u_axis = u_fits and (not v_fits or (face_u_span - total_u) >= (face_v_span - total_v))
    sizes = own_u_list if use_u_axis else own_v_list
    total = total_u if use_u_axis else total_v
    order = sorted(range(n), key=lambda i: positions[i][0] if use_u_axis else positions[i][1])

    new_coord = [0.0] * n
    cursor = -total / 2.0
    for i in order:
        new_coord[i] = cursor + sizes[i] / 2.0
        cursor += sizes[i] + gap

    result: dict[str, Relationship] = {}
    for idx, (id_, rel, _mesh) in enumerate(members):
        if use_u_axis:
            result[id_] = Relationship(kind=rel.kind, target_id=rel.target_id, face=rel.face,
                                        offset_u=new_coord[idx], offset_v=rel.offset_v, standoff=rel.standoff)
        else:
            result[id_] = Relationship(kind=rel.kind, target_id=rel.target_id, face=rel.face,
                                        offset_u=rel.offset_u, offset_v=new_coord[idx], standoff=rel.standoff)
    return result
