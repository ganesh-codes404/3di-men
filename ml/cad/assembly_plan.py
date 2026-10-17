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

RelationshipKind = Literal["attached_to_face", "concentric_with", "symmetric_about"]
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
    # concentric_with only: the shared coaxial axis -- the OTHER two
    # coordinates are copied from the target exactly; this axis's own
    # coordinate is target's + axis_offset (0 = exactly level with target).
    axis: Axis | None = None
    axis_offset: float = 0.0


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
    BASE, not the bounding-box center," so attached_to_face already lands
    a dome's base flush without any extra correction). A shape built with
    NO meaningful local-origin convention at all (confirmed real gap:
    add_extrusion's "l_shape" profile, whose own vertices just happen to
    start at a corner for an unrelated caller's hole-math convenience, see
    data/program_builders.py) cannot be distinguished from a deliberately
    base-anchored one by geometry alone -- there is no reliable way to
    "auto-correct" for the former without breaking the latter. That
    narrower gap is fixed at its own source instead where it's
    unambiguous: sandbox.py's "triangle" profile centers its own geometry
    like every other true primitive, rather than special-casing this
    shared resolver for one profile_kind's otherwise-ambiguous footprint."""
    target_bounds = target_mesh.bounds
    target_center = target_bounds.mean(axis=0)

    if relationship.kind == "concentric_with":
        idx = _AXIS_INDEX[relationship.axis or "z"]
        pos = list(target_center)
        pos[idx] = float(target_center[idx] + relationship.axis_offset)
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

        face_center_u = float((face.outer[:, 0].min() + face.outer[:, 0].max()) / 2.0)
        face_center_v = float((face.outer[:, 1].min() + face.outer[:, 1].max()) / 2.0)
        u, v = face_center_u + relationship.offset_u, face_center_v + relationship.offset_v
        attach_point = face.origin + u * face.u + v * face.v

        own_extents = own_mesh.bounds[1] - own_mesh.bounds[0]
        unit_normal = face.normal / np.linalg.norm(face.normal)
        # this part's own half-size measured ALONG the face normal direction
        # (a dot-product-weighted blend of its x/y/z extents, so this works
        # for a normal along any axis, not just +-z) -- its OWN center must
        # sit this far out from the face so its back face lands flush.
        own_half_extent_along_normal = float(np.dot(own_extents / 2.0, np.abs(unit_normal)))
        world_pos = attach_point + unit_normal * (own_half_extent_along_normal + relationship.standoff)
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

    - concentric_with: align local Z with the shared axis. Sign doesn't
      matter for any rotationally-symmetric part (a cylinder/gear look
      identical pointing +y or -y), so "x"/"y"/"z" map to their + direction.
    - attached_to_face: align local Z with the face's own OUTWARD normal
      (sign matters here -- a boss must rise AWAY from its face, not into
      the part it's attached to).
    - symmetric_about: no rotation (out of scope for this fix -- see
      resolve_position's own symmetric_about handling for the position-side
      bug this project fixes separately).
    """
    if relationship.kind == "concentric_with":
        axis = relationship.axis or "z"
        return _ALIGN_LOCAL_Z_TO[f"+{axis}"]
    if relationship.kind == "attached_to_face":
        face_dir = relationship.face or "+z"
        return _ALIGN_LOCAL_Z_TO[face_dir]
    return (0.0, 0.0, 0.0)
