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
    this project does not ask the LLM to do."""
    target_bounds = target_mesh.bounds
    target_center = target_bounds.mean(axis=0)

    if relationship.kind == "concentric_with":
        idx = _AXIS_INDEX[relationship.axis or "z"]
        pos = list(target_center)
        pos[idx] = float(target_center[idx] + relationship.axis_offset)
        return (float(pos[0]), float(pos[1]), float(pos[2]))

    if relationship.kind == "symmetric_about":
        idx = _AXIS_INDEX[relationship.axis or "x"]
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
