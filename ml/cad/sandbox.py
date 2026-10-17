"""The tool layer a learned model composes: a small set of generic geometry
primitives/macros operating on a stateful Sandbox, plus execution tracing so
every synthetic (text, params) pair the existing generator produces can also
get a (text, program) label for free (see data/program_dataset.py).

Design decisions (see the design conversation this was built from for the
full reasoning, condensed here):

- Handles are small integers assigned by creation order, not strings. This
  keeps "which shape does this step operate on" a bounded categorical choice
  (< the current step index) instead of open-ended string generation --
  the single biggest reason this stays a scikit-learn-shaped problem
  (program_model.py) rather than requiring a real generative decoder with
  grammar-masked decoding.
- Handles are single-use/linear: translate/rotate/boolean_* CONSUME their
  input handle(s) and produce a new one, rather than mutating in place. This
  matches what the code already effectively did (a cylinder gets moved once,
  used once, discarded) and lets validate() catch "reused a handle after it
  was consumed" as a cheap, zero-geometry structural error.
- Involute gear tooth generation and simple 2D-profile extrusion are MACROS,
  not decomposed into more atomic primitives. The test applied: a step earns
  primitive status only if different combinations of it produce meaningfully
  different novel geometry (box + cylinder + boolean pass this trivially).
  There is exactly one way to use involute-curve math (build one gear) --
  nothing to compose -- so exposing its internals as steps would only teach
  an artificial vocabulary that never reuses anywhere else.
- drill_hole_pattern's placement math (corner/bolt-circle/linear layouts)
  reads the TARGET's own current bounds to derive extents, rather than
  taking them as separate predicted arguments -- those extents are already
  implicit in whatever earlier add_box/add_cylinder call created the target,
  so re-predicting them would be redundant and would blow up the argument
  count needed for what should be a bounded, scalar-argument tool call.
- CSG-robustness padding (how much a cutter oversticks the target so
  boolean_difference doesn't hit a degenerate tangent face) is computed
  INSIDE drill_hole/drill_hole_pattern, and is never a separate argument --
  it's implementation detail, not something a model should have to predict.
  The cutter's length along the drill axis is floored at the TARGET's own
  bounding-box extent along that axis (not just the predicted `depth`) --
  a predicted model, unlike a hand-authored builder, can regress `depth`
  too small, and a too-short cutter fails silently: boolean_difference
  still succeeds and the mesh stays watertight, so Sandbox.execute() would
  report status="ok" even though the hole never breaks the surface. Every
  drill_hole/drill_hole_pattern call in this codebase already passes the
  target's own material thickness as `depth` (there is no blind-hole use
  case yet), so flooring at the bounding-box extent is a safe, conservative
  fix rather than a behavior change for correctly-predicted programs.
"""

from __future__ import annotations

import inspect
import typing
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import trimesh

Axis = Literal["x", "y", "z"]
PatternKind = Literal["single", "bolt_circle", "linear_array", "corner_array"]
ProfileKind = Literal["rect", "rounded_rect", "l_shape", "triangle", "ellipse"]
SweepPathKind = Literal["straight", "arc", "helix"]
FaceDirection = Literal["+x", "-x", "+y", "-y", "+z", "-z"]
HeadStyle = Literal["flat", "conical"]

_AXIS_VECTORS: dict[Axis, list[int]] = {"x": [1, 0, 0], "y": [0, 1, 0], "z": [0, 0, 1]}
_FACE_DIRECTION_VECTORS: dict[FaceDirection, list[int]] = {
    "+x": [1, 0, 0], "-x": [-1, 0, 0], "+y": [0, 1, 0], "-y": [0, -1, 0], "+z": [0, 0, 1], "-z": [0, 0, -1],
}

# Handle-flow classification, shared by Sandbox.validate() (below) and
# tool_schema() -- hoisted to module level rather than left as local
# variables inside validate() so both can reference the exact same source
# of truth (the manual-tool-console playground reflects THIS classification
# to know which of a step's arguments are handle references), instead of
# two lists that could silently drift apart when a new tool is added.
_CREATES = {
    "add_box", "add_cylinder", "add_extrusion", "add_gear_profile", "add_revolve", "add_shouldered_cylinder",
    "add_sphere", "add_torus", "add_capsule", "add_ellipsoid", "add_polygon_frustum", "add_dome", "add_ring",
    "add_loft", "add_sweep", "add_tube",
}
_CONSUMES_ONE = {
    "translate", "rotate", "mirror", "scale", "shell", "taper", "flare", "twist", "bend",
    "round_box_edges", "chamfer_box_edges", "mirror_merge",
    "radial_array_oriented", "linear_array_scaled",
    "smooth_subdivide", "add_wireframe_from_edges", "weld_vertices",
}
_CONSUMES_TWO = {"boolean_union", "boolean_difference", "boolean_intersection"}
_CONSUMES_TARGET = {
    "drill_hole", "drill_hole_pattern", "linear_pattern", "circular_pattern",
    "extrude_face", "inset_pocket", "fillet_edge", "chamfer_edge",
    "counterbore_hole", "drill_slot", "add_rib", "add_boss_with_bore",
    "counterbore_hole_pattern", "add_slot_pattern", "add_knurl_pattern", "add_hex_socket",
}
# Every tool a person can invoke directly from the manual tool console, in
# the order shown there -- deliberately excludes "finalize" (not a modeling
# op, just "pick a handle to preview/export," handled separately by the
# console itself) and the private/bookkeeping methods.
TOOL_OPS: list[str] = [
    "add_box", "add_cylinder", "add_revolve", "add_shouldered_cylinder", "add_extrusion", "add_gear_profile",
    "add_sphere", "add_torus", "add_capsule", "add_ellipsoid", "add_polygon_frustum", "add_dome", "add_ring",
    "add_loft", "add_sweep", "add_tube",
    "taper", "flare", "twist", "bend", "translate", "rotate", "mirror", "scale",
    "boolean_union", "boolean_difference", "boolean_intersection",
    "drill_hole", "drill_hole_pattern", "linear_pattern", "circular_pattern",
    "extrude_face", "inset_pocket", "shell", "fillet_edge", "chamfer_edge",
    "counterbore_hole", "drill_slot", "add_rib",
    "round_box_edges", "chamfer_box_edges", "mirror_merge", "add_boss_with_bore",
    "counterbore_hole_pattern", "add_slot_pattern", "add_knurl_pattern",
    "radial_array_oriented", "linear_array_scaled", "add_hex_socket",
    "smooth_subdivide", "add_wireframe_from_edges", "weld_vertices",
]


@dataclass
class ToolCall:
    op: str
    args: dict[str, Any]  # JSON-safe only -- no numpy scalars/arrays


class SandboxError(Exception):
    """Raised for structural errors (undefined/consumed handle, bad enum) --
    always caught by validate() before any geometry runs, in the intended
    usage; also raised directly if a caller skips validate()."""


def _rotate_to_axis(mesh: trimesh.Trimesh, axis: Axis) -> trimesh.Trimesh:
    """trimesh primitives default to a Z-axis orientation; rotate to X or Y."""
    if axis == "z":
        return mesh
    if axis == "x":
        mesh.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [0, 1, 0]))
    elif axis == "y":
        mesh.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0]))
    return mesh


def _rounded_rect_polygon(length: float, width: float, fillet_radius: float, segments_per_corner: int = 8) -> np.ndarray:
    """Duplicated from data/procedural_generator.py deliberately: that module
    stays the source of truth for the flat/deterministic path, this module
    is the source of truth for the tool-composition path -- they're allowed
    to share an algorithm without importing each other, since a future
    refactor point (unifying the two) is explicitly out of scope for now."""
    hl, hw = length / 2, width / 2
    r = max(0.0, min(fillet_radius, hl - 1e-6, hw - 1e-6))
    if r <= 1e-9:
        return np.array([[-hl, -hw], [hl, -hw], [hl, hw], [-hl, hw]])
    corners = [(hl - r, hw - r, 0), (-hl + r, hw - r, 90), (-hl + r, -hw + r, 180), (hl - r, -hw + r, 270)]
    pts = []
    for cx, cy, start_deg in corners:
        for i in range(segments_per_corner + 1):
            theta = np.deg2rad(start_deg + 90 * i / segments_per_corner)
            pts.append((cx + r * np.cos(theta), cy + r * np.sin(theta)))
    return np.array(pts)


def _l_shape_polygon(leg1: float, leg2: float, thickness: float) -> np.ndarray:
    return np.array([[0, 0], [leg1, 0], [leg1, thickness], [thickness, thickness], [thickness, leg2], [0, leg2]])


def _right_triangle_polygon(base: float, height: float) -> np.ndarray:
    """A right triangle, right angle at the corner, base along X and
    height along Y -- but shifted so its own BOUNDING BOX is centered at
    the local origin, matching add_box/add_cylinder/add_sphere's own
    convention (every OTHER true primitive in this file is centered this
    way). Deliberately NOT the origin-corner convention
    _l_shape_polygon uses -- that convention exists only because
    data/program_builders.py's build_bracket_program hard-codes hole
    positions assuming it, a dependency this brand-new profile has none
    of, so there is no reason to inherit the same limitation. A centered
    footprint means assembly_plan.py's existing attached_to_face/
    concentric_with math (which assumes a shape's local origin IS its own
    placement-relevant center -- true for every other primitive) places a
    triangular gusset correctly with no special-casing anywhere else.
    Added specifically to close a confirmed, previously-disclosed gap
    (compositional_decomposer.py's own module docstring: "a triangular
    prism... is genuinely no triangular profile anywhere in sandbox.py's
    tool roster") -- a real gusset is a triangular PRISM, never
    equivalent to a box with a triangular cutout."""
    return np.array([[0, 0], [base, 0], [0, height]]) - [base / 2.0, height / 2.0]


def _ellipse_polygon(radius_x: float, radius_y: float, segments: int = 32) -> np.ndarray:
    theta = np.linspace(0, 2 * np.pi, segments, endpoint=False)
    return np.column_stack([radius_x * np.cos(theta), radius_y * np.sin(theta)])


def _profile_polygon(kind: "ProfileKind", param_a: float, param_b: float, param_c: float) -> np.ndarray:
    """Shared profile-shape dispatch -- the single source of truth for what
    a (profile_kind, param_a, param_b, param_c) tuple means, used by
    add_extrusion AND add_loft/add_sweep/add_tube (below) so a loft between
    two differently-KINDED profiles and a plain extrusion of one profile
    stay perfectly consistent, never two almost-the-same implementations
    drifting apart. Meaning by profile_kind:
      rect / rounded_rect: param_a=length, param_b=width, param_c=fillet_radius (0 for rect)
      l_shape:              param_a=leg1,   param_b=leg2,  param_c=thickness
      triangle:              param_a=base,  param_b=height, param_c unused (right triangle,
                             right angle at the origin -- see _right_triangle_polygon)
      ellipse:               param_a=radius_x, param_b=radius_y, param_c unused"""
    if kind == "rect":
        return _rounded_rect_polygon(param_a, param_b, 0.0)
    if kind == "rounded_rect":
        return _rounded_rect_polygon(param_a, param_b, param_c)
    if kind == "l_shape":
        return _l_shape_polygon(param_a, param_b, param_c)
    if kind == "triangle":
        return _right_triangle_polygon(param_a, param_b)
    if kind == "ellipse":
        return _ellipse_polygon(param_a, param_b)
    raise SandboxError(f"unknown profile_kind {kind!r}")


def _resample_polygon_by_arclength(polygon: np.ndarray, n_points: int) -> np.ndarray:
    """Resamples a closed 2D polygon (N, 2) to EXACTLY `n_points`, evenly
    spaced by ARC LENGTH around its perimeter (not vertex index) -- the
    standard technique for lofting between two differently-shaped cross-
    sections (a rounded-rect bottom to an ellipse top, say): both get
    resampled to the SAME point count via this function first, so
    corresponding indices are meaningful to interpolate between, and arc-
    length (not uniform parameter) spacing keeps point density even
    around sharp corners instead of clustering wherever the source
    polygon's own vertices happened to be denser."""
    closed = np.vstack([polygon, polygon[:1]])
    seg_vecs = np.diff(closed, axis=0)
    seg_lens = np.linalg.norm(seg_vecs, axis=1)
    cum_len = np.concatenate([[0.0], np.cumsum(seg_lens)])
    total_len = cum_len[-1]
    if total_len <= 1e-9:
        raise SandboxError("degenerate profile (zero perimeter) cannot be resampled for a loft/sweep")
    targets = np.linspace(0.0, total_len, n_points, endpoint=False)
    out = np.zeros((n_points, 2))
    for i, t in enumerate(targets):
        seg_idx = np.searchsorted(cum_len, t, side="right") - 1
        seg_idx = min(max(seg_idx, 0), len(seg_lens) - 1)
        local_t = (t - cum_len[seg_idx]) / seg_lens[seg_idx] if seg_lens[seg_idx] > 1e-12 else 0.0
        out[i] = closed[seg_idx] + local_t * seg_vecs[seg_idx]
    return out


def _sweep_path_frames(
    path_kind: "SweepPathKind", path_param_a: float, path_param_b: float, path_param_c: float, path_segments: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Shared by add_sweep and add_tube: a BOUNDED, fixed-arity path
    definition (never a raw point list -- program_model.py's per-argument
    regressors need plain scalar arguments, the same constraint every
    other tool in this file already respects) that still reaches genuinely
    curved sweeps, not just straight extrusion:
      straight: path_param_a=length, b/c unused -- a straight line along local +z.
      arc:      path_param_a=radius, path_param_b=angle_deg, c unused -- a circular
                arc in the local XZ plane, starting at the origin tangent to +z.
      helix:    path_param_a=radius, path_param_b=pitch_mm (rise per full turn),
                path_param_c=turns -- a helix around local z, starting at the origin.
    Returns (positions, tangents), each (path_segments+1, 3), in the SWEEP's
    own local frame (not yet placed at any target center)."""
    n = path_segments
    if path_kind == "straight":
        length = path_param_a
        if length <= 1e-9:
            raise SandboxError("add_sweep/add_tube: straight path needs a positive path_param_a (length)")
        t = np.linspace(0.0, 1.0, n + 1)
        positions = np.column_stack([np.zeros(n + 1), np.zeros(n + 1), t * length])
        tangents = np.tile([0.0, 0.0, 1.0], (n + 1, 1))
        return positions, tangents

    if path_kind == "arc":
        radius, angle_deg = path_param_a, path_param_b
        if radius <= 1e-9:
            raise SandboxError("add_sweep/add_tube: arc path needs a positive path_param_a (radius)")
        if abs(angle_deg) <= 1e-9:
            raise SandboxError("add_sweep/add_tube: arc path needs a nonzero path_param_b (angle_deg)")
        phi = np.linspace(0.0, np.deg2rad(angle_deg), n + 1)
        positions = np.column_stack([radius * (1 - np.cos(phi)), np.zeros(n + 1), radius * np.sin(phi)])
        tangents = np.column_stack([np.sin(phi), np.zeros(n + 1), np.cos(phi)])
        return positions, tangents

    if path_kind == "helix":
        radius, pitch, turns = path_param_a, path_param_b, path_param_c
        if radius <= 1e-9:
            raise SandboxError("add_sweep/add_tube: helix path needs a positive path_param_a (radius)")
        if turns <= 1e-9:
            raise SandboxError("add_sweep/add_tube: helix path needs a positive path_param_c (turns)")
        total_angle = turns * 2 * np.pi
        t = np.linspace(0.0, 1.0, n + 1)
        angle = t * total_angle
        positions = np.column_stack([radius * np.cos(angle) - radius, radius * np.sin(angle), t * turns * pitch])
        d_angle = total_angle  # constant d(angle)/dt
        tangent_raw = np.column_stack([-radius * np.sin(angle) * d_angle, radius * np.cos(angle) * d_angle, np.full(n + 1, turns * pitch)])
        tangents = tangent_raw / np.linalg.norm(tangent_raw, axis=1, keepdims=True)
        return positions, tangents

    raise SandboxError(f"unknown path_kind {path_kind!r}")


def _loft_along_path(
    profile_2d_list: list[np.ndarray], positions: np.ndarray, tangents: np.ndarray, closed_ends: bool,
) -> trimesh.Trimesh:
    """Shared ring-building core for add_sweep/add_tube: places one 2D
    cross-section (already resampled to a consistent point count) at each
    path position, oriented perpendicular to that point's own tangent via
    a simple per-point reference-vector frame (stable for the bounded,
    smoothly-varying straight/arc/helix paths this project supports --
    not a full parallel-transport minimization, which these paths don't
    need). `profile_2d_list` has one entry per path point (the SAME
    profile for add_sweep; add_tube passes two concentric rings per
    point instead of capping, see add_tube). When `closed_ends` is True,
    the first and last cross-sections are triangulated and capped
    (add_sweep's solid sweep); add_tube caps its own annular ends itself
    and passes closed_ends=False here for the straight ring-building part."""
    n_points = len(positions)
    n_profile = len(profile_2d_list[0])
    up_ref = np.array([0.0, 0.0, 1.0])

    chunks = []
    ring_start = []
    next_idx = 0
    for i in range(n_points):
        tangent = tangents[i]
        ref = up_ref if abs(np.dot(tangent, up_ref)) < 0.95 else np.array([0.0, 1.0, 0.0])
        local_x = np.cross(ref, tangent)
        local_x /= np.linalg.norm(local_x)
        local_y = np.cross(tangent, local_x)
        profile = profile_2d_list[i]
        world = positions[i] + np.outer(profile[:, 0], local_x) + np.outer(profile[:, 1], local_y)
        chunks.append(world)
        ring_start.append(next_idx)
        next_idx += n_profile

    vertices_list = [np.vstack(chunks)]
    faces = []
    for k in range(n_points - 1):
        r0, r1 = ring_start[k], ring_start[k + 1]
        for i in range(n_profile):
            j = (i + 1) % n_profile
            faces.append([r0 + i, r0 + j, r1 + j])
            faces.append([r0 + i, r1 + j, r1 + i])

    if closed_ends:
        from shapely.geometry import Polygon

        from trimesh.creation import triangulate_polygon

        for end_i, ring_idx in [(0, ring_start[0]), (n_points - 1, ring_start[-1])]:
            cap_v2d, cap_f = triangulate_polygon(Polygon(profile_2d_list[end_i]), engine="triangle", force_vertices=True)
            tangent = tangents[end_i]
            ref = up_ref if abs(np.dot(tangent, up_ref)) < 0.95 else np.array([0.0, 1.0, 0.0])
            local_x = np.cross(ref, tangent)
            local_x /= np.linalg.norm(local_x)
            local_y = np.cross(tangent, local_x)
            cap_v3d = positions[end_i] + np.outer(cap_v2d[:, 0], local_x) + np.outer(cap_v2d[:, 1], local_y)
            cap_idx = next_idx
            vertices_list.append(cap_v3d)
            next_idx += len(cap_v3d)
            if end_i == 0:
                for f in cap_f:  # outward normal faces -tangent at the start cap
                    faces.append([cap_idx + f[0], cap_idx + f[2], cap_idx + f[1]])
            else:
                for f in cap_f:  # outward normal faces +tangent at the end cap
                    faces.append([cap_idx + f[0], cap_idx + f[1], cap_idx + f[2]])

    vertices = np.vstack(vertices_list)
    mesh = trimesh.Trimesh(vertices=vertices, faces=np.array(faces), process=True)
    return mesh


def _cutter_height(mesh: trimesh.Trimesh, axis: Axis, depth: float) -> float:
    """Floors the drill cutter's length at the target's own bounding-box
    extent along the drill axis, not just the predicted `depth` -- see the
    module docstring's CSG-robustness-padding note for why."""
    axis_index = {"x": 0, "y": 1, "z": 2}[axis]
    target_extent = float(mesh.bounds[1][axis_index] - mesh.bounds[0][axis_index])
    return max(depth, target_extent) + 10.0


def _frustum_mesh(bottom_radius: float, top_radius: float, height: float, sections: int = 32) -> trimesh.Trimesh:
    """A straight-sided tapered cylinder, Z-axis, centered at its own
    origin (bottom at -height/2, top at +height/2) -- the general (both
    radii nonzero) case factored out of add_revolve, since
    countersink_hole's conical cutter needs the exact same ring-to-ring
    geometry and neither caller needs add_revolve's degenerate (one-radius-
    zero) handling: a countersink's hole_radius is always > 0 by
    definition (it's the fastener shank), so there is no apex-cone case to
    handle here."""
    theta = np.linspace(0, 2 * np.pi, sections, endpoint=False)
    z_bottom, z_top = -height / 2, height / 2
    n = sections
    bottom_ring = np.column_stack([bottom_radius * np.cos(theta), bottom_radius * np.sin(theta), np.full(n, z_bottom)])
    top_ring = np.column_stack([top_radius * np.cos(theta), top_radius * np.sin(theta), np.full(n, z_top)])
    bottom_center, top_center = np.array([[0.0, 0.0, z_bottom]]), np.array([[0.0, 0.0, z_top]])
    vertices = np.vstack([bottom_ring, top_ring, bottom_center, top_center])
    bc, tc = 2 * n, 2 * n + 1
    faces = []
    for i in range(n):
        j = (i + 1) % n
        faces.append([i, j, n + j])
        faces.append([i, n + j, n + i])
        faces.append([bc, j, i])
        faces.append([tc, n + i, n + j])
    mesh = trimesh.Trimesh(vertices=vertices, faces=np.array(faces), process=True)
    if mesh.volume < 0:
        mesh.invert()
    return mesh


def _subdivide_for_deform(mesh: trimesh.Trimesh, axis_idx: int, segments: int = 24) -> trimesh.Trimesh:
    """Required before twist/bend/flare, not optional: a freshly-created
    primitive (add_box, add_cylinder) has only TWO rings of vertices along
    any given axis -- its "side" faces are single flat quads (2 triangles)
    spanning the WHOLE length. twist/bend apply a rotation that varies
    continuously along that axis; a flat quad with vertices only at its two
    ends cannot represent a continuously-varying rotation in between, so
    the straight edges connecting those two rings cut across the intended
    curved/helical surface instead of following it -- confirmed empirically
    as a REAL bug, not a hypothetical: bending a coarse 100x10x10 box by 90
    degrees along its length gave a computed volume 36% off from the
    correct value (self-intersecting/badly-distorted side faces), for
    exactly this reason. flare has the same risk to a lesser degree (its
    transform is a smooth-varying SCALE, not a rotation, so a coarse mesh
    distorts less badly, but still measurably -- ~2.6% volume error in the
    same test). Subdividing every edge down to a small fraction of the
    deformation axis's own span first ensures enough intermediate rings
    exist for the curve to actually be represented, regardless of how
    coarse the input primitive was."""
    span = float(mesh.bounds[1][axis_idx] - mesh.bounds[0][axis_idx]) or 1.0
    max_edge = max(span / segments, 1e-6)
    new_vertices, new_faces = trimesh.remesh.subdivide_to_size(mesh.vertices, mesh.faces, max_edge=max_edge)
    return trimesh.Trimesh(vertices=new_vertices, faces=new_faces, process=True)


class Sandbox:
    def __init__(self, trace: bool = False):
        self._shapes: dict[int, trimesh.Trimesh] = {}
        self._consumed: set[int] = set()
        self._next_id = 0
        self._trace_enabled = trace
        self._trace: list[ToolCall] = []

    # ---- bookkeeping ----

    def _new_handle(self, mesh: trimesh.Trimesh) -> int:
        h = self._next_id
        self._shapes[h] = mesh
        self._next_id += 1
        return h

    def _get(self, handle: int) -> trimesh.Trimesh:
        if handle not in self._shapes:
            raise SandboxError(f"undefined handle {handle}")
        if handle in self._consumed:
            raise SandboxError(f"handle {handle} was already consumed by an earlier step")
        return self._shapes[handle]

    def _consume(self, *handles: int) -> None:
        self._consumed.update(handles)

    def _record(self, op: str, args: dict[str, Any]) -> None:
        if self._trace_enabled:
            self._trace.append(ToolCall(op=op, args=args))

    def get_trace(self) -> list[ToolCall]:
        return list(self._trace)

    # ---- atomic primitives ----

    def add_box(self, length: float, width: float, height: float, center: tuple[float, float, float] = (0, 0, 0)) -> int:
        mesh = trimesh.creation.box(extents=[length, width, height])
        if center != (0, 0, 0):
            mesh.apply_translation(center)
        h = self._new_handle(mesh)
        self._record("add_box", {"length": length, "width": width, "height": height, "center": list(center)})
        return h

    def add_cylinder(self, radius: float, height: float, center: tuple[float, float, float] = (0, 0, 0), axis: Axis = "z") -> int:
        mesh = trimesh.creation.cylinder(radius=radius, height=height, sections=32)
        mesh = _rotate_to_axis(mesh, axis)
        if center != (0, 0, 0):
            mesh.apply_translation(center)
        h = self._new_handle(mesh)
        self._record("add_cylinder", {"radius": radius, "height": height, "center": list(center), "axis": axis})
        return h

    def add_revolve(
        self, bottom_radius: float, top_radius: float, height: float,
        center: tuple[float, float, float] = (0, 0, 0), sections: int = 32,
    ) -> int:
        """A third primitive alongside box/cylinder: a straight-sided
        tapered cylinder (a frustum -- a cone when one radius is 0), built
        by revolving a two-point profile 360 degrees around Z, matching
        Blender's "spin" tool. Genuinely new geometry no composition of the
        other tools reaches -- a smoothly tapered wall isn't a box, a
        cylinder, or a boolean of either. Built directly from vertices/faces
        (no CSG), so it's exact and cheap, unlike approximating a taper via
        many stacked cylinders."""
        eps = 1e-9
        if bottom_radius <= eps and top_radius <= eps:
            raise SandboxError("add_revolve needs at least one nonzero radius")

        # Handled as three explicit cases rather than one generic branch
        # with degenerate (zero-radius) rings collapsed by process=True's
        # vertex merging: that approach left a non-watertight cone (a
        # zero-radius top ring merges into one point, but the "second side
        # triangle" per segment and the top cap fan both then reference two
        # now-identical vertices -- degenerate triangles trimesh's cleanup
        # didn't fully repair). Building the degenerate cases as a genuine
        # apex-to-ring fan from the start avoids ever creating a degenerate
        # triangle in the first place.
        if bottom_radius > eps and top_radius > eps:
            mesh = _frustum_mesh(bottom_radius, top_radius, height, sections)
            if center != (0, 0, 0):
                mesh.apply_translation(center)
            h = self._new_handle(mesh)
            self._record("add_revolve", {"bottom_radius": bottom_radius, "top_radius": top_radius, "height": height, "center": list(center)})
            return h

        theta = np.linspace(0, 2 * np.pi, sections, endpoint=False)
        z_bottom, z_top = -height / 2, height / 2
        n = sections
        if top_radius <= eps:
            ring = np.column_stack([bottom_radius * np.cos(theta), bottom_radius * np.sin(theta), np.full(n, z_bottom)])
            apex, bottom_center = np.array([[0.0, 0.0, z_top]]), np.array([[0.0, 0.0, z_bottom]])
            vertices = np.vstack([ring, apex, bottom_center])
            apex_idx, bc = n, n + 1
            faces = []
            for i in range(n):
                j = (i + 1) % n
                faces.append([i, j, apex_idx])
                faces.append([bc, j, i])
        else:  # bottom_radius <= eps
            ring = np.column_stack([top_radius * np.cos(theta), top_radius * np.sin(theta), np.full(n, z_top)])
            apex, top_center = np.array([[0.0, 0.0, z_bottom]]), np.array([[0.0, 0.0, z_top]])
            vertices = np.vstack([ring, apex, top_center])
            apex_idx, tc = n, n + 1
            faces = []
            for i in range(n):
                j = (i + 1) % n
                faces.append([apex_idx, j, i])
                faces.append([tc, i, j])

        mesh = trimesh.Trimesh(vertices=vertices, faces=np.array(faces), process=True)
        if mesh.volume < 0:  # winding came out inward-facing in one of the branches above -- flip rather than hand-verify every case
            mesh.invert()
        if center != (0, 0, 0):
            mesh.apply_translation(center)
        h = self._new_handle(mesh)
        self._record("add_revolve", {"bottom_radius": bottom_radius, "top_radius": top_radius, "height": height, "center": list(center)})
        return h

    def add_shouldered_cylinder(
        self, bottom_radius: float, bottom_height: float, top_radius: float, top_height: float,
        transition_height: float, center: tuple[float, float, float] = (0, 0, 0), sections: int = 32,
    ) -> int:
        """A cylinder that CURVES smoothly from one diameter to another
        partway up, instead of stepping or straight-tapering -- the real
        shape a bottle's shoulder (body narrowing to its neck), a vase, or
        a table leg actually has. Neither existing primitive reaches this:
        add_revolve's profile is a straight line between two radii, so
        wherever it meets a constant-radius section (e.g. a cylinder
        unioned on) there is still a visible sharp kink; unioning two plain
        cylinders leaves an actual sharp 90-degree step. Both were tried
        for a "water bottle" and looked wrong for exactly this reason.

        Built from bottom_radius/bottom_height (a constant-radius base
        section), then a SMOOTHSTEP-interpolated transition over
        transition_height (radius(t) = bottom_radius + (top_radius -
        bottom_radius) * (3t^2 - 2t^3) for t in [0, 1]) -- smoothstep has
        ZERO SLOPE at both t=0 and t=1, which is exactly what makes the
        curve meet the cylindrical sections on either side TANGENTIALLY,
        with no kink at either seam -- then top_radius/top_height (a
        constant-radius end section). Built directly from vertices/faces
        (no CSG, exact and cheap), the same approach as add_revolve;
        hollowing out the result (a real bottle wall) is a separate,
        composable shell() step afterward, not duplicated here."""
        eps = 1e-9
        if bottom_radius <= eps or top_radius <= eps:
            raise SandboxError("add_shouldered_cylinder needs two nonzero radii -- use add_revolve for a taper down to a point")
        if transition_height <= eps:
            raise SandboxError("add_shouldered_cylinder needs a positive transition_height")

        n = sections
        theta = np.linspace(0, 2 * np.pi, n, endpoint=False)

        # Internal smoothness resolution for the curved section -- an
        # implementation quality knob, not a predicted argument, same
        # reasoning as remesh_flat_facets' own triangulation constants.
        transition_rings = 12
        z_levels = [0.0, bottom_height]
        radii = [bottom_radius, bottom_radius]
        for i in range(1, transition_rings):
            t = i / transition_rings
            smooth_t = 3 * t * t - 2 * t * t * t
            z_levels.append(bottom_height + t * transition_height)
            radii.append(bottom_radius + (top_radius - bottom_radius) * smooth_t)
        z_levels.append(bottom_height + transition_height)
        radii.append(top_radius)
        z_levels.append(bottom_height + transition_height + top_height)
        radii.append(top_radius)

        z_offset = -z_levels[-1] / 2.0  # centered like every other primitive in this file

        ring_start = []
        chunks = []
        next_idx = 0
        for z, r in zip(z_levels, radii):
            chunks.append(np.column_stack([r * np.cos(theta), r * np.sin(theta), np.full(n, z + z_offset)]))
            ring_start.append(next_idx)
            next_idx += n
        bottom_center_idx, top_center_idx = next_idx, next_idx + 1
        chunks.append(np.array([[0.0, 0.0, z_levels[0] + z_offset]]))
        chunks.append(np.array([[0.0, 0.0, z_levels[-1] + z_offset]]))
        vertices = np.vstack(chunks)

        faces = []
        for k in range(len(z_levels) - 1):
            r0, r1 = ring_start[k], ring_start[k + 1]
            for i in range(n):
                j = (i + 1) % n
                faces.append([r0 + i, r0 + j, r1 + j])
                faces.append([r0 + i, r1 + j, r1 + i])
        r0 = ring_start[0]
        for i in range(n):
            j = (i + 1) % n
            faces.append([bottom_center_idx, r0 + j, r0 + i])
        rN = ring_start[-1]
        for i in range(n):
            j = (i + 1) % n
            faces.append([top_center_idx, rN + i, rN + j])

        mesh = trimesh.Trimesh(vertices=vertices, faces=np.array(faces), process=True)
        if mesh.volume < 0:
            mesh.invert()
        if center != (0, 0, 0):
            mesh.apply_translation(center)
        h = self._new_handle(mesh)
        self._record("add_shouldered_cylinder", {
            "bottom_radius": bottom_radius, "bottom_height": bottom_height,
            "top_radius": top_radius, "top_height": top_height,
            "transition_height": transition_height, "center": list(center),
        })
        return h

    def add_loft(
        self, bottom_profile: ProfileKind, bottom_param_a: float, bottom_param_b: float, bottom_param_c: float,
        top_profile: ProfileKind, top_param_a: float, top_param_b: float, top_param_c: float,
        height: float, center: tuple[float, float, float] = (0, 0, 0), sections: int = 24,
    ) -> int:
        """A solid SMOOTHLY blended between two cross-section profiles --
        possibly of DIFFERENT profile_kinds (a rounded-rect bottom lofting
        to an ellipse top, say) -- generalizing add_shouldered_cylinder's
        smoothstep radius-blend from "two radii" to "two arbitrary 2D
        profiles." Needed for anything a straight taper/add_revolve can't
        reach: a body panel that genuinely changes CROSS-SECTION SHAPE
        along its length, not just its size (a car nose tapering from a
        wide rounded-rect cockpit opening to a narrow ellipse at the tip,
        a transition duct, a blended fillet between two different beam
        profiles).

        Both profiles are resampled to the SAME `sections` point count by
        ARC LENGTH (_resample_polygon_by_arclength) so corresponding
        indices are meaningful to interpolate between regardless of how
        many raw vertices each profile's own shape produces -- an L-shape
        (6 corners) can loft into an ellipse (a smooth curve) because both
        get reduced to the same point count first, not because their raw
        vertex counts happen to match. Interpolation is smoothstep
        (3t^2-2t^3, zero slope at both ends) between the two RESAMPLED
        point sets, matching add_shouldered_cylinder's own tangent-
        continuous convention -- a loft that meets a constant cross-
        section on either end (e.g. two add_loft calls unioned) does so
        without a visible kink."""
        if sections < 3:
            raise SandboxError(f"add_loft needs sections >= 3, got {sections}")
        if height <= 1e-9:
            raise SandboxError("add_loft needs a positive height")

        bottom_outline = _resample_polygon_by_arclength(
            _profile_polygon(bottom_profile, bottom_param_a, bottom_param_b, bottom_param_c), sections
        )
        top_outline = _resample_polygon_by_arclength(
            _profile_polygon(top_profile, top_param_a, top_param_b, top_param_c), sections
        )

        # Internal smoothness resolution for the blend -- a quality knob,
        # not a predicted argument, same reasoning as add_shouldered_
        # cylinder's own transition_rings.
        levels = 16
        z_offset = -height / 2.0
        ring_start = []
        chunks = []
        next_idx = 0
        for i in range(levels + 1):
            t = i / levels
            smooth_t = 3 * t * t - 2 * t * t * t
            blended_2d = bottom_outline + (top_outline - bottom_outline) * smooth_t
            z = t * height + z_offset
            chunks.append(np.column_stack([blended_2d[:, 0], blended_2d[:, 1], np.full(sections, z)]))
            ring_start.append(next_idx)
            next_idx += sections
        side_vertices = np.vstack(chunks)

        faces = []
        for k in range(levels):
            r0, r1 = ring_start[k], ring_start[k + 1]
            for i in range(sections):
                j = (i + 1) % sections
                faces.append([r0 + i, r0 + j, r1 + j])
                faces.append([r0 + i, r1 + j, r1 + i])

        from shapely.geometry import Polygon

        from trimesh.creation import triangulate_polygon

        bottom_cap_v, bottom_cap_f = triangulate_polygon(Polygon(bottom_outline), engine="triangle", force_vertices=True)
        top_cap_v, top_cap_f = triangulate_polygon(Polygon(top_outline), engine="triangle", force_vertices=True)
        # force_vertices=True guarantees the cap triangulation's own vertex
        # ARRAY is exactly the resampled outline, in the same order -- so
        # the cap's boundary indices line up 1:1 with ring 0 / ring N
        # without a separate vertex-matching search.
        bottom_cap_idx = next_idx
        bottom_cap_3d = np.column_stack([bottom_cap_v[:, 0], bottom_cap_v[:, 1], np.full(len(bottom_cap_v), z_offset)])
        next_idx += len(bottom_cap_v)
        top_cap_idx = next_idx
        top_cap_3d = np.column_stack([top_cap_v[:, 0], top_cap_v[:, 1], np.full(len(top_cap_v), height + z_offset)])
        next_idx += len(top_cap_v)

        vertices = np.vstack([side_vertices, bottom_cap_3d, top_cap_3d])
        # bottom cap faces reversed (outward normal points -z); top cap as-is (+z)
        for f in bottom_cap_f:
            faces.append([bottom_cap_idx + f[0], bottom_cap_idx + f[2], bottom_cap_idx + f[1]])
        for f in top_cap_f:
            faces.append([top_cap_idx + f[0], top_cap_idx + f[1], top_cap_idx + f[2]])

        mesh = trimesh.Trimesh(vertices=vertices, faces=np.array(faces), process=True)
        if mesh.volume < 0:
            mesh.invert()
        if center != (0, 0, 0):
            mesh.apply_translation(center)
        h = self._new_handle(mesh)
        self._record("add_loft", {
            "bottom_profile": bottom_profile, "bottom_param_a": bottom_param_a, "bottom_param_b": bottom_param_b,
            "bottom_param_c": bottom_param_c, "top_profile": top_profile, "top_param_a": top_param_a,
            "top_param_b": top_param_b, "top_param_c": top_param_c, "height": height,
            "center": list(center), "sections": sections,
        })
        return h

    def add_sweep(
        self, profile_kind: ProfileKind, param_a: float, param_b: float, param_c: float,
        path_kind: SweepPathKind, path_param_a: float, path_param_b: float, path_param_c: float,
        center: tuple[float, float, float] = (0, 0, 0), sections: int = 24, path_segments: int = 32,
    ) -> int:
        """Extrudes a 2D profile along a CURVED path, not just straight up
        like add_extrusion -- reaches shapes neither add_extrusion nor
        add_loft can: a wing element that sweeps through an arc, a curved
        duct, a bent structural member with a real (non-rectangular)
        cross-section. path_kind is a bounded, fixed-arity path
        definition (straight/arc/helix -- see _sweep_path_frames'
        docstring for each one's param meaning), never a raw point list or
        spline control points, for the same reason every tool in this file
        takes plain scalar arguments: program_model.py's per-argument
        regressors have no way to predict an open-ended list.

        The profile's own shape is CONSTANT along the path (unlike
        add_loft, which blends between two different end profiles) --
        combine the two (e.g. add_loft first, then bend the result, or
        chain multiple add_sweep segments) for a shape that needs both a
        changing cross-section AND a curved path."""
        if sections < 3:
            raise SandboxError(f"add_sweep needs sections >= 3, got {sections}")

        outline = _resample_polygon_by_arclength(_profile_polygon(profile_kind, param_a, param_b, param_c), sections)
        positions, tangents = _sweep_path_frames(path_kind, path_param_a, path_param_b, path_param_c, path_segments)
        profile_list = [outline] * len(positions)

        mesh = _loft_along_path(profile_list, positions, tangents, closed_ends=True)
        # center on the path's own bounding box, like every other primitive
        bbox_center = mesh.bounds.mean(axis=0)
        mesh.apply_translation(-bbox_center)
        if mesh.volume < 0:
            mesh.invert()
        if center != (0, 0, 0):
            mesh.apply_translation(center)
        h = self._new_handle(mesh)
        self._record("add_sweep", {
            "profile_kind": profile_kind, "param_a": param_a, "param_b": param_b, "param_c": param_c,
            "path_kind": path_kind, "path_param_a": path_param_a, "path_param_b": path_param_b,
            "path_param_c": path_param_c, "center": list(center), "sections": sections, "path_segments": path_segments,
        })
        return h

    def add_tube(
        self, outer_radius: float, inner_radius: float,
        path_kind: SweepPathKind, path_param_a: float, path_param_b: float, path_param_c: float,
        center: tuple[float, float, float] = (0, 0, 0), sections: int = 24, path_segments: int = 32,
    ) -> int:
        """A hollow (or solid, if inner_radius is small) tube swept along a
        curved path -- chassis tubing, a roll hoop, a suspension wishbone
        link, an exhaust pipe: a ROUND cross-section that needs to follow
        an arc or helix, not just go straight (add_cylinder) or hollow out
        a straight shape (shell). A direct add_sweep of an annulus would
        need a profile_kind this project doesn't have (an annulus isn't a
        single closed curve); built directly as two concentric swept
        rings capped with an annular washer at each end instead."""
        eps = 1e-9
        if inner_radius < 0 or outer_radius <= eps:
            raise SandboxError("add_tube needs a positive outer_radius and a non-negative inner_radius")
        if inner_radius >= outer_radius:
            raise SandboxError(f"add_tube: inner_radius ({inner_radius}) must be smaller than outer_radius ({outer_radius})")
        if sections < 3:
            raise SandboxError(f"add_tube needs sections >= 3, got {sections}")

        outer_outline = _ellipse_polygon(outer_radius, outer_radius, sections)
        positions, tangents = _sweep_path_frames(path_kind, path_param_a, path_param_b, path_param_c, path_segments)
        is_solid = inner_radius <= eps

        if is_solid:
            profile_list = [outer_outline] * len(positions)
            mesh = _loft_along_path(profile_list, positions, tangents, closed_ends=True)
        else:
            inner_outline = _ellipse_polygon(inner_radius, inner_radius, sections)
            outer_mesh = _loft_along_path([outer_outline] * len(positions), positions, tangents, closed_ends=False)
            inner_mesh = _loft_along_path([inner_outline] * len(positions), positions, tangents, closed_ends=False)
            if inner_mesh.volume < 0:
                inner_mesh.invert()
            inner_mesh.invert()  # inner wall's normals must face INWARD (toward the tube's axis) to close the solid
            # annular end caps (outer ring minus inner ring) at each end, via
            # the same triangulate_polygon technique add_loft/_loft_along_path use
            from shapely.geometry import Polygon

            from trimesh.creation import triangulate_polygon

            up_ref = np.array([0.0, 0.0, 1.0])
            end_caps_v, end_caps_f = [], []
            next_idx = len(outer_mesh.vertices) + len(inner_mesh.vertices)
            for end_i in (0, len(positions) - 1):
                annulus = Polygon(outer_outline.tolist(), [inner_outline.tolist()])
                cap_v2d, cap_f = triangulate_polygon(annulus, engine="triangle", force_vertices=True)
                tangent = tangents[end_i]
                ref = up_ref if abs(np.dot(tangent, up_ref)) < 0.95 else np.array([0.0, 1.0, 0.0])
                local_x = np.cross(ref, tangent)
                local_x /= np.linalg.norm(local_x)
                local_y = np.cross(tangent, local_x)
                cap_v3d = positions[end_i] + np.outer(cap_v2d[:, 0], local_x) + np.outer(cap_v2d[:, 1], local_y)
                faces_oriented = cap_f[:, ::-1] if end_i == 0 else cap_f
                end_caps_v.append(cap_v3d)
                end_caps_f.append(faces_oriented + next_idx)
                next_idx += len(cap_v3d)

            all_vertices = np.vstack([outer_mesh.vertices, inner_mesh.vertices, *end_caps_v])
            all_faces = np.vstack([
                outer_mesh.faces, inner_mesh.faces + len(outer_mesh.vertices), *end_caps_f,
            ])
            mesh = trimesh.Trimesh(vertices=all_vertices, faces=all_faces, process=True)

        bbox_center = mesh.bounds.mean(axis=0)
        mesh.apply_translation(-bbox_center)
        if mesh.volume < 0:
            mesh.invert()
        if center != (0, 0, 0):
            mesh.apply_translation(center)
        h = self._new_handle(mesh)
        self._record("add_tube", {
            "outer_radius": outer_radius, "inner_radius": inner_radius, "path_kind": path_kind,
            "path_param_a": path_param_a, "path_param_b": path_param_b, "path_param_c": path_param_c,
            "center": list(center), "sections": sections, "path_segments": path_segments,
        })
        return h

    # ---- more primitives: none of these six shapes is reachable by
    # composing the existing primitives + booleans (a sphere isn't a
    # box/cylinder/frustum; a torus has a hole THROUGH its own body, not
    # just a cut; an ellipsoid is a non-uniformly-scaled sphere, not a
    # scaled box; an N-sided polygon frustum has FLAT facets, unlike
    # add_revolve's smooth circular ones; a dome/hemisphere is a genuinely
    # curved cap; a ring/annulus is common enough (washers, bushings,
    # o-ring grooves) to deserve a single bounded call instead of a
    # cylinder+cylinder+boolean_difference chain every time). ----

    def add_sphere(self, radius: float, center: tuple[float, float, float] = (0, 0, 0)) -> int:
        mesh = trimesh.creation.icosphere(subdivisions=3, radius=radius)
        if center != (0, 0, 0):
            mesh.apply_translation(center)
        h = self._new_handle(mesh)
        self._record("add_sphere", {"radius": radius, "center": list(center)})
        return h

    def add_torus(
        self, major_radius: float, minor_radius: float,
        center: tuple[float, float, float] = (0, 0, 0), axis: Axis = "z",
    ) -> int:
        """A donut: `major_radius` is the distance from the torus's own
        center to the middle of the tube, `minor_radius` is the tube's own
        radius. Genuinely new topology (a handle/hole through the solid
        itself, Euler characteristic 0) that no boolean of the existing
        primitives reaches."""
        if minor_radius >= major_radius:
            raise SandboxError(f"minor_radius ({minor_radius}) must be smaller than major_radius ({major_radius}) or the tube overlaps itself")
        mesh = trimesh.creation.torus(major_radius=major_radius, minor_radius=minor_radius, major_sections=32, minor_sections=24)
        mesh = _rotate_to_axis(mesh, axis)
        if center != (0, 0, 0):
            mesh.apply_translation(center)
        h = self._new_handle(mesh)
        self._record("add_torus", {"major_radius": major_radius, "minor_radius": minor_radius, "center": list(center), "axis": axis})
        return h

    def add_capsule(self, radius: float, height: float, center: tuple[float, float, float] = (0, 0, 0), axis: Axis = "z") -> int:
        """A cylinder with hemispherical ends -- a rounded rod/pill shape
        (a pin, a handle, a rounded-end fastener). `height` is the shape's
        FULL overall length INCLUDING both hemispherical caps, matching
        add_cylinder's own `height` convention (its full length) rather
        than trimesh.creation.capsule's own convention (cylinder-section-
        only, caps centered at its two ends instead of at its own
        midpoint) -- centered and re-parameterized here to match this
        file's existing primitives, not trimesh's raw default."""
        cylinder_length = height - 2 * radius
        if cylinder_length < 0:
            raise SandboxError(f"height ({height}) must be at least 2 x radius ({2*radius}) -- use add_sphere for a shape shorter than that")
        mesh = trimesh.creation.capsule(height=cylinder_length, radius=radius)
        mesh.apply_translation([0, 0, -height / 2])  # trimesh's own capsule spans z=[-radius, cylinder_length+radius]; recenter to match this file's convention
        mesh = _rotate_to_axis(mesh, axis)
        if center != (0, 0, 0):
            mesh.apply_translation(center)
        h = self._new_handle(mesh)
        self._record("add_capsule", {"radius": radius, "height": height, "center": list(center), "axis": axis})
        return h

    def add_ellipsoid(
        self, radius_x: float, radius_y: float, radius_z: float, center: tuple[float, float, float] = (0, 0, 0),
    ) -> int:
        """A stretched sphere -- an egg, a lens, a dome-like blob -- built
        as a unit icosphere non-uniformly scaled, the same "primitive built
        from a simpler one plus a transform" reasoning add_ellipsoid's own
        callers already use elsewhere in this file (e.g. shell's own
        scaled-copy approach)."""
        mesh = trimesh.creation.icosphere(subdivisions=3, radius=1.0)
        mesh.apply_transform(np.diag([radius_x, radius_y, radius_z, 1.0]))
        if center != (0, 0, 0):
            mesh.apply_translation(center)
        h = self._new_handle(mesh)
        self._record("add_ellipsoid", {"radius_x": radius_x, "radius_y": radius_y, "radius_z": radius_z, "center": list(center)})
        return h

    def add_polygon_frustum(
        self, sides: int, bottom_radius: float, top_radius: float, height: float,
        center: tuple[float, float, float] = (0, 0, 0), axis: Axis = "z",
    ) -> int:
        """An N-sided FLAT-FACETED frustum -- a hex nut/bolt/standoff
        (sides=6), a pentagon post, an octagon column -- distinct from
        add_revolve's always-smooth circular profile. Reuses _frustum_mesh
        verbatim with a small `sides` count instead of a smooth 32 --
        the exact same ring-to-ring geometry already verified for
        add_revolve/countersink_hole, since a "circle" sampled at N points
        for small N simply IS a flat N-gon, not an approximation of one.
        Both radii must be nonzero (a true pyramid-to-a-point needs
        add_revolve's own degenerate-apex handling, not replicated here --
        a real, disclosed scope limit, not a silent gap)."""
        eps = 1e-9
        if sides < 3:
            raise SandboxError(f"add_polygon_frustum needs at least 3 sides, got {sides}")
        if bottom_radius <= eps or top_radius <= eps:
            raise SandboxError("add_polygon_frustum needs two nonzero radii -- a true pyramid-to-a-point isn't supported by this tool")
        mesh = _frustum_mesh(bottom_radius, top_radius, height, sections=sides)
        mesh = _rotate_to_axis(mesh, axis)
        if center != (0, 0, 0):
            mesh.apply_translation(center)
        h = self._new_handle(mesh)
        self._record("add_polygon_frustum", {
            "sides": sides, "bottom_radius": bottom_radius, "top_radius": top_radius,
            "height": height, "center": list(center), "axis": axis,
        })
        return h

    def add_dome(self, radius: float, height: float, center: tuple[float, float, float] = (0, 0, 0), axis: Axis = "z") -> int:
        """A hemisphere/dome cap -- built by boolean-intersecting a full
        icosphere with a half-space box (a robust, exact bisection at the
        sphere's own equator -- no new curved-profile vertex math needed),
        then Z-scaled to `height` so the dome's own tallness is
        independent of its base radius (a shallow dish vs. a tall cap).
        UNLIKE this file's other primitives, `center` is the center of the
        FLAT BASE circle, not the shape's own bounding-box center -- a
        dome/cap is inherently base-anchored (it sits ON something), the
        same reasoning extrude_face's boss anchors at the face it grows
        from rather than centering itself."""
        if radius <= 1e-9 or height <= 1e-9:
            raise SandboxError("add_dome needs a positive radius and height")
        sphere = trimesh.creation.icosphere(subdivisions=3, radius=radius)
        half_space = trimesh.creation.box(extents=[radius * 4, radius * 4, radius * 2])
        half_space.apply_translation([0, 0, radius])  # spans z in [0, 2*radius] -- keeps only the sphere's upper half
        dome = sphere.intersection(half_space, engine="manifold")
        dome.apply_transform(np.diag([1.0, 1.0, height / radius, 1.0]))  # base stays at z=0, apex moves from z=radius to z=height
        dome = _rotate_to_axis(dome, axis)
        if center != (0, 0, 0):
            dome.apply_translation(center)
        h = self._new_handle(dome)
        self._record("add_dome", {"radius": radius, "height": height, "center": list(center), "axis": axis})
        return h

    def add_ring(
        self, outer_radius: float, inner_radius: float, height: float,
        center: tuple[float, float, float] = (0, 0, 0), axis: Axis = "z",
    ) -> int:
        """A washer/annulus/bushing -- common enough (washers, spacers,
        o-ring grooves, bushings) to deserve one bounded primitive call
        instead of a cylinder+cylinder+boolean_difference chain every
        time. trimesh.creation.annulus already builds exactly this shape
        directly (no CSG needed, so no boolean-robustness concerns)."""
        if inner_radius >= outer_radius:
            raise SandboxError(f"inner_radius ({inner_radius}) must be smaller than outer_radius ({outer_radius})")
        mesh = trimesh.creation.annulus(r_min=inner_radius, r_max=outer_radius, height=height, sections=32)
        mesh = _rotate_to_axis(mesh, axis)
        if center != (0, 0, 0):
            mesh.apply_translation(center)
        h = self._new_handle(mesh)
        self._record("add_ring", {"outer_radius": outer_radius, "inner_radius": inner_radius, "height": height, "center": list(center), "axis": axis})
        return h

    def taper(self, target: int, axis: Axis, bottom_factor: float, top_factor: float) -> int:
        """Blender's "Simple Deform: Taper" -- linearly scales cross-sections
        perpendicular to `axis` from `bottom_factor` (at the shape's own
        minimum along axis) to `top_factor` (at its maximum), about the
        shape's own centerline. Pure vertex repositioning, no CSG and no
        new triangulation -- so it's exact and can never introduce the
        sliver-triangle problem remesh_flat_facets exists for; a genuinely
        flat top or bottom face STAYS flat (every vertex on it shares the
        same position-along-axis, hence the same scale factor, so a planar
        set of points scaled uniformly about a center stays planar)."""
        mesh = self._get(target).copy()
        axis_idx = {"x": 0, "y": 1, "z": 2}[axis]
        other_idx = [i for i in range(3) if i != axis_idx]
        lo, hi = mesh.bounds[0][axis_idx], mesh.bounds[1][axis_idx]
        span = (hi - lo) or 1.0
        t = (mesh.vertices[:, axis_idx] - lo) / span
        factor = bottom_factor + (top_factor - bottom_factor) * t
        center_other = mesh.bounds.mean(axis=0)[other_idx]

        new_vertices = mesh.vertices.copy()
        for k, oi in enumerate(other_idx):
            new_vertices[:, oi] = center_other[k] + (mesh.vertices[:, oi] - center_other[k]) * factor
        mesh.vertices = new_vertices

        self._consume(target)
        h = self._new_handle(mesh)
        self._record("taper", {"target": target, "axis": axis, "bottom_factor": bottom_factor, "top_factor": top_factor})
        return h

    def flare(self, target: int, axis: Axis, bottom_factor: float, top_factor: float) -> int:
        """Identical to taper except the scale factor is interpolated with
        a SMOOTHSTEP curve instead of linearly -- a genuinely different,
        curved profile (a bell/trumpet flare, a bulging vase silhouette)
        rather than a straight-sided cone. Same exact-and-cheap pure
        vertex-repositioning approach, same reasoning as taper's own
        docstring for why a flat end stays flat."""
        axis_idx = {"x": 0, "y": 1, "z": 2}[axis]
        mesh = _subdivide_for_deform(self._get(target), axis_idx)
        other_idx = [i for i in range(3) if i != axis_idx]
        lo, hi = mesh.bounds[0][axis_idx], mesh.bounds[1][axis_idx]
        span = (hi - lo) or 1.0
        t = (mesh.vertices[:, axis_idx] - lo) / span
        smooth_t = 3 * t * t - 2 * t * t * t
        factor = bottom_factor + (top_factor - bottom_factor) * smooth_t
        center_other = mesh.bounds.mean(axis=0)[other_idx]

        new_vertices = mesh.vertices.copy()
        for k, oi in enumerate(other_idx):
            new_vertices[:, oi] = center_other[k] + (mesh.vertices[:, oi] - center_other[k]) * factor
        mesh.vertices = new_vertices

        self._consume(target)
        h = self._new_handle(mesh)
        self._record("flare", {"target": target, "axis": axis, "bottom_factor": bottom_factor, "top_factor": top_factor})
        return h

    def twist(self, target: int, axis: Axis, angle_deg: float) -> int:
        """Blender's "Simple Deform: Twist" -- progressively rotates each
        cross-section perpendicular to `axis` around the shape's own
        centerline, from 0 degrees at one end to `angle_deg` at the other.
        Pure per-vertex 2D rotation (exact, no CSG) -- the same
        "parameterize by position along axis, transform the other two
        coordinates about the shape's own center" structure taper/flare
        already use, just a rotation instead of a scale."""
        axis_idx = {"x": 0, "y": 1, "z": 2}[axis]
        mesh = _subdivide_for_deform(self._get(target), axis_idx)
        other_idx = [i for i in range(3) if i != axis_idx]
        lo, hi = mesh.bounds[0][axis_idx], mesh.bounds[1][axis_idx]
        span = (hi - lo) or 1.0
        t = (mesh.vertices[:, axis_idx] - lo) / span
        theta = np.radians(angle_deg) * t
        center_other = mesh.bounds.mean(axis=0)[other_idx]

        u = mesh.vertices[:, other_idx[0]] - center_other[0]
        v = mesh.vertices[:, other_idx[1]] - center_other[1]
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        new_vertices = mesh.vertices.copy()
        new_vertices[:, other_idx[0]] = center_other[0] + u * cos_t - v * sin_t
        new_vertices[:, other_idx[1]] = center_other[1] + u * sin_t + v * cos_t
        mesh.vertices = new_vertices

        self._consume(target)
        h = self._new_handle(mesh)
        self._record("twist", {"target": target, "axis": axis, "angle_deg": angle_deg})
        return h

    def bend(self, target: int, axis: Axis, bend_direction: Axis, angle_deg: float) -> int:
        """Blender's "Simple Deform: Bend" -- curves a straight shape into
        an arc: `axis` is the shape's own straight length direction,
        `bend_direction` (a second, DIFFERENT axis) is which perpendicular
        direction the arc curves toward, `angle_deg` is the total sweep
        angle of the resulting arc. The shape's own length along `axis`
        becomes the arc's own length (arc_radius = original_span /
        angle_rad), so a small angle barely curves it and a large one
        curves it tightly -- exact per-vertex remapping onto the arc, no
        CSG. A near-zero angle is treated as a no-op (dividing by a
        near-zero angle to get the arc radius is not numerically safe).
        DISCLOSED, not silently avoided: a large bend angle relative to
        the shape's own cross-section thickness can fold the shape into
        self-intersecting geometry, the same inherent limitation Blender's
        own bend modifier has -- this tool does not detect or prevent
        that, only sandbox.py's own watertight/volume checks downstream
        will ever flag it as "suspect"."""
        if bend_direction == axis:
            raise SandboxError("bend_direction must be different from axis")

        axis_idx = {"x": 0, "y": 1, "z": 2}[axis]
        mesh = _subdivide_for_deform(self._get(target), axis_idx)
        bend_idx = {"x": 0, "y": 1, "z": 2}[bend_direction]
        lo, hi = mesh.bounds[0][axis_idx], mesh.bounds[1][axis_idx]
        span = (hi - lo) or 1.0
        mid = (lo + hi) / 2.0
        center_bend = mesh.bounds.mean(axis=0)[bend_idx]

        theta_max = np.radians(angle_deg)
        new_vertices = mesh.vertices.copy()
        if abs(theta_max) > 1e-9:
            s_centered = mesh.vertices[:, axis_idx] - mid
            x = mesh.vertices[:, bend_idx] - center_bend
            radius = span / theta_max
            theta = theta_max * s_centered / span
            new_vertices[:, axis_idx] = mid + (radius - x) * np.sin(theta)
            new_vertices[:, bend_idx] = center_bend + radius - (radius - x) * np.cos(theta)
        mesh.vertices = new_vertices

        self._consume(target)
        h = self._new_handle(mesh)
        self._record("bend", {"target": target, "axis": axis, "bend_direction": bend_direction, "angle_deg": angle_deg})
        return h

    def boolean_union(self, a: int, b: int) -> int:
        from mesh_quality import remesh_flat_facets

        mesh_a, mesh_b = self._get(a), self._get(b)
        result = remesh_flat_facets(mesh_a.union(mesh_b, engine="manifold"))
        self._consume(a, b)
        h = self._new_handle(result)
        self._record("boolean_union", {"a": a, "b": b})
        return h

    def boolean_difference(self, a: int, b: int) -> int:
        from mesh_quality import remesh_flat_facets

        mesh_a, mesh_b = self._get(a), self._get(b)
        result = remesh_flat_facets(mesh_a.difference(mesh_b, engine="manifold"))
        self._consume(a, b)
        h = self._new_handle(result)
        self._record("boolean_difference", {"a": a, "b": b})
        return h

    def boolean_intersection(self, a: int, b: int) -> int:
        """The third boolean op alongside union/difference -- e.g. intersecting
        two boxes at an angle gives a wedge, a box and a cylinder gives a
        rounded slot. Same composability as the other two: meaningfully
        different geometry depending on what's fed in, not a fixed shape."""
        from mesh_quality import remesh_flat_facets

        mesh_a, mesh_b = self._get(a), self._get(b)
        result = remesh_flat_facets(mesh_a.intersection(mesh_b, engine="manifold"))
        self._consume(a, b)
        h = self._new_handle(result)
        self._record("boolean_intersection", {"a": a, "b": b})
        return h

    def translate(self, target: int, dx: float, dy: float, dz: float) -> int:
        mesh = self._get(target).copy()
        mesh.apply_translation([dx, dy, dz])
        self._consume(target)
        h = self._new_handle(mesh)
        self._record("translate", {"target": target, "dx": dx, "dy": dy, "dz": dz})
        return h

    def rotate(self, target: int, axis: Axis, angle_deg: float) -> int:
        mesh = self._get(target).copy()
        mesh.apply_transform(trimesh.transformations.rotation_matrix(np.radians(angle_deg), _AXIS_VECTORS[axis]))
        self._consume(target)
        h = self._new_handle(mesh)
        self._record("rotate", {"target": target, "axis": axis, "angle_deg": angle_deg})
        return h

    def mirror(self, target: int, axis: Axis) -> int:
        """Reflects the shape across a plane through the origin perpendicular
        to `axis`. trimesh.apply_transform detects the negative determinant
        and re-winds faces itself, so normals/volume sign come out correct
        without any special-casing here (verified in tests, not assumed)."""
        mesh = self._get(target).copy()
        scale_vec = {"x": (-1.0, 1.0, 1.0), "y": (1.0, -1.0, 1.0), "z": (1.0, 1.0, -1.0)}[axis]
        transform = np.diag([*scale_vec, 1.0])
        mesh.apply_transform(transform)
        self._consume(target)
        h = self._new_handle(mesh)
        self._record("mirror", {"target": target, "axis": axis})
        return h

    def scale(self, target: int, factor_x: float, factor_y: float, factor_z: float) -> int:
        """Non-uniform scale about the shape's OWN bounding-box center, not
        the world origin -- scaling an off-center shape about the world
        origin would also translate it, which is never the intent of a
        pure resize."""
        mesh = self._get(target).copy()
        center = mesh.bounds.mean(axis=0)
        mesh.apply_translation(-center)
        mesh.apply_transform(np.diag([factor_x, factor_y, factor_z, 1.0]))
        mesh.apply_translation(center)
        self._consume(target)
        h = self._new_handle(mesh)
        self._record("scale", {"target": target, "factor_x": factor_x, "factor_y": factor_y, "factor_z": factor_z})
        return h

    # ---- macros ----

    def drill_hole(self, target: int, radius: float, depth: float, center: tuple[float, float, float], axis: Axis = "z") -> int:
        from mesh_quality import remesh_flat_facets

        mesh = self._get(target)
        cutter_height = _cutter_height(mesh, axis, depth)
        cutter = trimesh.creation.cylinder(radius=radius, height=cutter_height, sections=32)
        cutter = _rotate_to_axis(cutter, axis)
        cutter.apply_translation(center)
        result = remesh_flat_facets(mesh.difference(cutter, engine="manifold"))
        self._consume(target)
        h = self._new_handle(result)
        self._record("drill_hole", {"target": target, "radius": radius, "depth": depth, "center": list(center), "axis": axis})
        return h

    def drill_hole_pattern(
        self,
        target: int,
        pattern: PatternKind,
        count: int,
        radius: float,
        depth: float,
        axis: Axis = "z",
        margin: float = 15.0,
        circle_diameter: float = 0.0,  # 0.0 = not specified, derive from target's own bounds -- deliberately not
                                        # None, so every argument stays a plain float (JSON/ML-friendly, no Optional)
        z: float = 0.0,
    ) -> int:
        """Placement positions are derived from the TARGET's own current
        bounds (inset by margin) rather than taken as separate arguments --
        those extents are already implicit in whichever earlier add_box/
        add_cylinder call created the target, so re-predicting them would
        be redundant and would blow up the argument count for this step."""
        mesh = self._get(target)
        bounds = mesh.bounds
        length, width = float(bounds[1][0] - bounds[0][0]), float(bounds[1][1] - bounds[0][1])

        if pattern == "single" or count <= 1:
            positions = [(0.0, 0.0, z)]
        elif pattern == "corner_array":
            hx, hy = length / 2 - margin, width / 2 - margin
            corners = [(sx * hx, sy * hy, z) for sx in (-1, 1) for sy in (-1, 1)]
            positions = corners[: max(count, 1)]
        elif pattern == "bolt_circle":
            d = circle_diameter if circle_diameter > 0 else min(length, width) - 2 * margin
            r = d / 2
            positions = [(r * np.cos(2 * np.pi * i / count), r * np.sin(2 * np.pi * i / count), z) for i in range(count)]
        elif pattern == "linear_array":
            span = max(length - 2 * margin, 0.0)
            if count <= 1:
                positions = [(0.0, 0.0, z)]
            else:
                step = span / (count - 1)
                start = -span / 2
                positions = [(start + i * step, 0.0, z) for i in range(count)]
        else:
            raise SandboxError(f"unknown pattern {pattern!r}")

        from mesh_quality import remesh_flat_facets

        cutter_height = _cutter_height(mesh, axis, depth)
        for x, y, zc in positions:
            cutter = trimesh.creation.cylinder(radius=radius, height=cutter_height, sections=32)
            cutter = _rotate_to_axis(cutter, axis)
            cutter.apply_translation([x, y, zc])
            mesh = mesh.difference(cutter, engine="manifold")
        mesh = remesh_flat_facets(mesh)  # once, after all holes -- one quality pass over a polygon-with-N-holes beats N sequential passes
        self._consume(target)
        h = self._new_handle(mesh)
        self._record(
            "drill_hole_pattern",
            {"target": target, "pattern": pattern, "count": count, "radius": radius, "depth": depth,
             "axis": axis, "margin": margin, "circle_diameter": circle_diameter, "z": z},
        )
        return h

    def linear_pattern(self, target: int, axis: Axis, count: int, spacing: float) -> int:
        """Generalizes drill_hole_pattern's linear_array placement math to
        ARRAY/UNION ANY shape along an axis, not just a hole cutter -- e.g.
        a row of ribs, fins, or bosses on a plate, not something any of the
        fixed builders could express. Copies are centered on the target's
        own current position (an array of 1 is just the original, unmoved)."""
        from mesh_quality import remesh_flat_facets

        base_mesh = self._get(target)
        count = max(1, count)
        axis_vec = _AXIS_VECTORS[axis]
        result = None
        for i in range(count):
            offset = (i - (count - 1) / 2) * spacing
            copy = base_mesh.copy()
            copy.apply_translation([offset * axis_vec[0], offset * axis_vec[1], offset * axis_vec[2]])
            result = copy if result is None else result.union(copy, engine="manifold")
        result = remesh_flat_facets(result)
        self._consume(target)
        h = self._new_handle(result)
        self._record("linear_pattern", {"target": target, "axis": axis, "count": count, "spacing": spacing})
        return h

    def circular_pattern(self, target: int, count: int, radius: float) -> int:
        """Generalizes drill_hole_pattern's bolt_circle placement math to
        ARRAY/UNION ANY shape around a circle (about Z, translated only --
        same convention as bolt_circle, no per-copy reorientation) -- e.g.
        a ring of bosses or standoffs, not just bolt holes."""
        from mesh_quality import remesh_flat_facets

        base_mesh = self._get(target)
        count = max(1, count)
        result = None
        for i in range(count):
            angle = 2 * np.pi * i / count
            x, y = radius * np.cos(angle), radius * np.sin(angle)
            copy = base_mesh.copy()
            copy.apply_translation([x, y, 0.0])
            result = copy if result is None else result.union(copy, engine="manifold")
        result = remesh_flat_facets(result)
        self._consume(target)
        h = self._new_handle(result)
        self._record("circular_pattern", {"target": target, "count": count, "radius": radius})
        return h

    # ---- face-based tools: Blender-style "select a face, then act on it,"
    # not whole-solid CSG. Face SELECTION here is by which axis-ish
    # direction a face points ("+z", "-x", ...) -- a bounded, categorical
    # choice a model can predict, the same reasoning that already governs
    # every other argument in this file -- never an arbitrary face index,
    # which wouldn't be a stable, learnable argument. See
    # mesh_quality.find_facet_by_direction, which these both share. ----

    def extrude_face(self, target: int, direction: FaceDirection, distance: float) -> int:
        """Extrudes the flat face pointing `direction` outward by `distance`
        -- a protruding boss, built from THAT face's own outline (including
        any holes already in it, which stay open) rather than a whole new
        primitive glued on. Genuine face-based topology, not just whole-
        solid CSG: the new geometry's footprint is whatever that face's
        boundary actually is, not a separately-predicted shape."""
        from shapely.geometry import Polygon

        from mesh_quality import find_facet_by_direction, remesh_flat_facets

        mesh = self._get(target)
        face = find_facet_by_direction(mesh, _FACE_DIRECTION_VECTORS[direction])
        if face is None:
            raise SandboxError(f"no flat face points {direction!r} on this shape")

        polygon = Polygon(face.outer, face.holes)
        if not polygon.is_valid or polygon.area <= 1e-9:
            raise SandboxError("selected face has no valid area to extrude")
        target_area = max(1.0, polygon.area / 300)
        boss_local = trimesh.creation.extrude_polygon(
            polygon, height=distance, engine="triangle", triangle_args=f"pq30a{target_area:.3f}"
        )
        transform = np.eye(4)
        transform[:3, :3] = np.column_stack([face.u, face.v, face.normal])
        transform[:3, 3] = face.origin
        boss_local.apply_transform(transform)

        result = remesh_flat_facets(mesh.union(boss_local, engine="manifold"))
        self._consume(target)
        h = self._new_handle(result)
        self._record("extrude_face", {"target": target, "direction": direction, "distance": distance})
        return h

    def inset_pocket(self, target: int, direction: FaceDirection, depth: float, margin: float) -> int:
        """Insets the flat face pointing `direction` inward by `margin`
        (Blender's "inset face"), then cuts that smaller outline `depth`
        into the material -- a recessed panel/pocket, not a circular hole
        (see drill_hole for that). Falls back to a clean SandboxError (never
        a silent wrong shape) if `margin` insets the face away entirely."""
        from shapely.geometry import Polygon

        from mesh_quality import find_facet_by_direction, remesh_flat_facets

        mesh = self._get(target)
        face = find_facet_by_direction(mesh, _FACE_DIRECTION_VECTORS[direction])
        if face is None:
            raise SandboxError(f"no flat face points {direction!r} on this shape")

        outer_polygon = Polygon(face.outer)
        inset = outer_polygon.buffer(-margin)
        if inset.is_empty or inset.geom_type != "Polygon" or inset.area <= 1e-9:
            raise SandboxError(f"margin {margin} insets the face away entirely")

        # cutter spans from just outside the face plane (robustness padding,
        # same reasoning as drill_hole's own oversized cutter) to `depth`
        # into the material, built along -normal (into the solid).
        target_area = max(1.0, inset.area / 300)
        cutter_local = trimesh.creation.extrude_polygon(
            inset, height=depth + 2.0, engine="triangle", triangle_args=f"pq30a{target_area:.3f}"
        )
        cutter_local.apply_translation([0.0, 0.0, -2.0])
        transform = np.eye(4)
        transform[:3, :3] = np.column_stack([face.u, face.v, -face.normal])
        transform[:3, 3] = face.origin
        cutter_local.apply_transform(transform)

        result = remesh_flat_facets(mesh.difference(cutter_local, engine="manifold"))
        self._consume(target)
        h = self._new_handle(result)
        self._record("inset_pocket", {"target": target, "direction": direction, "depth": depth, "margin": margin})
        return h

    def _face_local_transform(self, face) -> np.ndarray:
        """Shared local-face-plane -> world transform: local x/y are the
        face's own in-plane basis (face.u, face.v), local +z is INTO the
        material (-face.normal) -- the same convention inset_pocket already
        established, factored out so counterbore_hole/drill_slot (both
        face-based cutters) don't re-derive it."""
        transform = np.eye(4)
        transform[:3, :3] = np.column_stack([face.u, face.v, -face.normal])
        transform[:3, 3] = face.origin
        return transform

    @staticmethod
    def _face_center_uv(face) -> tuple[float, float]:
        """face.origin (and so (u, v) = (0, 0)) is a CORNER of the facet's
        boundary polygon, not its center -- confirmed directly (a 60x40
        +z face's outer ring ran from (0,0) to (40,-60) in its own local
        coords, not centered on (0,0)) -- so a human/LLM-friendly "offset
        from the middle of this face" argument (counterbore_hole,
        drill_slot) needs this correction added in, or offset_u=0,
        offset_v=0 would silently land near a corner instead of the
        center it's documented to mean."""
        center = (face.outer.min(axis=0) + face.outer.max(axis=0)) / 2.0
        return float(center[0]), float(center[1])

    def counterbore_hole(
        self, target: int, direction: FaceDirection, offset_u: float, offset_v: float,
        hole_radius: float, head_radius: float, head_depth: float, head_style: HeadStyle,
    ) -> int:
        """A fastener clearance hole entering the `direction` face at
        (offset_u, offset_v) in that face's own in-plane coordinates (the
        same 2D frame find_facet_by_direction already establishes) -- a
        straight shank bore all the way through, plus a recessed pocket at
        the surface sized for the fastener's HEAD: flat-bottomed
        (head_style="flat", a counterbore for a socket-head fastener) or
        conical (head_style="conical", a countersink for a flat-head
        fastener). Both head shapes are one tool with a bounded enum choice
        -- the same reasoning as add_extrusion's profile_kind -- rather than
        two near-duplicate tools."""
        from mesh_quality import find_facet_by_direction, remesh_flat_facets

        mesh = self._get(target)
        face = find_facet_by_direction(mesh, _FACE_DIRECTION_VECTORS[direction])
        if face is None:
            raise SandboxError(f"no flat face points {direction!r} on this shape")
        if head_radius <= hole_radius:
            raise SandboxError(f"head_radius ({head_radius}) must be larger than hole_radius ({hole_radius})")

        axis_idx = int(np.argmax(np.abs(face.normal)))
        shank_len = float(mesh.bounds[1][axis_idx] - mesh.bounds[0][axis_idx]) + 10.0  # robust full-penetration padding, same reasoning as drill_hole
        pad = 2.0
        center_u, center_v = self._face_center_uv(face)
        u, v = center_u + offset_u, center_v + offset_v

        shank_local = trimesh.creation.cylinder(radius=hole_radius, height=shank_len, sections=32)
        shank_local.apply_translation([u, v, shank_len / 2 - pad])

        if head_style == "flat":
            # a cylinder's radius doesn't depend on how tall it is, so
            # padding the height for CSG robustness (same reasoning as the
            # shank above) doesn't distort the recess's actual shape at all.
            head_local = trimesh.creation.cylinder(radius=head_radius, height=head_depth + pad, sections=32)
            head_local.apply_translation([u, v, head_depth / 2 - pad / 2])
        elif head_style == "conical":
            # UNLIKE the flat cylinder, a frustum's radius changes linearly
            # along its height -- padding its height the same way the flat
            # case does would shift the taper and make the recess narrower
            # than head_radius at the true surface. So this one is built to
            # its EXACT head_depth (local z = 0, the surface, at the WIDE
            # head_radius end, down to local z = head_depth at the NARROW
            # hole_radius end where it meets the shank) with no padding of
            # its own -- the shank's own pad already provides the CSG
            # robustness overlap right at the entry face once unioned.
            head_local = _frustum_mesh(bottom_radius=head_radius, top_radius=hole_radius, height=head_depth, sections=32)
            head_local.apply_translation([u, v, head_depth / 2])
        else:
            raise SandboxError(f"unknown head_style {head_style!r}")

        transform = self._face_local_transform(face)
        shank_local.apply_transform(transform)
        head_local.apply_transform(transform)

        cutter = shank_local.union(head_local, engine="manifold")
        result = remesh_flat_facets(mesh.difference(cutter, engine="manifold"))
        self._consume(target)
        h = self._new_handle(result)
        self._record("counterbore_hole", {
            "target": target, "direction": direction, "offset_u": offset_u, "offset_v": offset_v,
            "hole_radius": hole_radius, "head_radius": head_radius, "head_depth": head_depth, "head_style": head_style,
        })
        return h

    def drill_slot(
        self, target: int, direction: FaceDirection, offset_u: float, offset_v: float,
        radius: float, slot_length: float, slot_axis: Literal["u", "v"], depth: float,
    ) -> int:
        """An elongated, rounded-end slot cut into the `direction` face --
        the common "adjustable mounting slot" feature drill_hole/
        drill_hole_pattern can't reach (those only ever place round holes).
        Built from two end-cap cylinders (radius) `slot_length` apart along
        the face's own u or v axis (a bounded categorical choice, never a
        free angle) unioned with a connecting box, exactly matching
        drill_hole's own robustness convention: the cutter's depth is
        floored at the target's own extent along the entry axis so a
        too-shallow predicted `depth` can never leave the slot not fully
        breaking through."""
        from mesh_quality import find_facet_by_direction, remesh_flat_facets

        mesh = self._get(target)
        face = find_facet_by_direction(mesh, _FACE_DIRECTION_VECTORS[direction])
        if face is None:
            raise SandboxError(f"no flat face points {direction!r} on this shape")

        axis_idx = int(np.argmax(np.abs(face.normal)))
        cut_len = float(mesh.bounds[1][axis_idx] - mesh.bounds[0][axis_idx]) + max(depth, 0.0) + 10.0
        pad = 2.0
        center_u, center_v = self._face_center_uv(face)
        u, v = center_u + offset_u, center_v + offset_v
        half_len = slot_length / 2.0
        du, dv = (half_len, 0.0) if slot_axis == "u" else (0.0, half_len)

        end_a = trimesh.creation.cylinder(radius=radius, height=cut_len, sections=32)
        end_a.apply_translation([u - du, v - dv, cut_len / 2 - pad])
        end_b = trimesh.creation.cylinder(radius=radius, height=cut_len, sections=32)
        end_b.apply_translation([u + du, v + dv, cut_len / 2 - pad])

        box_extents = [slot_length, radius * 2, cut_len] if slot_axis == "u" else [radius * 2, slot_length, cut_len]
        middle = trimesh.creation.box(extents=box_extents)
        middle.apply_translation([u, v, cut_len / 2 - pad])

        cutter_local = end_a.union(end_b, engine="manifold").union(middle, engine="manifold")
        transform = self._face_local_transform(face)
        cutter_local.apply_transform(transform)

        result = remesh_flat_facets(mesh.difference(cutter_local, engine="manifold"))
        self._consume(target)
        h = self._new_handle(result)
        self._record("drill_slot", {
            "target": target, "direction": direction, "offset_u": offset_u, "offset_v": offset_v,
            "radius": radius, "slot_length": slot_length, "slot_axis": slot_axis, "depth": depth,
        })
        return h

    def _face_pattern_positions(
        self, face, pattern: PatternKind, count: int, margin: float, circle_diameter: float,
    ) -> list[tuple[float, float]]:
        """Shared 2D placement math for face-based *_pattern tools -- the
        same four PatternKind layouts drill_hole_pattern already
        established (single/corner_array/bolt_circle/linear_array),
        computed in a FACE's own (u, v) local frame using the face's own
        footprint (inset by margin) rather than the target's overall world
        bounds, so a pattern can be placed on any flat face, not just one
        that happens to match the target's full bounding box."""
        center_u, center_v = self._face_center_uv(face)
        length = float(face.outer[:, 0].max() - face.outer[:, 0].min())
        width = float(face.outer[:, 1].max() - face.outer[:, 1].min())

        if pattern == "single" or count <= 1:
            positions = [(0.0, 0.0)]
        elif pattern == "corner_array":
            hx, hy = length / 2 - margin, width / 2 - margin
            corners = [(sx * hx, sy * hy) for sx in (-1, 1) for sy in (-1, 1)]
            positions = corners[: max(count, 1)]
        elif pattern == "bolt_circle":
            d = circle_diameter if circle_diameter > 0 else min(length, width) - 2 * margin
            r = d / 2
            positions = [(r * np.cos(2 * np.pi * i / count), r * np.sin(2 * np.pi * i / count)) for i in range(count)]
        elif pattern == "linear_array":
            span = max(length - 2 * margin, 0.0)
            if count <= 1:
                positions = [(0.0, 0.0)]
            else:
                step = span / (count - 1)
                start = -span / 2
                positions = [(start + i * step, 0.0) for i in range(count)]
        else:
            raise SandboxError(f"unknown pattern {pattern!r}")
        return [(center_u + u, center_v + v) for u, v in positions]

    def counterbore_hole_pattern(
        self, target: int, direction: FaceDirection, pattern: PatternKind, count: int,
        hole_radius: float, head_radius: float, head_depth: float, head_style: HeadStyle,
        margin: float = 15.0, circle_diameter: float = 0.0,
    ) -> int:
        """Generalizes counterbore_hole to a PatternKind layout of holes
        across one face in a single call, reusing drill_hole_pattern's
        placement math via _face_pattern_positions -- a bolt circle or
        corner array of fastener holes WITH recessed heads, instead of
        counterbore_hole's single hole or drill_hole_pattern's plain
        (headless) holes."""
        if head_radius <= hole_radius:
            raise SandboxError(f"head_radius ({head_radius}) must be larger than hole_radius ({hole_radius})")
        from mesh_quality import find_facet_by_direction, remesh_flat_facets

        mesh = self._get(target)
        face = find_facet_by_direction(mesh, _FACE_DIRECTION_VECTORS[direction])
        if face is None:
            raise SandboxError(f"no flat face points {direction!r} on this shape")

        axis_idx = int(np.argmax(np.abs(face.normal)))
        shank_len = float(mesh.bounds[1][axis_idx] - mesh.bounds[0][axis_idx]) + 10.0
        pad = 2.0
        positions = self._face_pattern_positions(face, pattern, count, margin, circle_diameter)

        cutter = None
        for u, v in positions:
            shank = trimesh.creation.cylinder(radius=hole_radius, height=shank_len, sections=32)
            shank.apply_translation([u, v, shank_len / 2 - pad])
            if head_style == "flat":
                head = trimesh.creation.cylinder(radius=head_radius, height=head_depth + pad, sections=32)
                head.apply_translation([u, v, head_depth / 2 - pad / 2])
            elif head_style == "conical":
                head = _frustum_mesh(bottom_radius=head_radius, top_radius=hole_radius, height=head_depth, sections=32)
                head.apply_translation([u, v, head_depth / 2])
            else:
                raise SandboxError(f"unknown head_style {head_style!r}")
            one = shank.union(head, engine="manifold")
            cutter = one if cutter is None else cutter.union(one, engine="manifold")

        transform = self._face_local_transform(face)
        cutter.apply_transform(transform)
        result = remesh_flat_facets(mesh.difference(cutter, engine="manifold"))
        self._consume(target)
        h = self._new_handle(result)
        self._record("counterbore_hole_pattern", {
            "target": target, "direction": direction, "pattern": pattern, "count": count,
            "hole_radius": hole_radius, "head_radius": head_radius, "head_depth": head_depth, "head_style": head_style,
            "margin": margin, "circle_diameter": circle_diameter,
        })
        return h

    def add_slot_pattern(
        self, target: int, direction: FaceDirection, pattern: PatternKind, count: int,
        radius: float, slot_length: float, slot_axis: Literal["u", "v"], depth: float,
        margin: float = 15.0, circle_diameter: float = 0.0,
    ) -> int:
        """Generalizes drill_slot to a PatternKind layout of slots across
        one face -- e.g. a row or bolt-circle of adjustable mounting
        slots, reusing the same placement math as
        counterbore_hole_pattern/drill_hole_pattern."""
        from mesh_quality import find_facet_by_direction, remesh_flat_facets

        mesh = self._get(target)
        face = find_facet_by_direction(mesh, _FACE_DIRECTION_VECTORS[direction])
        if face is None:
            raise SandboxError(f"no flat face points {direction!r} on this shape")

        axis_idx = int(np.argmax(np.abs(face.normal)))
        cut_len = float(mesh.bounds[1][axis_idx] - mesh.bounds[0][axis_idx]) + max(depth, 0.0) + 10.0
        pad = 2.0
        half_len = slot_length / 2.0
        du, dv = (half_len, 0.0) if slot_axis == "u" else (0.0, half_len)
        box_extents = [slot_length, radius * 2, cut_len] if slot_axis == "u" else [radius * 2, slot_length, cut_len]
        positions = self._face_pattern_positions(face, pattern, count, margin, circle_diameter)

        cutter = None
        for u, v in positions:
            end_a = trimesh.creation.cylinder(radius=radius, height=cut_len, sections=32)
            end_a.apply_translation([u - du, v - dv, cut_len / 2 - pad])
            end_b = trimesh.creation.cylinder(radius=radius, height=cut_len, sections=32)
            end_b.apply_translation([u + du, v + dv, cut_len / 2 - pad])
            middle = trimesh.creation.box(extents=box_extents)
            middle.apply_translation([u, v, cut_len / 2 - pad])
            one = end_a.union(end_b, engine="manifold").union(middle, engine="manifold")
            cutter = one if cutter is None else cutter.union(one, engine="manifold")

        transform = self._face_local_transform(face)
        cutter.apply_transform(transform)
        result = remesh_flat_facets(mesh.difference(cutter, engine="manifold"))
        self._consume(target)
        h = self._new_handle(result)
        self._record("add_slot_pattern", {
            "target": target, "direction": direction, "pattern": pattern, "count": count,
            "radius": radius, "slot_length": slot_length, "slot_axis": slot_axis, "depth": depth,
            "margin": margin, "circle_diameter": circle_diameter,
        })
        return h

    def add_knurl_pattern(
        self, target: int, direction: FaceDirection, offset_u: float, offset_v: float,
        radius: float, height: float, ridge_count: int,
    ) -> int:
        """A ring of thin radial ridges standing proud of the `direction`
        face, centered at (offset_u, offset_v) -- a grip-texture
        approximation of a Blender/machining "knurl" (straight radial
        ridges, not a true diamond-knurl crosshatch, which would need a
        genuine surface-relief operation this stack doesn't have -- a
        disclosed simplification, not a silent wrong shape)."""
        if ridge_count < 3:
            raise SandboxError(f"ridge_count ({ridge_count}) must be at least 3")
        from mesh_quality import find_facet_by_direction, remesh_flat_facets

        mesh = self._get(target)
        face = find_facet_by_direction(mesh, _FACE_DIRECTION_VECTORS[direction])
        if face is None:
            raise SandboxError(f"no flat face points {direction!r} on this shape")
        center_u, center_v = self._face_center_uv(face)
        u0, v0 = center_u + offset_u, center_v + offset_v
        ridge_width = max(radius * 0.08, 0.3)

        ridges = None
        for i in range(ridge_count):
            angle = 2 * np.pi * i / ridge_count
            ridge = trimesh.creation.box(extents=[radius, ridge_width, height])
            ridge.apply_translation([radius / 2.0, 0.0, -height / 2.0])  # local -z = outward
            ridge.apply_transform(trimesh.transformations.rotation_matrix(angle, [0, 0, 1]))
            ridge.apply_translation([u0, v0, 0.0])
            ridges = ridge if ridges is None else ridges.union(ridge, engine="manifold")

        transform = self._face_local_transform(face)
        ridges.apply_transform(transform)
        result = remesh_flat_facets(mesh.union(ridges, engine="manifold"))
        self._consume(target)
        h = self._new_handle(result)
        self._record("add_knurl_pattern", {
            "target": target, "direction": direction, "offset_u": offset_u, "offset_v": offset_v,
            "radius": radius, "height": height, "ridge_count": ridge_count,
        })
        return h

    def radial_array_oriented(self, target: int, count: int, radius: float) -> int:
        """Like circular_pattern, but each copy is also ROTATED about Z by
        its own placement angle before being moved into position, so it
        faces OUTWARD from the circle's center instead of keeping the
        original shape's orientation -- e.g. a ring of spokes/pointers/
        teeth that must each point away from the hub, not a ring of
        identically-oriented copies."""
        from mesh_quality import remesh_flat_facets

        base_mesh = self._get(target)
        count = max(1, count)
        result = None
        for i in range(count):
            angle = 2 * np.pi * i / count
            copy = base_mesh.copy()
            copy.apply_transform(trimesh.transformations.rotation_matrix(angle, [0, 0, 1]))
            x, y = radius * np.cos(angle), radius * np.sin(angle)
            copy.apply_translation([x, y, 0.0])
            result = copy if result is None else result.union(copy, engine="manifold")
        result = remesh_flat_facets(result)
        self._consume(target)
        h = self._new_handle(result)
        self._record("radial_array_oriented", {"target": target, "count": count, "radius": radius})
        return h

    def linear_array_scaled(self, target: int, axis: Axis, count: int, spacing: float, scale_step: float) -> int:
        """Like linear_pattern, but each successive copy is uniformly
        scaled about its own center by (1 + i*scale_step) -- e.g. a row of
        progressively larger/smaller fins or teeth, not just N identical
        copies. Raises if any copy's scale factor would end up <= 0."""
        from mesh_quality import remesh_flat_facets

        base_mesh = self._get(target)
        count = max(1, count)
        axis_vec = _AXIS_VECTORS[axis]
        center0 = base_mesh.bounds.mean(axis=0)
        result = None
        for i in range(count):
            factor = 1.0 + i * scale_step
            if factor <= 0:
                raise SandboxError(f"scale_step ({scale_step}) makes copy {i}'s scale factor non-positive ({factor:g})")
            offset = (i - (count - 1) / 2) * spacing
            copy = base_mesh.copy()
            copy.apply_translation(-center0)
            copy.apply_scale(factor)
            copy.apply_translation(center0)
            copy.apply_translation([offset * axis_vec[0], offset * axis_vec[1], offset * axis_vec[2]])
            result = copy if result is None else result.union(copy, engine="manifold")
        result = remesh_flat_facets(result)
        self._consume(target)
        h = self._new_handle(result)
        self._record("linear_array_scaled", {
            "target": target, "axis": axis, "count": count, "spacing": spacing, "scale_step": scale_step,
        })
        return h

    def add_hex_socket(
        self, target: int, direction: FaceDirection, offset_u: float, offset_v: float,
        across_flats: float, depth: float,
    ) -> int:
        """A hex-shaped recessed pocket -- an Allen-key/hex-bit socket --
        cut into the `direction` face at (offset_u, offset_v). Reuses
        _frustum_mesh with equal bottom/top radius and sections=6 (the
        same "flat-faceted prism via a degenerate frustum" trick
        add_polygon_frustum uses). `across_flats` is the standard
        flat-to-flat hex measurement; the circumradius _frustum_mesh needs
        is across_flats / sqrt(3)."""
        from mesh_quality import find_facet_by_direction, remesh_flat_facets

        mesh = self._get(target)
        face = find_facet_by_direction(mesh, _FACE_DIRECTION_VECTORS[direction])
        if face is None:
            raise SandboxError(f"no flat face points {direction!r} on this shape")
        center_u, center_v = self._face_center_uv(face)
        u, v = center_u + offset_u, center_v + offset_v
        circumradius = across_flats / np.sqrt(3.0)
        pad = 2.0

        cutter = _frustum_mesh(bottom_radius=circumradius, top_radius=circumradius, height=depth + pad, sections=6)
        cutter.apply_translation([u, v, depth / 2 - pad / 2])
        transform = self._face_local_transform(face)
        cutter.apply_transform(transform)

        result = remesh_flat_facets(mesh.difference(cutter, engine="manifold"))
        self._consume(target)
        h = self._new_handle(result)
        self._record("add_hex_socket", {
            "target": target, "direction": direction, "offset_u": offset_u, "offset_v": offset_v,
            "across_flats": across_flats, "depth": depth,
        })
        return h

    # Above this, shell()'s axis-scaled-copy approximation is rejected
    # outright rather than silently returned -- see measure_wall_thickness_deviation's
    # own docstring for why 90th-percentile deviation, not mean/median, and
    # why 0.35 specifically: confirmed directly that a plain box scores
    # 0.0 and a real shelled gear profile (the approximation's own
    # documented worst case) scores 0.72 -- comfortably separated, so 0.35
    # sits well clear of normal box/prismatic-shape noise while still
    # catching a genuinely bad approximation, not just a merely imperfect one.
    _SHELL_MAX_THICKNESS_DEVIATION = 0.35

    def shell(self, target: int, wall_thickness: float) -> int:
        """Hollows out the shape, leaving `wall_thickness` of material
        uniformly -- Blender's "solidify"/shell. Implemented as an
        APPROXIMATION, disclosed rather than hidden: this scales a copy of
        the shape down about its own center (independently per axis, from
        its own bounding-box half-extents) and subtracts it, rather than a
        true constant-distance surface offset -- trimesh/manifold3d have no
        general mesh-offset operation to build a real one on. This is
        accurate for box-like/prismatic shapes (an enclosure, a plate) and
        increasingly approximate for shapes with fine surface detail at a
        scale comparable to `wall_thickness` (e.g. a gear's teeth would
        shrink proportionally, not stay a constant thickness).

        That approximation is now MEASURED, not just disclosed in a
        docstring: mesh_quality.measure_wall_thickness_deviation ray-casts
        real wall-thickness samples from the actual result and, if the
        90th-percentile deviation from the requested thickness is too
        large, this raises SandboxError instead of returning the bad
        approximation as if it were a correct shell -- "fail explicitly
        rather than presenting an approximation as an exact shell," not a
        silent best-effort result."""
        from mesh_quality import measure_wall_thickness_deviation, remesh_flat_facets

        mesh = self._get(target)
        half_extents = (mesh.bounds[1] - mesh.bounds[0]) / 2.0
        factors = np.array([
            max(0.05, (h - wall_thickness) / h) if h > 1e-6 else 1.0
            for h in half_extents
        ])
        center = mesh.bounds.mean(axis=0)
        inner = mesh.copy()
        inner.apply_translation(-center)
        inner.apply_transform(np.diag([*factors, 1.0]))
        inner.apply_translation(center)

        result = remesh_flat_facets(mesh.difference(inner, engine="manifold"))

        deviation = measure_wall_thickness_deviation(result, wall_thickness)
        if deviation is not None and deviation > self._SHELL_MAX_THICKNESS_DEVIATION:
            raise SandboxError(
                f"shell: the requested wall_thickness ({wall_thickness:g}mm) cannot be produced reliably on "
                f"this shape by shell()'s axis-scaled-copy approximation -- measured 90th-percentile wall-"
                f"thickness deviation is {deviation:.0%}, over the {self._SHELL_MAX_THICKNESS_DEVIATION:.0%} "
                "bound this tool accepts. This shape likely has fine surface detail (teeth, ribs, small "
                "features) at a scale comparable to the requested wall thickness, where this approximation "
                "is known to break down -- see shell()'s own docstring."
            )

        self._consume(target)
        h = self._new_handle(result)
        self._record("shell", {"target": target, "wall_thickness": wall_thickness})
        return h

    # ---- mesh-refinement tools: smoothing / edge-frame conversion / vertex
    # cleanup -- gaps found by cross-referencing this project's own acquired
    # CAD/mesh documentation (knowledge/sources.py) against the existing
    # tool roster: Blender's Subdivision Surface, Wireframe, and Weld
    # modifiers each had real usage in the acquired docs with no equivalent
    # tool here. All three operate on an EXISTING shape's own mesh (not a
    # free-form point list, respecting program_model.py's fixed-arity
    # constraint), same as shell/fillet_edge above. ----

    def smooth_subdivide(self, target: int, subdivisions: int = 1, iterations: int = 10) -> int:
        """Rounds off a shape's body/edges -- Blender's Subdivision Surface
        modifier, which both subdivides (adds geometry) AND smooths
        (relaxes vertex positions toward a limit surface); doing only the
        smoothing step on a low-poly shape is confirmed, not assumed, to be
        wrong -- Taubin-smoothing a plain 8-vertex box with no prior
        subdivision collapses it to ~6% of its original volume (tested
        directly against this project's own trimesh version), since there
        isn't enough geometry for the filter to preserve shape against.
        Subdividing first (each level ~4x's the face count) gives the
        smoothing filter something reasonable to work with -- confirmed to
        preserve >98% of volume at 2 subdivision levels + 10 iterations on
        the same test box. Taubin smoothing specifically (not plain
        Laplacian) because it's volume-preserving by construction, unlike
        Laplacian smoothing's well-known shrinkage problem."""
        import trimesh.smoothing as _smoothing

        if not (0 <= subdivisions <= 3):
            raise SandboxError(f"smooth_subdivide needs subdivisions in [0, 3], got {subdivisions} "
                                "(each level ~4x's the face count -- higher is impractically slow)")
        if not (1 <= iterations <= 50):
            raise SandboxError(f"smooth_subdivide needs iterations in [1, 50], got {iterations}")

        mesh = self._get(target).copy()
        for _ in range(subdivisions):
            mesh = mesh.subdivide()
        _smoothing.filter_taubin(mesh, iterations=iterations)

        self._consume(target)
        h = self._new_handle(mesh)
        self._record("smooth_subdivide", {"target": target, "subdivisions": subdivisions, "iterations": iterations})
        return h

    _MAX_WIREFRAME_EDGES = 500  # see docstring below -- a real, disclosed bound, not silently truncated

    def add_wireframe_from_edges(self, target: int, edge_radius: float, sections: int = 6) -> int:
        """Converts a shape into a tube-frame following its own edges --
        Blender's Wireframe modifier. Builds one cylinder per unique mesh
        edge (trimesh.creation.cylinder's `segment` param connects two real
        3D points directly, taken from the shape's OWN existing topology,
        never free-form LLM-authored coordinates) and unions them all --
        the exact same sequential `result.union(copy, engine="manifold")`
        pattern linear_pattern/circular_pattern already use above, not a
        new combining strategy. Bounded by _MAX_WIREFRAME_EDGES: this is an
        O(edge count) chain of boolean unions, genuinely slow past a few
        hundred edges -- raises a clear, actionable error instead of
        silently hanging on a shape with thousands of edges (e.g. one
        that's already been heavily subdivided)."""
        from mesh_quality import remesh_flat_facets

        if edge_radius <= 1e-9:
            raise SandboxError(f"add_wireframe_from_edges needs a positive edge_radius, got {edge_radius}")
        if sections < 3:
            raise SandboxError(f"add_wireframe_from_edges needs sections >= 3, got {sections}")

        mesh = self._get(target)
        edges = mesh.edges_unique
        if len(edges) == 0:
            raise SandboxError("add_wireframe_from_edges: shape has no edges")
        if len(edges) > self._MAX_WIREFRAME_EDGES:
            raise SandboxError(
                f"add_wireframe_from_edges: shape has {len(edges)} edges, over the {self._MAX_WIREFRAME_EDGES} "
                "bound this tool supports (each edge is its own boolean union -- intended for simple, "
                "moderate-edge-count shapes like a box or bracket, not a dense/subdivided mesh)"
            )

        result = None
        for i0, i1 in edges:
            p0, p1 = mesh.vertices[i0], mesh.vertices[i1]
            if np.linalg.norm(p1 - p0) < 1e-9:
                continue  # a degenerate zero-length edge -- skip rather than fail the whole tool on it
            tube = trimesh.creation.cylinder(radius=edge_radius, segment=[p0, p1], sections=sections)
            result = tube if result is None else result.union(tube, engine="manifold")
        if result is None:
            raise SandboxError("add_wireframe_from_edges: every edge was degenerate -- nothing to build")

        result = remesh_flat_facets(result)
        self._consume(target)
        h = self._new_handle(result)
        self._record("add_wireframe_from_edges", {"target": target, "edge_radius": edge_radius, "sections": sections})
        return h

    def weld_vertices(self, target: int, tolerance: float = 0.01) -> int:
        """Merges vertices within `tolerance` (mm) of each other -- Blender's
        Weld modifier; a mesh-cleanup pass for near-duplicate vertices left
        behind by other operations (e.g. a mirror_merge seam, or
        floating-point residue from a boolean chain), not a shape-creating
        tool. trimesh's own merge_vertices takes a `digits_vertex` rounding
        precision rather than a direct distance, so `tolerance` is
        converted via -log10 (tolerance=0.01mm -> 2 decimal digits);
        mesh.process() afterward is trimesh's own standard cleanup pass
        (removes any now-degenerate/duplicate faces the merge produced),
        not a second home-grown cleanup step."""
        if tolerance <= 1e-9:
            raise SandboxError(f"weld_vertices needs a positive tolerance, got {tolerance}")

        mesh = self._get(target).copy()
        digits = max(0, int(round(-np.log10(tolerance))))
        mesh.merge_vertices(digits_vertex=digits)
        mesh.process(validate=True)

        self._consume(target)
        h = self._new_handle(mesh)
        self._record("weld_vertices", {"target": target, "tolerance": tolerance})
        return h

    # ---- edge tools: round or flatten the edge where two PERPENDICULAR
    # faces meet -- the two faces are selected the same bounded way as
    # extrude_face/inset_pocket (by direction, never an arbitrary edge
    # index). Only handles axis-aligned faces (every shape this tool layer
    # builds is), which is what makes the edge itself computable at all:
    # for two perpendicular axis-aligned planes, the edge between them is
    # exactly the line where both planes' coordinates hold simultaneously,
    # running along the one remaining axis -- no general edge-finding
    # (arbitrary meshes, non-axis-aligned faces) is attempted. ----

    def _edge_geometry(self, mesh: trimesh.Trimesh, direction1: FaceDirection, direction2: FaceDirection):
        from mesh_quality import find_facet_by_direction

        face1 = find_facet_by_direction(mesh, _FACE_DIRECTION_VECTORS[direction1])
        face2 = find_facet_by_direction(mesh, _FACE_DIRECTION_VECTORS[direction2])
        if face1 is None or face2 is None:
            raise SandboxError(f"no flat face points {direction1!r} or {direction2!r} on this shape")
        if abs(float(np.dot(face1.normal, face2.normal))) > 0.1:
            raise SandboxError(f"{direction1!r} and {direction2!r} are not perpendicular faces -- no edge between them")

        axis1, axis2 = int(np.argmax(np.abs(face1.normal))), int(np.argmax(np.abs(face2.normal)))
        axis3 = 3 - axis1 - axis2
        sign1 = 1.0 if face1.normal[axis1] > 0 else -1.0
        sign2 = 1.0 if face2.normal[axis2] > 0 else -1.0
        coord1, coord2 = float(face1.origin[axis1]), float(face2.origin[axis2])
        bounds = mesh.bounds
        span3 = float(bounds[1][axis3] - bounds[0][axis3])
        mid3 = float((bounds[1][axis3] + bounds[0][axis3]) / 2.0)
        pad = span3 * 0.1 + 1.0
        # Returns BOTH the raw edge span and a padded version: fillet_edge/
        # chamfer_edge (SUBTRACTIVE cutters) want the padded one -- a cutter
        # overshooting past the real edge length is harmless, there's
        # nothing beyond the target's own material to wrongly remove.
        # add_rib (ADDITIVE) needs the raw, unpadded span instead -- a rib
        # built to the padded length would protrude past the part's actual
        # ends once unioned on, confirmed empirically (a real test case
        # added ~384mm^3 more than the intended rib volume, exactly
        # matching the padding's own contribution).
        return axis1, axis2, axis3, sign1, sign2, coord1, coord2, mid3, span3, span3 + 2 * pad

    def fillet_edge(self, target: int, direction1: FaceDirection, direction2: FaceDirection, radius: float) -> int:
        """Rounds the edge where the `direction1` and `direction2` faces
        meet -- Blender's bevel, with a circular (not flat) profile.
        Built from a corner box minus a quarter-cylinder tangent to both
        original faces (both already-verified primitives, no new
        triangulation math needed), then subtracted from the shape as one
        cutter -- so the internal add_box/add_cylinder/boolean_difference
        construction is never exposed as separate trace steps, matching how
        add_gear_profile's involute math stays a single macro call too."""
        from mesh_quality import remesh_flat_facets

        mesh = self._get(target)
        axis1, axis2, axis3, sign1, sign2, coord1, coord2, mid3, _span3, cutter_span3 = self._edge_geometry(mesh, direction1, direction2)

        box_extents = [0.0, 0.0, 0.0]
        box_extents[axis1], box_extents[axis2], box_extents[axis3] = radius, radius, cutter_span3
        box_center = [0.0, 0.0, 0.0]
        box_center[axis1], box_center[axis2], box_center[axis3] = coord1 - sign1 * radius / 2, coord2 - sign2 * radius / 2, mid3
        corner_box = trimesh.creation.box(extents=box_extents)
        corner_box.apply_translation(box_center)

        cyl = trimesh.creation.cylinder(radius=radius, height=cutter_span3, sections=32)
        cyl = _rotate_to_axis(cyl, {0: "x", 1: "y", 2: "z"}[axis3])
        cyl_center = [0.0, 0.0, 0.0]
        cyl_center[axis1], cyl_center[axis2], cyl_center[axis3] = coord1 - sign1 * radius, coord2 - sign2 * radius, mid3
        cyl.apply_translation(cyl_center)

        cutter = corner_box.difference(cyl, engine="manifold")
        result = remesh_flat_facets(mesh.difference(cutter, engine="manifold"))
        self._consume(target)
        h = self._new_handle(result)
        self._record("fillet_edge", {"target": target, "direction1": direction1, "direction2": direction2, "radius": radius})
        return h

    def _edge_wedge(self, mesh: trimesh.Trimesh, direction1: FaceDirection, direction2: FaceDirection, distance: float, into_material: bool = True):
        """The right-triangle wedge (legs of `distance`, extruded the full
        span of the edge where `direction1`/`direction2` meet) shared by
        chamfer_edge and add_rib -- same edge-locating math (_edge_geometry),
        opposite triangle orientation:
        - chamfer_edge (into_material=True, the default): legs point
          `coord - sign*distance`, i.e. TOWARD whichever side of each face
          already has material -- correct for a normal CONVEX corner, where
          the wedge must lie inside the existing solid to be subtracted.
        - add_rib (into_material=False): legs point `coord + sign*distance`,
          the MIRROR IMAGE through the corner point -- correct for a
          CONCAVE/internal corner (e.g. an L-bracket's inner bend), where
          the space being bridged is EMPTY, not solid. Confirmed empirically,
          not assumed: reusing chamfer's own wedge unmodified for add_rib
          landed almost entirely inside material already there (added
          ~384 of an expected ~1600 mm^3 on a real L-bracket test) --
          exactly the "already inside a convex corner's own material"
          shape, the wrong one for reinforcing a concave corner's void.
        """
        from shapely.geometry import Polygon

        axis1, axis2, axis3, sign1, sign2, coord1, coord2, mid3, span3, cutter_span3 = self._edge_geometry(mesh, direction1, direction2)
        leg_sign = 1.0 if into_material else -1.0
        # subtractive (chamfer): padded length is harmless to overshoot.
        # additive (rib): must stop exactly at the real edge ends, or the
        # union protrudes past the part's actual geometry -- see this
        # method's docstring.
        wedge_span = cutter_span3 if into_material else span3

        corner = (coord1, coord2)
        leg1 = (coord1 - leg_sign * sign1 * distance, coord2)
        leg2 = (coord1, coord2 - leg_sign * sign2 * distance)
        target_area = max(1.0, distance * distance / 20)
        wedge_2d = trimesh.creation.extrude_polygon(
            Polygon([corner, leg1, leg2]), height=wedge_span, engine="triangle", triangle_args=f"pq30a{target_area:.3f}"
        )
        wedge_vertices = np.zeros_like(wedge_2d.vertices)
        wedge_vertices[:, axis1] = wedge_2d.vertices[:, 0]
        wedge_vertices[:, axis2] = wedge_2d.vertices[:, 1]
        wedge_vertices[:, axis3] = wedge_2d.vertices[:, 2] + (mid3 - wedge_span / 2)
        wedge = trimesh.Trimesh(vertices=wedge_vertices, faces=wedge_2d.faces, process=True)
        if wedge.volume < 0:
            # (axis1, axis2, axis3) is sometimes an orientation-reversing
            # permutation of (x, y, z) -- e.g. z-then-y gives (2, 1, 0), a
            # mirror, not a rotation -- which flips the extruded polygon
            # inside-out even though its winding stays internally
            # consistent. Same defensive fix as add_revolve's.
            wedge.invert()
        return wedge, axis1, axis2, axis3

    def chamfer_edge(self, target: int, direction1: FaceDirection, direction2: FaceDirection, distance: float) -> int:
        """Flattens (rather than rounds) the edge where `direction1` and
        `direction2` meet -- Blender's bevel with a flat, not circular,
        profile. A wedge cutter (a right-triangle cross-section, legs of
        `distance`, extruded along the edge) removes the corner."""
        from mesh_quality import remesh_flat_facets

        mesh = self._get(target)
        wedge, *_ = self._edge_wedge(mesh, direction1, direction2, distance)

        result = remesh_flat_facets(mesh.difference(wedge, engine="manifold"))
        self._consume(target)
        h = self._new_handle(result)
        self._record("chamfer_edge", {"target": target, "direction1": direction1, "direction2": direction2, "distance": distance})
        return h

    def add_rib(self, target: int, direction1: FaceDirection, direction2: FaceDirection, size: float) -> int:
        """Adds a triangular gusset/support rib bridging the CONCAVE
        (internal) corner where `direction1` and `direction2` meet -- e.g.
        the inner bend of an L-bracket, or a shelled enclosure's inner
        wall-to-floor junction -- structural reinforcement across that
        empty corner, a full-length wedge (legs of `size`) along the whole
        edge. Reuses _edge_wedge's edge-locating math, but with the
        MIRRORED wedge orientation (into_material=False) -- see that
        method's docstring for why chamfer_edge's own wedge, reused
        unmodified, does not work here (it lies inside already-existing
        material, not the empty space being bridged)."""
        from mesh_quality import remesh_flat_facets

        mesh = self._get(target)
        wedge, *_ = self._edge_wedge(mesh, direction1, direction2, size, into_material=False)

        result = remesh_flat_facets(mesh.union(wedge, engine="manifold"))
        self._consume(target)
        h = self._new_handle(result)
        self._record("add_rib", {"target": target, "direction1": direction1, "direction2": direction2, "size": size})
        return h

    _BOX_VOLUME_FRACTION_TOLERANCE = 0.005  # 0.5% -- a box has no curved/faceted
    # faces, so unlike e.g. add_sweep's analytic-vs-faceted error, a genuine
    # axis-aligned box's meshed volume should match its bbox volume almost
    # exactly; this tolerance is for floating-point/remeshing noise only.

    def _is_axis_aligned_box(self, mesh: trimesh.Trimesh) -> bool:
        """A watertight solid's volume equals its own minimal axis-aligned
        bounding box's volume IF AND ONLY IF it occupies that entire
        bounding box with no empty space -- which, for a solid contained
        within its own minimal bbox by definition, is only possible when
        the solid IS that box (any corner cut, pocket, rounding, or
        non-box feature either reduces enclosed volume below the bbox
        volume or would have enlarged the bbox itself). This makes
        volume-vs-bbox-volume a sufficient, not just suggestive, test --
        not a heuristic stand-in for "probably box-shaped."""
        if not mesh.is_watertight or mesh.volume <= 0:
            return False
        length, width, height = mesh.extents
        bbox_volume = float(length * width * height)
        if bbox_volume <= 1e-9:
            return False
        return abs(mesh.volume - bbox_volume) / bbox_volume <= self._BOX_VOLUME_FRACTION_TOLERANCE

    def _minkowski_box(self, target: int, offset: float, corner_local_vertices: np.ndarray, op_name: str) -> int:
        """Shared machinery for round_box_edges/chamfer_box_edges: both are
        the Minkowski sum of the target's own bounding box with a small
        shape (a sphere for round, an octahedron for chamfer), which is
        exactly the convex hull of that small shape placed at each of the
        8 (shrunk-inward-by-offset) box corners -- a well-known exact
        construction, and one that sidesteps the tangent-face degenerate-
        boolean risk a literal 12-edge fillet/chamfer_edge chain would hit.

        Confirmed EXACT (not approximate) specifically because of the
        _is_axis_aligned_box precondition below: reading the target's own
        bounding box and reconstructing from it is only mathematically
        equivalent to "operating on the real geometry" when the target
        genuinely IS its own bounding box. Every other target is REJECTED
        outright (a typed SandboxError) rather than silently replaced with
        its bounding box's rounded/chamfered reconstruction -- a real,
        confirmed bug this session's own accuracy audit found (the earlier
        behavior discarded the actual shape on any non-box target with no
        warning), fixed here as a precondition rather than a disclosed
        approximation, since "reject unsupported targets before modifying
        geometry" is this project's own stated preference over silently
        returning a different shape."""
        mesh = self._get(target)
        if not self._is_axis_aligned_box(mesh):
            raise SandboxError(
                f"{op_name}: target {target} is not an axis-aligned box (volume {mesh.volume:.1f} vs. "
                f"bounding-box volume {float(np.prod(mesh.extents)):.1f}) -- this operation reconstructs "
                "from the bounding box, which is only exact for a genuine box-shaped target; refusing to "
                "silently replace a non-box shape with its bounding box"
            )
        length, width, height = mesh.extents
        hx, hy, hz = length / 2.0 - offset, width / 2.0 - offset, height / 2.0 - offset
        if hx <= 0 or hy <= 0 or hz <= 0:
            raise SandboxError(
                f"{op_name}: offset ({offset}) is too large for this shape's own "
                f"bounding box ({length:g} x {width:g} x {height:g})"
            )
        center = mesh.bounds.mean(axis=0)
        corner_signs = [(sx, sy, sz) for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]
        all_vertices = np.vstack([
            corner_local_vertices + np.array([sx * hx, sy * hy, sz * hz])
            for sx, sy, sz in corner_signs
        ])
        result = trimesh.Trimesh(vertices=all_vertices).convex_hull
        result.apply_translation(center)
        self._consume(target)
        h = self._new_handle(result)
        return h

    def round_box_edges(self, target: int, radius: float) -> int:
        """Rounds all 12 edges and 8 corners of a box-like target in one
        call -- Blender's Bevel modifier applied uniformly to a cube.
        See _minkowski_box for the construction and its box-shaped-target
        caveat."""
        sphere = trimesh.creation.icosphere(subdivisions=3, radius=radius)
        h = self._minkowski_box(target, radius, sphere.vertices, "round_box_edges")
        self._record("round_box_edges", {"target": target, "radius": radius})
        return h

    def chamfer_box_edges(self, target: int, distance: float) -> int:
        """Flat-chamfers all 12 edges and 8 corners of a box-like target in
        one call -- the uniform "cut corners" look, as opposed to
        round_box_edges' curved fillets. Minkowski sum with an octahedron
        (rather than a sphere) is the construction that produces flat
        chamfer facets instead of curved ones; see _minkowski_box."""
        octahedron_local = np.array([
            [distance, 0, 0], [-distance, 0, 0],
            [0, distance, 0], [0, -distance, 0],
            [0, 0, distance], [0, 0, -distance],
        ])
        h = self._minkowski_box(target, distance, octahedron_local, "chamfer_box_edges")
        self._record("chamfer_box_edges", {"target": target, "distance": distance})
        return h

    def mirror_merge(self, target: int, axis: Axis) -> int:
        """Mirrors the shape across a plane through the origin perpendicular
        to `axis` (the same reflection convention as `mirror`) and unions
        the mirrored copy onto the original in one call -- Blender's Mirror
        modifier with "Merge" enabled. Useful for building a symmetric
        half of a part and doubling it, rather than modeling both sides
        by hand."""
        from mesh_quality import remesh_flat_facets

        mesh = self._get(target)
        mirrored = mesh.copy()
        scale_vec = {"x": (-1.0, 1.0, 1.0), "y": (1.0, -1.0, 1.0), "z": (1.0, 1.0, -1.0)}[axis]
        mirrored.apply_transform(np.diag([*scale_vec, 1.0]))

        result = remesh_flat_facets(mesh.union(mirrored, engine="manifold"))
        self._consume(target)
        h = self._new_handle(result)
        self._record("mirror_merge", {"target": target, "axis": axis})
        return h

    def add_boss_with_bore(
        self, target: int, direction: FaceDirection, offset_u: float, offset_v: float,
        boss_radius: float, boss_height: float, bore_radius: float,
    ) -> int:
        """A raised cylindrical boss with a concentric through-hole -- the
        classic "screw boss" molded into an enclosure wall for a
        self-tapping screw. One macro call rather than an extrude-style
        boss union followed by a separate drill_hole, so the bore is
        guaranteed perfectly concentric with the boss (two independent
        steps risk the predictor picking slightly different offsets for
        each). Uses the same face-local-transform convention as
        counterbore_hole/drill_slot (local +z = into the material,
        local -z = outward, away from the surface)."""
        if bore_radius >= boss_radius:
            raise SandboxError(f"bore_radius ({bore_radius}) must be smaller than boss_radius ({boss_radius})")
        from mesh_quality import find_facet_by_direction, remesh_flat_facets

        mesh = self._get(target)
        face = find_facet_by_direction(mesh, _FACE_DIRECTION_VECTORS[direction])
        if face is None:
            raise SandboxError(f"no flat face points {direction!r} on this shape")
        center_u, center_v = self._face_center_uv(face)
        u, v = center_u + offset_u, center_v + offset_v
        transform = self._face_local_transform(face)

        boss_local = trimesh.creation.cylinder(radius=boss_radius, height=boss_height, sections=32)
        boss_local.apply_translation([u, v, -boss_height / 2.0])  # local -z = outward
        boss_local.apply_transform(transform)

        combined = remesh_flat_facets(mesh.union(boss_local, engine="manifold"))

        axis_idx = int(np.argmax(np.abs(face.normal)))
        target_extent = float(mesh.bounds[1][axis_idx] - mesh.bounds[0][axis_idx])
        pad = 2.0
        bore_length = boss_height + pad + target_extent + pad
        bore_local = trimesh.creation.cylinder(radius=bore_radius, height=bore_length, sections=32)
        # spans local z from -(boss_height + pad) (just outside the boss tip)
        # to bore_length - (boss_height + pad) (deep into the target)
        bore_local.apply_translation([u, v, bore_length / 2.0 - (boss_height + pad)])
        bore_local.apply_transform(transform)

        result = remesh_flat_facets(combined.difference(bore_local, engine="manifold"))
        self._consume(target)
        h = self._new_handle(result)
        self._record("add_boss_with_bore", {
            "target": target, "direction": direction, "offset_u": offset_u, "offset_v": offset_v,
            "boss_radius": boss_radius, "boss_height": boss_height, "bore_radius": bore_radius,
        })
        return h

    def add_extrusion(
        self, profile_kind: ProfileKind, param_a: float, param_b: float, param_c: float = 0.0,
        height: float = 1.0, center: tuple[float, float, float] = (0, 0, 0),
    ) -> int:
        """Flat param_a/b/c rather than a nested dict -- deliberately, so
        program_model.py's per-argument estimators (one flat regressor per
        named argument) don't need special-casing for a param whose meaning
        varies by an enum value. See _profile_polygon's own docstring for
        the full meaning-by-profile_kind table (shared with add_loft/
        add_sweep/add_tube, not re-specified here)."""
        from shapely.geometry import Polygon

        outline = _profile_polygon(profile_kind, param_a, param_b, param_c)
        footprint_area = float(Polygon(outline).area)

        target_triangle_area = max(4.0, footprint_area / 400)  # quality heuristic, never a predicted argument
        mesh = trimesh.creation.extrude_polygon(
            Polygon(outline), height=height, engine="triangle", triangle_args=f"pq30a{target_triangle_area:.1f}"
        )
        mesh.apply_translation([0, 0, -height / 2])
        if center != (0, 0, 0):
            mesh.apply_translation(center)
        h = self._new_handle(mesh)
        self._record("add_extrusion", {"profile_kind": profile_kind, "param_a": param_a, "param_b": param_b,
                                        "param_c": param_c, "height": height, "center": list(center)})
        return h

    def add_gear_profile(
        self, module_mm: float, tooth_count: int, face_width_mm: float, bore_diameter_mm: float,
        pressure_angle_deg: float = 20.0, center: tuple[float, float, float] = (0, 0, 0),
    ) -> int:
        from data.procedural_generator import GearParams, build_gear

        params = GearParams(
            module_mm=module_mm, tooth_count=tooth_count, face_width_mm=face_width_mm,
            bore_diameter_mm=bore_diameter_mm, pressure_angle_deg=pressure_angle_deg,
        )
        mesh = build_gear(params)  # involute math has no reuse value as decomposed steps -- see module docstring
        if center != (0, 0, 0):
            mesh.apply_translation(center)
        h = self._new_handle(mesh)
        self._record(
            "add_gear_profile",
            {"module_mm": module_mm, "tooth_count": tooth_count, "face_width_mm": face_width_mm,
             "bore_diameter_mm": bore_diameter_mm, "pressure_angle_deg": pressure_angle_deg, "center": list(center)},
        )
        return h

    # ---- finalize ----

    def finalize(self, target: int) -> trimesh.Trimesh:
        mesh = self._get(target)
        self._record("finalize", {"target": target})
        return mesh

    # ---- validation (zero geometry cost) ----

    @staticmethod
    def validate(program: list[ToolCall]) -> list[str]:
        """Dry-run referential well-formedness: every handle a step reads
        must have been created by an earlier step and not yet consumed by
        another. Runs no geometry at all -- pure bookkeeping, so it's safe
        to call on an untrusted/predicted program before ever touching CSG."""
        issues: list[str] = []
        alive: set[int] = set()
        next_id = 0

        for i, call in enumerate(program):
            if call.op in _CREATES:
                alive.add(next_id)
                next_id += 1
            elif call.op in _CONSUMES_ONE or call.op in _CONSUMES_TARGET:
                target = call.args.get("target")
                if target not in alive:
                    issues.append(f"step {i} ({call.op}): handle {target} is undefined or already consumed")
                else:
                    alive.discard(target)
                alive.add(next_id)
                next_id += 1
            elif call.op in _CONSUMES_TWO:
                a, b = call.args.get("a"), call.args.get("b")
                for h in (a, b):
                    if h not in alive:
                        issues.append(f"step {i} ({call.op}): handle {h} is undefined or already consumed")
                alive.discard(a)
                alive.discard(b)
                alive.add(next_id)
                next_id += 1
            elif call.op == "finalize":
                target = call.args.get("target")
                if target not in alive:
                    issues.append(f"step {i} (finalize): handle {target} is undefined or already consumed")
            else:
                issues.append(f"step {i}: unknown op {call.op!r}")
        return issues

    @staticmethod
    def execute(program: list[ToolCall]) -> "ProgramExecutionResult":
        """Transactional, with a real recoverable checkpoint -- not just in
        name. CONFIRMED, not assumed, that the previous version's recovery
        was largely non-functional: every Sandbox tool method returns an
        int HANDLE, not a trimesh.Trimesh, so the old code's
        `elif isinstance(result, trimesh.Trimesh): last_mesh = result`
        branch was dead for every op except `finalize` -- meaning a
        mid-program exception on any step BEFORE finalize (the normal
        case; finalize is always the last step) had NO real checkpoint to
        fall back to and degraded straight to status="rejected", mesh=None,
        discarding every successfully-built intermediate step's work. This
        version tracks the real last-good mesh after EVERY successful,
        immediately-validated step (via sb._shapes, not the return value's
        type), so a step N failure can recover step N-1's actual geometry,
        not nothing.

        Also validates EACH step's own output immediately (empty-mesh and
        watertightness checks) rather than only at the very end -- a
        corrupted intermediate mesh (e.g. a boolean op that produced a
        non-watertight result) is now caught at the step that caused it,
        with THAT step identified in the error, instead of silently
        flowing into further operations and only being noticed, with no
        way to tell which step was actually responsible, after the whole
        program finished."""
        from evaluation.geometry_eval import evaluate_mesh

        issues = Sandbox.validate(program)
        if issues:
            return ProgramExecutionResult(status="rejected", mesh=None, executed_steps=0, failing_step=0, error="; ".join(issues), geometry_report=None)

        sb = Sandbox(trace=False)
        last_mesh: trimesh.Trimesh | None = None  # set only by an explicit `finalize` call -- a program
                                                    # that never finalizes has not "completed," even if every
                                                    # individual step succeeded (unchanged from before)
        last_good_mesh: trimesh.Trimesh | None = None  # the real recoverable checkpoint -- the most recent
                                                          # successfully-built AND immediately-validated mesh,
                                                          # from ANY step, not just finalize

        def _mesh_for_result(result):
            if isinstance(result, trimesh.Trimesh):
                return result
            if isinstance(result, int) and result in sb._shapes:
                return sb._shapes[result]
            return None

        for i, call in enumerate(program):
            try:
                method = getattr(sb, call.op)
                result = method(**call.args)
                if call.op == "finalize":
                    last_mesh = result
                current = _mesh_for_result(result)
                if current is not None:
                    if len(current.vertices) == 0:
                        raise SandboxError(f"{call.op} (step {i}): produced an empty mesh")
                    if not current.is_watertight:
                        raise SandboxError(f"{call.op} (step {i}): result is not watertight immediately after this operation")
                    last_good_mesh = current
            except Exception as e:
                if last_good_mesh is not None:
                    return ProgramExecutionResult(status="partial", mesh=last_good_mesh, executed_steps=i, failing_step=i, error=str(e), geometry_report=None)
                return ProgramExecutionResult(status="rejected", mesh=None, executed_steps=i, failing_step=i, error=str(e), geometry_report=None)

        if last_mesh is None:
            return ProgramExecutionResult(status="rejected", mesh=None, executed_steps=len(program), failing_step=None, error="program produced no mesh", geometry_report=None)

        if len(last_mesh.vertices) == 0:
            # a boolean op can fully consume its target (e.g. a hole far
            # larger than the part) and leave a technically-valid but empty
            # mesh -- evaluate_mesh() assumes non-empty bounds, so this must
            # be caught before calling it, not treated as a crash
            return ProgramExecutionResult(status="suspect", mesh=last_mesh, executed_steps=len(program), failing_step=None,
                                           error="program executed but produced an empty mesh", geometry_report=None)

        report = evaluate_mesh(last_mesh)
        if not report.watertight or last_mesh.volume <= 0:
            return ProgramExecutionResult(status="suspect", mesh=last_mesh, executed_steps=len(program), failing_step=None, error=None, geometry_report=report)
        return ProgramExecutionResult(status="ok", mesh=last_mesh, executed_steps=len(program), failing_step=None, error=None, geometry_report=report)


@dataclass
class ProgramExecutionResult:
    status: Literal["ok", "rejected", "partial", "suspect"]
    mesh: "trimesh.Trimesh | None"
    executed_steps: int
    failing_step: int | None
    error: str | None
    geometry_report: Any = None  # evaluation.geometry_eval.GeometryReport | None


def _arg_spec_for_param(name: str, param: inspect.Parameter, annotation: Any) -> dict:
    """One parameter's shape, for a UI to render an input control without
    hardcoding anything about the tool roster -- reflected straight off the
    real method signature, so a new tool's manual-console form appears
    automatically the moment it's added to TOOL_OPS, with no second,
    hand-maintained schema to fall out of sync."""
    spec: dict[str, Any] = {"name": name, "required": param.default is inspect.Parameter.empty}
    if param.default is not inspect.Parameter.empty:
        default = param.default
        spec["default"] = list(default) if isinstance(default, tuple) else default

    origin = typing.get_origin(annotation)
    if origin is Literal:
        spec["type"] = "enum"
        spec["choices"] = list(typing.get_args(annotation))
    elif annotation is int:
        spec["type"] = "integer"
    elif annotation is float:
        spec["type"] = "number"
    elif origin is tuple:
        spec["type"] = "vec3"
    else:
        spec["type"] = "number"  # conservative fallback -- every non-enum, non-int, non-vec3 arg in this file is a float
    return spec


def tool_schema() -> list[dict]:
    """Reflects Sandbox's real tool methods into a JSON-ready description
    (op name, handle-flow category, and each argument's name/type/default/
    enum-choices) -- the single source of truth the manual tool console's
    dynamic form-builder reads, and the LLM's decomposition prompt vocabulary
    is written to stay consistent with by hand. Built off the SAME method
    signatures Sandbox.execute() actually calls with **call.args, so this
    can never describe an argument the real tool doesn't accept."""
    schema = []
    for op in TOOL_OPS:
        method = getattr(Sandbox, op)
        hints = typing.get_type_hints(method)
        sig = inspect.signature(method)
        category = (
            "create" if op in _CREATES
            else "consumes_two" if op in _CONSUMES_TWO
            else "consumes_one" if op in _CONSUMES_ONE
            else "consumes_target"  # every TOOL_OPS entry not in one of the other three sets is in _CONSUMES_TARGET
        )
        # target/a/b are references to an earlier step's OUTPUT handle, not
        # a plain integer a person types in -- overriding their type to
        # "handle" here (rather than leaving them "integer") is what lets
        # the console render them as a dropdown of currently-alive handles
        # instead of a free-typed number a typo could point at nothing.
        handle_arg_names = {"target"} if category != "consumes_two" else {"a", "b"}
        args = []
        for name, param in sig.parameters.items():
            if name == "self":
                continue
            spec = _arg_spec_for_param(name, param, hints.get(name))
            if name in handle_arg_names:
                spec["type"] = "handle"
            args.append(spec)
        schema.append({"op": op, "category": category, "args": args})
    return schema
