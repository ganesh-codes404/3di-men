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
ProfileKind = Literal["rect", "rounded_rect", "l_shape"]
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
_CREATES = {"add_box", "add_cylinder", "add_extrusion", "add_gear_profile", "add_revolve", "add_shouldered_cylinder"}
_CONSUMES_ONE = {"translate", "rotate", "mirror", "scale", "shell", "taper"}
_CONSUMES_TWO = {"boolean_union", "boolean_difference", "boolean_intersection"}
_CONSUMES_TARGET = {
    "drill_hole", "drill_hole_pattern", "linear_pattern", "circular_pattern",
    "extrude_face", "inset_pocket", "fillet_edge", "chamfer_edge",
    "counterbore_hole", "drill_slot", "add_rib",
}
# Every tool a person can invoke directly from the manual tool console, in
# the order shown there -- deliberately excludes "finalize" (not a modeling
# op, just "pick a handle to preview/export," handled separately by the
# console itself) and the private/bookkeeping methods.
TOOL_OPS: list[str] = [
    "add_box", "add_cylinder", "add_revolve", "add_shouldered_cylinder", "add_extrusion", "add_gear_profile",
    "taper", "translate", "rotate", "mirror", "scale",
    "boolean_union", "boolean_difference", "boolean_intersection",
    "drill_hole", "drill_hole_pattern", "linear_pattern", "circular_pattern",
    "extrude_face", "inset_pocket", "shell", "fillet_edge", "chamfer_edge",
    "counterbore_hole", "drill_slot", "add_rib",
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
        shrink proportionally, not stay a constant thickness) -- acceptable
        for what this tool is actually used for (hollowing out an
        enclosure-like solid), not a general-purpose BREP shell."""
        from mesh_quality import remesh_flat_facets

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
        self._consume(target)
        h = self._new_handle(result)
        self._record("shell", {"target": target, "wall_thickness": wall_thickness})
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

    def add_extrusion(
        self, profile_kind: ProfileKind, param_a: float, param_b: float, param_c: float = 0.0,
        height: float = 1.0, center: tuple[float, float, float] = (0, 0, 0),
    ) -> int:
        """Flat param_a/b/c rather than a nested dict -- deliberately, so
        program_model.py's per-argument estimators (one flat regressor per
        named argument) don't need special-casing for a param whose meaning
        varies by an enum value. Meaning by profile_kind:
          rect / rounded_rect: param_a=length, param_b=width, param_c=fillet_radius (0 for rect)
          l_shape:             param_a=leg1,   param_b=leg2,  param_c=thickness
        """
        from shapely.geometry import Polygon

        if profile_kind == "rect":
            outline = _rounded_rect_polygon(param_a, param_b, 0.0)
            footprint_area = param_a * param_b
        elif profile_kind == "rounded_rect":
            outline = _rounded_rect_polygon(param_a, param_b, param_c)
            footprint_area = param_a * param_b
        elif profile_kind == "l_shape":
            outline = _l_shape_polygon(param_a, param_b, param_c)
            footprint_area = param_a * param_b
        else:
            raise SandboxError(f"unknown profile_kind {profile_kind!r}")

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
        from evaluation.geometry_eval import evaluate_mesh

        issues = Sandbox.validate(program)
        if issues:
            return ProgramExecutionResult(status="rejected", mesh=None, executed_steps=0, failing_step=0, error="; ".join(issues), geometry_report=None)

        sb = Sandbox(trace=False)
        last_mesh: trimesh.Trimesh | None = None
        for i, call in enumerate(program):
            try:
                method = getattr(sb, call.op)
                result = method(**call.args)
                if call.op == "finalize":
                    last_mesh = result
                elif isinstance(result, trimesh.Trimesh):
                    last_mesh = result
            except Exception as e:
                if last_mesh is not None:
                    return ProgramExecutionResult(status="partial", mesh=last_mesh, executed_steps=i, failing_step=i, error=str(e), geometry_report=None)
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
