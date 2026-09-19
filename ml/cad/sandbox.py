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

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import trimesh

Axis = Literal["x", "y", "z"]
PatternKind = Literal["single", "bolt_circle", "linear_array", "corner_array"]
ProfileKind = Literal["rect", "rounded_rect", "l_shape"]

_AXIS_VECTORS: dict[Axis, list[int]] = {"x": [1, 0, 0], "y": [0, 1, 0], "z": [0, 0, 1]}


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

        _CREATES = {"add_box", "add_cylinder", "add_extrusion", "add_gear_profile"}
        _CONSUMES_ONE = {"translate", "rotate"}
        _CONSUMES_TWO = {"boolean_union", "boolean_difference"}
        _CONSUMES_TARGET = {"drill_hole", "drill_hole_pattern"}

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
