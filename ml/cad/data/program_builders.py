"""Per-family Sandbox program constructors -- the tool-composition-path
counterpart to procedural_generator.py's build_*(params) functions.

Deliberately a SEPARATE, parallel implementation rather than adding a
`sandbox=` parameter to the existing build_* functions and branching inside
them. Reasoning: the existing functions are the primary, deeply-tested path
(65 passing tests depend on their exact behavior); intertwining a second
code path inside them risks a regression in code that currently has zero
risk of one. Instead, each function here independently constructs the
equivalent Sandbox program from the same params dataclass, and
test_procedural_generator_sandbox_regression.py verifies the two paths
produce equivalent (watertight, comparable volume/bounds) geometry across
many sampled params -- "equivalent," not byte-identical, since Sandbox's
tools compute their own CSG-robustness padding and mesh-quality heuristics
internally rather than replicating each builder's specific fudge factors
(see sandbox.py's module docstring).

Each function returns the FINAL HANDLE (not the mesh) so callers can choose
to sandbox.finalize() it themselves and/or read sandbox.get_trace() for the
full recorded program.
"""

from __future__ import annotations

from data.procedural_generator import BracketParams, EnclosureParams, FlangeParams, GearParams, MountingPlateParams, ShaftParams
from sandbox import Sandbox


def build_plate_program(params: MountingPlateParams, sandbox: Sandbox) -> int:
    plate = sandbox.add_extrusion(
        "rounded_rect", param_a=params.length_mm, param_b=params.width_mm, param_c=params.fillet_radius_mm,
        height=params.thickness_mm,
    )
    pattern = "corner_array" if params.hole_count == 4 else "linear_array"
    plate = sandbox.drill_hole_pattern(
        plate, pattern=pattern, count=params.hole_count, radius=params.hole_diameter_mm / 2,
        depth=params.thickness_mm, margin=params.hole_edge_margin_mm,
    )
    return plate


def build_flange_program(params: FlangeParams, sandbox: Sandbox) -> int:
    disk = sandbox.add_cylinder(radius=params.outer_diameter_mm / 2, height=params.thickness_mm)
    disk = sandbox.drill_hole(disk, radius=params.bore_diameter_mm / 2, depth=params.thickness_mm, center=(0, 0, 0))
    disk = sandbox.drill_hole_pattern(
        disk, pattern="bolt_circle", count=params.bolt_hole_count, radius=params.bolt_hole_diameter_mm / 2,
        depth=params.thickness_mm, circle_diameter=params.bolt_circle_diameter_mm,
    )
    return disk


def build_bracket_program(params: BracketParams, sandbox: Sandbox) -> int:
    bracket = sandbox.add_extrusion(
        "l_shape", param_a=params.leg1_length_mm, param_b=params.leg2_length_mm, param_c=params.thickness_mm,
        height=params.length_mm,
    )
    # arm holes vary along Z (the extrusion/length axis) at a fixed offset on
    # the OTHER axis -- drill_hole_pattern's linear_array only varies X at a
    # fixed Y, which doesn't fit either arm, so these use plain drill_hole in
    # a loop instead (holes_per_arm is capped at 1-2, so this stays well
    # within the K=10 step budget without needing to generalize the pattern
    # tool for one family's use)
    r = params.hole_diameter_mm / 2
    t = params.thickness_mm
    n = params.holes_per_arm
    margin = params.hole_edge_margin_mm
    usable = max(params.length_mm - 2 * margin, 0.0)
    if n <= 1:
        z_positions = [0.0]
    else:
        step = usable / (n - 1)
        z_positions = [-usable / 2 + i * step for i in range(n)]

    for z in z_positions:
        bracket = sandbox.drill_hole(bracket, radius=r, depth=t, center=(params.leg1_length_mm / 2, t / 2, z), axis="y")
    for z in z_positions:
        bracket = sandbox.drill_hole(bracket, radius=r, depth=t, center=(t / 2, params.leg2_length_mm / 2, z), axis="x")
    return bracket


def build_shaft_program(params: ShaftParams, sandbox: Sandbox) -> int:
    shaft = sandbox.add_cylinder(radius=params.outer_diameter_mm / 2, height=params.length_mm)
    if params.inner_diameter_mm > 0:
        shaft = sandbox.drill_hole(shaft, radius=params.inner_diameter_mm / 2, depth=params.length_mm, center=(0, 0, 0))
    if params.keyway_width_mm > 0:
        keyway_length = params.keyway_length_mm or params.length_mm
        od = params.outer_diameter_mm
        keyway = sandbox.add_box(
            params.keyway_depth_mm + 1.0, params.keyway_width_mm, keyway_length,
            center=(od / 2 - params.keyway_depth_mm / 2 + 0.5, 0, 0),
        )
        shaft = sandbox.boolean_difference(shaft, keyway)
    return shaft


def build_enclosure_program(params: EnclosureParams, sandbox: Sandbox) -> int:
    L, W, H, wt = params.length_mm, params.width_mm, params.height_mm, params.wall_thickness_mm
    outer = sandbox.add_box(L, W, H)

    cavity_top = H / 2 + 1.0 if params.open_face == "top" else H / 2 - wt
    cavity_bottom = -H / 2 + wt
    cavity_height = cavity_top - cavity_bottom
    cavity_center_z = (cavity_top + cavity_bottom) / 2
    cavity = sandbox.add_box(L - 2 * wt, W - 2 * wt, cavity_height, center=(0, 0, cavity_center_z))
    shell = sandbox.boolean_difference(outer, cavity)

    # vent_hole_count and vent_hole_diameter_mm are independently omittable
    # by prompt_dataset.py's field-omission logic -- "6 holes of 0mm
    # diameter" is a real, reachable combination (a zero-radius drill cutter
    # isn't a valid volume), not a hypothetical, so both must be checked
    if params.vent_hole_count > 0 and params.vent_hole_diameter_mm > 0 and params.open_face != "top":
        r = params.vent_hole_diameter_mm / 2
        shell = sandbox.drill_hole_pattern(
            shell, pattern="linear_array", count=params.vent_hole_count, radius=r,
            depth=wt, axis="z", margin=r * 1.5, z=H / 2 - wt / 2,
        )
    return shell


def build_gear_program(params: GearParams, sandbox: Sandbox) -> int:
    return sandbox.add_gear_profile(
        module_mm=params.module_mm, tooth_count=params.tooth_count, face_width_mm=params.face_width_mm,
        bore_diameter_mm=params.bore_diameter_mm, pressure_angle_deg=params.pressure_angle_deg,
    )


PROGRAM_BUILDERS = {
    "plate": build_plate_program,
    "flange": build_flange_program,
    "bracket": build_bracket_program,
    "shaft": build_shaft_program,
    "enclosure": build_enclosure_program,
    "gear": build_gear_program,
}


def traced_build(family: str, params) -> tuple["trimesh.Trimesh", list]:
    """Runs the family's program builder with tracing on, returning
    (mesh, program) -- the (text, program) label pairing point for
    data/program_dataset.py."""
    sandbox = Sandbox(trace=True)
    handle = PROGRAM_BUILDERS[family](params, sandbox)
    mesh = sandbox.finalize(handle)
    return mesh, sandbox.get_trace()
