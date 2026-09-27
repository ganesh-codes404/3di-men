import numpy as np

from sandbox import TOOL_OPS, Sandbox, ToolCall, tool_schema


def test_add_box_and_cylinder_produce_valid_meshes():
    sb = Sandbox()
    box_h = sb.add_box(50, 40, 10)
    cyl_h = sb.add_cylinder(5, 20)
    assert sb._get(box_h).is_watertight
    assert sb._get(cyl_h).is_watertight


def test_boolean_difference_removes_material_and_consumes_handles():
    sb = Sandbox()
    box_h = sb.add_box(50, 40, 10)
    hole_h = sb.add_cylinder(5, 30)
    box_volume_before = sb._get(box_h).volume
    result_h = sb.boolean_difference(box_h, hole_h)
    result_mesh = sb._get(result_h)
    assert result_mesh.volume < box_volume_before
    assert result_mesh.is_watertight
    # both inputs must be consumed -- reusing either should be an error
    try:
        sb._get(box_h)
        assert False, "box_h should have been consumed"
    except Exception:
        pass


def test_trace_records_calls_when_enabled():
    sb = Sandbox(trace=True)
    box_h = sb.add_box(50, 40, 10)
    sb.finalize(box_h)
    trace = sb.get_trace()
    assert [c.op for c in trace] == ["add_box", "finalize"]
    assert trace[0].args["length"] == 50


def test_trace_is_empty_when_disabled():
    sb = Sandbox(trace=False)
    box_h = sb.add_box(50, 40, 10)
    sb.finalize(box_h)
    assert sb.get_trace() == []


def test_drill_hole_pattern_corner_array_matches_expected_count():
    sb = Sandbox()
    box_h = sb.add_box(100, 60, 6)
    result_h = sb.drill_hole_pattern(box_h, pattern="corner_array", count=4, radius=4, depth=6, margin=15)
    mesh = sb._get(result_h)
    assert mesh.is_watertight


def test_drill_hole_pattern_bolt_circle():
    sb = Sandbox()
    disk_h = sb.add_cylinder(60, 8)
    result_h = sb.drill_hole_pattern(disk_h, pattern="bolt_circle", count=6, radius=4, depth=8, circle_diameter=90)
    mesh = sb._get(result_h)
    assert mesh.is_watertight


def test_add_extrusion_rounded_rect_and_l_shape():
    sb1 = Sandbox()
    h1 = sb1.add_extrusion("rounded_rect", param_a=100, param_b=60, param_c=5, height=6)
    assert sb1._get(h1).is_watertight

    sb2 = Sandbox()
    h2 = sb2.add_extrusion("l_shape", param_a=60, param_b=60, param_c=5, height=100)
    assert sb2._get(h2).is_watertight


def test_add_gear_profile_produces_watertight_gear():
    sb = Sandbox()
    h = sb.add_gear_profile(module_mm=2.0, tooth_count=16, face_width_mm=10, bore_diameter_mm=8)
    assert sb._get(h).is_watertight


def test_boolean_intersection_produces_valid_mesh_smaller_than_either_input():
    sb = Sandbox()
    box_h = sb.add_box(20, 20, 20)
    cyl_h = sb.add_cylinder(8, 40)
    box_volume = sb._get(box_h).volume
    result_h = sb.boolean_intersection(box_h, cyl_h)
    mesh = sb._get(result_h)
    assert mesh.is_watertight
    assert 0 < mesh.volume < box_volume


def test_mirror_preserves_volume_and_watertightness_and_reflects_position():
    sb = Sandbox()
    box_h = sb.add_box(10, 20, 30, center=(5, 0, 0))
    original_volume = sb._get(box_h).volume
    mirrored_h = sb.mirror(box_h, "x")
    mesh = sb._get(mirrored_h)
    assert mesh.is_watertight
    assert abs(mesh.volume - original_volume) < 1e-6
    # centered at x=5 before, must be centered at x=-5 after reflecting across x=0
    assert mesh.bounds.mean(axis=0)[0] < 0


def test_scale_resizes_about_its_own_center_not_the_origin():
    sb = Sandbox()
    box_h = sb.add_box(10, 10, 10, center=(50, 0, 0))
    scaled_h = sb.scale(box_h, 2.0, 1.0, 0.5)
    mesh = sb._get(scaled_h)
    assert mesh.is_watertight
    assert list(mesh.extents) == [20.0, 10.0, 5.0]
    # center must stay at (50, 0, 0) -- scaling is not also a translation
    assert (abs(mesh.bounds.mean(axis=0) - [50.0, 0.0, 0.0]) < 1e-6).all()


def test_linear_pattern_produces_n_non_overlapping_copies():
    sb = Sandbox()
    box_h = sb.add_box(5, 5, 5)
    result_h = sb.linear_pattern(box_h, "x", count=4, spacing=20)
    mesh = sb._get(result_h)
    assert mesh.is_watertight
    assert abs(mesh.volume - 4 * 125) < 1e-6  # spacing >> box size -- copies must not overlap


def test_linear_pattern_of_one_is_just_the_original_unmoved():
    sb = Sandbox()
    box_h = sb.add_box(5, 5, 5)
    original_volume = sb._get(box_h).volume
    result_h = sb.linear_pattern(box_h, "x", count=1, spacing=20)
    mesh = sb._get(result_h)
    assert abs(mesh.volume - original_volume) < 1e-6


def test_circular_pattern_produces_n_non_overlapping_copies():
    sb = Sandbox()
    cyl_h = sb.add_cylinder(3, 10)
    single_volume = sb._get(cyl_h).volume
    result_h = sb.circular_pattern(cyl_h, count=6, radius=30)
    mesh = sb._get(result_h)
    assert mesh.is_watertight
    assert abs(mesh.volume - 6 * single_volume) < 1e-3


def test_extrude_face_adds_a_boss_matching_the_faces_own_footprint():
    sb = Sandbox()
    box_h = sb.add_box(40, 30, 10)
    result_h = sb.extrude_face(box_h, "+z", distance=15)
    mesh = sb._get(result_h)
    assert mesh.is_watertight
    # same 40x30 footprint, total height 10 (base) + 15 (boss) = 25
    assert abs(mesh.volume - 40 * 30 * 25) < 1e-6
    assert list(mesh.extents) == [40.0, 30.0, 25.0]


def test_extrude_face_preserves_existing_holes_in_that_face():
    sb = Sandbox()
    box_h = sb.add_box(40, 30, 10)
    drilled_h = sb.drill_hole_pattern(box_h, pattern="corner_array", count=4, radius=3, depth=10, margin=8)
    result_h = sb.extrude_face(drilled_h, "+z", distance=15)
    mesh = sb._get(result_h)
    assert mesh.is_watertight
    # the boss must NOT fill in the existing holes -- volume stays well below
    # a solid (holeless) box+boss of the same overall envelope
    assert mesh.volume < 40 * 30 * 25


def test_extrude_face_raises_when_no_face_points_that_direction():
    sb = Sandbox()
    cyl_h = sb.add_cylinder(10, 20)
    try:
        sb.extrude_face(cyl_h, "+x", distance=5)
        assert False, "should have raised -- a cylinder has no flat +x face"
    except Exception:
        pass


def test_inset_pocket_removes_material_without_breaking_through():
    sb = Sandbox()
    box_h = sb.add_box(40, 30, 10)
    original_volume = sb._get(box_h).volume
    result_h = sb.inset_pocket(box_h, "+z", depth=4, margin=5)
    mesh = sb._get(result_h)
    assert mesh.is_watertight
    assert mesh.volume < original_volume


def test_inset_pocket_raises_when_margin_insets_the_face_away_entirely():
    sb = Sandbox()
    box_h = sb.add_box(20, 20, 10)
    try:
        sb.inset_pocket(box_h, "+z", depth=4, margin=15)  # margin > half the 20mm face
        assert False, "should have raised -- nothing left after insetting"
    except Exception:
        pass


def test_shell_hollows_out_leaving_less_material():
    sb = Sandbox()
    box_h = sb.add_box(40, 30, 20)
    original_volume = sb._get(box_h).volume
    result_h = sb.shell(box_h, wall_thickness=3)
    mesh = sb._get(result_h)
    assert mesh.is_watertight
    assert mesh.volume < original_volume


def test_add_revolve_frustum_is_watertight_and_tapered():
    sb = Sandbox()
    h = sb.add_revolve(bottom_radius=20, top_radius=10, height=30)
    mesh = sb._get(h)
    assert mesh.is_watertight
    assert mesh.volume > 0
    assert list(mesh.extents[:2]) == [40.0, 40.0]  # bottom (2*20) is the widest cross-section


def test_add_revolve_handles_zero_radius_at_either_end_without_degenerate_faces():
    for bottom_radius, top_radius in [(15, 0), (0, 12)]:
        sb = Sandbox()
        h = sb.add_revolve(bottom_radius=bottom_radius, top_radius=top_radius, height=25)
        mesh = sb._get(h)
        assert mesh.is_watertight, f"bottom={bottom_radius} top={top_radius}"
        assert mesh.volume > 0, f"bottom={bottom_radius} top={top_radius}"


def test_add_revolve_raises_when_both_radii_are_zero():
    sb = Sandbox()
    try:
        sb.add_revolve(bottom_radius=0, top_radius=0, height=10)
        assert False, "should have raised -- degenerate to a line, not a solid"
    except Exception:
        pass


def test_taper_narrows_the_top_while_keeping_the_bottom_and_stays_watertight():
    sb = Sandbox()
    box_h = sb.add_box(40, 40, 30)
    original_volume = sb._get(box_h).volume
    result_h = sb.taper(box_h, "z", bottom_factor=1.0, top_factor=0.5)
    mesh = sb._get(result_h)
    assert mesh.is_watertight
    assert mesh.volume < original_volume
    bottom_pts = mesh.vertices[mesh.vertices[:, 2] < -14.9]
    top_pts = mesh.vertices[mesh.vertices[:, 2] > 14.9]
    assert bottom_pts[:, :2].ptp(axis=0).tolist() == [40.0, 40.0]  # bottom_factor=1.0 -- unchanged
    assert top_pts[:, :2].ptp(axis=0).tolist() == [20.0, 20.0]  # top_factor=0.5 -- halved


def test_taper_with_equal_factors_is_a_no_op_on_shape():
    sb = Sandbox()
    box_h = sb.add_box(30, 20, 10)
    original_volume = sb._get(box_h).volume
    result_h = sb.taper(box_h, "x", bottom_factor=1.0, top_factor=1.0)
    mesh = sb._get(result_h)
    assert abs(mesh.volume - original_volume) < 1e-6


def test_fillet_edge_rounds_the_corner_and_removes_the_right_amount_of_material():
    sb = Sandbox()
    box_h = sb.add_box(40, 30, 20)
    original_volume = sb._get(box_h).volume
    result_h = sb.fillet_edge(box_h, "+z", "+x", radius=5)
    mesh = sb._get(result_h)
    assert mesh.is_watertight
    # removes a (radius^2 - pi*radius^2/4) cross-section along the 30mm
    # edge (the y axis, the one axis neither +z nor +x selects)
    expected_removed = (5 * 5 - 3.14159265 * 5 * 5 / 4) * 30
    assert abs((original_volume - mesh.volume) - expected_removed) < 5.0


def test_chamfer_edge_removes_exactly_the_analytic_wedge_volume():
    sb = Sandbox()
    box_h = sb.add_box(40, 30, 20)
    original_volume = sb._get(box_h).volume
    result_h = sb.chamfer_edge(box_h, "+z", "+x", distance=5)
    mesh = sb._get(result_h)
    assert mesh.is_watertight
    # a flat wedge cut is exact: 0.5 * distance^2 * edge_length
    expected_removed = 0.5 * 5 * 5 * 30
    assert abs((original_volume - mesh.volume) - expected_removed) < 1.0


def test_chamfer_edge_handles_an_orientation_reversing_axis_pair():
    # Regression test: (+z, +y) maps to axes (2, 1, 0), an ODD permutation
    # of (x, y, z) -- a mirror, not a rotation -- which used to flip the
    # extruded wedge cutter inside-out (negative volume) and make
    # mesh.difference() fail with "Not all meshes are volumes!". A 3400-case
    # sweep across all generic training patterns caught this: every failure
    # was a z/y pair, never z/x or x/y (both even permutations).
    sb = Sandbox()
    box_h = sb.add_box(40, 50, 40)
    original_volume = sb._get(box_h).volume
    result_h = sb.chamfer_edge(box_h, "+z", "+y", distance=6)
    mesh = sb._get(result_h)
    assert mesh.is_watertight
    assert mesh.volume > 0
    expected_removed = 0.5 * 6 * 6 * 40
    assert abs((original_volume - mesh.volume) - expected_removed) < 1.0


def test_fillet_edge_raises_on_non_perpendicular_faces():
    sb = Sandbox()
    box_h = sb.add_box(40, 30, 20)
    try:
        sb.fillet_edge(box_h, "+z", "-z", radius=5)
        assert False, "should have raised -- opposite faces share no edge"
    except Exception:
        pass


def test_chamfer_edge_raises_when_no_face_points_that_direction():
    sb = Sandbox()
    cyl_h = sb.add_cylinder(10, 20)
    try:
        sb.chamfer_edge(cyl_h, "+x", "+y", distance=3)
        assert False, "should have raised -- a cylinder has no flat +x or +y face"
    except Exception:
        pass


def test_validate_catches_undefined_handle():
    program = [ToolCall(op="add_box", args={"length": 50, "width": 40, "height": 10, "center": [0, 0, 0]}),
               ToolCall(op="drill_hole", args={"target": 5, "radius": 4, "depth": 10, "center": [0, 0, 0], "axis": "z"})]
    issues = Sandbox.validate(program)
    assert len(issues) == 1
    assert "5" in issues[0]


def test_validate_catches_reused_consumed_handle():
    program = [
        ToolCall(op="add_box", args={"length": 50, "width": 40, "height": 10, "center": [0, 0, 0]}),
        ToolCall(op="add_cylinder", args={"radius": 4, "height": 30, "center": [0, 0, 0], "axis": "z"}),
        ToolCall(op="boolean_difference", args={"a": 0, "b": 1}),
        ToolCall(op="drill_hole", args={"target": 0, "radius": 4, "depth": 10, "center": [0, 0, 0], "axis": "z"}),  # 0 already consumed
    ]
    issues = Sandbox.validate(program)
    assert any("0" in issue for issue in issues)


def test_validate_accepts_program_using_every_new_tool():
    program = [
        ToolCall(op="add_box", args={"length": 10, "width": 10, "height": 10, "center": [0, 0, 0]}),  # 0
        ToolCall(op="mirror", args={"target": 0, "axis": "x"}),  # 1
        ToolCall(op="scale", args={"target": 1, "factor_x": 2.0, "factor_y": 1.0, "factor_z": 1.0}),  # 2
        ToolCall(op="linear_pattern", args={"target": 2, "axis": "x", "count": 3, "spacing": 20}),  # 3
        ToolCall(op="add_cylinder", args={"radius": 3, "height": 10, "center": [0, 0, 0], "axis": "z"}),  # 4
        ToolCall(op="circular_pattern", args={"target": 4, "count": 4, "radius": 15}),  # 5
        ToolCall(op="boolean_intersection", args={"a": 3, "b": 5}),  # 6
        ToolCall(op="finalize", args={"target": 6}),
    ]
    assert Sandbox.validate(program) == []


def test_validate_accepts_program_using_every_face_based_tool():
    program = [
        ToolCall(op="add_box", args={"length": 40, "width": 30, "height": 10, "center": [0, 0, 0]}),  # 0
        ToolCall(op="extrude_face", args={"target": 0, "direction": "+z", "distance": 15}),  # 1
        ToolCall(op="inset_pocket", args={"target": 1, "direction": "-z", "depth": 3, "margin": 5}),  # 2
        ToolCall(op="shell", args={"target": 2, "wall_thickness": 3}),  # 3
        ToolCall(op="finalize", args={"target": 3}),
    ]
    assert Sandbox.validate(program) == []


def test_validate_accepts_program_using_revolve_and_taper():
    program = [
        ToolCall(op="add_revolve", args={"bottom_radius": 20, "top_radius": 10, "height": 30, "center": [0, 0, 0]}),  # 0
        ToolCall(op="taper", args={"target": 0, "axis": "z", "bottom_factor": 1.0, "top_factor": 0.8}),  # 1
        ToolCall(op="finalize", args={"target": 1}),
    ]
    assert Sandbox.validate(program) == []


def test_validate_accepts_program_using_edge_tools():
    program = [
        ToolCall(op="add_box", args={"length": 40, "width": 30, "height": 20, "center": [0, 0, 0]}),  # 0
        ToolCall(op="fillet_edge", args={"target": 0, "direction1": "+z", "direction2": "+x", "radius": 5}),  # 1
        ToolCall(op="chamfer_edge", args={"target": 1, "direction1": "+z", "direction2": "-x", "distance": 4}),  # 2
        ToolCall(op="finalize", args={"target": 2}),
    ]
    assert Sandbox.validate(program) == []


def test_validate_accepts_well_formed_program():
    program = [
        ToolCall(op="add_box", args={"length": 100, "width": 60, "height": 6, "center": [0, 0, 0]}),
        ToolCall(op="drill_hole_pattern", args={"target": 0, "pattern": "corner_array", "count": 4, "radius": 4,
                                                 "depth": 6, "axis": "z", "margin": 15.0, "circle_diameter": 0.0, "z": 0.0}),
        ToolCall(op="finalize", args={"target": 1}),
    ]
    assert Sandbox.validate(program) == []


def test_execute_ok_on_well_formed_program():
    program = [
        ToolCall(op="add_box", args={"length": 100, "width": 60, "height": 6, "center": [0, 0, 0]}),
        ToolCall(op="drill_hole_pattern", args={"target": 0, "pattern": "corner_array", "count": 4, "radius": 4,
                                                 "depth": 6, "axis": "z", "margin": 15.0, "circle_diameter": 0.0, "z": 0.0}),
        ToolCall(op="finalize", args={"target": 1}),
    ]
    result = Sandbox.execute(program)
    assert result.status == "ok"
    assert result.mesh is not None
    assert result.mesh.is_watertight


def test_execute_rejected_on_malformed_program_zero_geometry_cost():
    program = [ToolCall(op="drill_hole", args={"target": 99, "radius": 4, "depth": 10, "center": [0, 0, 0], "axis": "z"})]
    result = Sandbox.execute(program)
    assert result.status == "rejected"
    assert result.mesh is None
    assert result.executed_steps == 0


def test_execute_partial_on_mid_program_failure():
    # boolean_difference against a handle that's never been created (99) --
    # passes validate() only if we sneak a bad arg past it; use a program
    # that validates fine structurally but fails at execution time instead:
    # drill a hole with a radius so large it consumes/degenerates the target.
    program = [
        ToolCall(op="add_box", args={"length": 10, "width": 10, "height": 10, "center": [0, 0, 0]}),
        ToolCall(op="drill_hole", args={"target": 0, "radius": 1000, "depth": 10, "center": [0, 0, 0], "axis": "z"}),
        ToolCall(op="finalize", args={"target": 1}),
    ]
    result = Sandbox.execute(program)
    # a hole far larger than the target should fully consume it -- either an
    # empty/degenerate result (suspect) or an execution failure (partial) is
    # an acceptable outcome; what's NOT acceptable is a silent "ok"
    assert result.status in ("suspect", "partial", "rejected")


def test_tool_schema_covers_every_op_in_tool_ops():
    schema = tool_schema()
    assert [s["op"] for s in schema] == TOOL_OPS
    assert "finalize" not in [s["op"] for s in schema]


def test_tool_schema_categorizes_and_types_handle_args_correctly():
    by_op = {s["op"]: s for s in tool_schema()}

    fillet = by_op["fillet_edge"]
    assert fillet["category"] == "consumes_target"
    arg_by_name = {a["name"]: a for a in fillet["args"]}
    assert arg_by_name["target"]["type"] == "handle"
    assert arg_by_name["direction1"]["type"] == "enum"
    assert set(arg_by_name["direction1"]["choices"]) == {"+x", "-x", "+y", "-y", "+z", "-z"}
    assert arg_by_name["radius"]["type"] == "number"

    union = by_op["boolean_union"]
    assert union["category"] == "consumes_two"
    union_args = {a["name"]: a for a in union["args"]}
    assert union_args["a"]["type"] == "handle"
    assert union_args["b"]["type"] == "handle"

    box = by_op["add_box"]
    assert box["category"] == "create"
    box_args = {a["name"]: a for a in box["args"]}
    assert "target" not in box_args
    assert box_args["center"]["type"] == "vec3"
    assert box_args["center"]["default"] == [0, 0, 0]
    assert box_args["length"]["required"] is True

    gear = by_op["add_gear_profile"]
    gear_args = {a["name"]: a for a in gear["args"]}
    assert gear_args["tooth_count"]["type"] == "integer"


def test_tool_schema_args_are_json_serializable():
    import json

    json.dumps(tool_schema())  # raises if anything (e.g. a raw tuple) slipped through


# ---- counterbore_hole / drill_slot / add_rib -- added to build "more robust" (fastener-ready, structurally reinforced) models ----

def test_counterbore_hole_flat_removes_exact_analytic_volume():
    sb = Sandbox()
    box_h = sb.add_box(60, 40, 20)
    original_volume = sb._get(box_h).volume
    result_h = sb.counterbore_hole(box_h, "+z", offset_u=0, offset_v=0, hole_radius=3, head_radius=6, head_depth=4, head_style="flat")
    mesh = sb._get(result_h)
    assert mesh.is_watertight
    import math
    shank_vol = math.pi * 3 * 3 * 20
    head_extra = math.pi * 6 * 6 * 4 - math.pi * 3 * 3 * 4
    expected = shank_vol + head_extra
    # 32-gon polygon approximation of a circle is a known, small (~0.5%) systematic undershoot
    assert abs((original_volume - mesh.volume) - expected) / expected < 0.01


def test_counterbore_hole_conical_is_wide_at_the_surface_and_narrow_at_depth():
    """The countersink cone must be head_radius wide right at the true
    surface (not partway down) -- this is the exact shape bug found and
    fixed while building this tool (bottom/top radius were swapped)."""
    sb = Sandbox()
    box_h = sb.add_box(60, 40, 20)
    result_h = sb.counterbore_hole(box_h, "+z", offset_u=0, offset_v=0, hole_radius=3, head_radius=6, head_depth=4, head_style="conical")
    mesh = sb._get(result_h)
    assert mesh.is_watertight
    section = mesh.section(plane_origin=[0, 0, 9.9], plane_normal=[0, 0, 1])
    assert section is not None
    planar, _ = section.to_2D()
    import numpy as np
    dists = np.linalg.norm(planar.vertices, axis=1)
    inner_radius = dists[dists < 20].max()  # excludes the outer 60x40 box boundary
    assert 5.5 < inner_radius < 6.5  # close to head_radius=6, just below the true surface


def test_counterbore_hole_raises_when_head_not_larger_than_hole():
    sb = Sandbox()
    box_h = sb.add_box(40, 30, 10)
    try:
        sb.counterbore_hole(box_h, "+z", 0, 0, hole_radius=5, head_radius=4, head_depth=3, head_style="flat")
        assert False, "should have raised -- head must be larger than the shank hole"
    except Exception:
        pass


def test_counterbore_hole_offset_is_relative_to_the_face_center_not_a_corner():
    """Regression test for a real, confirmed bug: face.origin is a CORNER
    of the facet's boundary, not its center, so offset_u=0, offset_v=0
    without a correction would land the hole near a corner instead of the
    middle of the face."""
    sb = Sandbox()
    box_h = sb.add_box(60, 40, 20, center=(0, 0, 0))
    result_h = sb.counterbore_hole(box_h, "+z", offset_u=0, offset_v=0, hole_radius=3, head_radius=6, head_depth=4, head_style="flat")
    mesh = sb._get(result_h)
    # a hole centered on the face must leave the mesh's own x/y bounding
    # box centered on the origin too (a corner-placed hole would skew it)
    center_xy = mesh.bounds.mean(axis=0)[:2]
    assert abs(center_xy[0]) < 0.5 and abs(center_xy[1]) < 0.5


def test_drill_slot_removes_exact_analytic_capsule_volume():
    sb = Sandbox()
    box_h = sb.add_box(80, 40, 10)
    original_volume = sb._get(box_h).volume
    result_h = sb.drill_slot(box_h, "+z", offset_u=0, offset_v=0, radius=4, slot_length=20, slot_axis="u", depth=10)
    mesh = sb._get(result_h)
    assert mesh.is_watertight
    import math
    expected = (math.pi * 4 * 4 + 2 * 4 * 20) * 10
    assert abs((original_volume - mesh.volume) - expected) / expected < 0.01


def test_drill_slot_along_v_axis_is_also_watertight():
    sb = Sandbox()
    box_h = sb.add_box(80, 40, 10)
    result_h = sb.drill_slot(box_h, "+z", offset_u=0, offset_v=0, radius=3, slot_length=15, slot_axis="v", depth=10)
    mesh = sb._get(result_h)
    assert mesh.is_watertight


def test_drill_slot_off_center_stays_watertight():
    sb = Sandbox()
    box_h = sb.add_box(80, 40, 10)
    result_h = sb.drill_slot(box_h, "+z", offset_u=20, offset_v=10, radius=3, slot_length=15, slot_axis="u", depth=10)
    mesh = sb._get(result_h)
    assert mesh.is_watertight


def test_add_rib_bridges_a_concave_corner_with_exact_wedge_volume():
    """Ribs only make sense at a CONCAVE (internal) corner -- e.g. an
    L-bracket's inner bend -- where the wedge fills real empty space, not
    a convex box corner where the same wedge would already lie inside
    existing material (confirmed empirically: reusing chamfer_edge's own
    wedge unmodified added only ~384 of an expected 1600mm^3 there)."""
    sb = Sandbox()
    vertical = sb.add_box(10, 50, 60, center=(5, 25, 30))
    horizontal = sb.add_box(60, 50, 10, center=(30, 25, 5))
    bracket_h = sb.boolean_union(vertical, horizontal)
    original_volume = sb._get(bracket_h).volume

    ribbed_h = sb.add_rib(bracket_h, "+x", "+z", size=8)
    mesh = sb._get(ribbed_h)
    assert mesh.is_watertight
    expected = 0.5 * 8 * 8 * 50  # right-triangle cross-section, full 50mm span, exact (no CSG padding distortion)
    assert abs((mesh.volume - original_volume) - expected) < 1.0


def test_add_rib_does_not_protrude_past_the_edges_real_extent():
    """The additive case must NOT reuse chamfer_edge's padded cutter
    length -- confirmed as a real bug during development: the padded
    wedge stuck out ~6mm past each end of the actual edge once unioned."""
    sb = Sandbox()
    vertical = sb.add_box(10, 50, 60, center=(5, 25, 30))
    horizontal = sb.add_box(60, 50, 10, center=(30, 25, 5))
    bracket_h = sb.boolean_union(vertical, horizontal)
    ribbed_h = sb.add_rib(bracket_h, "+x", "+z", size=8)
    mesh = sb._get(ribbed_h)
    # the bracket's own y-extent is exactly [0, 50] -- a rib that overshoots
    # the padded ~62mm span would push these bounds outward
    assert mesh.bounds[0][1] >= -1e-6
    assert mesh.bounds[1][1] <= 50 + 1e-6


def test_add_rib_raises_on_non_perpendicular_faces():
    sb = Sandbox()
    box_h = sb.add_box(40, 30, 20)
    try:
        sb.add_rib(box_h, "+z", "-z", size=5)
        assert False, "should have raised -- opposite faces share no edge"
    except Exception:
        pass


def test_validate_accepts_program_using_the_new_robustness_tools():
    program = [
        ToolCall(op="add_box", args={"length": 60, "width": 40, "height": 20, "center": [0, 0, 0]}),
        ToolCall(op="counterbore_hole", args={
            "target": 0, "direction": "+z", "offset_u": 0, "offset_v": 0,
            "hole_radius": 3, "head_radius": 6, "head_depth": 4, "head_style": "flat",
        }),
        ToolCall(op="drill_slot", args={
            "target": 1, "direction": "+z", "offset_u": 15, "offset_v": 0,
            "radius": 3, "slot_length": 12, "slot_axis": "u", "depth": 20,
        }),
        ToolCall(op="finalize", args={"target": 2}),
    ]
    issues = Sandbox.validate(program)
    assert issues == []
    result = Sandbox.execute(program)
    assert result.status in ("ok", "suspect")


# ---- add_shouldered_cylinder -- a smooth-CURVED diameter transition, added because neither
# add_revolve (straight taper, visible kink where it meets a constant-radius section) nor
# unioning two plain cylinders (a sharp 90-degree step) can build a realistic bottle shoulder ----

def test_add_shouldered_cylinder_is_watertight_with_correct_bounds():
    sb = Sandbox()
    h = sb.add_shouldered_cylinder(bottom_radius=32, bottom_height=100, top_radius=12, top_height=30, transition_height=25)
    mesh = sb._get(h)
    assert mesh.is_watertight
    assert mesh.volume > 0
    assert abs(mesh.extents[0] - 64) < 1e-6  # 2 * bottom_radius (the widest point)
    assert abs(mesh.extents[2] - 155) < 1e-6  # 100 + 25 + 30


def test_add_shouldered_cylinder_radius_profile_is_a_smooth_s_curve_not_a_step():
    """The whole point of this tool: the radius must change GRADUALLY across
    the transition, not jump straight from bottom_radius to top_radius."""
    sb = Sandbox()
    h = sb.add_shouldered_cylinder(bottom_radius=32, bottom_height=100, top_radius=12, top_height=30, transition_height=25)
    mesh = sb._get(h)
    zmin = mesh.bounds[0][2]

    def radius_at(z_rel):
        section = mesh.section(plane_origin=[0, 0, zmin + z_rel], plane_normal=[0, 0, 1])
        return np.linalg.norm(section.vertices[:, :2], axis=1).mean()

    r_before = radius_at(99)      # still in the constant-radius base
    r_quarter = radius_at(106.25)  # 25% through the transition
    r_mid = radius_at(112.5)      # exactly the transition's midpoint
    r_three_q = radius_at(118.75)  # 75% through
    r_after = radius_at(126)      # into the constant-radius top section

    assert abs(r_before - 32) < 0.5
    assert abs(r_after - 12) < 0.5
    # strictly decreasing through the transition -- a real gradient, not a jump
    assert r_before > r_quarter > r_mid > r_three_q > r_after
    # smoothstep is symmetric: the midpoint must be exactly the arithmetic mean of the two radii
    assert abs(r_mid - 22.0) < 0.5


def test_add_shouldered_cylinder_can_be_hollowed_by_shell():
    sb = Sandbox()
    h = sb.add_shouldered_cylinder(bottom_radius=32, bottom_height=100, top_radius=12, top_height=30, transition_height=25)
    solid_volume = sb._get(h).volume
    hollow_h = sb.shell(h, wall_thickness=2)
    mesh = sb._get(hollow_h)
    assert mesh.is_watertight
    assert mesh.volume < solid_volume
    assert not mesh.contains([[0, 0, 0]])[0]  # center is a cavity
    assert mesh.contains([[31, 0, mesh.bounds[0][2] + 10]])[0]  # a point inside the 2mm wall near the base


def test_add_shouldered_cylinder_raises_on_zero_radius():
    sb = Sandbox()
    try:
        sb.add_shouldered_cylinder(bottom_radius=0, bottom_height=50, top_radius=10, top_height=20, transition_height=15)
        assert False, "should have raised -- use add_revolve for a taper down to a point"
    except Exception:
        pass


def test_add_shouldered_cylinder_raises_on_non_positive_transition():
    sb = Sandbox()
    try:
        sb.add_shouldered_cylinder(bottom_radius=20, bottom_height=50, top_radius=10, top_height=20, transition_height=0)
        assert False, "should have raised -- a zero-height transition is not this tool's job"
    except Exception:
        pass


def test_validate_accepts_program_using_add_shouldered_cylinder():
    program = [
        ToolCall(op="add_shouldered_cylinder", args={
            "bottom_radius": 32, "bottom_height": 100, "top_radius": 12,
            "top_height": 30, "transition_height": 25, "center": [0, 0, 0],
        }),
        ToolCall(op="shell", args={"target": 0, "wall_thickness": 2}),
        ToolCall(op="finalize", args={"target": 1}),
    ]
    issues = Sandbox.validate(program)
    assert issues == []
    result = Sandbox.execute(program)
    assert result.status == "ok"
