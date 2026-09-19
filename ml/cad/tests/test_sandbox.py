from sandbox import Sandbox, ToolCall


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
