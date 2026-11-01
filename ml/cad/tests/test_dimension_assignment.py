"""Direct unit tests of dimension_assignment.py -- hand-built programs
and requirements, no model loading. See tests/test_dimension_
regression.py for the end-to-end production-pipeline tests."""

from dimension_assignment import assign_requirements
from dimension_requirements import DimensionRequirement, FeatureCountRequirement, RequirementStatus
from sandbox import ToolCall


def _dim(value_mm, parameter, **kw):
    return DimensionRequirement(id="REQ-T", source_phrase="t", value=value_mm, original_unit="mm", normalized_value=value_mm, normalized_unit="mm", parameter=parameter, **kw)


def test_base_shape_diameter_is_halved_into_radius():
    req = _dim(40.0, "outer_diameter")
    program = [ToolCall(op="add_cylinder", args={"radius": 1.0, "height": 1.0, "axis": "z", "center": [0, 0, 0]})]
    result = assign_requirements([req], [], program)
    assert result.program[0].args["radius"] == 20.0
    assert req.status is RequirementStatus.ASSIGNED
    assert req.assigned_op_index == 0


def test_a_requirement_the_chosen_program_has_no_matching_op_for_is_unassigned_not_guessed():
    req = _dim(20.0, "main_diameter")
    program = [ToolCall(op="add_cylinder", args={"radius": 1.0, "height": 1.0, "axis": "z", "center": [0, 0, 0]})]
    result = assign_requirements([req], [], program)
    assert result.program[0].args["radius"] == 1.0  # untouched
    assert req.status is RequirementStatus.UNASSIGNED


def test_unresolved_requirement_from_extraction_is_passed_through_untouched():
    req = DimensionRequirement(id="REQ-T", source_phrase="t", value=20.0, original_unit="mm", normalized_value=20.0, parameter=None, status=RequirementStatus.UNRESOLVED)
    program = [ToolCall(op="add_cylinder", args={"radius": 1.0, "height": 1.0, "axis": "z", "center": [0, 0, 0]})]
    result = assign_requirements([req], [], program)
    assert result.program[0].args["radius"] == 1.0
    assert req.status is RequirementStatus.UNRESOLVED
    assert any(entry.outcome == "UNRESOLVED" for entry in result.log)


def test_a_single_hole_diameter_requirement_applies_to_every_individual_hole_op():
    """Confirmed real gap: "three 8mm holes" built as 3 separate
    drill_hole calls only had the FIRST one corrected; the other two
    kept whatever the regressor separately guessed."""
    req = _dim(8.0, "hole_diameter")
    program = [
        ToolCall(op="add_box", args={"length": 70.0, "width": 40.0, "height": 3.0, "center": [0, 0, 0]}),
        ToolCall(op="drill_hole", args={"target": 0, "radius": 1.0, "depth": 5.0, "axis": "z", "center": [0, 0, 0]}),
        ToolCall(op="drill_hole", args={"target": 1, "radius": 2.0, "depth": 5.0, "axis": "z", "center": [0, 0, 0]}),
        ToolCall(op="drill_hole", args={"target": 2, "radius": 3.0, "depth": 5.0, "axis": "z", "center": [0, 0, 0]}),
    ]
    result = assign_requirements([req], [], program)
    assert [c.args["radius"] for c in result.program[1:]] == [4.0, 4.0, 4.0]


def test_edge_offset_targets_only_the_first_match_not_every_hole_op():
    """edge_offset is a single PATTERN-level value (the margin), unlike
    hole_diameter/hole_depth -- it must not fan out across multiple ops."""
    req = _dim(20.0, "edge_offset")
    program = [
        ToolCall(op="add_box", args={"length": 70.0, "width": 40.0, "height": 3.0, "center": [0, 0, 0]}),
        ToolCall(op="drill_hole_pattern", args={"target": 0, "pattern": "corner_array", "count": 4, "radius": 3.0, "depth": 5.0, "axis": "z", "margin": 1.0, "circle_diameter": 0.0, "z": 0.0}),
    ]
    result = assign_requirements([req], [], program)
    assert result.program[1].args["margin"] == 20.0


def test_count_requirement_overrides_the_pattern_ops_own_count_and_pattern_type():
    req = FeatureCountRequirement(id="REQ-C", source_phrase="t", count=5, pattern_hint="linear_array")
    program = [
        ToolCall(op="add_box", args={"length": 100.0, "width": 50.0, "height": 5.0, "center": [0, 0, 0]}),
        ToolCall(op="drill_hole_pattern", args={"target": 0, "pattern": "corner_array", "count": 2, "radius": 2.0, "depth": 5.0, "axis": "z", "margin": 10.0, "circle_diameter": 0.0, "z": 0.0}),
    ]
    result = assign_requirements([], [req], program)
    assert result.program[1].args["count"] == 5
    assert result.program[1].args["pattern"] == "linear_array"
    assert req.status is RequirementStatus.ASSIGNED


def test_count_requirement_with_no_hole_producing_op_at_all_is_unassigned():
    req = FeatureCountRequirement(id="REQ-C", source_phrase="t", count=3)
    program = [ToolCall(op="add_box", args={"length": 70.0, "width": 40.0, "height": 3.0, "center": [0, 0, 0]})]
    result = assign_requirements([], [req], program)
    assert req.status is RequirementStatus.UNASSIGNED


def test_shouldered_cylinder_overall_length_rescales_proportionally_not_arbitrarily():
    req = _dim(100.0, "overall_length")
    program = [ToolCall(op="add_shouldered_cylinder", args={"bottom_radius": 10.0, "bottom_height": 30.0, "top_radius": 6.0, "top_height": 10.0, "transition_height": 10.0, "center": [0, 0, 0]})]
    result = assign_requirements([req], [], program)
    args = result.program[0].args
    total = args["bottom_height"] + args["top_height"] + args["transition_height"]
    assert abs(total - 100.0) < 1e-6
    # ratio preserved: original was 30:10:10 (3:1:1)
    assert abs(args["bottom_height"] / args["top_height"] - 3.0) < 1e-6
