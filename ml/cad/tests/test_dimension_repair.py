"""Direct unit tests of dimension_repair.py -- hand-built programs with
a deliberately broken hole placement, no model loading. See tests/
test_dimension_regression.py for the end-to-end production-pipeline
tests."""

from dimension_requirements import DimensionRequirement, FeatureCountRequirement, RequirementStatus
from dimension_repair import repair_hole_count_mismatch, repair_missing_stepped_section
from sandbox import ToolCall


def test_repair_is_a_noop_when_the_count_already_matches():
    req = FeatureCountRequirement(id="REQ-C", source_phrase="t", count=4)
    program = [
        ToolCall(op="add_box", args={"length": 100.0, "width": 60.0, "height": 6.0, "center": [0, 0, 0]}),
        ToolCall(op="drill_hole_pattern", args={"target": 0, "pattern": "corner_array", "count": 4, "radius": 4.0, "depth": 6.0, "axis": "z", "margin": 15.0, "circle_diameter": 0.0, "z": 0.0}),
        ToolCall(op="finalize", args={"target": 1}),
    ]
    outcome = repair_hole_count_mismatch(program, req)
    assert outcome.attempts == []
    assert outcome.detected_count == 4
    assert req.status is RequirementStatus.GEOMETRY_VERIFIED


def test_repair_shrinks_margin_when_it_is_too_large_to_fit_any_hole():
    """A margin larger than half the plate's own smaller dimension (60mm
    wide, margin=70 -- confirmed live: this alone drops detected_hole_
    count to 0) pushes every corner position outside the plate's own
    footprint entirely."""
    req = FeatureCountRequirement(id="REQ-C", source_phrase="t", count=4)
    program = [
        ToolCall(op="add_box", args={"length": 100.0, "width": 60.0, "height": 6.0, "center": [0, 0, 0]}),
        ToolCall(op="drill_hole_pattern", args={"target": 0, "pattern": "corner_array", "count": 4, "radius": 4.0, "depth": 6.0, "axis": "z", "margin": 70.0, "circle_diameter": 0.0, "z": 0.0}),
        ToolCall(op="finalize", args={"target": 1}),
    ]
    outcome = repair_hole_count_mismatch(program, req)
    assert outcome.detected_count == 4
    assert req.status is RequirementStatus.GEOMETRY_VERIFIED
    assert any(a.strategy == "retry_pattern_with_safe_margin" and a.improved for a in outcome.attempts)


def test_repair_replaces_wildly_mispositioned_individual_holes_with_a_pattern():
    req = FeatureCountRequirement(id="REQ-C", source_phrase="t", count=3)
    program = [
        ToolCall(op="add_box", args={"length": 70.0, "width": 40.0, "height": 3.0, "center": [0, 0, 0]}),
        ToolCall(op="drill_hole", args={"target": 0, "radius": 4.0, "depth": 6.0, "axis": "y", "center": [26.5, -2.0, -20.8]}),
        ToolCall(op="drill_hole", args={"target": 1, "radius": 4.0, "depth": 6.0, "axis": "y", "center": [16.6, 7.8, 22.5]}),
        ToolCall(op="drill_hole", args={"target": 2, "radius": 4.0, "depth": 6.0, "axis": "x", "center": [-0.2, 24.8, -34.2]}),
        ToolCall(op="finalize", args={"target": 3}),
    ]
    outcome = repair_hole_count_mismatch(program, req)
    assert outcome.detected_count == 3
    assert req.status is RequirementStatus.GEOMETRY_VERIFIED
    assert any(c.op == "drill_hole_pattern" for c in outcome.program)
    assert not any(c.op == "drill_hole" for c in outcome.program)


def test_repair_discloses_failure_rather_than_silently_accepting_a_mismatch():
    """A degenerate case repair genuinely cannot fix (radius too large
    to ever fit `count` holes on this plate at all) -- must end as
    FAILED, not quietly reverted to pretend nothing was wrong."""
    req = FeatureCountRequirement(id="REQ-C", source_phrase="t", count=4)
    program = [
        ToolCall(op="add_box", args={"length": 20.0, "width": 20.0, "height": 3.0, "center": [0, 0, 0]}),
        ToolCall(op="drill_hole_pattern", args={"target": 0, "pattern": "corner_array", "count": 4, "radius": 50.0, "depth": 3.0, "axis": "z", "margin": 5.0, "circle_diameter": 0.0, "z": 0.0}),
    ]
    outcome = repair_hole_count_mismatch(program, req)
    assert req.status is RequirementStatus.FAILED
    assert outcome.detected_count != 4


def test_repair_swaps_an_incompatible_l_shape_profile_to_rect():
    req = FeatureCountRequirement(id="REQ-C", source_phrase="t", count=2)
    program = [
        ToolCall(op="add_extrusion", args={"profile_kind": "l_shape", "param_a": 78.0, "param_b": 68.0, "param_c": 4.5, "height": 5.0, "center": [0, 0, 0]}),
        ToolCall(op="drill_hole_pattern", args={"target": 0, "pattern": "linear_array", "count": 2, "radius": 3.0, "depth": 6.0, "axis": "z", "margin": 17.0, "circle_diameter": 0.0, "z": 0.0}),
        ToolCall(op="finalize", args={"target": 1}),
    ]
    outcome = repair_hole_count_mismatch(program, req)
    assert outcome.program[0].args["profile_kind"] == "rect"
    assert req.status is RequirementStatus.GEOMETRY_VERIFIED
    assert outcome.detected_count == 2


def test_missing_stepped_section_is_a_noop_without_both_diameters_stated():
    reqs = [DimensionRequirement(id="REQ-1", source_phrase="t", value=20.0, original_unit="mm", normalized_value=20.0, parameter="outer_diameter")]
    program = [ToolCall(op="add_cylinder", args={"radius": 10.0, "height": 50.0, "axis": "z", "center": [0, 0, 0]})]
    outcome = repair_missing_stepped_section(program, reqs)
    assert outcome.applied is False
    assert outcome.program is program


def test_missing_stepped_section_swaps_add_cylinder_for_add_shouldered_cylinder():
    reqs = [
        DimensionRequirement(id="REQ-1", source_phrase="t", value=20.0, original_unit="mm", normalized_value=20.0, parameter="main_diameter"),
        DimensionRequirement(id="REQ-2", source_phrase="t", value=12.0, original_unit="mm", normalized_value=12.0, parameter="reduced_diameter"),
    ]
    program = [ToolCall(op="add_cylinder", args={"radius": 14.0, "height": 87.0, "axis": "z", "center": [0, 0, 0]}), ToolCall(op="finalize", args={"target": 0})]
    outcome = repair_missing_stepped_section(program, reqs)
    assert outcome.applied is True
    assert outcome.program[0].op == "add_shouldered_cylinder"
    assert outcome.program[1].op == "finalize"


def test_missing_stepped_section_does_not_touch_an_already_correct_base_op():
    reqs = [
        DimensionRequirement(id="REQ-1", source_phrase="t", value=20.0, original_unit="mm", normalized_value=20.0, parameter="main_diameter"),
        DimensionRequirement(id="REQ-2", source_phrase="t", value=12.0, original_unit="mm", normalized_value=12.0, parameter="reduced_diameter"),
    ]
    program = [ToolCall(op="add_shouldered_cylinder", args={"bottom_radius": 10.0, "bottom_height": 40.0, "top_radius": 6.0, "top_height": 40.0, "transition_height": 10.0, "center": [0, 0, 0]})]
    outcome = repair_missing_stepped_section(program, reqs)
    assert outcome.applied is False
    assert outcome.program is program
