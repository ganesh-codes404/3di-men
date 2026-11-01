"""Direct unit tests of dimension_audit.py -- hand-built requirements
and a real, small, Sandbox-built mesh (no model loading)."""

from dimension_audit import build_dimension_audit_report
from dimension_requirements import DimensionRequirement, FeatureCountRequirement, RequirementStatus
from evaluation.geometry_eval import evaluate_mesh
from sandbox import Sandbox


def _build(ops):
    from sandbox import ToolCall

    result = Sandbox.execute(ops)
    assert result.mesh is not None
    return result.mesh, evaluate_mesh(result.mesh).as_dict()


def test_verified_line_when_actual_matches_requested():
    from sandbox import ToolCall

    mesh, report = _build([
        ToolCall(op="add_box", args={"length": 70.0, "width": 40.0, "height": 3.0, "center": [0, 0, 0]}),
        ToolCall(op="finalize", args={"target": 0}),
    ])
    req = DimensionRequirement(id="R1", source_phrase="3mm thick", value=3.0, original_unit="mm", normalized_value=3.0, parameter="thickness", status=RequirementStatus.ASSIGNED)
    audit = build_dimension_audit_report([req], [], mesh, report)
    assert audit.lines[0].status == "VERIFIED"
    assert audit.mandatory_verified == 1
    assert audit.mandatory_total == 1


def test_mismatch_line_when_actual_extent_differs():
    from sandbox import ToolCall

    mesh, report = _build([
        ToolCall(op="add_box", args={"length": 70.0, "width": 40.0, "height": 3.0, "center": [0, 0, 0]}),
        ToolCall(op="finalize", args={"target": 0}),
    ])
    req = DimensionRequirement(id="R1", source_phrase="10mm thick", value=10.0, original_unit="mm", normalized_value=10.0, parameter="thickness", status=RequirementStatus.ASSIGNED)
    audit = build_dimension_audit_report([req], [], mesh, report)
    assert audit.lines[0].status == "MISMATCH"
    assert audit.mandatory_verified == 0


def test_unresolved_requirement_never_counts_as_verified():
    req = DimensionRequirement(id="R1", source_phrase="20mm bracket", value=20.0, original_unit="mm", normalized_value=20.0, parameter=None, status=RequirementStatus.UNRESOLVED)
    audit = build_dimension_audit_report([req], [], None, None)
    assert audit.lines[0].status == "UNRESOLVED"
    assert audit.mandatory_verified == 0
    assert audit.mandatory_total == 1


def test_hole_diameter_is_checked_against_the_closest_detected_hole():
    from sandbox import ToolCall

    mesh, report = _build([
        ToolCall(op="add_box", args={"length": 70.0, "width": 40.0, "height": 3.0, "center": [0, 0, 0]}),
        ToolCall(op="drill_hole", args={"target": 0, "radius": 4.0, "depth": 6.0, "axis": "z", "center": [0, 0, 0]}),
        ToolCall(op="finalize", args={"target": 1}),
    ])
    req = DimensionRequirement(id="R1", source_phrase="8mm hole", value=8.0, original_unit="mm", normalized_value=8.0, parameter="hole_diameter", status=RequirementStatus.ASSIGNED, assigned_arg_name="radius")
    audit = build_dimension_audit_report([req], [], mesh, report)
    assert audit.lines[0].status == "VERIFIED"


def test_hole_diameter_mismatch_when_no_hole_was_actually_detected():
    from sandbox import ToolCall

    mesh, report = _build([
        ToolCall(op="add_box", args={"length": 70.0, "width": 40.0, "height": 3.0, "center": [0, 0, 0]}),
        ToolCall(op="finalize", args={"target": 0}),
    ])
    req = DimensionRequirement(id="R1", source_phrase="8mm hole", value=8.0, original_unit="mm", normalized_value=8.0, parameter="hole_diameter", status=RequirementStatus.ASSIGNED, assigned_arg_name="radius")
    audit = build_dimension_audit_report([req], [], mesh, report)
    assert audit.lines[0].status == "MISMATCH"


def test_feature_count_line_verified_when_detected_count_matches():
    from sandbox import ToolCall

    mesh, report = _build([
        ToolCall(op="add_box", args={"length": 100.0, "width": 60.0, "height": 6.0, "center": [0, 0, 0]}),
        ToolCall(op="drill_hole_pattern", args={"target": 0, "pattern": "corner_array", "count": 4, "radius": 4.0, "depth": 6.0, "axis": "z", "margin": 15.0, "circle_diameter": 0.0, "z": 0.0}),
        ToolCall(op="finalize", args={"target": 1}),
    ])
    req = FeatureCountRequirement(id="R2", source_phrase="four holes", count=4, status=RequirementStatus.GEOMETRY_VERIFIED)
    audit = build_dimension_audit_report([], [req], mesh, report)
    assert audit.lines[0].status == "VERIFIED"


def test_render_produces_the_briefs_own_documented_format():
    req = DimensionRequirement(id="R1", source_phrase="Base thickness", value=5.0, original_unit="mm", normalized_value=5.0, parameter=None, status=RequirementStatus.UNRESOLVED)
    audit = build_dimension_audit_report([req], [], None, None)
    text = audit.render()
    assert "DIMENSION AUDIT" in text
    assert "Requested:" in text
    assert "Actual:" in text
    assert "Status:" in text
    assert "Overall:" in text
