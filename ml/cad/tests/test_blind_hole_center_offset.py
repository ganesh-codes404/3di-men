"""Direct tests of sandbox.blind_hole_center_offset -- the helper that
makes drill_hole/drill_hole_pattern's own `depth` argument actually
produce a blind (not through) hole when paired with the right center
offset. See dimension_assignment.py's own use of it for the real,
live-reproduced bug this closes (a stated "8mm deep" hole built as a
through-hole because `depth` alone, with no center offset, can never
stop a cutter short -- drill_hole's cutter always floors its own
length at the target's full extent)."""

from sandbox import Sandbox, ToolCall, blind_hole_center_offset
from evaluation.geometry_eval import evaluate_mesh


def test_offset_is_zero_when_depth_reaches_the_full_extent():
    assert blind_hole_center_offset(target_extent=20.0, depth=20.0) == 0.0
    assert blind_hole_center_offset(target_extent=20.0, depth=25.0) == 0.0  # deeper than the part -- degenerates to through


def test_offset_is_positive_for_a_genuinely_shallower_depth():
    offset = blind_hole_center_offset(target_extent=20.0, depth=8.0)
    assert offset > 0.0


def test_offset_actually_produces_a_blind_hole_not_a_through_one():
    target_extent, depth = 20.0, 8.0
    offset = blind_hole_center_offset(target_extent, depth)
    program = [
        ToolCall(op="add_cylinder", args={"radius": 25.0, "height": target_extent, "axis": "z", "center": [0.0, 0.0, 0.0]}),
        ToolCall(op="drill_hole", args={"target": 0, "radius": 5.0, "depth": depth, "axis": "z", "center": [0.0, 0.0, offset]}),
        ToolCall(op="finalize", args={"target": 1}),
    ]
    result = Sandbox.execute(program)
    assert result.status == "ok"
    report = evaluate_mesh(result.mesh)
    assert report.hole_through == [False]


def test_zero_offset_with_a_shallow_depth_produces_a_through_hole_instead():
    """The exact confirmed bug: depth alone (no offset) never makes a
    hole blind, because the cutter's own length floors at the target's
    full extent regardless of depth."""
    program = [
        ToolCall(op="add_cylinder", args={"radius": 25.0, "height": 20.0, "axis": "z", "center": [0.0, 0.0, 0.0]}),
        ToolCall(op="drill_hole", args={"target": 0, "radius": 5.0, "depth": 8.0, "axis": "z", "center": [0.0, 0.0, 0.0]}),
        ToolCall(op="finalize", args={"target": 1}),
    ]
    result = Sandbox.execute(program)
    report = evaluate_mesh(result.mesh)
    assert report.hole_through == [True]
