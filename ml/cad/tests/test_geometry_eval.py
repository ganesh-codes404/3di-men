"""Tests for evaluation/geometry_eval.py's axis-agnostic hole detection.

Confirmed real gap these cover: the detector used to only ever test the Z
axis (see _detect_holes_and_cylinders's own prior docstring, and
data/procedural_generator.py's BracketParams, whose two arms drill
horizontally -- Y-axis and X-axis -- and were invisible to it,
detected_hole_count reading 0 for any bracket regardless of how correctly
it was actually drilled). Every mesh here is real trimesh geometry (a
genuine boolean difference), not a mock -- these tests exist to prove the
actual measurement is correct, not just that the function runs."""

import numpy as np
import pytest
import trimesh

from evaluation.geometry_eval import evaluate_mesh


def _box_with_through_hole(axis: str, diameter: float = 10.0) -> trimesh.Trimesh:
    """A 40x40x40 box with one round hole drilled all the way through along
    `axis` -- the cylinder is deliberately longer (60mm) than the box's own
    40mm extent so it punches cleanly out both faces."""
    box = trimesh.creation.box([40.0, 40.0, 40.0])
    hole = trimesh.creation.cylinder(radius=diameter / 2, height=60.0, sections=48)
    if axis == "x":
        hole.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [0, 1, 0]))
    elif axis == "y":
        hole.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0]))
    return box.difference(hole, engine="manifold")


def _box_with_blind_hole(axis: str, depth: float = 15.0, diameter: float = 10.0) -> trimesh.Trimesh:
    """A 40x40x40 box with a round blind bore, drilled `depth` mm into the
    +axis face -- NOT all the way through the box's own 40mm extent. The
    cylinder sticks 2mm proud of that face so the boolean difference opens
    a clean mouth instead of relying on an exact coplanar coincidence."""
    box = trimesh.creation.box([40.0, 40.0, 40.0])
    protrusion = 2.0
    height = depth + protrusion
    hole = trimesh.creation.cylinder(radius=diameter / 2, height=height, sections=48)
    if axis == "x":
        hole.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [0, 1, 0]))
    elif axis == "y":
        hole.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0]))
    center = 20.0 - depth / 2 + protrusion / 2  # box half-extent is 20
    shift = [0.0, 0.0, 0.0]
    shift["xyz".index(axis)] = center
    hole.apply_translation(shift)
    return box.difference(hole, engine="manifold")


def test_hole_along_z_axis_is_still_detected():
    """Regression: the pre-existing Z-axis case must keep working exactly
    as before this generalization."""
    mesh = _box_with_through_hole("z")
    report = evaluate_mesh(mesh)
    assert report.detected_hole_count == 1
    assert report.hole_axes == ["z"]
    assert report.hole_diameters_mm[0] == pytest.approx(10.0, abs=0.5)
    assert report.hole_through == [True]


def test_hole_along_x_axis_is_detected():
    """Confirmed real gap this closes: previously invisible (detected_hole_count
    read 0) because the old detector only ever tested the Z axis."""
    mesh = _box_with_through_hole("x")
    report = evaluate_mesh(mesh)
    assert report.detected_hole_count == 1
    assert report.hole_axes == ["x"]
    assert report.hole_diameters_mm[0] == pytest.approx(10.0, abs=0.5)
    assert report.hole_through == [True]


def test_hole_along_y_axis_is_detected():
    mesh = _box_with_through_hole("y")
    report = evaluate_mesh(mesh)
    assert report.detected_hole_count == 1
    assert report.hole_axes == ["y"]
    assert report.hole_diameters_mm[0] == pytest.approx(10.0, abs=0.5)


def test_two_holes_on_different_axes_are_both_detected_without_double_counting():
    """A part with holes on two different axes at once (exactly the
    bracket's own shape, simplified to a single box) -- each must be
    attributed to its own real axis, and claimed faces from one axis's
    pass must not leak into / get double-counted by another."""
    box = trimesh.creation.box([40.0, 40.0, 40.0])
    hole_z = trimesh.creation.cylinder(radius=5.0, height=60.0, sections=48)
    hole_x = trimesh.creation.cylinder(radius=4.0, height=60.0, sections=48)
    hole_x.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [0, 1, 0]))
    hole_x.apply_translation([0, 12, 0])  # offset sideways so it doesn't intersect the Z hole
    mesh = box.difference(hole_z, engine="manifold").difference(hole_x, engine="manifold")
    report = evaluate_mesh(mesh)
    assert report.detected_hole_count == 2
    assert sorted(report.hole_axes) == ["x", "z"]
    diam_by_axis = dict(zip(report.hole_axes, report.hole_diameters_mm))
    assert diam_by_axis["z"] == pytest.approx(10.0, abs=0.5)
    assert diam_by_axis["x"] == pytest.approx(8.0, abs=0.5)


def test_blind_hole_is_not_marked_through():
    mesh = _box_with_blind_hole("z")
    report = evaluate_mesh(mesh)
    assert report.detected_hole_count == 1
    assert report.hole_through == [False]


def test_bracket_arms_detected_on_their_real_horizontal_axes():
    """Direct regression for the exact case data/procedural_generator.py's
    BracketParams docstring used to document as a known, deferred gap:
    arm 1's holes run along Y, arm 2's run along X -- neither is Z, so the
    old Z-only detector read detected_hole_count=0 for this real part."""
    from data.procedural_generator import BracketParams, build_bracket

    params = BracketParams(
        leg1_length_mm=60, leg2_length_mm=60, thickness_mm=5, length_mm=80,
        hole_diameter_mm=6, hole_edge_margin_mm=15, holes_per_arm=1,
    )
    mesh = build_bracket(params)
    assert mesh.is_watertight

    report = evaluate_mesh(mesh)
    assert report.detected_hole_count == 2
    assert sorted(report.hole_axes) == ["x", "y"]
    for d in report.hole_diameters_mm:
        assert d == pytest.approx(6.0, abs=0.75)
