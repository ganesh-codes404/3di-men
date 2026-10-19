"""Tests for printability.py -- Print Lab's geometry-based analysis
service. Every check here is computed directly from real trimesh geometry
(no mocked numbers) -- these tests exist to prove the ACTUAL measurements
are correct, not just that the functions run without raising."""

import numpy as np
import pytest
import trimesh

from printability import (
    DEFAULT_PRINTER_PROFILES,
    PrinterProfile,
    analyze_printability,
    check_build_volume_fit,
    check_mesh_integrity,
    check_overhangs,
    estimate_material,
    slicer_availability,
    suggest_orientations,
)


# ---------------- mesh integrity ----------------

def test_mesh_integrity_on_a_valid_watertight_cube():
    cube = trimesh.creation.box([20, 20, 20])
    report = check_mesh_integrity(cube)
    assert report.watertight is True
    assert report.manifold is True
    assert report.body_count == 1
    assert report.non_manifold_edge_count == 0
    assert report.degenerate_face_count == 0
    assert report.is_empty is False
    assert report.bounding_box_mm == pytest.approx((20.0, 20.0, 20.0))
    assert report.volume_mm3 == pytest.approx(8000.0)
    assert report.printable_as_is is True


def test_mesh_integrity_detects_multiple_disconnected_components():
    a = trimesh.creation.box([10, 10, 10])
    b = trimesh.creation.box([10, 10, 10])
    b.apply_translation([100, 0, 0])  # far apart -- genuinely disconnected, not touching
    combined = trimesh.util.concatenate([a, b])
    report = check_mesh_integrity(combined)
    assert report.body_count == 2
    assert report.watertight is True  # each shell is individually closed


def test_mesh_integrity_flags_a_non_watertight_mesh():
    cube = trimesh.creation.box([20, 20, 20])
    open_mesh = trimesh.Trimesh(vertices=cube.vertices, faces=cube.faces[:-2])  # drop a couple faces -> opens it up
    report = check_mesh_integrity(open_mesh)
    assert report.watertight is False
    assert report.printable_as_is is False


def test_mesh_integrity_on_an_empty_mesh_is_honest_not_a_crash():
    empty = trimesh.Trimesh(vertices=[], faces=[])
    report = check_mesh_integrity(empty)
    assert report.is_empty is True
    assert report.printable_as_is is False
    assert report.volume_mm3 is None


# ---------------- build volume fit ----------------

def test_build_volume_fit_within_bounds():
    model = trimesh.creation.box([50, 50, 50])
    printer = DEFAULT_PRINTER_PROFILES["generic_small"]  # 220x220x250
    report = check_build_volume_fit(model, printer)
    assert report.fits is True
    assert report.overrun_mm == (0.0, 0.0, 0.0)


def test_build_volume_fit_reports_which_axis_overruns_and_by_how_much():
    model = trimesh.creation.box([300, 50, 50])  # X exceeds a small printer
    printer = PrinterProfile(name="tiny", build_volume_mm=(100.0, 100.0, 100.0))
    report = check_build_volume_fit(model, printer)
    assert report.fits is False
    assert report.overrun_mm[0] == pytest.approx(200.0)  # 300 - 100
    assert report.overrun_mm[1] == 0.0 and report.overrun_mm[2] == 0.0


def test_build_volume_fit_changes_with_orientation():
    """A model that doesn't fit lying down might fit standing up -- the
    fit check must reflect the mesh's CURRENT orientation, not some fixed
    assumption, so rotating it changes the result."""
    model = trimesh.creation.box([300, 20, 20])  # too long in X
    printer = PrinterProfile(name="tiny", build_volume_mm=(100.0, 100.0, 350.0))
    assert check_build_volume_fit(model, printer).fits is False

    rotated = model.copy()
    rotated.apply_transform(trimesh.transformations.rotation_matrix(np.radians(90), (0, 1, 0)))
    assert check_build_volume_fit(rotated, printer).fits is True


# ---------------- overhangs ----------------

def test_flat_box_has_zero_overhang():
    """Confirmed real bug this guards against: a plain box's flat bottom
    face points straight down (the steepest possible angle) but rests
    directly on the build plate -- it must never be counted as an
    overhang needing support."""
    box = trimesh.creation.box([50, 50, 20])
    report = check_overhangs(box)
    assert report.overhang_area_mm2 == pytest.approx(0.0, abs=1e-6)
    assert report.likely_needs_support is False


def test_cantilevered_overhang_is_detected():
    stem = trimesh.creation.box([10, 10, 30])
    top = trimesh.creation.box([60, 60, 10])
    top.apply_translation([0, 0, 20])
    tshape = stem.union(top, engine="manifold")
    report = check_overhangs(tshape)
    assert report.overhang_area_mm2 > 100.0
    assert report.likely_needs_support is True


def test_overhang_face_indices_point_at_the_real_overhanging_faces():
    """Confirmed real gap this closes: Print Lab's viewer could only show
    the AGGREGATE overhang area as a number, with no way to highlight
    WHERE on the model it actually is -- the per-face mask was computed
    then discarded. These must be real face indices into the same mesh
    (not a separate derived shape), each one actually failing the
    overhang-angle test on its own."""
    stem = trimesh.creation.box([10, 10, 30])
    top = trimesh.creation.box([60, 60, 10])
    top.apply_translation([0, 0, 20])
    tshape = stem.union(top, engine="manifold")
    report = check_overhangs(tshape)
    assert len(report.overhang_face_indices) > 0
    assert max(report.overhang_face_indices) < len(tshape.faces)
    # every reported index must independently fail the same angle test
    build_dir = np.array([0.0, 0.0, 1.0])
    threshold = -np.cos(np.radians(45.0))
    cos_angle = tshape.face_normals[report.overhang_face_indices] @ build_dir
    assert (cos_angle < threshold).all()
    # and their total area must equal the reported aggregate
    assert tshape.area_faces[report.overhang_face_indices].sum() == pytest.approx(report.overhang_area_mm2)


def test_overhang_angle_threshold_is_configurable():
    """A shallower (smaller) allowed angle flags MORE surface area as
    needing support than a steeper (larger) one, for the same geometry."""
    stem = trimesh.creation.box([10, 10, 30])
    top = trimesh.creation.box([60, 60, 10])
    top.apply_translation([0, 0, 20])
    tshape = stem.union(top, engine="manifold")
    strict = check_overhangs(tshape, overhang_angle_deg=10.0)
    lenient = check_overhangs(tshape, overhang_angle_deg=80.0)
    assert strict.overhang_area_mm2 >= lenient.overhang_area_mm2


# ---------------- material estimate ----------------

def test_material_estimate_with_known_density():
    cube = trimesh.creation.box([10, 10, 10])  # 1000 mm^3 = 1 cm^3
    estimate = estimate_material(cube, density_g_per_cm3=1.24)  # PLA-ish
    assert estimate.solid_volume_mm3 == pytest.approx(1000.0)
    assert estimate.estimated_mass_g == pytest.approx(1.24)


def test_material_estimate_without_density_never_guesses_a_mass():
    cube = trimesh.creation.box([10, 10, 10])
    estimate = estimate_material(cube, density_g_per_cm3=None)
    assert estimate.solid_volume_mm3 == pytest.approx(1000.0)
    assert estimate.estimated_mass_g is None


# ---------------- orientation suggestions ----------------

def test_suggest_orientations_prefers_the_lowest_overhang_candidate():
    stem = trimesh.creation.box([10, 10, 30])
    top = trimesh.creation.box([60, 60, 10])
    top.apply_translation([0, 0, 20])
    tshape = stem.union(top, engine="manifold")
    printer = DEFAULT_PRINTER_PROFILES["generic_medium"]
    candidates = suggest_orientations(tshape, printer)
    assert len(candidates) == 6
    # sorted best-first: the top candidate's score must be <= every other candidate's
    assert all(candidates[0].score <= c.score for c in candidates)


def test_suggest_orientations_marks_candidates_that_dont_fit():
    model = trimesh.creation.box([500, 500, 500])  # too big for any orientation of a small printer
    printer = DEFAULT_PRINTER_PROFILES["generic_small"]
    candidates = suggest_orientations(model, printer)
    assert all(c.fits_build_volume is False for c in candidates)


# ---------------- slicer availability ----------------

def test_slicer_availability_never_assumes_a_slicer_is_present():
    result = slicer_availability()
    assert set(result.keys()) == {"cura", "prusaslicer"}
    for info in result.values():
        assert "available" in info and "executable_path" in info
        assert isinstance(info["available"], bool)


# ---------------- full report ----------------

def test_analyze_printability_on_a_clean_small_part():
    cube = trimesh.creation.box([30, 30, 30])
    report = analyze_printability(cube, DEFAULT_PRINTER_PROFILES["generic_small"], density_g_per_cm3=1.24)
    assert report.integrity.printable_as_is is True
    assert report.build_volume_fit.fits is True
    assert report.material.estimated_mass_g is not None
    assert report.as_dict()["print_time_estimate"] == "unavailable -- requires a real slicer estimate (see limitations)"
    assert not any(w["severity"] == "error" for w in report.warnings)


def test_analyze_printability_on_an_empty_mesh_is_honest_not_a_crash():
    empty = trimesh.Trimesh(vertices=[], faces=[])
    report = analyze_printability(empty, DEFAULT_PRINTER_PROFILES["generic_small"])
    assert report.integrity.is_empty is True
    assert any(w["severity"] == "error" for w in report.warnings)


def test_analyze_printability_flags_an_oversized_model():
    huge = trimesh.creation.box([500, 500, 500])
    report = analyze_printability(huge, DEFAULT_PRINTER_PROFILES["generic_small"])
    assert report.build_volume_fit.fits is False
    assert any(w["severity"] == "error" and "build volume" in w["message"] for w in report.warnings)
