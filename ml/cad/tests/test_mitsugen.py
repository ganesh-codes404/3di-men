"""Tests for the Mitsugen slice-based 3D-printing generation pipeline --
a DISTINCT product/architecture from 3di-men's CAD assembly pipeline (see
mitsugen/__init__.py). Covers the mission brief's own 14 required
regression scenarios, deterministically -- this whole pipeline never
calls an LLM, so every test here runs fast and exercises the real
geometry code, not a mock."""

import numpy as np
import pytest

from mitsugen.builder import apply_hollowing, build_solid_from_slices
from mitsugen.pipeline import generate_mitsugen_model
from mitsugen.profile import Profile2D, validate_adjacent_profiles, validate_profile
from mitsugen.slice_planner import plan_slices
from mitsugen.spec import MitsugenShapeSpec, compose_spec_from_prompt


# ---------------- 1. simple extruded profile ----------------

def test_simple_cylinder_builds_a_correct_solid():
    """A constant-radius shape (no taper) is a degenerate case of the
    radial-sweep strategy -- a simple extrusion. Must build a valid
    solid with the exact requested dimensions."""
    spec = MitsugenShapeSpec(object_description="cylinder", height_mm=100.0, waypoints_mm=[(0.0, 30.0), (1.0, 30.0)], max_slices=12)
    plan = plan_slices(spec)
    build = build_solid_from_slices(plan)
    assert build.error is None
    assert build.mesh.is_watertight
    assert build.mesh.extents == pytest.approx([60.0, 60.0, 100.0], abs=1.0)
    expected_volume = np.pi * 30.0 ** 2 * 100.0
    assert build.mesh.volume == pytest.approx(expected_volume, rel=0.02)


# ---------------- 2. cross-section changes along the construction axis ----------------

def test_tapered_vase_profile_changes_radius_along_the_axis():
    """A real taper (base radius != top radius) must actually produce a
    model whose cross-section genuinely narrows -- not a cylinder, not
    a cone approximated as a cylinder."""
    spec = MitsugenShapeSpec(object_description="vase", height_mm=200.0, waypoints_mm=[(0.0, 55.0), (1.0, 20.0)], max_slices=12)
    plan = plan_slices(spec)
    build = build_solid_from_slices(plan)
    assert build.error is None
    assert build.mesh.is_watertight
    # bottom half of the mesh must be measurably wider than the top half
    bottom_slice = build.mesh.section(plane_origin=[0, 0, 10], plane_normal=[0, 0, 1])
    top_slice = build.mesh.section(plane_origin=[0, 0, 190], plane_normal=[0, 0, 1])
    assert bottom_slice is not None and top_slice is not None
    bottom_radius = bottom_slice.extents[0] / 2.0
    top_radius = top_slice.extents[0] / 2.0
    assert bottom_radius > top_radius * 1.5  # genuinely, substantially narrower at the top


# ---------------- 3. opening / internal boundary (hollow vessel) ----------------

def test_hollow_vessel_has_an_open_top_and_a_real_cavity():
    spec = MitsugenShapeSpec(object_description="vase", height_mm=200.0, waypoints_mm=[(0.0, 55.0), (1.0, 30.0)], max_slices=12)
    plan = plan_slices(spec)
    solid_build = build_solid_from_slices(plan)
    hollow_build = apply_hollowing(solid_build, plan, wall_thickness_mm=3.0)
    assert hollow_build.hollow_applied is True
    assert hollow_build.mesh.is_watertight  # the cavity + outer shell differenced into one valid watertight solid
    assert hollow_build.mesh.volume < solid_build.mesh.volume * 0.5  # a real, substantial cavity, not a token one


def test_hollowing_with_an_impossible_wall_thickness_does_not_corrupt_the_solid():
    """A wall thickness too large to leave any positive inner cavity must
    not be silently accepted as "hollowed" -- the solid version is kept,
    with the limitation disclosed."""
    spec = MitsugenShapeSpec(object_description="thin vase", height_mm=50.0, waypoints_mm=[(0.0, 5.0), (1.0, 5.0)], max_slices=4)
    plan = plan_slices(spec)
    solid_build = build_solid_from_slices(plan)
    hollow_build = apply_hollowing(solid_build, plan, wall_thickness_mm=10.0)  # thicker than the whole radius
    assert hollow_build.hollow_applied is False
    assert hollow_build.error is not None
    assert hollow_build.mesh.is_watertight  # still a valid solid -- not corrupted


# ---------------- 4. fewer slices than the configured maximum ----------------

def test_simple_shape_uses_fewer_slices_than_the_configured_maximum():
    """Explicitly required: "do not automatically use N_max for every
    object." A plain cylinder needs only 2 slices regardless of how
    high max_slices is set."""
    spec = MitsugenShapeSpec(object_description="cylinder", height_mm=100.0, waypoints_mm=[(0.0, 30.0), (1.0, 30.0)], max_slices=50)
    plan = plan_slices(spec)
    assert plan.actual_slice_count == 2
    assert plan.actual_slice_count < spec.max_slices
    assert plan.within_tolerance is True


# ---------------- 5. shape exceeds the slice budget at the configured tolerance ----------------

def test_shape_exceeding_the_slice_budget_reports_an_honest_limitation():
    """A tight tolerance with a very low slice budget, on a shape with a
    real transition, must not silently claim success -- it must report
    exactly why, and by how much it missed, never fabricate accuracy."""
    spec = MitsugenShapeSpec(
        object_description="vase", height_mm=200.0,
        waypoints_mm=[(0.0, 55.0), (0.5, 55.0), (1.0, 20.0)],
        max_slices=3, tolerance_fraction=0.01,
    )
    plan = plan_slices(spec)
    assert plan.within_tolerance is False
    assert plan.limitation is not None
    assert "increase" in plan.limitation.lower() or "simplify" in plan.limitation.lower()
    assert plan.actual_slice_count <= spec.max_slices  # still respects the hard cap even though it can't meet tolerance


# ---------------- 6. invalid / self-intersecting profiles ----------------

def test_validate_profile_rejects_too_few_points():
    degenerate = Profile2D(slice_id=0, outline=np.array([[0.0, 0.0], [1.0, 0.0]]), radius_mm=1.0)
    report = validate_profile(degenerate)
    assert report.valid is False
    assert "3 points" in report.reason


def test_validate_profile_rejects_non_finite_coordinates():
    bad = Profile2D(slice_id=0, outline=np.array([[0.0, 0.0], [1.0, float("nan")], [0.0, 1.0]]), radius_mm=1.0)
    report = validate_profile(bad)
    assert report.valid is False
    assert "non-finite" in report.reason


def test_validate_profile_rejects_a_self_intersecting_outline():
    """A bowtie/figure-eight shaped polygon -- a real self-intersection,
    not a hypothetical one."""
    bowtie = Profile2D(slice_id=0, outline=np.array([[0.0, 0.0], [1.0, 1.0], [1.0, 0.0], [0.0, 1.0]]), radius_mm=1.0)
    report = validate_profile(bowtie)
    assert report.valid is False


def test_validate_profile_accepts_a_real_circular_profile():
    from mitsugen.profile import make_circular_profile

    profile = make_circular_profile(0, radius_mm=20.0)
    report = validate_profile(profile)
    assert report.valid is True
    assert report.reason is None


# ---------------- 7. incompatible adjacent profiles ----------------

def test_validate_adjacent_profiles_rejects_a_degenerate_neighbor():
    from mitsugen.profile import make_circular_profile

    good = make_circular_profile(0, radius_mm=20.0)
    degenerate = Profile2D(slice_id=1, outline=np.array([[0.0, 0.0]]), radius_mm=0.0)
    report = validate_adjacent_profiles(good, degenerate)
    assert report.valid is False


def test_validate_adjacent_profiles_accepts_two_real_circles():
    from mitsugen.profile import make_circular_profile

    a = make_circular_profile(0, radius_mm=20.0)
    b = make_circular_profile(1, radius_mm=15.0)
    assert validate_adjacent_profiles(a, b).valid is True


# ---------------- 8. exceeds the selected printer's build volume ----------------

def test_pipeline_flags_a_model_that_exceeds_the_printer_build_volume():
    result = generate_mitsugen_model(
        "a vase, 200mm tall, 300mm wide at the base, narrowing to 100mm at the top",
        max_slices=10, printer_build_volume_mm=(150.0, 150.0, 250.0),
    )
    assert result.overall_status == "FAIL"
    fit_finding = next(f for f in result.findings if "build volume" in f.requirement)
    assert fit_finding.passed is False


def test_pipeline_passes_a_model_that_fits_a_generous_printer():
    result = generate_mitsugen_model(
        "a vase, 100mm tall, 40mm wide at the base, narrowing to 20mm at the top",
        max_slices=10, printer_build_volume_mm=(300.0, 300.0, 300.0),
    )
    assert result.overall_status == "PASS"


# ---------------- 9. connected geometry (no unintended disconnection) ----------------

def test_built_solid_is_always_a_single_connected_body():
    """The ring-loft construction connects every consecutive pair of
    slices with real triangulated side faces -- never a stack of
    disconnected slabs. body_count must be exactly 1."""
    spec = MitsugenShapeSpec(object_description="vase", height_mm=150.0, waypoints_mm=[(0.0, 40.0), (0.5, 60.0), (1.0, 25.0)], max_slices=8)
    plan = plan_slices(spec)
    build = build_solid_from_slices(plan)
    assert build.error is None
    assert build.mesh.body_count == 1


def test_builder_rejects_too_few_slices_rather_than_guessing():
    from mitsugen.slice_planner import PlannedSlice, SlicePlan

    broken_plan = SlicePlan(
        slices=[PlannedSlice(slice_id=0, order=0, height_fraction=0.0, position_mm=0.0, radius_mm=10.0)],
        requested_max_slices=5, actual_slice_count=1, approximation_error_fraction=0.0, within_tolerance=True, limitation=None,
    )
    build = build_solid_from_slices(broken_plan)
    assert build.mesh is None
    assert build.error is not None


# ---------------- 10. explicit dimensions preserved despite semantic defaults ----------------

def test_explicit_dimensions_override_the_named_objects_own_typical_size():
    """Non-negotiable principle, Mitsugen's own copy of it: a named
    object ("vase") carries typical sizing, but explicit numbers in the
    SAME prompt must win, never be silently blended with the default."""
    spec = compose_spec_from_prompt("a vase, 500mm tall, 10mm wide at the base, narrowing to 5mm at the top")
    assert spec.explicit_mm == {"height_mm": 500.0, "base_radius_mm": 10.0, "top_radius_mm": 5.0}
    assert spec.inferred_defaults_mm == {}
    assert spec.height_mm == 500.0  # not the vase default of 220mm
    assert spec.waypoints_mm[0][1] == 10.0  # not the vase default of 55mm


def test_well_known_object_with_no_dimensions_uses_a_disclosed_inferred_default():
    spec = compose_spec_from_prompt("a vase")
    assert spec.explicit_mm == {}
    assert "height_mm" in spec.inferred_defaults_mm
    assert spec.height_mm > 0  # a realistic, disclosed default -- not left at 0/unbuildable


def test_unrecognized_object_with_no_dimensions_is_honestly_unsupported():
    spec = compose_spec_from_prompt("a wobbly thing")
    assert spec.unsupported_reason is not None


# ---------------- 11. export/reimport ----------------

def test_generated_model_exports_and_reimports_correctly():
    import trimesh

    spec = MitsugenShapeSpec(object_description="vase", height_mm=120.0, waypoints_mm=[(0.0, 40.0), (1.0, 25.0)], max_slices=8)
    plan = plan_slices(spec)
    build = build_solid_from_slices(plan)
    exported = build.mesh.export(file_type="stl")
    import io

    reimported = trimesh.load_mesh(io.BytesIO(exported), file_type="stl")
    assert reimported.is_watertight
    assert np.allclose(sorted(reimported.extents), sorted(build.mesh.extents))
    assert abs(reimported.volume - build.mesh.volume) < 1.0


# ---------------- 12. opened in the Mitsugen viewer (backend half: server route) ----------------

def test_server_generate_route_returns_a_usable_mesh_url():
    import server

    server.app.config["TESTING"] = True
    with server.app.test_client() as c:
        res = c.post("/api/mitsugen/generate", json={
            "prompt": "a vase, 150mm tall, 60mm wide at the base, narrowing to 30mm at the top", "max_slices": 8,
        })
        assert res.status_code == 200
        data = res.get_json()
        assert data["mesh_url"] is not None
        mesh_res = c.get(data["mesh_url"])
        assert mesh_res.status_code == 200


def test_server_mitsugen_page_is_reachable():
    import server

    server.app.config["TESTING"] = True
    with server.app.test_client() as c:
        res = c.get("/mitsugen")
        assert res.status_code == 200


# ---------------- 13. Mitsugen's generation path is separate from 3di-men's assembly pipeline ----------------

def test_mitsugen_never_imports_3di_mens_cad_assembly_engine():
    """Explicit requirement: "Do not allow the Mitsugen UI to call
    3di-men's CAD assembly solver as its primary geometry-generation
    engine." Verified by inspecting the actual import graph of every
    mitsugen/*.py module, not asserted by convention alone."""
    import ast
    from pathlib import Path

    forbidden = {"sandbox", "program_model", "compositional_decomposer", "assembly_plan", "parallel_builder", "llm_decompose"}
    mitsugen_dir = Path(__file__).resolve().parent.parent / "mitsugen"
    found_imports = set()
    for py_file in mitsugen_dir.glob("*.py"):
        tree = ast.parse(py_file.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                found_imports.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                found_imports.add(node.module.split(".")[0])
    overlap = found_imports & forbidden
    assert overlap == set(), f"mitsugen/*.py imports 3di-men's CAD engine module(s): {overlap}"


# ---------------- 14. maximum construction slice count is never exceeded ----------------

@pytest.mark.parametrize("max_slices,waypoints", [
    (2, [(0.0, 30.0), (1.0, 30.0)]),
    (3, [(0.0, 50.0), (1.0, 10.0)]),
    (5, [(0.0, 50.0), (0.3, 60.0), (0.7, 20.0), (1.0, 40.0)]),
    (20, [(0.0, 50.0), (1.0, 10.0)]),
])
def test_actual_slice_count_never_exceeds_the_configured_maximum(max_slices, waypoints):
    spec = MitsugenShapeSpec(object_description="test", height_mm=100.0, waypoints_mm=waypoints, max_slices=max_slices)
    plan = plan_slices(spec)
    assert plan.actual_slice_count <= max_slices
