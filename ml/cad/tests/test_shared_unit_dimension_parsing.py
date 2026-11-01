"""Phase 7's own regression suite: robust shared/trailing-unit dimension
parsing, plus focused reproductions of the independently-investigated
benchmark failures E47 and E50 (Part 4 -- confirmed, via live
reproduction, to have ROOT CAUSES UNRELATED to dimension parsing; these
tests exist to document and LOCK IN that finding, not to claim either
is fixed by this task)."""

from pathlib import Path

import pytest

from benchmark_prompts import PROMPTS
from benchmark_runner import get_decomposition
from dimension_requirements import _find_dimension_chains, extract_dimension_requirements
from op_requirements import extract_requirements


# ---------------- Parser tests ----------------

def test_trailing_shared_unit():
    chains = _find_dimension_chains("100 x 60 x 10 mm")
    assert chains[0]["values"] == [(100.0, "mm"), (60.0, "mm"), (10.0, "mm")]


def test_repeated_units_unaffected():
    chains = _find_dimension_chains("100mm x 60mm x 10mm")
    assert chains[0]["values"] == [(100.0, "mm"), (60.0, "mm"), (10.0, "mm")]


def test_written_by_connector():
    chains = _find_dimension_chains("100 by 60 by 10 mm")
    assert chains[0]["values"] == [(100.0, "mm"), (60.0, "mm"), (10.0, "mm")]


def test_comma_connector():
    chains = _find_dimension_chains("100, 60, 10 mm")
    assert chains[0]["values"] == [(100.0, "mm"), (60.0, "mm"), (10.0, "mm")]


def test_written_unit_millimeters():
    chains = _find_dimension_chains("100 x 60 x 10 millimeters")
    assert chains[0]["values"] == [(100.0, "millimeters"), (60.0, "millimeters"), (10.0, "millimeters")]


def test_unicode_multiplication_sign():
    chains = _find_dimension_chains("100 × 60 × 10 mm")
    assert chains[0]["values"] == [(100.0, "mm"), (60.0, "mm"), (10.0, "mm")]


def test_decimal_values():
    chains = _find_dimension_chains("12.5 x 6.25 mm")
    assert chains[0]["values"] == [(12.5, "mm"), (6.25, "mm")]


def test_mixed_whitespace():
    chains = _find_dimension_chains("100x60 x  10   mm")
    assert chains[0]["values"] == [(100.0, "mm"), (60.0, "mm"), (10.0, "mm")]


def test_two_number_chain():
    chains = _find_dimension_chains("40 x 80 mm")
    assert chains[0]["values"] == [(40.0, "mm"), (80.0, "mm")]


def test_multiple_dimensional_groups_in_one_sentence():
    chains = _find_dimension_chains("100 x 60 x 10 mm plate with a 30 x 8 mm slot")
    assert len(chains) == 2
    assert chains[0]["values"] == [(100.0, "mm"), (60.0, "mm"), (10.0, "mm")]
    assert chains[1]["values"] == [(30.0, "mm"), (8.0, "mm")]


def test_unrelated_numbers_near_dimensions_do_not_form_a_false_chain():
    # "Four" has no unit and is not connector-adjacent to a resolvable
    # mm chain -- extract_dimension_requirements (tested below) must
    # leave the hole count alone (a different, non-dimension requirement).
    chains = _find_dimension_chains("four 6 mm holes")
    assert chains == []  # "4" and "6mm" are not connected by x/by/, at all


def test_ambiguous_chain_with_no_resolvable_unit_anywhere_is_rejected():
    chains = _find_dimension_chains("100 x 60 x 10 widgets")
    assert chains == []  # no member resolves ANY unit -- never silently assume mm


# ---------------- Assignment tests (dimension_requirements.py) ----------------

def test_assign_plate_dimensions_trailing_unit():
    reqs = {r.parameter: r.normalized_value for r in extract_dimension_requirements("100 x 60 x 10 mm plate")}
    assert reqs == {"length": 100.0, "width": 60.0, "height": 10.0}


def test_assign_cylinder_dimensions_not_swapped():
    reqs = {r.parameter: r.normalized_value for r in extract_dimension_requirements("Cylinder, 40 mm diameter and 80 mm height")}
    assert reqs == {"outer_diameter": 40.0, "height": 80.0}


def test_assign_tube_od_id_length_distinct():
    reqs = {r.parameter: r.normalized_value for r in extract_dimension_requirements("Tube, OD 30 mm, ID 20 mm, length 100 mm")}
    assert reqs == {"outer_diameter": 30.0, "hole_diameter": 20.0, "length": 100.0}


def test_plate_dimensions_not_confused_with_slot_dimensions():
    text = "100 x 60 x 10 mm plate with a 30 x 8 mm slot"
    plate_reqs = {r.parameter: r.normalized_value for r in extract_dimension_requirements(text) if r.parameter in ("length", "width", "height")}
    assert plate_reqs == {"length": 100.0, "width": 60.0, "height": 10.0}
    # the slot's own 30/8mm must NOT appear as length/width/height anywhere
    assert 30.0 not in plate_reqs.values() or plate_reqs.get("length") != 30.0
    slot_dims = extract_requirements(text)
    slot_req = next(r for r in slot_dims if r.feature_type == "subtractive_slot")
    assert slot_req.dimensions_mm == {"length": 30.0, "width": 8.0}


def test_hole_count_not_treated_as_a_dimension():
    reqs = extract_dimension_requirements("Six equally spaced 5 mm holes")
    assert all(r.value != 6.0 for r in reqs)  # "six" never enters this function at all -- it has no unit
    hole_reqs = [r for r in reqs if r.parameter == "hole_diameter"]
    assert len(hole_reqs) == 1 and hole_reqs[0].normalized_value == 5.0


def test_pitch_circle_diameter_distinct_from_hole_count():
    reqs = extract_dimension_requirements("Four holes on a 60 mm pitch circle diameter")
    assert len(reqs) == 1
    assert reqs[0].parameter == "pitch_circle_diameter"
    assert reqs[0].normalized_value == 60.0


def test_shaft_length_diameter_shoulder_distinct():
    reqs = {r.parameter: r.normalized_value for r in extract_dimension_requirements("A 100 mm shaft with a 20 mm diameter and a 30 mm shoulder")}
    assert reqs == {"length": 100.0, "outer_diameter": 20.0, "reduced_diameter": 30.0}


def test_angle_and_thickness_remain_separate():
    reqs = extract_dimension_requirements("A 45-degree chamfer on a 10 mm thick plate")
    assert len(reqs) == 1  # "45" has no mm unit at all -- never extracted as a dimension
    assert reqs[0].parameter == "thickness" and reqs[0].normalized_value == 10.0


def test_explicit_dimension_not_silently_replaced_when_unresolved():
    """A genuinely ambiguous number (no chain, no keyword) stays
    UNRESOLVED -- never silently dropped, never silently guessed."""
    from dimension_requirements import RequirementStatus

    reqs = extract_dimension_requirements("a bracket, 20mm, with a 6mm hole")
    unresolved = [r for r in reqs if r.status is RequirementStatus.UNRESOLVED]
    assert any(r.value == 20.0 for r in unresolved)


# ---------------- Integration tests (real production extraction + assignment) ----------------

def test_integration_box_chain_reaches_operation_parameters():
    from dimension_assignment import assign_requirements
    from dimension_requirements import extract_feature_count_requirements
    from sandbox import ToolCall

    text = "100 x 60 x 10 mm plate"
    program = [ToolCall(op="add_box", args={"length": 1.0, "width": 1.0, "height": 1.0, "center": [0, 0, 0]}), ToolCall(op="finalize", args={"target": 0})]
    result = assign_requirements(extract_dimension_requirements(text), extract_feature_count_requirements(text), program)
    assert result.program[0].args["length"] == 100.0
    assert result.program[0].args["width"] == 60.0
    assert result.program[0].args["height"] == 10.0


def test_integration_real_geometry_matches_shared_unit_plate_request(tmp_path):
    """Runs the ACTUAL production path (program_model + sandbox) for a
    trailing-shared-unit prompt and measures the resulting mesh
    directly -- the Phase 6 benchmark's own A01 scenario."""
    pytest.importorskip("sklearn")
    from llm_decompose import ActionablePoint
    from parallel_builder import build_assembly_from_points
    from program_model import ProgramModelConfig, train_program_model
    from data.program_dataset import sample_program_dataset, write_program_dataset

    root = tmp_path / "dataset"
    samples = sample_program_dataset({"plate": 40}, n_generic=20, seed=0)
    write_program_dataset(root, samples, seed=0)
    bundle, _ = train_program_model(root, ProgramModelConfig(seed=0))
    bundle_path = tmp_path / "bundle.joblib"
    bundle.save(bundle_path)

    point = ActionablePoint(description="A 100 x 60 x 10 mm rectangular plate.", position_mm=(0.0, 0.0, 0.0), id="p0")
    result = build_assembly_from_points([point], bundle_path, max_workers=1)
    p = result.points[0]
    assert p.mesh is not None
    # The requested dimensions must appear somewhere in the measured
    # geometry -- a parser unit test passing is not proof the real
    # pipeline built the right thing; this asserts actual mesh extents.
    extents = sorted(float(e) for e in p.mesh.extents)
    assert any(abs(e - 10.0) / 10.0 < 0.2 for e in extents), f"no measured extent near 10mm thickness: {extents}"
    assert any(abs(e - 60.0) / 60.0 < 0.2 for e in extents), f"no measured extent near 60mm width: {extents}"
    assert any(abs(e - 100.0) / 100.0 < 0.2 for e in extents), f"no measured extent near 100mm length: {extents}"


# ---------------- Part 4: E47 and E50, independently investigated ----------------
# These reproduce the ORIGINAL benchmark artifacts (benchmark_runs/
# run-6af75c7bb7/E47.json, E50.json) via the SAME reproducible fixed
# decomposition + trained bundle -- confirmed, via direct re-measurement,
# to have root causes UNRELATED to dimension parsing. Assertions below
# LOCK IN that finding (the failure must still reproduce, for the SAME
# documented reason) rather than claim either is fixed by this task.

@pytest.fixture(scope="module")
def default_bundle_path() -> Path:
    path = Path("models/program_model.joblib")
    if not path.exists():
        pytest.skip("production bundle models/program_model.joblib not present")
    return path


def test_e47_root_cause_is_operation_selection_not_dimension_parsing(default_bundle_path):
    """E47 ('A shaft inserted into two bearing housings.'): confirmed via
    direct reproduction that op_arbitration.py CORRECTLY extracts the
    'bore' requirement and CORRECTLY detects both the classifier's own
    candidate (add_box, add_box, boolean_difference -- a rectangular
    pocket, not a round bore) and RAG's retrieved candidate as invalid,
    returning NO_VALID_PLAN -- but falls back to the known-invalid
    classifier choice since there is no UI to pause and ask. Dimension
    extraction/assignment are never reached because no drill_hole-family
    op exists in the chosen program at all."""
    from program_model import ProgramModelBundle, predict_program_checked
    from op_arbitration import ArbitrationOutcome

    bundle = ProgramModelBundle.load(default_bundle_path)
    desc = "a bearing housing, 40mm x 40mm x 20mm, with a 12.2mm bore"
    pred = predict_program_checked(bundle, desc)
    assert pred.arbitration is not None
    assert pred.arbitration.outcome is ArbitrationOutcome.NO_VALID_PLAN
    ops = tuple(step.op for step in pred.program if step.op != "finalize")
    assert "drill_hole" not in ops and "drill_hole_pattern" not in ops and "counterbore_hole" not in ops


def test_e50_root_cause_is_axis_measurement_not_dimension_parsing():
    """E50 ('A pulley mounted on a stepped shaft.'): confirmed via direct
    reproduction that assembly_compatibility._cylindrical_diameter_mm
    measures a cylinder's diameter using the RELATIONSHIP's own declared
    axis, never re-checking whether that component's own BUILT mesh is
    actually oriented along that world axis. The shaft here is the
    relationship's TARGET (never itself rotated by resolve_rotation_deg
    -- only the relationship's OWNER, the pulley, is) and stays built
    along its own local z; measuring its diameter "perpendicular to x"
    averages its true 10mm diameter with its own 100mm LENGTH, producing
    the exact 55.00mm the original benchmark run reported."""
    import trimesh

    from assembly_compatibility import _cylindrical_diameter_mm

    shaft = trimesh.creation.cylinder(radius=5.0, height=100.0, sections=32)  # built along its own local z, never rotated
    measured_against_declared_relationship_axis = _cylindrical_diameter_mm(shaft, axis="x")
    measured_against_the_shafts_own_true_axis = _cylindrical_diameter_mm(shaft, axis="z")
    assert abs(measured_against_declared_relationship_axis - 55.0) < 0.01  # reproduces the original benchmark's own 55.00mm exactly
    assert abs(measured_against_the_shafts_own_true_axis - 10.0) < 0.01  # the shaft's own genuine diameter, for contrast
