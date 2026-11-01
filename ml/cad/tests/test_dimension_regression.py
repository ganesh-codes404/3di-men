"""The 10 required regression tests (Part 9) for reliable dimension
assignment, feature counts, and hard geometric constraints -- run
against the REAL production entry point (predict_program_checked)
through the REAL trained bundle (models/program_model.joblib), never a
mocked classifier. Every test executes its final program through the
real Sandbox and independently measures the resulting geometry
(extents, hole count via evaluation.geometry_eval.evaluate_mesh's real
hole detection, hole diameters, hole depth/through-vs-blind) -- never
just a comparison of predicted ToolCall.args against what was asked
for, per this brief's own "do not mark a test passed merely because
parsing returned the correct values" rule.

Reproduces, and confirms fixed, the exact 3 failures this task's own
audit found live against this bundle (see dimension_requirements.py's
and dimension_assignment.py's own module docstrings for the full
before/after of each):

Failure A ("a 5mm thick base and two 6mm mounting holes"): add_extrusion
predicted height=58.3mm for a stated 5mm, because the old override
required the text to supply as many literal numbers as the op has
numeric slots before overriding ANY of them.

Failure B (a rectangular plate with a stated hole count): feature counts
were asserted only on the op's own `count` ARGUMENT, never independently
checked against the number of holes the built geometry actually has.

Failure C ("40mm outer diameter, 12mm through-bore, and 30mm height"):
add_cylinder predicted height=12mm (the BORE's own diameter, not the
base shape's stated height) -- the old override zipped literals in
document order against ARG_SCHEMA's field order with no semantic check.
"""

from pathlib import Path

import pytest

from dimension_repair import repair_hole_count_mismatch, repair_missing_stepped_section
from dimension_requirements import extract_dimension_requirements, extract_feature_count_requirements
from evaluation.geometry_eval import evaluate_mesh
from program_model import ProgramModelBundle, predict_program_checked
from sandbox import Sandbox

BUNDLE_PATH = Path(__file__).resolve().parent.parent / "models" / "program_model.joblib"


@pytest.fixture(scope="module")
def bundle() -> ProgramModelBundle:
    if not BUNDLE_PATH.exists():
        pytest.skip(f"no trained bundle at {BUNDLE_PATH} -- run scripts/train_program_model.py first")
    return ProgramModelBundle.load(BUNDLE_PATH)


def _build_and_repair(bundle: ProgramModelBundle, text: str):
    """The full production-intended pipeline: predict (which already
    runs dimension assignment + the stepped-section repair internally,
    see program_model._apply_dimension_requirements), then repair any
    hole-count/placement mismatch (which needs a built mesh to check
    against, so it runs here, after prediction, not inside it)."""
    pred = predict_program_checked(bundle, text)
    program = pred.program
    for count_req in extract_feature_count_requirements(text):
        outcome = repair_hole_count_mismatch(program, count_req)
        program = outcome.program
    result = Sandbox.execute(program)
    report = evaluate_mesh(result.mesh) if result.mesh is not None else None
    return program, result, report


# ---------------- Test 1: numeric assignment ----------------

def test_1_numeric_assignment_thickness_and_hole_diameter_and_count(bundle):
    text = "A bracket with a 5 mm thick base and two 6 mm mounting holes."
    program, result, report = _build_and_repair(bundle, text)
    assert result.status == "ok"
    assert abs(result.mesh.extents[2] - 5.0) < 0.5  # thickness
    assert report.detected_hole_count == 2
    assert all(abs(d - 6.0) < 1.0 for d in report.hole_diameters_mm)


# ---------------- Test 2: four-hole pattern ----------------

def test_2_four_hole_pattern_near_each_corner(bundle):
    text = "A rectangular mounting plate measuring 120 mm by 80 mm and 5 mm thick, with four 6 mm holes, one near each corner."
    program, result, report = _build_and_repair(bundle, text)
    assert result.status == "ok"
    assert abs(result.mesh.extents[0] - 120.0) < 0.5
    assert abs(result.mesh.extents[1] - 80.0) < 0.5
    assert abs(result.mesh.extents[2] - 5.0) < 0.5
    assert report.detected_hole_count == 4  # exactly 4 -- no unexpected 5th hole
    assert all(abs(d - 6.0) < 1.0 for d in report.hole_diameters_mm)


# ---------------- Test 3: cylinder with bore ----------------

def test_3_cylinder_outer_diameter_bore_and_height_never_cross_contaminate(bundle):
    """The headline confirmed failure (Failure C): the bore's own 12mm
    used to leak into the base cylinder's height field."""
    text = "A cylindrical component with 40 mm outer diameter, 12 mm through-bore, and 30 mm height."
    program, result, report = _build_and_repair(bundle, text)
    assert result.status == "ok"
    assert abs(result.mesh.extents[0] - 40.0) < 0.5
    assert abs(result.mesh.extents[1] - 40.0) < 0.5
    assert abs(result.mesh.extents[2] - 30.0) < 0.5  # never 12mm
    assert any(abs(d - 12.0) < 1.5 for d in report.hole_diameters_mm)
    assert any(report.hole_through)  # the bore passes through the entire component


# ---------------- Test 4: blind hole ----------------

def test_4_blind_hole_leaves_material_at_the_bottom(bundle):
    text = "A 50 mm diameter, 20 mm tall cylinder with a 10 mm diameter blind hole, 8 mm deep."
    program, result, report = _build_and_repair(bundle, text)
    assert result.status == "ok"
    assert abs(result.mesh.extents[0] - 50.0) < 0.5
    assert abs(result.mesh.extents[1] - 50.0) < 0.5
    assert abs(result.mesh.extents[2] - 20.0) < 0.5
    assert any(abs(d - 10.0) < 1.5 for d in report.hole_diameters_mm)
    assert report.hole_through == [False]  # blind -- never pierces through


# ---------------- Test 5: stepped shaft ----------------

def test_5_stepped_shaft_preserves_both_diameters_and_overall_length(bundle):
    """Confirmed Problem-1/Problem-2 boundary case: op_arbitration.
    arbitrate() already correctly flags the classifier's plain-cylinder
    prediction as NO_VALID_PLAN (missing stepped_section), but nothing
    downstream used to act on that disclosure -- dimension_repair.
    repair_missing_stepped_section (wired into predict_program itself)
    is what actually swaps in add_shouldered_cylinder."""
    text = "A shaft with a 20 mm diameter main section, 12 mm diameter stepped section, and 100 mm overall length."
    program, result, report = _build_and_repair(bundle, text)
    assert result.status == "ok"
    assert any(c.op == "add_shouldered_cylinder" for c in program)  # the required step exists
    assert abs(result.mesh.extents[0] - 20.0) < 0.5  # 2 * bottom_radius -- the main (wider) diameter
    assert abs(result.mesh.extents[2] - 100.0) < 0.5  # overall length
    # both diameters preserved: the narrow end must be measurably
    # narrower than the wide end, not a uniform cylinder
    zmin, zmax = result.mesh.bounds[0][2], result.mesh.bounds[1][2]
    import numpy as np

    def radius_at(z):
        section = result.mesh.section(plane_origin=[0, 0, z], plane_normal=[0, 0, 1])
        return float(np.linalg.norm(section.vertices[:, :2], axis=1).mean())

    r_wide = radius_at(zmin + 0.02 * (zmax - zmin))
    r_narrow = radius_at(zmax - 0.02 * (zmax - zmin))
    assert r_wide > r_narrow + 2.0


# ---------------- Test 6: repeated linear holes ----------------

def test_6_five_equally_spaced_linear_holes(bundle):
    text = "A 100 mm long plate with five 4 mm diameter holes equally spaced along its centreline."
    program, result, report = _build_and_repair(bundle, text)
    assert result.status == "ok"
    assert abs(result.mesh.extents[0] - 100.0) < 0.5
    assert report.detected_hole_count == 5
    assert all(abs(d - 4.0) < 1.0 for d in report.hole_diameters_mm)
    # "equally spaced" -- the pattern op's own type must actually be a
    # linear row, not corner_array (which cannot place 5 points at all)
    assert any(c.op == "drill_hole_pattern" and c.args.get("pattern") == "linear_array" for c in program)


# ---------------- Test 7: unit conversion ----------------

def test_7_unit_conversion_cm_to_mm(bundle):
    text = "A plate 10 cm long, 50 mm wide and 5 mm thick."
    program, result, report = _build_and_repair(bundle, text)
    assert result.status == "ok"
    assert abs(result.mesh.extents[0] - 100.0) < 0.5  # 10cm -> 100mm
    assert abs(result.mesh.extents[1] - 50.0) < 0.5
    assert abs(result.mesh.extents[2] - 5.0) < 0.5


# ---------------- Test 8: ambiguous input ----------------

def test_8_ambiguous_dimension_is_exposed_not_invented(bundle):
    text = "A 20 mm bracket with a 6 mm hole."
    dimension_reqs = extract_dimension_requirements(text)
    ambiguous = [r for r in dimension_reqs if r.value == 20.0]
    assert len(ambiguous) == 1
    assert ambiguous[0].status.name == "UNRESOLVED"  # never silently decided to be thickness/length/anything else

    unambiguous = [r for r in dimension_reqs if r.value == 6.0]
    assert unambiguous[0].parameter == "hole_diameter"

    program, result, report = _build_and_repair(bundle, text)
    assert result.status == "ok"
    assert report.detected_hole_count == 1
    assert any(abs(d - 6.0) < 1.0 for d in report.hole_diameters_mm)


# ---------------- Test 9: adversarial numeric proximity ----------------

def test_9_adversarial_proximity_thickness_diameter_count_and_offset_stay_separate(bundle):
    text = "A mounting plate, 5 mm thick, with four holes of 6 mm diameter and 20 mm edge offsets."
    program, result, report = _build_and_repair(bundle, text)
    assert result.status == "ok"
    assert abs(result.mesh.extents[2] - 5.0) < 0.5
    assert report.detected_hole_count == 4
    assert all(abs(d - 6.0) < 1.0 for d in report.hole_diameters_mm)
    # the 20mm edge offset must have landed on the pattern's own margin,
    # never on thickness or hole diameter
    pattern_call = next(c for c in program if c.op in ("drill_hole_pattern", "counterbore_hole_pattern"))
    assert abs(pattern_call.args["margin"] - 20.0) < 0.5


# ---------------- Test 10: explicit dimensions vs defaults ----------------

def test_10_explicit_dimensions_are_never_overwritten_by_defaults(bundle):
    text = "A 70 mm by 40 mm plate with a 3 mm thickness and three 8 mm holes."
    program, result, report = _build_and_repair(bundle, text)
    assert result.status == "ok"
    assert abs(result.mesh.extents[0] - 70.0) < 0.5
    assert abs(result.mesh.extents[1] - 40.0) < 0.5
    assert abs(result.mesh.extents[2] - 3.0) < 0.5
    assert report.detected_hole_count == 3  # exactly 3 -- no stray 4th hole left over from the raw prediction
    assert all(abs(d - 8.0) < 1.0 for d in report.hole_diameters_mm)
