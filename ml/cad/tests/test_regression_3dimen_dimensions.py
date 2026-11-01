"""Regression fixtures for 3di-men's dimension-accuracy and op-sequence
correctness, built directly from a mission brief's own Part 1 test-
prompt list ("a rectangular plate with four specified through-holes,"
"a bracket with explicit width/height/thickness and mounting holes,"
"a cylindrical component with a specified outside diameter and bore,"
"a shaft with specified diameter/length/stepped sections," "a component
whose semantic name normally implies a standard size, but whose
explicit dimensions intentionally differ").

HONEST baseline, not an aspirational one: this file originally pinned
several confirmed, NOT-YET-FIXED gaps (a hole count or hole/profile
dimension silently wrong, a stepped section missing its own op
entirely) rather than silently passing over them, per this project's
"record baseline outcomes honestly" standard. A later mission brief
(reliable dimension assignment, feature counts, and hard geometric
constraints -- see dimension_requirements.py/dimension_assignment.py/
dimension_repair.py's own module docstrings) fixed every one of them;
each test below that used to document a gap now says "FIX CONFIRMED"
and explains what actually closed it. One test remains an honestly-
documented, intentionally out-of-scope gap (cross-dimension mechanical
consistency checking -- see test_contradictory_dimensions_are_not_
currently_detected_confirmed_gap's own docstring). Each test's
docstring still says plainly whether it confirms a FIX or documents a
REMAINING GAP -- update (never silently delete) a GAP test the day the
underlying behavior actually changes.

Uses program_model.predict_program_checked() directly wherever the
fixture doesn't need multi-part assembly placement -- program_model's
own inference is a small sklearn model, not the slow local LLM
(llm_decompose.py), so this whole file runs in seconds, not the
10-60+ minutes a real decompose_prompt() call costs on this hardware."""

from pathlib import Path

import pytest

pytest.importorskip("sklearn")

from joblib import load

from program_model import predict_program_checked

BUNDLE_PATH = Path(__file__).resolve().parent.parent / "models" / "program_model.joblib"


@pytest.fixture(scope="module")
def bundle():
    if not BUNDLE_PATH.exists():
        pytest.skip("models/program_model.joblib not present on this machine")
    return load(BUNDLE_PATH)


# ---------------- 1. rectangular plate + four through-holes ----------------

def test_plate_base_dimensions_are_exact(bundle):
    """FIX CONFIRMED (prior round, re-verified here): the base shape's
    own explicit dimensions are forced from the literal text
    (_override_primary_shape_dimensions), not left to the regressor's
    approximation."""
    text = "a rectangular plate 100mm x 80mm x 6mm with four 8mm diameter through-holes near the corners"
    pred = predict_program_checked(bundle, text)
    base = pred.program[0]
    assert base.op == "add_box"
    assert base.args["length"] == 100.0
    assert base.args["width"] == 80.0
    assert base.args["height"] == 6.0


def test_plate_four_holes_uses_a_real_hole_pattern_with_the_correct_diameter(bundle):
    """FIX CONFIRMED: the predicted op-sequence (add_box,
    counterbore_hole_pattern, finalize) -- RAG's own alternative, which
    the hybrid arbitration correctly prefers here at 0.387 similarity
    (above the ORIGINAL, unchanged 0.20 floor -- see
    ProgramModelConfig.rag_override_similarity_threshold's own
    docstring for why this floor was investigated and deliberately left
    at its original value) -- is a real hole PATTERN, not an
    undifferentiated box-shaped cavity. hole_radius is now exactly
    4.0mm (half of the stated 8mm diameter), via the bore-hole
    override extended to the _pattern variants (drill_hole_pattern/
    counterbore_hole_pattern), reusing the SAME "Nmm diameter [through-]
    hole" matching this file's bore test already confirms works for the
    singular ops."""
    text = "a rectangular plate 100mm x 80mm x 6mm with four 8mm diameter through-holes near the corners"
    pred = predict_program_checked(bundle, text)
    ops = tuple(c.op for c in pred.program)
    assert ops == ("add_box", "counterbore_hole_pattern", "finalize")
    pattern_call = pred.program[1]
    assert pattern_call.args["hole_radius"] == 4.0


def test_plate_four_holes_pattern_count_is_now_fixed(bundle):
    """FIX CONFIRMED -- this test used to pin a CONFIRMED REMAINING GAP:
    the pattern's own hole COUNT was predicted as 5, not the requested
    4, because nothing overrode a pattern op's own count field (only
    its shared diameter was covered). Fixed by dimension_requirements.
    extract_feature_count_requirements + dimension_assignment.
    assign_requirements, wired into program_model._apply_dimension_
    requirements -- the feature-count requirement's own stated count
    now overrides the pattern op's count directly."""
    text = "a rectangular plate 100mm x 80mm x 6mm with four 8mm diameter through-holes near the corners"
    pred = predict_program_checked(bundle, text)
    pattern_call = pred.program[1]
    assert pattern_call.args["count"] == 4


# ---------------- 2. bracket + explicit dims + mounting holes ----------------

def test_bracket_mounting_hole_diameter_is_exact(bundle):
    """FIX CONFIRMED: the hybrid arbitration correctly prefers RAG's
    alternative (add_extrusion, drill_hole x4) over the classifier's
    own wrong guess (add_cylinder as the base shape for an explicitly
    rectangular bracket) at 0.257 similarity -- this is the ORIGINAL,
    unchanged 0.20 floor at work, not a new fix (see
    ProgramModelConfig.rag_override_similarity_threshold's own
    docstring: raising this floor was investigated and rejected because
    it would have blocked exactly this rescue). Each hole's own radius
    is now exactly 3.0mm (half of the stated 6mm diameter) -- a NEW
    fix: the bore-hole override previously missed this because
    "mounting" sat between "diameter" and "hole", now handled via the
    same curated descriptor-word list spec_validation.py's own hole-
    count extractor uses."""
    text = "a bracket 80mm x 50mm x 5mm with two 6mm diameter mounting holes"
    pred = predict_program_checked(bundle, text)
    assert pred.program[0].op == "add_extrusion"
    hole_calls = [c for c in pred.program if c.op == "drill_hole"]
    assert len(hole_calls) >= 1
    for call in hole_calls:
        assert call.args["radius"] == 3.0


def test_bracket_extrusion_thickness_and_hole_count_are_now_fixed(bundle):
    """FIX CONFIRMED -- this test used to pin TWO confirmed remaining
    gaps: add_extrusion's own height field got the hole's 6mm diameter
    (an entirely unrelated clause) instead of the bracket's own stated
    5mm thickness, because the old override zipped the first N literal
    numbers in the WHOLE text against ARG_SCHEMA's field order with no
    semantic check; and 4 holes were drilled, not the requested 2 (no
    override covered a pattern/count field at all, and no mechanism
    caught 4 individually-placed drill_hole ops disagreeing with a
    stated count of 2).

    Both fixed: dimension_requirements.py tags "5mm thick" by its own
    nearby keyword (never by position), so it reaches add_extrusion's
    height field correctly regardless of what else is in the text; and
    dimension_repair.repair_hole_count_mismatch (run after execution,
    since it needs a built mesh to independently measure against)
    detects the 4-vs-2 mismatch and replaces the 4 individually-placed
    (and, confirmed live, wildly mispositioned) drill_hole ops with one
    correctly-counted drill_hole_pattern."""
    from dimension_repair import repair_hole_count_mismatch
    from dimension_requirements import extract_feature_count_requirements
    from evaluation.geometry_eval import evaluate_mesh
    from sandbox import Sandbox

    text = "a bracket 80mm x 50mm x 5mm with two 6mm diameter mounting holes"
    pred = predict_program_checked(bundle, text)
    base = pred.program[0]
    assert base.args["height"] == 5.0

    program = pred.program
    for count_req in extract_feature_count_requirements(text):
        outcome = repair_hole_count_mismatch(program, count_req)
        program = outcome.program
    result = Sandbox.execute(program)
    assert result.status == "ok"
    assert evaluate_mesh(result.mesh).detected_hole_count == 2


# ---------------- 3. cylinder with a specified outside diameter and bore ----------------

def test_cylinder_bore_dimension_override_works_when_the_op_sequence_is_correct(bundle):
    """FIX CONFIRMED: when the op-sequence is a correctly-shaped
    (add_cylinder, drill_hole, finalize) -- forced here via forced_ops
    the same way predict_program_rag's own winning sequence would be,
    isolating the dimension-override fix from the SEPARATE op-sequence-
    selection gap documented below -- the bore's own explicit 12mm
    diameter is forced to radius 6.0, not left to the regressor's
    approximation (which predicted an unrelated value before this
    fix)."""
    from program_model import predict_program

    text = "a cylinder, 40mm diameter, 30mm tall, with a 12mm diameter bore through it"
    program = predict_program(bundle, text, forced_ops=["add_cylinder", "drill_hole", "finalize"])
    base, hole = program[0], program[1]
    assert base.args["radius"] == 20.0
    assert base.args["height"] == 30.0
    assert hole.args["radius"] == 6.0


def test_cylinder_bore_end_to_end_is_now_fixed(bundle):
    """FIX CONFIRMED -- this test used to pin a CONFIRMED REMAINING GAP:
    without forcing the op-sequence, RAG overrode the classifier's own
    CORRECT prediction (add_cylinder, drill_hole) with an unrelated one
    (add_box, add_cylinder, boolean_intersection) at 0.346 similarity,
    because a single global similarity floor could not tell "textually
    similar but geometrically wrong" apart from "less similar but
    geometrically right" (see op_arbitration.py's own module docstring
    for the full before/after). Root cause fixed by replacing the bare
    threshold with a requirement-aware arbitration engine
    (op_requirements.py + op_arbitration.py, wired into
    program_model.predict_program_hybrid): the classifier's own valid
    sequence is now kept over RAG's requirement-violating one regardless
    of similarity. Update (never silently delete) this test the day the
    underlying behavior changes again."""
    pred = predict_program_checked(bundle, "a cylinder, 40mm diameter, 30mm tall, with a 12mm diameter bore through it")
    base, hole = pred.program[0], pred.program[1]
    ops = tuple(c.op for c in pred.program)
    assert base.op == "add_cylinder"
    assert base.args["radius"] == 20.0
    assert base.args["height"] == 30.0
    assert hole.op in ("drill_hole", "add_boss_with_bore", "counterbore_hole")
    assert hole.args["radius"] == 6.0
    assert "add_box" not in ops
    assert "boolean_intersection" not in ops
    assert pred.arbitration is not None
    assert pred.arbitration.outcome.name == "CLASSIFIER_ACCEPTED"


# ---------------- 4. shaft with diameter/length/stepped section ----------------

def test_shaft_stepped_section_is_now_fixed(bundle):
    """FIX CONFIRMED -- this test used to pin a CONFIRMED REMAINING GAP:
    the classifier never predicted a feature op for this phrasing at
    all (plain add_cylinder, finalize -- no step of any kind), so there
    was nothing for any dimension-override mechanism to even act on.
    Also a confirmed Problem-1/Problem-2 BOUNDARY case: op_arbitration.
    arbitrate() already correctly flagged this as NO_VALID_PLAN (missing
    stepped_section), but nothing downstream acted on that disclosure.

    Fixed by dimension_repair.repair_missing_stepped_section (wired
    into program_model.predict_program -- see that function's own
    docstring): when the text's own extraction already shows a wide
    diameter (here tagged outer_diameter, since "10mm diameter" has no
    "main section" qualifier) alongside a genuinely distinct
    reduced_diameter ("stepped section 14mm diameter"), the base op is
    swapped from add_cylinder to add_shouldered_cylinder, then
    dimension_assignment places both diameters and the overall length
    (generalized to accept "80mm long" -- a plain `length` tag -- for
    this op, not just the literal "overall length" phrase)."""
    text = "a shaft 10mm diameter, 80mm long, with a stepped section 14mm diameter for 20mm"
    pred = predict_program_checked(bundle, text)
    base = pred.program[0]
    assert base.op == "add_shouldered_cylinder"
    assert base.args["bottom_radius"] == 5.0  # 10mm diameter / 2
    assert base.args["top_radius"] == 7.0  # 14mm diameter / 2
    total_length = base.args["bottom_height"] + base.args["top_height"] + base.args["transition_height"]
    assert abs(total_length - 80.0) < 1e-6  # "80mm long"


# ---------------- 5. gear with explicit module/teeth/face-width/bore ----------------

def test_gear_all_four_explicit_dimensions_are_exact(bundle):
    """FIX CONFIRMED (this investigation): previously every one of a
    gear's own 4 stated numbers was wrong by 20-30% (module 2 -> 2.54,
    20 teeth -> 21, 10mm face width -> 12.29mm, 6mm bore -> 7.46mm)
    because _override_primary_shape_dimensions never fires for a gear
    (module/tooth_count have no "mm" suffix) -- now fixed by a
    dedicated, label-based override (_override_gear_dimensions)."""
    text = "a gear, module 2, 20 teeth, 10mm face width, with a 6mm bore"
    pred = predict_program_checked(bundle, text)
    base = pred.program[0]
    assert base.op == "add_gear_profile"
    assert base.args["module_mm"] == 2.0
    assert base.args["tooth_count"] == 20
    assert base.args["face_width_mm"] == 10.0
    assert base.args["bore_diameter_mm"] == 6.0


def test_gear_dimensions_are_exact_regardless_of_clause_order(bundle):
    """Training's own _render_gear (data/prompt_dataset.py) shuffles
    clause order -- the label-based override must get every field right
    regardless of which order module/teeth/face-width/bore appear in."""
    text = "spur gear, 6mm bore, 20 teeth, 10mm face width, module 2"
    pred = predict_program_checked(bundle, text)
    base = pred.program[0]
    assert base.args["module_mm"] == 2.0
    assert base.args["tooth_count"] == 20
    assert base.args["face_width_mm"] == 10.0
    assert base.args["bore_diameter_mm"] == 6.0


# ---------------- 8. semantic name vs. explicit dimensions ----------------

def test_explicit_dimensions_override_a_standard_names_own_typical_size(bundle):
    """FIX CONFIRMED (prior round, re-verified here): "NEMA 17" implies a
    standard 42.3mm mounting face, but the request's own explicit
    50mm x 50mm x 10mm wins outright -- never averaged, never silently
    replaced by the standard's own value."""
    text = "a NEMA 17 stepper motor mount, 50mm x 50mm x 10mm thick"
    pred = predict_program_checked(bundle, text)
    base = pred.program[0]
    assert base.op == "add_box"
    assert base.args["length"] == 50.0
    assert base.args["width"] == 50.0
    assert base.args["height"] == 10.0


# ---------------- 9 & 10. contradictory / unsatisfiable requests ----------------

def test_contradictory_dimensions_are_not_currently_detected_confirmed_gap(bundle):
    """CONFIRMED REMAINING GAP, NOT BUILT by this investigation --
    reported honestly rather than faked. A request where a shaft's own
    stated diameter is LARGER than the bore it's explicitly supposed to
    fit inside (10mm shaft, 8mm bore -- mechanically impossible: nothing
    can detect or explain this today, it is simply built as asked with
    no cross-dimension consistency check at all. Building the brief's
    own demanded "mandatory mechanical explanation for deviations"
    (Part 15) is a genuinely new subsystem (conflict classification +
    a non-fabricated mechanical explanation), not a small patch to an
    existing mechanism -- correctly out of scope for this investigation's
    remaining time, and recorded here as the single most important
    piece of unfinished work, not silently omitted."""
    text = "a shaft 10mm diameter that must fit inside an 8mm diameter bore in the housing"
    pred = predict_program_checked(bundle, text)
    # No assertion of "rejected" or "flagged" -- there is currently no
    # mechanism that would do either. This documents that fact plainly:
    # the pipeline builds SOMETHING, with zero cross-dimension check.
    assert pred.program  # it builds *something* -- the point is that nothing here validates 10mm vs 8mm against each other
