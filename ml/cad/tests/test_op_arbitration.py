"""Direct tests of op_arbitration.py's own decision logic -- no model
training/loading (that's tests/test_program_model.py's
predict_program_hybrid tests, which exercise this through the real
pipeline). These tests pin down the exact decision tree
arbitrate()/evaluate_candidate() implement using hand-built op-
sequences and texts, including the THREE real, previously-confirmed
production failure/success cases this engine exists to fix (see
op_requirements.py's and op_arbitration.py's own module docstrings for
the live-reproduced evidence behind each one)."""

from op_arbitration import ArbitrationOutcome, evaluate_candidate, arbitrate
from op_requirements import extract_requirements


# ---------------- evaluate_candidate ----------------

def test_candidate_with_wrong_base_geometry_is_invalid():
    text = "a rectangular plate 100mm x 80mm x 6mm with four 8mm diameter through-holes near the corners"
    reqs = extract_requirements(text)
    ev = evaluate_candidate(("add_cylinder", "drill_hole", "finalize"), reqs)
    assert not ev.valid
    assert any("base geometry" in v for v in ev.violations)


def test_candidate_missing_a_required_feature_is_invalid():
    text = "a cylinder, 40mm diameter, 30mm tall, with a 12mm diameter bore through it"
    reqs = extract_requirements(text)
    ev = evaluate_candidate(("add_cylinder", "finalize"), reqs)
    assert not ev.valid
    assert any("subtractive_hole" in v for v in ev.violations)


def test_candidate_satisfying_every_requirement_is_valid_with_full_coverage():
    text = "a cylinder, 40mm diameter, 30mm tall, with a 12mm diameter bore through it"
    reqs = extract_requirements(text)
    ev = evaluate_candidate(("add_cylinder", "drill_hole", "finalize"), reqs)
    assert ev.valid
    assert ev.requirement_coverage == 1.0


def test_candidate_with_no_requirements_to_check_is_always_valid():
    ev = evaluate_candidate(("add_sphere", "finalize"), [])
    assert ev.valid
    assert ev.requirement_coverage == 1.0


def test_under_count_without_a_pattern_op_is_a_violation():
    text = "a rectangular plate 100mm x 80mm x 6mm with four 8mm diameter through-holes near the corners"
    reqs = extract_requirements(text)
    ev = evaluate_candidate(("add_box", "drill_hole", "finalize"), reqs)  # only 1 drill_hole, text asked for 4, no pattern op
    assert not ev.valid
    assert any("requires 4" in v for v in ev.violations)


def test_under_count_is_not_flagged_when_a_pattern_op_is_present():
    text = "a rectangular plate 100mm x 80mm x 6mm with four 8mm diameter through-holes near the corners"
    reqs = extract_requirements(text)
    ev = evaluate_candidate(("add_box", "drill_hole_pattern", "finalize"), reqs)
    assert ev.valid


def test_op_with_no_registered_geometry_kind_is_never_rejected_for_base_geometry():
    # add_loft/add_sweep are deliberately absent from CREATES_GEOMETRY_KIND
    text = "a cylinder, 40mm diameter, 30mm tall"
    reqs = extract_requirements(text)
    ev = evaluate_candidate(("add_loft", "finalize"), reqs)
    assert ev.valid


# ---------------- arbitrate(): the three real production cases ----------------

def test_cylinder_with_bore_keeps_the_correct_classifier_answer_over_a_wrong_but_similar_rag_match():
    """The confirmed-broken case this whole engine exists to fix: the
    classifier is RIGHT, RAG's retrieved neighbor is WRONG despite a
    textually-plausible 0.346 similarity. Must resolve to
    CLASSIFIER_ACCEPTED, never RAG's box+intersection pattern."""
    text = "a cylinder, 40mm diameter, 30mm tall, with a 12mm diameter bore through it"
    classifier_ops = ("add_cylinder", "drill_hole", "finalize")
    rag_ops = ("add_box", "add_cylinder", "boolean_intersection", "finalize")
    result = arbitrate(classifier_ops, rag_ops, 0.346, text)
    assert result.outcome is ArbitrationOutcome.CLASSIFIER_ACCEPTED
    assert result.chosen_ops == classifier_ops


def test_plate_hole_pattern_rag_rescue_is_preserved():
    """One of the two confirmed-GOOD existing RAG rescues: the classifier
    forgets the hole pattern entirely; RAG's retrieved sequence has it.
    Must still resolve to RAG_RESCUE -- this is the regression this
    engine must never break while fixing the cylinder case above."""
    text = "a rectangular plate 100mm x 80mm x 6mm with four 8mm diameter through-holes near the corners"
    classifier_ops = ("add_box", "add_box", "boolean_difference", "finalize")
    rag_ops = ("add_box", "counterbore_hole_pattern", "finalize")
    result = arbitrate(classifier_ops, rag_ops, 0.387, text)
    assert result.outcome is ArbitrationOutcome.RAG_RESCUE
    assert result.chosen_ops == rag_ops


def test_bracket_base_shape_rag_rescue_is_preserved_even_at_lower_similarity_than_the_bad_cylinder_case():
    """The other confirmed-GOOD existing RAG rescue, and the one that
    proves similarity magnitude alone could never have been the right
    discriminator: this rescue's own similarity (0.257) is LOWER than
    the bad cylinder override's (0.346) above, yet this one must be
    kept and that one must be rejected -- requirement compatibility,
    not similarity, is what correctly separates them."""
    text = "a bracket 80mm x 50mm x 5mm with two 6mm diameter mounting holes"
    classifier_ops = ("add_cylinder", "drill_hole", "drill_hole_pattern", "finalize")  # wrong base geometry for a rectangular bracket
    rag_ops = ("add_extrusion", "drill_hole", "drill_hole", "drill_hole", "drill_hole", "finalize")
    result = arbitrate(classifier_ops, rag_ops, 0.257, text)
    assert result.outcome is ArbitrationOutcome.RAG_RESCUE
    assert result.chosen_ops == rag_ops


# ---------------- arbitrate(): the rest of the decision tree ----------------

def test_identical_sequences_short_circuit_to_classifier_accepted():
    text = "a cylinder, 40mm diameter, 30mm tall"
    ops = ("add_cylinder", "finalize")
    result = arbitrate(ops, ops, 0.9, text)
    assert result.outcome is ArbitrationOutcome.CLASSIFIER_ACCEPTED
    assert result.chosen_ops == ops


def test_both_invalid_returns_no_valid_plan_but_still_a_usable_fallback():
    text = "a rectangular plate 100mm x 80mm x 6mm with four 8mm diameter through-holes near the corners"
    classifier_ops = ("add_cylinder", "finalize")  # wrong base geometry, no hole feature
    rag_ops = ("add_sphere", "finalize")  # also wrong base geometry, no hole feature
    result = arbitrate(classifier_ops, rag_ops, 0.9, text)
    assert result.outcome is ArbitrationOutcome.NO_VALID_PLAN
    assert result.chosen_ops == classifier_ops  # disclosed, non-crashing default -- never None


def test_valid_classifier_beats_an_invalid_rag_regardless_of_similarity():
    text = "a cylinder, 40mm diameter, 30mm tall, with a 12mm diameter bore through it"
    classifier_ops = ("add_cylinder", "drill_hole", "finalize")
    rag_ops = ("add_box", "finalize")  # wrong base geometry, no hole feature
    result = arbitrate(classifier_ops, rag_ops, 0.99, text)  # even a very high similarity must not win
    assert result.outcome is ArbitrationOutcome.CLASSIFIER_ACCEPTED
    assert result.chosen_ops == classifier_ops


def test_an_optional_unsatisfied_requirement_lowers_coverage_without_invalidating_the_candidate():
    """Every requirement extract_requirements() currently produces is
    `required=True` (missing one is a hard violation -- see
    test_candidate_missing_a_required_feature_is_invalid above), so in
    today's real pipeline `valid` and `coverage == 1.0` always coincide;
    the coverage-based (as opposed to length-based) tier of arbitrate()'s
    both-valid tie-break is consequently unreachable through the public
    arbitrate(text) path right now -- a disclosed characteristic of the
    current requirement model, not a bug (see op_arbitration.py's own
    module docstring). This pins the scoring primitive itself directly,
    for the day a `required=False` ("nice to have") requirement type
    gets added."""
    from op_requirements import Requirement

    reqs = [
        Requirement(id="base_geometry", feature_type="base_geometry", geometry_type="cylindrical", explicit=True, source_text="cylinder"),
        Requirement(id="edge_treatment", feature_type="edge_treatment", geometry_type=None, explicit=True, source_text="chamfered", required=False, compatible_ops=frozenset({"chamfer_edge"})),
    ]
    without_optional = evaluate_candidate(("add_cylinder", "finalize"), reqs)
    with_optional = evaluate_candidate(("add_cylinder", "chamfer_edge", "finalize"), reqs)
    assert without_optional.valid and with_optional.valid
    assert without_optional.requirement_coverage < with_optional.requirement_coverage == 1.0


def test_both_valid_equal_coverage_shorter_rag_sequence_wins_as_rag_assisted():
    text = "a cylinder, 40mm diameter, 30mm tall, with a 12mm diameter bore through it"
    classifier_ops = ("add_cylinder", "drill_hole", "add_dome", "finalize")  # valid but longer than needed
    rag_ops = ("add_cylinder", "drill_hole", "finalize")
    result = arbitrate(classifier_ops, rag_ops, 0.5, text)
    assert result.outcome is ArbitrationOutcome.RAG_ASSISTED
    assert result.chosen_ops == rag_ops


def test_both_valid_equal_coverage_and_length_but_different_sequences_needs_disambiguation():
    text = "a cylinder, 40mm diameter, 30mm tall, with a 12mm diameter bore through it"
    classifier_ops = ("add_cylinder", "drill_hole", "finalize")
    rag_ops = ("add_cylinder", "counterbore_hole", "finalize")  # same length, both satisfy the bore requirement, genuinely different op
    result = arbitrate(classifier_ops, rag_ops, 0.5, text)
    assert result.outcome is ArbitrationOutcome.NEEDS_DISAMBIGUATION
    assert result.chosen_ops == classifier_ops  # disclosed ambiguity, not a silent invented decision


def test_classifier_and_rag_agreement_does_not_bypass_validity_checking():
    """Confirmed real gap (live production bundle): "a shaft with a
    chamfered edge, 25mm diameter, 120mm long" -- the classifier AND RAG
    independently land on the SAME (add_cylinder, drill_hole, finalize)
    sequence, which silently drops the requested chamfer entirely.
    Agreement must not be treated as automatic validity; the shared
    sequence still has to satisfy the text's own requirements."""
    text = "a shaft with a chamfered edge, 25mm diameter, 120mm long"
    ops = ("add_cylinder", "drill_hole", "finalize")  # no chamfer_edge/chamfer_box_edges anywhere
    result = arbitrate(ops, ops, 0.9, text)
    assert result.outcome is ArbitrationOutcome.NO_VALID_PLAN


def test_classifier_and_rag_agreement_on_a_valid_sequence_is_still_accepted_for_free():
    text = "a cylinder, 40mm diameter, 30mm tall"
    ops = ("add_cylinder", "finalize")
    result = arbitrate(ops, ops, 0.9, text)
    assert result.outcome is ArbitrationOutcome.CLASSIFIER_ACCEPTED
    assert result.chosen_ops == ops


def test_merged_plan_outcome_is_defined_but_never_produced_by_the_current_engine():
    # Documents a known, disclosed limitation (see op_arbitration.py's own
    # module docstring) rather than silently dropping the enum value the
    # brief requires to exist.
    assert ArbitrationOutcome.MERGED_PLAN.value == "MERGED_PLAN"
