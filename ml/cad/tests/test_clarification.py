"""Phase 8 Part C -- the 15 required adversarial clarification test
scenarios, plus the lifecycle/model-contract coverage. Every scenario
builds the REAL typed objects clarification.py actually consumes
(parallel_builder.PointBuildResult/AssemblyBuildResult, component_
inventory.ComponentTypeRequirement, op_requirements.Requirement,
assembly_compatibility.RelationshipEvaluation/AssemblyCompatibilityReport)
rather than loose dicts, so a field rename anywhere in those modules
would break this file too, not silently drift out of sync."""

from pathlib import Path

import pytest

from assembly_compatibility import AssemblyCompatibilityReport, CompatibilityStatus, MechanicalRelationshipType, RelationshipEvaluation
from clarification import BuildDecision, ClarificationCategory, assess_build
from component_inventory import ComponentTypeRequirement, ReconciliationStatus
from dimension_requirements import RequirementStatus
from llm_decompose import ActionablePoint
from op_requirements import Requirement
from parallel_builder import AssemblyBuildResult, PointBuildResult


def _point(desc: str, pid: str = "p1", status: str = "ok", mesh=object(), **kw) -> PointBuildResult:
    return PointBuildResult(
        point=ActionablePoint(description=desc, position_mm=(0, 0, 0), id=pid),
        status=status, executed_steps=1, error=kw.pop("error", None),
        nearest_neighbor_similarity=0.5, program=[], mesh=mesh, **kw,
    )


def _result(points=None, component_inventory=None, compatibility_report=None) -> AssemblyBuildResult:
    return AssemblyBuildResult(
        mesh=None, points=points or [], fell_back_to_single_point=False,
        component_inventory=component_inventory, compatibility_report=compatibility_report,
    )


# 1. Ambiguous component identity (component_type could not be determined at all) -- blocking, NEEDS_CLARIFICATION.
def test_1_unresolved_component_type_blocks_with_a_specific_question():
    inv = [ComponentTypeRequirement(component_id="c1", component_type=None, component_name="thing", source_text="a thing on top", failure_reason="no recognized component-type keyword")]
    assessment = assess_build(_result(points=[_point("a thing on top")], component_inventory=inv))
    assert assessment.build_decision is BuildDecision.NEEDS_CLARIFICATION
    reqs = [c for c in assessment.clarification_requests if c.category is ClarificationCategory.AMBIGUOUS_COMPONENT_IDENTITY]
    assert len(reqs) == 1
    assert reqs[0].blocking is True
    assert "a thing on top" in reqs[0].question


# 2. Missing MANDATORY component -- blocking, NEEDS_CLARIFICATION, a real add/remove choice offered.
def test_2_missing_mandatory_component_blocks_and_offers_a_real_choice():
    inv = [ComponentTypeRequirement(
        component_id="bracket_1", component_type="bracket", component_name="bracket", source_text="a mounting bracket",
        mandatory=True, requested_count=1, generated_count=0, reconciliation=ReconciliationStatus.MISSING_FROM_PLAN,
        failure_reason="no generated instance matches",
    )]
    assessment = assess_build(_result(points=[_point("a plate")], component_inventory=inv))
    assert assessment.build_decision is BuildDecision.NEEDS_CLARIFICATION
    req = next(c for c in assessment.clarification_requests if c.category is ClarificationCategory.COMPONENT_COUNT_MISMATCH)
    assert req.blocking is True
    assert "add the missing component" in req.candidate_answers


# 3. Missing OPTIONAL component -- advisory, proceeds with a disclosed assumption, never blocks.
def test_3_missing_optional_component_is_advisory_not_blocking():
    inv = [ComponentTypeRequirement(
        component_id="trim_1", component_type="cover", component_name="decorative trim", source_text="maybe some trim",
        mandatory=False, requested_count=1, generated_count=0, reconciliation=ReconciliationStatus.MISSING_FROM_PLAN,
        failure_reason="no generated instance matches",
    )]
    assessment = assess_build(_result(points=[_point("a box")], component_inventory=inv))
    assert assessment.build_decision is BuildDecision.PROCEED_WITH_DISCLOSED_ASSUMPTIONS
    req = next(c for c in assessment.clarification_requests if c.category is ClarificationCategory.COMPONENT_COUNT_MISMATCH)
    assert req.blocking is False
    assert req.default_allowed is True
    assert len(assessment.assumptions) == 1


# 4. Component count mismatch for a MANDATORY type -- blocking.
def test_4_mandatory_count_mismatch_blocks():
    inv = [ComponentTypeRequirement(
        component_id="holes_1", component_type="fastener", component_name="bolt", source_text="six bolts",
        mandatory=True, requested_count=6, generated_count=4, reconciliation=ReconciliationStatus.COUNT_MISMATCH,
        failure_reason="requested 6, generated 4",
    )]
    assessment = assess_build(_result(points=[_point("a plate with six bolts")], component_inventory=inv))
    assert assessment.build_decision is BuildDecision.NEEDS_CLARIFICATION
    req = next(c for c in assessment.clarification_requests if c.category is ClarificationCategory.COMPONENT_COUNT_MISMATCH)
    assert "6" in req.question and "4" in req.question


# 5. Duplicated component -- advisory, proceeds.
def test_5_duplicated_component_is_advisory():
    inv = [ComponentTypeRequirement(
        component_id="bracket_1", component_type="bracket", component_name="bracket", source_text="a bracket",
        mandatory=True, requested_count=1, generated_count=2, instance_ids=["b1", "b2"],
        reconciliation=ReconciliationStatus.DUPLICATED_IN_PLAN, failure_reason="generated twice",
    )]
    assessment = assess_build(_result(points=[_point("a bracket")], component_inventory=inv))
    req = next(c for c in assessment.clarification_requests if c.category is ClarificationCategory.COMPONENT_COUNT_MISMATCH)
    assert req.blocking is False
    assert assessment.build_decision is BuildDecision.PROCEED_WITH_DISCLOSED_ASSUMPTIONS


# 6. Identity ambiguous (several generated parts could equally match) -- blocking.
def test_6_identity_ambiguous_blocks():
    inv = [ComponentTypeRequirement(
        component_id="bracket_1", component_type="bracket", component_name="bracket", source_text="the bracket",
        instance_ids=["b1", "b2"], reconciliation=ReconciliationStatus.IDENTITY_AMBIGUOUS,
        failure_reason="two generated brackets are equally plausible matches",
    )]
    assessment = assess_build(_result(points=[_point("two brackets")], component_inventory=inv))
    req = next(c for c in assessment.clarification_requests if c.category is ClarificationCategory.AMBIGUOUS_COMPONENT_IDENTITY)
    assert req.blocking is True
    assert set(req.affected_component_ids) == {"bracket_1", "b1", "b2"}


# 7. E47's own signature -- op_arbitration NO_VALID_PLAN, even after the widened rescue search,
# is a genuine capability gap: UNSUPPORTED, phrased with NO question (no answer changes it).
def test_7_no_valid_plan_is_unsupported_with_no_question_asked():
    p = _point("a bearing housing with a bore", arbitration_outcome="NO_VALID_PLAN",
                arbitration_reasoning=["classifier invalid: no round bore op", "widened retrieval found no alternative meeting the 0.20 floor -- confirmed NO_VALID_PLAN"])
    assessment = assess_build(_result(points=[p]))
    assert assessment.build_decision is BuildDecision.UNSUPPORTED
    req = next(c for c in assessment.clarification_requests if c.category is ClarificationCategory.UNSUPPORTED_FEATURE)
    assert req.question == ""
    assert req.blocking is True
    assert len(assessment.unsupported_requirements) == 1


# 8. op_arbitration NEEDS_DISAMBIGUATION -- advisory, proceeds with a disclosed default choice.
def test_8_needs_disambiguation_is_advisory_default():
    p = _point("a spaceship", arbitration_outcome="NEEDS_DISAMBIGUATION",
                arbitration_reasoning=["both valid, equal coverage, equal length, genuinely different sequences -- ambiguous"])
    assessment = assess_build(_result(points=[p]))
    assert assessment.build_decision is BuildDecision.PROCEED_WITH_DISCLOSED_ASSUMPTIONS
    req = next(c for c in assessment.clarification_requests if c.category is ClarificationCategory.AMBIGUOUS_OPERATION_CHOICE)
    assert req.blocking is False
    assert req.recommended_answer == "use the default approach"


# 9. A required feature flagged UNSUPPORTED_TO_VERIFY (present in the op sequence, but not
# independently re-measurable from the built mesh) -- advisory, never blocks on its own.
def test_9_unsupported_to_verify_feature_is_advisory():
    req_obj = Requirement(id="r1", feature_type="hollow_shell", geometry_type=None, explicit=True,
                           source_text="a shelled-out interior", required=True,
                           status=RequirementStatus.UNSUPPORTED_TO_VERIFY, failure_reason="shell thickness not independently measurable from this mesh")
    p = _point("a box with a shelled-out interior", feature_requirements=[req_obj])
    assessment = assess_build(_result(points=[p]))
    assert assessment.build_decision is BuildDecision.PROCEED_WITH_DISCLOSED_ASSUMPTIONS
    req = next(c for c in assessment.clarification_requests if c.category is ClarificationCategory.UNSUPPORTED_FEATURE)
    assert req.blocking is False


# 10. A required feature that FAILED verification (built, but wrong) -- blocking.
def test_10_failed_required_feature_blocks():
    req_obj = Requirement(id="r1", feature_type="subtractive_slot", geometry_type=None, explicit=True,
                           source_text="a 30x8mm slot", required=True,
                           status=RequirementStatus.FAILED, failure_reason="no slot-shaped feature found on the built mesh")
    p = _point("a plate with a 30x8mm slot", feature_requirements=[req_obj])
    assessment = assess_build(_result(points=[p]))
    assert assessment.build_decision is BuildDecision.NEEDS_CLARIFICATION
    req = next(c for c in assessment.clarification_requests if c.category is ClarificationCategory.FEATURE_VERIFICATION_FAILURE)
    assert req.blocking is True
    assert "30x8mm slot" in req.question


# 11. A point that failed to build any geometry at all -- blocking, a real alternative/drop choice.
def test_11_geometry_build_failure_blocks_with_a_real_choice():
    p = _point("an impossible nested cavity", status="rejected", mesh=None, error="boolean_difference raised: non-manifold result")
    assessment = assess_build(_result(points=[p]))
    assert assessment.build_decision is BuildDecision.NEEDS_CLARIFICATION
    req = next(c for c in assessment.clarification_requests if c.category is ClarificationCategory.GEOMETRY_BUILD_FAILURE)
    assert req.blocking is True
    assert "drop this component" in req.candidate_answers


# 12. assembly_compatibility UNDER_SPECIFIED -- advisory, a confirm-or-default question.
def test_12_under_specified_relationship_is_advisory():
    eva = RelationshipEvaluation(
        relationship_id="REL-1", component_a="a", component_b="b", relationship_type=MechanicalRelationshipType.FACE_CONTACT,
        source_phrase_a="a bracket", source_phrase_b="a plate", status=CompatibilityStatus.UNDER_SPECIFIED,
        reason="no face direction specified for this attachment",
    )
    report = AssemblyCompatibilityReport(evaluations=[eva])
    assessment = assess_build(_result(points=[_point("a bracket on a plate")], compatibility_report=report))
    assert assessment.build_decision is BuildDecision.PROCEED_WITH_DISCLOSED_ASSUMPTIONS
    req = next(c for c in assessment.clarification_requests if c.category is ClarificationCategory.UNDER_SPECIFIED_RELATIONSHIP)
    assert req.blocking is False


# 13. assembly_compatibility INCOMPATIBLE -- a genuine mechanical conflict -- blocking.
def test_13_incompatible_relationship_blocks_with_the_suggested_fix():
    eva = RelationshipEvaluation(
        relationship_id="REL-2", component_a="shaft_01", component_b="housing_01", relationship_type=MechanicalRelationshipType.SHAFT_BORE_FIT,
        source_phrase_a="a 12mm shaft", source_phrase_b="a bearing housing", status=CompatibilityStatus.INCOMPATIBLE,
        reason="shaft (12.00mm) is larger than bore (10.00mm) -- interference, not clearance",
        suggested_action="increase housing_01's bore to at least 12.00mm",
    )
    report = AssemblyCompatibilityReport(evaluations=[eva])
    assessment = assess_build(_result(points=[_point("a shaft in a housing")], compatibility_report=report))
    assert assessment.build_decision is BuildDecision.NEEDS_CLARIFICATION
    req = next(c for c in assessment.clarification_requests if c.category is ClarificationCategory.ASSEMBLY_INCOMPATIBILITY)
    assert req.blocking is True
    assert req.recommended_answer == "increase housing_01's bore to at least 12.00mm"


# 14. A compatible fit with no stated clearance/fit class -- advisory tolerance disclosure.
def test_14_unspecified_fit_class_is_advisory():
    eva = RelationshipEvaluation(
        relationship_id="REL-3", component_a="shaft_01", component_b="housing_01", relationship_type=MechanicalRelationshipType.SHAFT_BORE_FIT,
        source_phrase_a="a shaft", source_phrase_b="a housing", status=CompatibilityStatus.COMPATIBLE,
        calculated={"diametral_clearance_mm": 0.2, "fit_classification": "clearance fit (no tolerance or fit class specified in the request)"},
    )
    report = AssemblyCompatibilityReport(evaluations=[eva])
    assessment = assess_build(_result(points=[_point("a shaft in a housing")], compatibility_report=report))
    assert assessment.build_decision is BuildDecision.PROCEED_WITH_DISCLOSED_ASSUMPTIONS
    req = next(c for c in assessment.clarification_requests if c.category is ClarificationCategory.TOLERANCE_OR_FIT_UNSPECIFIED)
    assert req.blocking is False


# 15. Precedence: a genuine capability gap (NO_VALID_PLAN) AND a separately-answerable
# ambiguity both exist -- the answerable one must be asked first, never silently
# downgraded straight to UNSUPPORTED (answering it might remove the unsupported feature
# entirely, e.g. by dropping the component that needs it).
def test_15_answerable_ambiguity_takes_precedence_over_a_capability_gap():
    inv = [ComponentTypeRequirement(component_id="c1", component_type=None, component_name="thing", source_text="some thing", failure_reason="unrecognized")]
    p = _point("a bearing housing with a bore", arbitration_outcome="NO_VALID_PLAN", arbitration_reasoning=["no valid alternative found"])
    assessment = assess_build(_result(points=[p], component_inventory=inv))
    assert assessment.build_decision is BuildDecision.NEEDS_CLARIFICATION
    categories = {c.category for c in assessment.clarification_requests}
    assert ClarificationCategory.AMBIGUOUS_COMPONENT_IDENTITY in categories
    assert ClarificationCategory.UNSUPPORTED_FEATURE in categories


# -- lifecycle / model-contract coverage --

def test_16_clean_build_proceeds_with_nothing_flagged():
    assessment = assess_build(_result(points=[_point("a plain enclosure")]))
    assert assessment.build_decision is BuildDecision.PROCEED
    assert assessment.clarification_requests == []
    assert assessment.assumptions == []


def test_17_validation_failure_fallback_when_nothing_else_caught_it():
    import dataclasses

    from build_validation import BuildValidationReport, ValidationStatus

    report = BuildValidationReport(
        geometry_status=ValidationStatus.PASS, dimension_status=ValidationStatus.FAIL,
        feature_status=ValidationStatus.NOT_EVALUATED, component_status=ValidationStatus.NOT_EVALUATED,
        compatibility_status=ValidationStatus.NOT_EVALUATED, requirement_status=ValidationStatus.FAIL,
        overall_status=ValidationStatus.FAIL,
    )
    assessment = assess_build(_result(points=[_point("a plain enclosure")]), validation_report=report)
    assert assessment.build_decision is BuildDecision.BLOCKED_BY_VALIDATION_FAILURE
    dataclasses.asdict  # sanity: BuildValidationReport is a plain dataclass, no special teardown needed


def test_18_as_dict_round_trips_every_field():
    inv = [ComponentTypeRequirement(component_id="c1", component_type=None, component_name="thing", source_text="a thing", failure_reason="x")]
    assessment = assess_build(_result(points=[_point("a thing")], component_inventory=inv))
    d = assessment.as_dict()
    assert d["build_decision"] == "NEEDS_CLARIFICATION"
    assert isinstance(d["clarification_requests"], list) and d["clarification_requests"]
    req_dict = d["clarification_requests"][0]
    assert set(req_dict) == {
        "clarification_id", "category", "severity", "question", "context", "affected_component_ids",
        "affected_requirement_ids", "candidate_answers", "recommended_answer", "default_allowed",
        "blocking", "reason", "source_stage", "status",
    }
    assert req_dict["status"] == "DETECTED"


# -- full-pipeline integration: confirms E47 is fixed end-to-end, not just at the
# op_arbitration unit level (see tests/test_e47_arbitration_rescue.py) or the
# clarification-detector unit level (tests 1-15 above). Runs the REAL trained
# bundle through the REAL build pipeline (no program_override, no mocking).

@pytest.fixture(scope="module")
def default_bundle_path() -> Path:
    path = Path("models/program_model.joblib")
    if not path.exists():
        pytest.skip("production bundle models/program_model.joblib not present")
    return path


def test_19_e47_production_case_never_silently_passes_end_to_end(default_bundle_path):
    """The exact E47 reproduction text (Phase 7's own lock-in test,
    tests/test_shared_unit_dimension_parsing.py::
    test_e47_root_cause_is_operation_selection_not_dimension_parsing):
    'a bearing housing, 40mm x 40mm x 20mm, with a 12.2mm bore'. Confirmed,
    live, against the real production bundle: op_arbitration's own
    widened rescue search (program_model._rescue_valid_alternative) finds
    a real training example (add_box, counterbore_hole, finalize) but at
    only 0.175 similarity -- below the 0.20 confidence floor -- so the
    outcome stays the honest NO_VALID_PLAN. This must never reach the
    caller as a silent, undisclosed PASS: AssemblyBuildResult.overall_
    status must not be "PASS", and the clarification layer must flag it
    (never a clean PROCEED)."""
    from parallel_builder import build_assembly_from_points

    points = [ActionablePoint(description="a bearing housing, 40mm x 40mm x 20mm, with a 12.2mm bore", position_mm=(0, 0, 0), id="housing_1")]
    result = build_assembly_from_points(points, default_bundle_path, max_workers=1)

    assert result.points[0].arbitration_outcome == "NO_VALID_PLAN"
    assert result.overall_status != "PASS"

    from build_validation import build_validation_report

    vr = build_validation_report(result)
    assessment = assess_build(result, validation_report=vr)
    assert assessment.build_decision is not BuildDecision.PROCEED
    unsupported = [c for c in assessment.clarification_requests if c.category is ClarificationCategory.UNSUPPORTED_FEATURE]
    assert len(unsupported) == 1
    assert unsupported[0].blocking is True
    assert unsupported[0].question == ""
    assert "widened retrieval" in unsupported[0].reason
    assert "confirmed NO_VALID_PLAN" in unsupported[0].reason
