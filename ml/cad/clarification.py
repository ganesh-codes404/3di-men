"""Phase 8 Part B -- the Intelligent Clarification System.

Deliberately NOT a new, competing requirement/status representation:
every detection function below reads structured results an existing
module ALREADY computed on every build (op_requirements.Requirement /
feature_verification's own Required Feature Ledger, component_inventory.
ComponentTypeRequirement, assembly_compatibility.RelationshipEvaluation,
op_arbitration.ArbitrationResult via PointBuildResult.arbitration_*,
build_validation.BuildValidationReport's own rollup) -- this module's
only job is to decide, from those ALREADY-COMPUTED facts, whether 3di-
men should (a) proceed, (b) proceed while disclosing an assumption it
made, (c) ask a specific, answerable question before proceeding, or (d)
say plainly that it cannot currently build what was asked. Matches the
project's own stated goal for this system: "I know what you mean," "I
need one detail before proceeding," or "I cannot currently generate
that feature" -- chosen from the system's actual engineering state, not
guessed.

Ask-vs-infer policy (the actual decision rule every detector below
applies, not left implicit):
- A gap is ONLY ever escalated to a user-facing question
  (ClarificationCategory, blocking=True) when answering it would
  actually change what gets built, AND 3di-men has no way to infer a
  safe default on its own -- e.g. "did you mean this noun to be a
  bracket or a mount" (component_type is None: two structurally
  different parts), or "which of these two structurally different,
  equally-valid op sequences did you want" (NEEDS_DISAMBIGUATION).
- A gap that has a genuinely reasonable, disclosable default is NEVER
  escalated to a blocking question -- it is answered by inference, and
  the inference is disclosed as an assumption instead (e.g. an
  UNDER_SPECIFIED face direction defaulting to the most common
  convention, a NEEDS_DISAMBIGUATION tie broken toward the classifier's
  own sequence). Good question (asked): "You asked for a bracket with
  mounting holes but didn't say how many -- how many holes, and what
  spacing?" (changes the actual geometry, no safe default exists). Bad
  question (never asked): "Should the shaft be gray or blue?" (does not
  affect any requirement this engine tracks) or "Do you want it to be
  a good design?" (not a real, answerable engineering question).
- A gap that reflects a genuine CAPABILITY limit (arbitration found no
  valid op sequence even after widening its own search -- see program_
  model._rescue_valid_alternative) is never phrased as a question at
  all (no answer would change the engine's ability to build it right
  now) -- it is reported as UNSUPPORTED, honestly, per this module's
  own BuildDecision.UNSUPPORTED outcome."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class ClarificationCategory(Enum):
    AMBIGUOUS_COMPONENT_IDENTITY = "AMBIGUOUS_COMPONENT_IDENTITY"  # component_inventory: component_type could not be determined, or IDENTITY_AMBIGUOUS
    COMPONENT_COUNT_MISMATCH = "COMPONENT_COUNT_MISMATCH"  # component_inventory: MISSING_FROM_PLAN / COUNT_MISMATCH / DUPLICATED_IN_PLAN
    AMBIGUOUS_DIMENSION = "AMBIGUOUS_DIMENSION"  # dimension_requirements: UNRESOLVED (an explicit number with no determinable parameter)
    GEOMETRY_BUILD_FAILURE = "GEOMETRY_BUILD_FAILURE"  # a point's own program never produced valid geometry at all (status == "rejected")
    UNSUPPORTED_FEATURE = "UNSUPPORTED_FEATURE"  # a required feature with no compatible op in ANY available sequence (op_arbitration NO_VALID_PLAN, or feature_requirements UNSUPPORTED_TO_VERIFY)
    FEATURE_VERIFICATION_FAILURE = "FEATURE_VERIFICATION_FAILURE"  # feature_requirements: FAILED -- built, but the built geometry doesn't actually satisfy it
    UNDER_SPECIFIED_RELATIONSHIP = "UNDER_SPECIFIED_RELATIONSHIP"  # assembly_compatibility: UNDER_SPECIFIED (e.g. no face/axis stated)
    ASSEMBLY_INCOMPATIBILITY = "ASSEMBLY_INCOMPATIBILITY"  # assembly_compatibility: INCOMPATIBLE -- a real mechanical conflict between two built components
    AMBIGUOUS_OPERATION_CHOICE = "AMBIGUOUS_OPERATION_CHOICE"  # op_arbitration: NEEDS_DISAMBIGUATION -- two structurally different, equally-valid sequences
    TOLERANCE_OR_FIT_UNSPECIFIED = "TOLERANCE_OR_FIT_UNSPECIFIED"  # assembly_compatibility: a diametral fit with no stated clearance/fit class at all


class ClarificationStatus(Enum):
    DETECTED = "DETECTED"
    PRESENTED = "PRESENTED"
    ANSWERED = "ANSWERED"
    REQUIREMENTS_UPDATED = "REQUIREMENTS_UPDATED"
    REPLANNED = "REPLANNED"
    VERIFIED = "VERIFIED"
    CANCELLED = "CANCELLED"
    EXPIRED = "EXPIRED"
    BLOCKED = "BLOCKED"
    SUPERSEDED = "SUPERSEDED"


class BuildDecision(Enum):
    PROCEED = "PROCEED"
    PROCEED_WITH_DISCLOSED_ASSUMPTIONS = "PROCEED_WITH_DISCLOSED_ASSUMPTIONS"
    NEEDS_CLARIFICATION = "NEEDS_CLARIFICATION"
    UNSUPPORTED = "UNSUPPORTED"
    BLOCKED_BY_VALIDATION_FAILURE = "BLOCKED_BY_VALIDATION_FAILURE"


@dataclass
class ClarificationRequest:
    clarification_id: str
    category: ClarificationCategory
    severity: str  # "blocking" | "advisory" -- "blocking" means the build decision cannot be a
        # clean PROCEED while this is unanswered; "advisory" means it was (or will be) resolved
        # by a disclosed default, never silently absorbed.
    question: str  # a SPECIFIC, answerable question about THIS build -- never a generic
        # "are you sure?" (see this module's own docstring on the ask-vs-infer policy)
    context: str  # why this came up -- the source text / measured fact that triggered it
    affected_component_ids: list[str] = field(default_factory=list)
    affected_requirement_ids: list[str] = field(default_factory=list)
    candidate_answers: list[str] = field(default_factory=list)
    recommended_answer: "str | None" = None
    default_allowed: bool = False  # True when proceeding without an answer is safe -- the
        # recommended_answer is then the disclosed assumption
    blocking: bool = False
    reason: str = ""
    source_stage: str = ""  # which validator/module detected this -- e.g. "component_inventory",
        # "op_arbitration", "assembly_compatibility"
    status: ClarificationStatus = ClarificationStatus.DETECTED

    def as_dict(self) -> dict:
        return {
            "clarification_id": self.clarification_id,
            "category": self.category.value,
            "severity": self.severity,
            "question": self.question,
            "context": self.context,
            "affected_component_ids": self.affected_component_ids,
            "affected_requirement_ids": self.affected_requirement_ids,
            "candidate_answers": self.candidate_answers,
            "recommended_answer": self.recommended_answer,
            "default_allowed": self.default_allowed,
            "blocking": self.blocking,
            "reason": self.reason,
            "source_stage": self.source_stage,
            "status": self.status.value,
        }


@dataclass
class ClarificationAssessment:
    clarification_requests: list[ClarificationRequest]
    build_decision: BuildDecision
    assumptions: list[str]  # human-readable disclosures for every non-blocking gap this
        # assessment resolved by inference rather than by asking
    unresolved_requirements: list[str]
    unsupported_requirements: list[str]

    def as_dict(self) -> dict:
        return {
            "clarification_requests": [c.as_dict() for c in self.clarification_requests],
            "build_decision": self.build_decision.value,
            "assumptions": self.assumptions,
            "unresolved_requirements": self.unresolved_requirements,
            "unsupported_requirements": self.unsupported_requirements,
        }


def _rid(counter: list) -> str:
    counter[0] += 1
    return f"CLR-{counter[0]:03d}"


def _detect_component_inventory_clarifications(result, counter: list) -> list[ClarificationRequest]:
    inventory = getattr(result, "component_inventory", None)
    if not inventory:
        return []
    out: list[ClarificationRequest] = []
    for c in inventory:
        if c.component_type is None:
            out.append(ClarificationRequest(
                clarification_id=_rid(counter), category=ClarificationCategory.AMBIGUOUS_COMPONENT_IDENTITY,
                severity="blocking",
                question=f"What kind of component did you mean by {c.source_text!r}? I don't recognize a buildable component type in that phrase.",
                context=c.source_text, affected_component_ids=[c.component_id],
                candidate_answers=[], default_allowed=False, blocking=True,
                reason=c.failure_reason or "no recognized component-type keyword found in this phrase",
                source_stage="component_inventory",
            ))
            continue
        recon = c.reconciliation.value if c.reconciliation is not None else None
        if recon == "MISSING_FROM_PLAN":
            out.append(ClarificationRequest(
                clarification_id=_rid(counter), category=ClarificationCategory.COMPONENT_COUNT_MISMATCH,
                severity="blocking" if c.mandatory else "advisory",
                question=(
                    f"You asked for {c.requested_count} {c.component_name}(s) ({c.source_text!r}), but none were "
                    "generated. Should I add it, or did you mean to remove it from the design?"
                ),
                context=c.source_text, affected_component_ids=[c.component_id],
                candidate_answers=["add the missing component", "remove it from the requirements"],
                default_allowed=not c.mandatory,
                recommended_answer="add the missing component" if c.mandatory else "remove it from the requirements",
                blocking=c.mandatory, reason=c.failure_reason or "requested component has no matching generated instance",
                source_stage="component_inventory",
            ))
        elif recon == "COUNT_MISMATCH":
            out.append(ClarificationRequest(
                clarification_id=_rid(counter), category=ClarificationCategory.COMPONENT_COUNT_MISMATCH,
                severity="blocking" if c.mandatory else "advisory",
                question=(
                    f"You asked for {c.requested_count} {c.component_name}(s), but {c.generated_count} were "
                    f"generated ({c.source_text!r}). Should the count be corrected to {c.requested_count}?"
                ),
                context=c.source_text, affected_component_ids=[c.component_id],
                candidate_answers=[f"build exactly {c.requested_count}", f"keep {c.generated_count} as built"],
                default_allowed=not c.mandatory, recommended_answer=f"build exactly {c.requested_count}",
                blocking=c.mandatory, reason=c.failure_reason or "requested and generated counts disagree",
                source_stage="component_inventory",
            ))
        elif recon == "IDENTITY_AMBIGUOUS":
            out.append(ClarificationRequest(
                clarification_id=_rid(counter), category=ClarificationCategory.AMBIGUOUS_COMPONENT_IDENTITY,
                severity="blocking",
                question=(
                    f"Several generated components could equally be {c.component_name!r} ({c.source_text!r}) -- "
                    "which one did you mean, or are these actually meant to be distinct parts?"
                ),
                context=c.source_text, affected_component_ids=[c.component_id] + list(c.instance_ids),
                default_allowed=False, blocking=True,
                reason=c.failure_reason or "more than one plausible match, none of them distinguishable by description alone",
                source_stage="component_inventory",
            ))
        elif recon == "DUPLICATED_IN_PLAN":
            out.append(ClarificationRequest(
                clarification_id=_rid(counter), category=ClarificationCategory.COMPONENT_COUNT_MISMATCH,
                severity="advisory",
                question=f"{c.component_name!r} ({c.source_text!r}) appears to have been generated more times than requested -- should the extra copy be removed?",
                context=c.source_text, affected_component_ids=[c.component_id] + list(c.instance_ids),
                candidate_answers=["remove the extra copy", "keep all generated copies"],
                default_allowed=True, recommended_answer="keep all generated copies",
                blocking=False, reason=c.failure_reason or "generated count exceeds requested count",
                source_stage="component_inventory",
            ))
    return out


def _detect_arbitration_clarifications(result, counter: list) -> list[ClarificationRequest]:
    """E47's own disclosure channel: PointBuildResult.arbitration_outcome,
    now carried through from op_arbitration by parallel_builder.py (see
    that module's own Phase 8 Part B wiring note). NO_VALID_PLAN here
    means arbitration already tried the default top-5 candidates AND
    program_model._rescue_valid_alternative's widened search, and still
    found nothing that satisfies this point's own extracted
    requirements with enough confidence to trust -- a genuine capability
    gap, never phrased as a question (no answer changes what the model
    is currently able to build), always UNSUPPORTED."""
    out: list[ClarificationRequest] = []
    for p in getattr(result, "points", []) or []:
        if p.arbitration_outcome == "NO_VALID_PLAN":
            out.append(ClarificationRequest(
                clarification_id=_rid(counter), category=ClarificationCategory.UNSUPPORTED_FEATURE,
                severity="blocking",
                question="",  # no question -- see this function's own docstring: a capability gap, not an ambiguity
                context=p.point.description,
                affected_component_ids=[p.point.id] if p.point.id else [],
                default_allowed=False, blocking=True,
                reason="; ".join(p.arbitration_reasoning) or "no op sequence (classifier, RAG, or a widened retrieval search) satisfies this request's own extracted requirements with enough confidence to trust",
                source_stage="op_arbitration",
            ))
        elif p.arbitration_outcome == "NEEDS_DISAMBIGUATION":
            out.append(ClarificationRequest(
                clarification_id=_rid(counter), category=ClarificationCategory.AMBIGUOUS_OPERATION_CHOICE,
                severity="advisory",
                question=(
                    f"There are two equally-valid ways to build {p.point.description!r} with no clear preference "
                    "stated -- is the default approach acceptable, or did you have a specific method in mind?"
                ),
                context=p.point.description, affected_component_ids=[p.point.id] if p.point.id else [],
                default_allowed=True, recommended_answer="use the default approach",
                blocking=False, reason="; ".join(p.arbitration_reasoning) or "classifier and RAG both valid, equal coverage, genuinely different sequences",
                source_stage="op_arbitration",
            ))
    return out


def _detect_feature_clarifications(result, counter: list) -> list[ClarificationRequest]:
    out: list[ClarificationRequest] = []
    for p in getattr(result, "points", []) or []:
        for r in getattr(p, "feature_requirements", []) or []:
            if r.status.value == "UNSUPPORTED_TO_VERIFY" and r.required:
                out.append(ClarificationRequest(
                    clarification_id=_rid(counter), category=ClarificationCategory.UNSUPPORTED_FEATURE,
                    severity="advisory", question="",
                    context=r.source_text, affected_component_ids=[r.component_id] if r.component_id else [],
                    affected_requirement_ids=[r.id], default_allowed=True,
                    recommended_answer="proceed; this feature's presence could not be independently re-measured from the built mesh",
                    blocking=False, reason=r.failure_reason, source_stage="feature_verification",
                ))
            elif r.status.value == "FAILED" and r.required:
                out.append(ClarificationRequest(
                    clarification_id=_rid(counter), category=ClarificationCategory.FEATURE_VERIFICATION_FAILURE,
                    severity="blocking",
                    question=f"The required feature from {r.source_text!r} did not match the built geometry ({r.failure_reason}) -- should I retry with different parameters, or was the requirement itself incorrect?",
                    context=r.source_text, affected_component_ids=[r.component_id] if r.component_id else [],
                    affected_requirement_ids=[r.id], default_allowed=False, blocking=True,
                    reason=r.failure_reason, source_stage="feature_verification",
                ))
        if p.status == "rejected":
            out.append(ClarificationRequest(
                clarification_id=_rid(counter), category=ClarificationCategory.GEOMETRY_BUILD_FAILURE,
                severity="blocking",
                question=f"{p.point.description!r} could not be built at all ({p.error}) -- should I try an alternative approach, or drop this component?",
                context=p.point.description, affected_component_ids=[p.point.id] if p.point.id else [],
                candidate_answers=["try an alternative approach", "drop this component"],
                default_allowed=False, blocking=True, reason=p.error or "", source_stage="sandbox_execution",
            ))
    return out


def _detect_compatibility_clarifications(result, counter: list) -> list[ClarificationRequest]:
    compat = getattr(result, "compatibility_report", None)
    if compat is None or not compat.evaluations:
        return []
    out: list[ClarificationRequest] = []
    for e in compat.evaluations:
        status = e.status.value
        if status == "UNDER_SPECIFIED":
            out.append(ClarificationRequest(
                clarification_id=_rid(counter), category=ClarificationCategory.UNDER_SPECIFIED_RELATIONSHIP,
                severity="advisory",
                question=f"The relationship between {e.component_a} and {e.component_b} doesn't state enough to fully verify ({e.reason}) -- can you confirm the intended axis/face, or is the default assumption fine?",
                context=f"{e.source_phrase_a} / {e.source_phrase_b}", affected_component_ids=[e.component_a, e.component_b],
                default_allowed=True, recommended_answer="proceed with the inferred default",
                blocking=False, reason=e.reason, source_stage="assembly_compatibility",
            ))
        elif status == "INCOMPATIBLE":
            out.append(ClarificationRequest(
                clarification_id=_rid(counter), category=ClarificationCategory.ASSEMBLY_INCOMPATIBILITY,
                severity="blocking",
                question=f"{e.component_a} and {e.component_b} do not actually fit together as built ({e.reason}). {e.suggested_action or 'How should this be corrected?'}",
                context=f"{e.source_phrase_a} / {e.source_phrase_b}", affected_component_ids=[e.component_a, e.component_b],
                candidate_answers=[e.suggested_action] if e.suggested_action else [],
                recommended_answer=e.suggested_action, default_allowed=False, blocking=True,
                reason=e.reason, source_stage="assembly_compatibility",
            ))
        elif status == "COMPATIBLE" and "fit_classification" in e.calculated and "line-to-line" not in str(e.calculated.get("fit_classification", "")):
            out.append(ClarificationRequest(
                clarification_id=_rid(counter), category=ClarificationCategory.TOLERANCE_OR_FIT_UNSPECIFIED,
                severity="advisory",
                question=f"No fit class or clearance was stated for {e.component_a}/{e.component_b} -- the built clearance ({e.calculated.get('diametral_clearance_mm')}mm) is being treated as acceptable. Is a specific fit (press, slip, line-to-line) actually required?",
                context=f"{e.source_phrase_a} / {e.source_phrase_b}", affected_component_ids=[e.component_a, e.component_b],
                default_allowed=True, recommended_answer="treat the built clearance as acceptable",
                blocking=False, reason="no clearance or fit class stated in either component's own description",
                source_stage="assembly_compatibility",
            ))
    return out


def _detect_dimension_clarifications(result, counter: list) -> list[ClarificationRequest]:
    from dimension_requirements import extract_dimension_requirements

    out: list[ClarificationRequest] = []
    for p in getattr(result, "points", []) or []:
        if p.mesh is None:
            continue
        for d in extract_dimension_requirements(p.point.description):
            if d.status.value == "UNRESOLVED":
                out.append(ClarificationRequest(
                    clarification_id=_rid(counter), category=ClarificationCategory.AMBIGUOUS_DIMENSION,
                    severity="blocking",
                    question=f"{d.source_phrase!r} states a number ({d.value:g}{d.original_unit}) but I can't tell which dimension it refers to -- which one did you mean?",
                    context=p.point.description, affected_component_ids=[p.point.id] if p.point.id else [],
                    affected_requirement_ids=[d.id], default_allowed=False, blocking=True,
                    reason="an explicit numeric value could not be confidently assigned to any specific parameter",
                    source_stage="dimension_requirements",
                ))
    return out


def assess_build(result, validation_report=None) -> ClarificationAssessment:
    """The single entry point: takes a parallel_builder.AssemblyBuildResult
    (already fully built) and produces every clarification this build
    raises, plus the overall build_decision. Never triggers new
    generation, never re-runs geometry -- every detector above reads
    only what the build already computed.

    `validation_report` is build_validation.BuildValidationReport, if the
    caller already computed one (server.py's /api/build does) -- used
    ONLY as a final fallback: a hard FAIL this module's own 5 detectors
    didn't specifically categorize (e.g. a generic spec_validation.py
    dimension mismatch with no corresponding UNRESOLVED/FAILED typed
    requirement) still must never resolve to a silent PROCEED. Never
    recomputed here -- Phase 6's own aggregation is reused, not
    duplicated."""
    counter = [0]
    requests: list[ClarificationRequest] = []
    requests += _detect_component_inventory_clarifications(result, counter)
    requests += _detect_arbitration_clarifications(result, counter)
    requests += _detect_feature_clarifications(result, counter)
    requests += _detect_compatibility_clarifications(result, counter)
    requests += _detect_dimension_clarifications(result, counter)

    assumptions = [
        f"{c.category.value}: {c.recommended_answer} ({c.reason})" if c.reason else f"{c.category.value}: {c.recommended_answer}"
        for c in requests if c.default_allowed and c.recommended_answer
    ]
    unresolved = [c.question or c.reason for c in requests if c.category is ClarificationCategory.AMBIGUOUS_DIMENSION]
    unsupported = [c.reason for c in requests if c.category is ClarificationCategory.UNSUPPORTED_FEATURE]

    capability_gap = any(
        c.category is ClarificationCategory.UNSUPPORTED_FEATURE and c.source_stage == "op_arbitration" and c.blocking
        for c in requests
    )
    answerable_blocking = [c for c in requests if c.blocking and not (c.category is ClarificationCategory.UNSUPPORTED_FEATURE and c.source_stage == "op_arbitration")]

    if answerable_blocking:
        # A separately-answerable ambiguity always gets asked first, even
        # when a genuine capability gap ALSO exists -- answering it might
        # change the requirement that's currently unsupported (e.g.
        # dropping the unbuildable feature entirely), so this is never
        # silently downgraded straight to UNSUPPORTED.
        decision = BuildDecision.NEEDS_CLARIFICATION
    elif capability_gap:
        decision = BuildDecision.UNSUPPORTED
    elif validation_report is not None and validation_report.overall_status.value == "FAIL":
        decision = BuildDecision.BLOCKED_BY_VALIDATION_FAILURE
    elif requests:
        decision = BuildDecision.PROCEED_WITH_DISCLOSED_ASSUMPTIONS
    else:
        decision = BuildDecision.PROCEED

    return ClarificationAssessment(
        clarification_requests=requests, build_decision=decision,
        assumptions=assumptions, unresolved_requirements=unresolved, unsupported_requirements=unsupported,
    )
