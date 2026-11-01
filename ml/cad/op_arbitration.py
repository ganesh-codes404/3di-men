"""Arbitration engine between the classifier's own independent op-
sequence proposal and RAG's retrieved one. Replaces
predict_program_hybrid's old bare-similarity-threshold check (see
program_model.ProgramModelConfig.rag_override_similarity_threshold's
own docstring for the full investigation of why that approach was
tried and rejected: raising the threshold enough to block one
confirmed-bad RAG override cost two OTHER confirmed-good RAG rescues,
because a single global number cannot separate "this retrieved example
is geometrically wrong" from "this retrieved example is geometrically
right but merely textually distant").

The classifier and RAG are both treated as PROPOSALS here, never
independent authorities: each is checked against the SAME structured
requirement list (op_requirements.extract_requirements) for hard
violations before either is trusted at all, then scored on requirement
coverage only if both pass. Semantic similarity to a retrieved example
is never sufficient to override an explicit requirement -- only actual
compatibility with what the text asked for is."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from op_requirements import CREATES_GEOMETRY_KIND, Requirement, extract_requirements

# A pattern op's own "count" ARGUMENT (not visible at the op-sequence
# level this engine works at) is what actually determines how many
# instances of a feature get built -- checked downstream by the
# existing per-argument regressors/overrides, not here. A sequence
# containing one of these is never flagged for an under-count, even
# though its visible op count is 1.
_PATTERN_OPS = frozenset({"drill_hole_pattern", "counterbore_hole_pattern", "add_slot_pattern", "circular_pattern", "linear_pattern"})


def _base_geometry_compatible(op: "str | None", required_kind: str) -> bool:
    if op is None:
        return False
    kind = CREATES_GEOMETRY_KIND.get(op)
    if kind is None:
        return True  # add_loft/add_sweep/any op this registry has no opinion about -- never rejected for a requirement it can't speak to
    return kind == required_kind


def _sequence_contains_any(ops: "tuple[str, ...]", compatible_ops: frozenset) -> bool:
    return any(op in compatible_ops for op in ops)


@dataclass
class CandidateEvaluation:
    ops: "tuple[str, ...]"
    valid: bool
    violations: list[str] = field(default_factory=list)
    requirement_coverage: float = 1.0  # fraction of requirements satisfied -- the scoring signal when both candidates are valid


def evaluate_candidate(ops: "tuple[str, ...]", requirements: list[Requirement]) -> CandidateEvaluation:
    """Never raises. An op-sequence with zero requirements to check
    against (the text gave this engine nothing recognizable to verify)
    is always valid -- this engine only REJECTS a candidate for
    contradicting something the text actually asked for, never for
    failing to satisfy a requirement that was never extracted.

    Note: op_requirements.extract_requirements() currently only ever
    produces `required=True` requirements, so `valid` and
    `requirement_coverage == 1.0` always coincide in practice today --
    missing ANY requirement is a hard violation, not a partial-credit
    deduction. `requirement_coverage` only does genuine partial-credit
    scoring for a `required=False` requirement, which nothing in this
    codebase currently constructs; arbitrate()'s both-valid tie-break is
    consequently decided by sequence length today, not by coverage
    differences. Kept as a real (not dead) code path for when an
    optional/inferred requirement type is eventually added, rather than
    collapsing `valid`/`coverage` into one signal now and having to
    re-split them later."""
    violations: list[str] = []
    satisfied = 0
    for req in requirements:
        if req.feature_type == "base_geometry":
            base_op = ops[0] if ops else None
            if not _base_geometry_compatible(base_op, req.geometry_type):
                violations.append(
                    f"base geometry requirement '{req.geometry_type}' (from {req.source_text!r}) "
                    f"violated by base op {base_op!r}"
                )
                continue
            satisfied += 1
        else:
            has_feature = _sequence_contains_any(ops, req.compatible_ops)
            if not has_feature:
                if req.required:
                    violations.append(
                        f"required feature '{req.feature_type}' (from {req.source_text!r}) has no "
                        f"compatible operation in {ops}"
                    )
                # An unsatisfied OPTIONAL requirement is never a hard
                # rejection, but it must not count toward `satisfied`
                # either -- it genuinely wasn't met, so coverage should
                # reflect that, not silently award full credit for
                # something absent.
                continue
            if req.count is not None and req.count > 1 and not any(op in _PATTERN_OPS for op in ops):
                actual_count = sum(1 for op in ops if op in req.compatible_ops)
                if actual_count < req.count:
                    violations.append(
                        f"requires {req.count} of feature '{req.feature_type}' (from {req.source_text!r}) "
                        f"but sequence has only {actual_count} individual feature op(s) and no pattern operation"
                    )
                    continue
            satisfied += 1
    coverage = satisfied / len(requirements) if requirements else 1.0
    return CandidateEvaluation(ops=ops, valid=not violations, violations=violations, requirement_coverage=coverage)


class ArbitrationOutcome(Enum):
    CLASSIFIER_ACCEPTED = "CLASSIFIER_ACCEPTED"
    RAG_ASSISTED = "RAG_ASSISTED"
    RAG_RESCUE = "RAG_RESCUE"
    NEEDS_DISAMBIGUATION = "NEEDS_DISAMBIGUATION"
    NO_VALID_PLAN = "NO_VALID_PLAN"
    # MERGED_PLAN is intentionally not produced by arbitrate() below --
    # splicing two op-sequences together while preserving handle-flow
    # dependencies (sandbox.py's own _CONSUMES_ONE/_CONSUMES_TWO/
    # _CONSUMES_TARGET bookkeeping) is a real, separate undertaking this
    # brief's own Part 10 scope line ("do not expand this task into a
    # complete assembly-system rewrite") excludes. The value name stays
    # defined, not silently dropped, so a future, deliberately-scoped
    # extension has a documented place to land -- see this module's own
    # final-report notes on why it's not implemented here.
    MERGED_PLAN = "MERGED_PLAN"


@dataclass
class ArbitrationResult:
    outcome: ArbitrationOutcome
    chosen_ops: "tuple[str, ...] | None"
    classifier_evaluation: CandidateEvaluation
    rag_evaluation: "CandidateEvaluation | None"
    reasoning: list[str] = field(default_factory=list)


def arbitrate(
    classifier_ops: "tuple[str, ...]", rag_ops: "tuple[str, ...]", rag_similarity: float, text: str,
) -> ArbitrationResult:
    """Never raises, never returns chosen_ops=None for NO_VALID_PLAN
    without the caller being able to tell -- the outcome field IS the
    disclosure; a caller that needs a program anyway (every current
    caller does -- there is no "ask the user mid-decomposition" UI yet)
    falls back to the classifier's own sequence either way, same
    "disclosed degradation, never a crash" discipline this project
    applies everywhere else, but the AMBIGUITY or FAILURE is never
    silently hidden from whoever inspects this result."""
    requirements = extract_requirements(text)
    reasoning: list[str] = [f"requirements extracted: {[(r.feature_type, r.geometry_type or r.source_text) for r in requirements]}"]

    classifier_eval = evaluate_candidate(classifier_ops, requirements)
    reasoning.append(
        f"classifier {classifier_ops}: valid={classifier_eval.valid}, coverage={classifier_eval.requirement_coverage:.2f}"
        + ("; " + "; ".join(classifier_eval.violations) if classifier_eval.violations else "")
    )

    if classifier_ops == rag_ops:
        # Agreement is a strong, free confidence signal ONLY when the
        # agreed-upon sequence actually satisfies the text's own
        # requirements. Confirmed real gap this guards (live production
        # bundle, not hypothetical): for "a shaft with a chamfered edge,
        # 25mm diameter, 120mm long," classifier AND RAG both independently
        # land on (add_cylinder, drill_hole, finalize) -- agreeing with
        # each other while BOTH silently drop the requested chamfer and add
        # an unrequested hole. Skipping validity on agreement (the original
        # design here) would let a shared wrong answer sail through
        # unchecked precisely because nothing disagreed with it. Still
        # never recomputes rag_eval separately -- it's definitionally
        # identical to classifier_eval -- just doesn't skip the check.
        if classifier_eval.valid:
            reasoning.append("classifier and RAG agree on a requirement-satisfying sequence -- no arbitration needed")
            return ArbitrationResult(ArbitrationOutcome.CLASSIFIER_ACCEPTED, classifier_ops, classifier_eval, classifier_eval, reasoning)
        reasoning.append("classifier and RAG agree, but their shared sequence violates the text's own requirements -- agreement alone is not validity; no valid plan from either source")
        return ArbitrationResult(ArbitrationOutcome.NO_VALID_PLAN, classifier_ops, classifier_eval, classifier_eval, reasoning)

    rag_eval = evaluate_candidate(rag_ops, requirements)
    reasoning.append(
        f"RAG {rag_ops} (similarity={rag_similarity:.3f}): valid={rag_eval.valid}, coverage={rag_eval.requirement_coverage:.2f}"
        + ("; " + "; ".join(rag_eval.violations) if rag_eval.violations else "")
    )

    if not classifier_eval.valid and not rag_eval.valid:
        reasoning.append("neither candidate satisfies the structured requirements -- no valid plan; falling back to the classifier's own sequence as the least-bad disclosed default")
        return ArbitrationResult(ArbitrationOutcome.NO_VALID_PLAN, classifier_ops, classifier_eval, rag_eval, reasoning)

    if not classifier_eval.valid and rag_eval.valid:
        reasoning.append("classifier sequence violates requirements; RAG provides a compatible rescue")
        return ArbitrationResult(ArbitrationOutcome.RAG_RESCUE, rag_ops, classifier_eval, rag_eval, reasoning)

    if classifier_eval.valid and not rag_eval.valid:
        reasoning.append("RAG sequence violates requirements -- similarity alone cannot override a valid classifier plan")
        return ArbitrationResult(ArbitrationOutcome.CLASSIFIER_ACCEPTED, classifier_ops, classifier_eval, rag_eval, reasoning)

    # Both are valid and genuinely disagree -- score by requirement
    # coverage, then by simplicity (fewer ops) as a mild tiebreaker; a
    # TRUE tie (equal coverage, equal length) is exposed rather than
    # silently decided, per this module's own "expose ambiguity rather
    # than invent" principle -- but still returns the classifier's own
    # sequence as the usable default, since no UI exists yet to pause
    # and ask.
    if rag_eval.requirement_coverage > classifier_eval.requirement_coverage:
        reasoning.append("both valid; RAG has strictly better requirement coverage")
        return ArbitrationResult(ArbitrationOutcome.RAG_ASSISTED, rag_ops, classifier_eval, rag_eval, reasoning)
    if rag_eval.requirement_coverage < classifier_eval.requirement_coverage:
        reasoning.append("both valid; classifier has strictly better requirement coverage")
        return ArbitrationResult(ArbitrationOutcome.CLASSIFIER_ACCEPTED, classifier_ops, classifier_eval, rag_eval, reasoning)
    if len(rag_ops) < len(classifier_ops):
        reasoning.append("both valid with equal coverage; RAG is simpler (fewer operations)")
        return ArbitrationResult(ArbitrationOutcome.RAG_ASSISTED, rag_ops, classifier_eval, rag_eval, reasoning)
    if len(rag_ops) > len(classifier_ops):
        reasoning.append("both valid with equal coverage; classifier is simpler (fewer operations)")
        return ArbitrationResult(ArbitrationOutcome.CLASSIFIER_ACCEPTED, classifier_ops, classifier_eval, rag_eval, reasoning)
    reasoning.append("both valid, equal coverage, equal length, genuinely different sequences -- ambiguous; defaulting to the classifier's own independently-derived sequence, disclosed as ambiguous rather than silently decided")
    return ArbitrationResult(ArbitrationOutcome.NEEDS_DISAMBIGUATION, classifier_ops, classifier_eval, rag_eval, reasoning)
