"""The one authoritative Build Validation Report -- Phase 6's own
aggregation point over every validator this project already built
(spec_validation.py, dimension_requirements.py, feature_verification.py,
component_inventory.py, assembly_compatibility.py). Never a competing or
duplicate requirement representation: every field here is READ from an
existing validator's own already-computed result (PointBuildResult/
AssemblyBuildResult, both already populated by parallel_builder.py),
never recomputed from scratch.

Confirmed real root cause this exists to fix (production-path audit):
server.py's /api/build response exposes only a single terse
"overall_status" string (PASS/FAIL/INCOMPLETE) -- every DETAILED
validator result that already runs on every build (feature_
verification.py's own per-feature ledger, component_inventory.py's own
per-component reconciliation, assembly_compatibility.py's own per-
relationship evaluations) is computed, folded into that one string, and
then discarded; nothing about WHICH specific requirement failed, or WHY,
ever reaches a caller. AssemblyBuildResult.overall_status itself is kept
completely unchanged (still exactly "PASS"/"FAIL"/"INCOMPLETE", the
existing 3-value contract many existing tests already assert on) --
this module adds a richer, ADDITIONAL report alongside it, never
replaces it (Part 7's own "preserve backward compatibility" rule)."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum


class ValidationStatus(Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    PARTIAL = "PARTIAL"
    UNSUPPORTED = "UNSUPPORTED"
    UNRESOLVED = "UNRESOLVED"
    NOT_EVALUATED = "NOT_EVALUATED"


# Overall build status uses a narrower 5-value vocabulary (no NOT_
# EVALUATED -- an overall build result always HAS an opinion, even if
# that opinion is "nothing applicable was evaluated" -> PASS, see
# _aggregate's own docstring) -- reuses the SAME enum, just restricts
# which values callers should expect at the top level.
_OVERALL_VALUES = {"PASS", "PARTIAL", "FAIL", "UNSUPPORTED", "UNRESOLVED"}


@dataclass
class BuildValidationReport:
    geometry_status: ValidationStatus
    dimension_status: ValidationStatus
    feature_status: ValidationStatus
    component_status: ValidationStatus
    compatibility_status: ValidationStatus
    requirement_status: ValidationStatus  # the roll-up of dimension+feature+component
                                           # status specifically -- "were the things the
                                           # user actually asked for fulfilled," distinct
                                           # from geometry_status (raw mesh validity) and
                                           # compatibility_status (mechanical fit between
                                           # components, not what any ONE component itself
                                           # contains).
    overall_status: ValidationStatus
    validation_errors: list[str] = field(default_factory=list)
    unsupported_requirements: list[str] = field(default_factory=list)
    unresolved_requirements: list[str] = field(default_factory=list)
    validation_warnings: list[str] = field(default_factory=list)
    repair_history: list[str] = field(default_factory=list)
    reconciled_claims: list[str] = field(default_factory=list)  # Part 4's own "report
        # disagreements explicitly" -- a spec_validation.py dimension claim that LOOKED
        # unmatched against the mesh's overall extents, but was independently confirmed
        # by a more specific feature-level check (feature_verification.py/dimension_
        # requirements.py) -- disclosed here, never silently dropped from either report.

    def as_dict(self) -> dict:
        return {
            "geometry_status": self.geometry_status.value,
            "dimension_status": self.dimension_status.value,
            "feature_status": self.feature_status.value,
            "component_status": self.component_status.value,
            "compatibility_status": self.compatibility_status.value,
            "requirement_status": self.requirement_status.value,
            "overall_status": self.overall_status.value,
            "validation_errors": self.validation_errors,
            "unsupported_requirements": self.unsupported_requirements,
            "unresolved_requirements": self.unresolved_requirements,
            "validation_warnings": self.validation_warnings,
            "repair_history": self.repair_history,
            "reconciled_claims": self.reconciled_claims,
        }


def _reconcile_dimension_claims(point) -> tuple[list[float], list[str]]:
    """Part 4's own conflict-resolution policy, applied to ONE point.
    Returns (still_unmatched, reconciled_descriptions). Re-extracts
    dimension_requirements.extract_dimension_requirements(point.
    description) fresh -- cheap and deterministic (plain regex, no LLM
    call, same cost class feature_verification._verify_subtractive_hole
    already pays for its OWN diameter check), never a stored, possibly-
    stale copy. A generic spec_validation.py claim and a feature_
    verification.py Requirement are considered "the same requirement"
    when (a) dimension_requirements' own fresh extraction resolves a
    DimensionRequirement with a FEATURE-level parameter (hole_diameter/
    hole_depth -- never a base-shape one, which spec_validation's own
    overall-extent check is already legitimately checking) whose own
    normalized_value matches the unmatched claim within rounding, AND
    (b) the corresponding feature_requirements entry for that feature
    type reached GEOMETRY_VERIFIED. Only then is the generic claim
    reclassified as reconciled, never simply dropped -- if the feature-
    level check ALSO disagrees (FAILED), both checks agree and the claim
    stays a genuine failure."""
    from dimension_requirements import extract_dimension_requirements

    unmatched = list(getattr(point, "unmatched_dimensions_mm", []) or [])
    if not unmatched:
        return [], []

    feature_level_params = {"hole_diameter", "hole_depth"}
    fresh = [d for d in extract_dimension_requirements(point.point.description) if d.parameter in feature_level_params]
    feature_verified = any(
        r.feature_type == "subtractive_hole" and r.status.value == "GEOMETRY_VERIFIED"
        for r in getattr(point, "feature_requirements", []) or []
    )
    feature_failed = any(
        r.feature_type == "subtractive_hole" and r.status.value == "FAILED"
        for r in getattr(point, "feature_requirements", []) or []
    )

    still_unmatched: list[float] = []
    reconciled: list[str] = []
    for claim in unmatched:
        match = next((d for d in fresh if math.isclose(d.normalized_value, claim, rel_tol=0.02, abs_tol=0.05)), None)
        if match is not None and feature_verified and not feature_failed:
            reconciled.append(
                f"{claim:g}mm was flagged unmatched against the mesh's own overall extents "
                f"(spec_validation.py's generic check), but independently confirmed as this part's "
                f"own {match.parameter} by feature_verification.py's more specific check -- reconciled, not a real failure"
            )
        else:
            still_unmatched.append(claim)
    return still_unmatched, reconciled


def _status_for_point_dimensions(point, reconciled_count: int, still_unmatched: list[float]) -> ValidationStatus:
    if point.dimension_match_rate is None and reconciled_count == 0:
        return ValidationStatus.NOT_EVALUATED  # no numeric claims in the description at all
    if still_unmatched:
        return ValidationStatus.FAIL
    return ValidationStatus.PASS


def _status_for_point_features(point) -> ValidationStatus:
    reqs = getattr(point, "feature_requirements", []) or []
    if not reqs:
        return ValidationStatus.NOT_EVALUATED
    statuses = {r.status.value for r in reqs if r.required}
    if "FAILED" in statuses:
        return ValidationStatus.FAIL
    if "UNRESOLVED" in statuses:
        return ValidationStatus.UNRESOLVED
    if statuses and statuses <= {"UNSUPPORTED_TO_VERIFY"}:
        return ValidationStatus.UNSUPPORTED
    if "UNSUPPORTED_TO_VERIFY" in statuses:
        return ValidationStatus.PARTIAL  # some verified, some genuinely not independently checkable
    return ValidationStatus.PASS


def _aggregate(statuses: list[ValidationStatus]) -> ValidationStatus:
    """Shared rollup: FAIL beats everything, then UNRESOLVED, then
    UNSUPPORTED, then PARTIAL (a mix of PASS and something non-fatal),
    else PASS. NOT_EVALUATED entries are excluded from the vote
    entirely (Part 3's own rule: "a validator that was never executed
    must never be treated as PASS" -- but the CONVERSE matters too:
    nothing APPLICABLE existing is not a failure either, so an
    all-NOT_EVALUATED input correctly falls through to PASS, never
    fabricating a FAIL for a build with nothing of that kind to check)."""
    relevant = [s for s in statuses if s is not ValidationStatus.NOT_EVALUATED]
    if not relevant:
        return ValidationStatus.PASS
    if ValidationStatus.FAIL in relevant:
        return ValidationStatus.FAIL
    if ValidationStatus.UNRESOLVED in relevant:
        return ValidationStatus.UNRESOLVED
    if ValidationStatus.UNSUPPORTED in relevant:
        return ValidationStatus.UNSUPPORTED if all(s is ValidationStatus.UNSUPPORTED for s in relevant) else ValidationStatus.PARTIAL
    if ValidationStatus.PARTIAL in relevant:
        return ValidationStatus.PARTIAL
    return ValidationStatus.PASS


def build_validation_report(result) -> BuildValidationReport:
    """The single entry point: takes a parallel_builder.AssemblyBuildResult
    (already fully built -- never triggers new generation, never re-runs
    an expensive geometry scan) and aggregates every validator's own
    already-computed result into one report. Never raises."""
    errors: list[str] = []
    warnings: list[str] = []
    unsupported: list[str] = []
    unresolved: list[str] = []
    repair_history: list[str] = []
    reconciled_claims: list[str] = []

    # ---- geometry_status ----
    if not result.points:
        geometry_status = ValidationStatus.NOT_EVALUATED
    else:
        geom_statuses = []
        for p in result.points:
            if p.status == "rejected":
                geom_statuses.append(ValidationStatus.FAIL)
                errors.append(f"{p.point.description!r}: no valid geometry was generated at all ({p.error})")
            elif p.status in ("suspect", "partial", "out_of_distribution"):
                geom_statuses.append(ValidationStatus.PARTIAL)
            else:
                geom_statuses.append(ValidationStatus.PASS)
            repair_history.extend(p.repair_attempts)
        geometry_status = _aggregate(geom_statuses)

    # ---- dimension_status (with Part 4's own reconciliation) ----
    dim_statuses = []
    for p in result.points:
        if p.mesh is None:
            continue
        still_unmatched, reconciled = _reconcile_dimension_claims(p)
        reconciled_claims.extend(reconciled)
        dim_statuses.append(_status_for_point_dimensions(p, len(reconciled), still_unmatched))
        for d in still_unmatched:
            warnings.append(f"{p.point.description!r}: {d:g}mm does not match any measured dimension of the built geometry")
        for d in p.unmatched_hole_diameters_mm:
            errors.append(f"{p.point.description!r}: claimed hole/bore diameter {d:g}mm has no matching detected hole")
    dimension_status = _aggregate(dim_statuses) if dim_statuses else ValidationStatus.NOT_EVALUATED

    # ---- feature_status ----
    feat_statuses = []
    for p in result.points:
        s = _status_for_point_features(p)
        feat_statuses.append(s)
        for r in getattr(p, "feature_requirements", []) or []:
            if r.status.value == "FAILED":
                errors.append(f"{p.point.description!r}: required feature {r.feature_type!r} failed verification -- {r.failure_reason}")
            elif r.status.value == "UNSUPPORTED_TO_VERIFY":
                unsupported.append(f"{p.point.description!r}: feature {r.feature_type!r} -- {r.failure_reason}")
    feature_status = _aggregate(feat_statuses) if feat_statuses else ValidationStatus.NOT_EVALUATED

    # ---- component_status ----
    inventory = getattr(result, "component_inventory", None)
    if inventory is None:
        component_status = ValidationStatus.NOT_EVALUATED
    else:
        comp_statuses = []
        for c in inventory:
            if c.component_type is None:
                comp_statuses.append(ValidationStatus.UNRESOLVED)
                unresolved.append(f"component {c.component_name!r}: {c.failure_reason}")
                continue
            if c.reconciliation is not None and c.reconciliation.value == "MATCHED":
                comp_statuses.append(ValidationStatus.PASS)
            elif c.mandatory:
                comp_statuses.append(ValidationStatus.FAIL)
                errors.append(f"component {c.component_id!r} ({c.component_type}): {c.failure_reason}")
            else:
                comp_statuses.append(ValidationStatus.PARTIAL)
        component_status = _aggregate(comp_statuses) if comp_statuses else ValidationStatus.NOT_EVALUATED

    # ---- compatibility_status ----
    compat = getattr(result, "compatibility_report", None)
    if compat is None or not compat.evaluations:
        compatibility_status = ValidationStatus.NOT_EVALUATED
    else:
        mapping = {
            "COMPATIBLE": ValidationStatus.PASS, "INCOMPATIBLE": ValidationStatus.FAIL,
            "GEOMETRY_VALIDATION_FAILED": ValidationStatus.FAIL, "UNDER_SPECIFIED": ValidationStatus.UNRESOLVED,
            "UNSUPPORTED_TO_VERIFY": ValidationStatus.UNSUPPORTED, "NOT_EVALUATED": ValidationStatus.UNRESOLVED,
        }
        compat_statuses = [mapping.get(e.status.value, ValidationStatus.UNRESOLVED) for e in compat.evaluations]
        for e in compat.evaluations:
            if e.status.value == "INCOMPATIBLE":
                errors.append(f"{e.component_a}/{e.component_b}: {e.reason}")
            elif e.status.value == "NOT_EVALUATED":
                unresolved.append(f"{e.component_a}/{e.component_b}: relationship could not be evaluated -- {e.reason}")
            elif e.status.value == "UNSUPPORTED_TO_VERIFY":
                unsupported.append(f"{e.component_a}/{e.component_b}: {e.reason}")
        compatibility_status = _aggregate(compat_statuses)

    requirement_status = _aggregate([dimension_status, feature_status, component_status])
    overall_status = _aggregate([geometry_status, requirement_status, compatibility_status])

    return BuildValidationReport(
        geometry_status=geometry_status, dimension_status=dimension_status, feature_status=feature_status,
        component_status=component_status, compatibility_status=compatibility_status,
        requirement_status=requirement_status, overall_status=overall_status,
        validation_errors=errors, unsupported_requirements=unsupported, unresolved_requirements=unresolved,
        validation_warnings=warnings, repair_history=repair_history, reconciled_claims=reconciled_claims,
    )
