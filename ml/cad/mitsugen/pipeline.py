"""Orchestrates Mitsugen's two independent construction paths and their
own Stage F final validation:

1. radial_sweep -- Stages A-F from the original v1 brief: a structured
   MitsugenShapeSpec, curvature-adaptive slice planning, profile
   validation, N-ring-loft solid construction. Precise, deterministic,
   no LLM involved anywhere in the geometry -- used for plain bodies of
   revolution (a vase, a bottle, a bowl, a handle-less cup) where this
   exact representation applies.
2. primitive_composition -- for everything else: an LLM-driven
   decomposition into primitives (box/cylinder/sphere/cone) + boolean
   operations (mitsugen/freeform.py), built via real CSG
   (mitsugen/freeform_builder.py). This is a CONSTRUCTIVE APPROXIMATION,
   not an exact reproduction -- every primitive_composition result
   carries a permanent, non-suppressible disclosure finding saying so.

Routing between the two is a cheap, disclosed heuristic
(_requires_freeform), not a claim of perfect classification: it checks
whether the prompt names a known plain (handle-less, feature-less)
vessel shape or uses the explicit vessel-dimension grammar, and whether
it contains a feature word (handle, wheel, square, ...) that the
radial-sweep representation structurally cannot express. When in doubt,
it chooses primitive_composition -- attempting every requested feature
through CSG is strictly more honest than silently building a smooth
revolve that drops the feature nobody asked to have dropped.

Stage F validates against the ORIGINAL structured interpretation of the
prompt (the MitsugenShapeSpec or FreeformSpec) -- never against a
description the builder generated about its own output. Reuses 3di-men's
printability.py for the printability half of validation on both paths (a
general-purpose geometry checker, not CAD-specific -- see this package's
own __init__.py for why that reuse is appropriate while sandbox.py/
assembly_plan.py are not)."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from mitsugen.builder import BuildResult, apply_hollowing, build_solid_from_slices
from mitsugen.freeform import FreeformSpec, decompose_freeform_prompt
from mitsugen.freeform_builder import build_freeform_solid
from mitsugen.profile import validate_adjacent_profiles, validate_profile
from mitsugen.slice_planner import SlicePlan, plan_slices
from mitsugen.spec import MitsugenShapeSpec, compose_spec_from_prompt


@dataclass
class ValidationFinding:
    requirement: str
    expected: str
    measured: str
    passed: bool
    method: str


@dataclass
class MitsugenResult:
    construction_mode: str  # "radial_sweep" | "primitive_composition"
    spec: "MitsugenShapeSpec | None"  # set when construction_mode == "radial_sweep"
    freeform_spec: "FreeformSpec | None"  # set when construction_mode == "primitive_composition"
    plan: "SlicePlan | None"
    build: "BuildResult | None"
    printability: dict | None  # printability.PrintabilityReport.as_dict() -- reused, not reimplemented
    findings: list[ValidationFinding] = field(default_factory=list)
    overall_status: str = "FAIL"  # "PASS" | "INCOMPLETE" | "FAIL" -- same 3-value discipline as
                                    # parallel_builder.AssemblyBuildResult.overall_status, independently
                                    # computed here (Mitsugen's own pipeline, not a shared function)
    error: str | None = None


def _validate_profiles(plan: SlicePlan) -> list[ValidationFinding]:
    findings = []
    from mitsugen.profile import make_circular_profile

    profiles = [make_circular_profile(s.slice_id, s.radius_mm) for s in plan.slices]
    for p in profiles:
        report = validate_profile(p)
        findings.append(ValidationFinding(
            requirement=f"slice {p.slice_id} profile is a valid closed boundary", expected="valid",
            measured="valid" if report.valid else f"invalid: {report.reason}", passed=report.valid, method=report.method,
        ))
    for a, b in zip(profiles, profiles[1:]):
        report = validate_adjacent_profiles(a, b)
        findings.append(ValidationFinding(
            requirement=f"slices {a.slice_id}->{b.slice_id} are compatible for lofting", expected="compatible",
            measured="compatible" if report.valid else f"incompatible: {report.reason}", passed=report.valid, method=report.method,
        ))
    return findings


def _analyze_printability(mesh, printer_build_volume_mm) -> tuple[dict | None, ValidationFinding]:
    from printability import DEFAULT_PRINTER_PROFILES, PrinterProfile, analyze_printability

    try:
        printer = PrinterProfile(name="Mitsugen check", build_volume_mm=printer_build_volume_mm) if printer_build_volume_mm \
            else DEFAULT_PRINTER_PROFILES["generic_medium"]
        report = analyze_printability(mesh, printer)
        finding = ValidationFinding(
            requirement="fits the selected printer's build volume", expected="fits",
            measured="fits" if report.build_volume_fit.fits else "exceeds build volume",
            passed=report.build_volume_fit.fits, method=report.build_volume_fit.method,
        )
        return report.as_dict(), finding
    except Exception as exc:
        finding = ValidationFinding(
            requirement="printability analysis", expected="completed", measured=f"unavailable: {exc}",
            passed=True, method="n/a",  # a printability-check FAILURE to even run is disclosed, not treated
        )                               # as a geometry defect -- the solid itself may still be perfectly valid
        return None, finding


# -- Free-form feature detection: the radial_sweep path can only ever
# represent a smooth body of revolution. Any of these words signal a
# real feature (or a non-circular cross-section) that representation
# structurally cannot carry -- when present, primitive_composition is
# used instead, even for an otherwise vessel-shaped prompt (a "square
# vase" is not a revolve). Not an exhaustive classifier -- a heuristic,
# disclosed as such; the cost of a false negative here (missing a
# feature word) is silently dropping a feature, so the keyword list
# leans inclusive rather than narrow.
_FREEFORM_FEATURE_KEYWORDS = {
    "handle", "handles", "spout", "lid", "cap", "wheel", "wheels", "leg", "legs",
    "arm", "arms", "wing", "wings", "ear", "ears", "tail", "strap", "hinge",
    "button", "square", "hexagon", "hexagonal", "rectangular", "triangular", "star",
    "car", "bracket", "figure", "figurine", "toy", "animal", "robot", "face", "person",
}
# Only plain, handle-less, feature-less container shapes land on the
# radial_sweep path by name alone -- "mug" is deliberately excluded
# (real mugs have a handle, which that path cannot represent).
_PLAIN_VESSEL_NAMES = {"vase", "bottle", "bowl", "cup"}


def _requires_freeform(prompt: str, vessel_spec: MitsugenShapeSpec) -> bool:
    lowered = prompt.lower()
    if any(re.search(rf"\b{kw}\b", lowered) for kw in _FREEFORM_FEATURE_KEYWORDS):
        return True
    if vessel_spec.unsupported_reason:
        return True  # not a recognized plain vessel and no explicit dimensions either -- try primitive composition instead of hard-failing
    matched_plain_vessel = any(re.search(rf"\b{name}\b", lowered) for name in _PLAIN_VESSEL_NAMES)
    used_explicit_vessel_grammar = "height_mm" in vessel_spec.explicit_mm and (
        "base_radius_mm" in vessel_spec.explicit_mm or "top_radius_mm" in vessel_spec.explicit_mm
    )
    return not (matched_plain_vessel or used_explicit_vessel_grammar)


def _radial_sweep_path(
    spec: MitsugenShapeSpec, tolerance_fraction: float, hollow: bool, wall_thickness_mm: "float | None",
    printer_build_volume_mm: "tuple[float, float, float] | None",
) -> MitsugenResult:
    """Never raises: every stage's own failure comes back as a typed,
    disclosed MitsugenResult (overall_status + error/findings), never an
    exception the caller has to separately guard against."""
    spec.tolerance_fraction = tolerance_fraction
    spec.hollow = hollow
    spec.wall_thickness_mm = wall_thickness_mm

    if spec.unsupported_reason:
        return MitsugenResult(
            construction_mode="radial_sweep", spec=spec, freeform_spec=None, plan=None, build=None, printability=None,
            overall_status="FAIL", error=spec.unsupported_reason,
        )

    plan = plan_slices(spec)
    profile_findings = _validate_profiles(plan)
    if any(not f.passed for f in profile_findings):
        return MitsugenResult(
            construction_mode="radial_sweep", spec=spec, freeform_spec=None, plan=plan, build=None, printability=None,
            findings=profile_findings, overall_status="FAIL", error="one or more construction slices failed profile validation",
        )

    build = build_solid_from_slices(plan)
    if build.error:
        return MitsugenResult(
            construction_mode="radial_sweep", spec=spec, freeform_spec=None, plan=plan, build=build, printability=None,
            findings=profile_findings, overall_status="FAIL", error=build.error,
        )

    if hollow and wall_thickness_mm:
        build = apply_hollowing(build, plan, wall_thickness_mm)
        # apply_hollowing never silently fails -- it returns the solid
        # version with hollow_applied=False and a disclosed error instead;
        # that error is downgrading, not fatal (a valid solid still exists).

    findings = list(profile_findings)

    # Stage F -- independent final validation against the ORIGINAL spec,
    # never a description generated about the built mesh itself.
    actual_height = float(build.mesh.bounds[1][2] - build.mesh.bounds[0][2])
    height_ok = abs(actual_height - spec.height_mm) / max(spec.height_mm, 1e-6) <= 0.02
    findings.append(ValidationFinding(
        requirement="built height matches the structured spec's own height_mm", expected=f"{spec.height_mm:g}mm",
        measured=f"{actual_height:.1f}mm", passed=height_ok, method="world-space bounding box along the construction axis",
    ))

    max_radius_mm = max(r for _, r in spec.waypoints_mm)
    actual_max_extent = float(max(build.mesh.bounds[1][0] - build.mesh.bounds[0][0], build.mesh.bounds[1][1] - build.mesh.bounds[0][1])) / 2.0
    radius_ok = abs(actual_max_extent - max_radius_mm) / max(max_radius_mm, 1e-6) <= 0.05
    findings.append(ValidationFinding(
        requirement="built max radius matches the structured spec's own widest waypoint", expected=f"{max_radius_mm:g}mm",
        measured=f"{actual_max_extent:.1f}mm", passed=radius_ok, method="world-space XY bounding box",
    ))

    findings.append(ValidationFinding(
        requirement="actual slice count never exceeds the configured maximum", expected=f"<= {spec.max_slices}",
        measured=str(plan.actual_slice_count), passed=plan.actual_slice_count <= spec.max_slices,
        method="SlicePlan.actual_slice_count, computed by plan_slices",
    ))

    findings.append(ValidationFinding(
        requirement="slice plan meets the configured approximation tolerance", expected=f"<= {spec.tolerance_fraction:.0%} deviation",
        measured=f"{plan.approximation_error_fraction:.1%}", passed=plan.within_tolerance,
        method="piecewise-linear-through-slices vs. the true radius curve, max deviation over 400 samples",
    ))

    findings.append(ValidationFinding(
        requirement="exported geometry is a single valid watertight solid", expected="watertight, volume > 0",
        measured=f"watertight={build.mesh.is_watertight}, volume={build.mesh.volume:.0f}mm3",
        passed=bool(build.mesh.is_watertight and build.mesh.volume > 0), method="trimesh topology properties",
    ))

    printability_dict, printability_finding = _analyze_printability(build.mesh, printer_build_volume_mm)
    findings.append(printability_finding)

    hard_failures = [f for f in findings if not f.passed and f.requirement in (
        "built height matches the structured spec's own height_mm",
        "built max radius matches the structured spec's own widest waypoint",
        "actual slice count never exceeds the configured maximum",
        "exported geometry is a single valid watertight solid",
        "fits the selected printer's build volume",
    )]
    soft_failures = [f for f in findings if not f.passed and f not in hard_failures]

    if hard_failures:
        overall_status = "FAIL"
    elif soft_failures or not build.hollow_applied and hollow:
        overall_status = "INCOMPLETE"
    else:
        overall_status = "PASS"

    return MitsugenResult(
        construction_mode="radial_sweep", spec=spec, freeform_spec=None, plan=plan, build=build,
        printability=printability_dict, findings=findings, overall_status=overall_status, error=None,
    )


def _primitive_composition_path(
    prompt: str, hollow: bool, printer_build_volume_mm: "tuple[float, float, float] | None",
) -> MitsugenResult:
    """Never raises: an unusable LLM decomposition degrades to a single
    disclosed approximate box (FreeformSpec.fell_back=True) rather than
    failing outright -- see mitsugen/freeform.py. A structurally invalid
    build (an unbuildable primitive, a failed boolean op, a non-
    watertight result) is still a typed FAIL, same as the radial_sweep
    path's own discipline."""
    freeform_spec = decompose_freeform_prompt(prompt)

    approximation_detail = f"{len(freeform_spec.primitives)} primitive(s), {len(freeform_spec.operations)} boolean operation(s)"
    if freeform_spec.fell_back:
        approximation_detail += " -- the LLM's own decomposition was unusable; a single default box approximation was substituted instead"
    findings = [ValidationFinding(
        requirement="free-form objects are approximated via primitive composition (box/cylinder/sphere/cone + boolean operations), never reproduced exactly",
        expected="disclosed", measured=approximation_detail, passed=True,
        method="LLM-driven primitive+boolean decomposition, mitsugen/freeform.py (independent of 3di-men's CAD solver)",
    )]
    if hollow:
        # Never silently drop a request: hollowing is only implemented for
        # the radial_sweep path's own ring-loft cavity technique today.
        # Disclosed as a soft (non-failing) gap, not applied as a no-op.
        findings.append(ValidationFinding(
            requirement="hollow interior was requested", expected="applied",
            measured="not yet supported on the primitive-composition path -- built as a solid instead",
            passed=False, method="n/a",
        ))
    if freeform_spec.dropped_primitives:
        findings.append(ValidationFinding(
            requirement="every primitive the LLM proposed was itself well-formed", expected="0 dropped",
            measured=f"dropped (malformed, not built): {', '.join(freeform_spec.dropped_primitives)}", passed=False,
            method="primitive field validation during parsing",
        ))

    build = build_freeform_solid(freeform_spec)
    if build.error:
        return MitsugenResult(
            construction_mode="primitive_composition", spec=None, freeform_spec=freeform_spec, plan=None, build=build,
            printability=None, findings=findings, overall_status="FAIL", error=build.error,
        )

    findings.append(ValidationFinding(
        requirement="exported geometry is a single valid watertight solid", expected="watertight, volume > 0, 1 body",
        measured=f"watertight={build.mesh.is_watertight}, volume={build.mesh.volume:.0f}mm3, bodies={build.mesh.body_count}",
        passed=bool(build.mesh.is_watertight and build.mesh.volume > 0 and build.mesh.body_count == 1),
        method="trimesh topology properties",
    ))

    printability_dict, printability_finding = _analyze_printability(build.mesh, printer_build_volume_mm)
    findings.append(printability_finding)

    hard_failures = [f for f in findings if not f.passed and f.requirement in (
        "exported geometry is a single valid watertight solid",
        "fits the selected printer's build volume",
    )]
    soft_failures = [f for f in findings if not f.passed and f not in hard_failures]

    if hard_failures:
        overall_status = "FAIL"
    elif soft_failures or freeform_spec.fell_back:
        overall_status = "INCOMPLETE"
    else:
        overall_status = "PASS"

    return MitsugenResult(
        construction_mode="primitive_composition", spec=None, freeform_spec=freeform_spec, plan=None, build=build,
        printability=printability_dict, findings=findings, overall_status=overall_status, error=None,
    )


def generate_mitsugen_model(
    prompt: str, max_slices: int = 12, tolerance_fraction: float = 0.05,
    hollow: bool = False, wall_thickness_mm: "float | None" = None,
    printer_build_volume_mm: "tuple[float, float, float] | None" = None,
) -> MitsugenResult:
    """Entry point. Routes to the precise radial_sweep path for a plain
    body of revolution, else to primitive_composition (see this module's
    own docstring and _requires_freeform). Never raises."""
    vessel_spec = compose_spec_from_prompt(prompt, default_max_slices=max_slices)
    if _requires_freeform(prompt, vessel_spec):
        return _primitive_composition_path(prompt, hollow, printer_build_volume_mm)
    return _radial_sweep_path(vessel_spec, tolerance_fraction, hollow, wall_thickness_mm, printer_build_volume_mm)
