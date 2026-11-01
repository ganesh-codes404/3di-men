"""The Assembly Compatibility and Constraint Engine -- validates that
components an assembly's own INTENDED relationships connect actually
fit/mesh/mate with each other, not just that each one individually
satisfies its own stated dimensions (Problem 2's own scope) or that the
op-sequence choosing its geometry was correct (Problem 1's). Confirmed
real gap this closes: nothing anywhere in this pipeline previously
checked "is this shaft the right SIZE for the bore it's concentric_with"
or "do these two gears' own module/tooth-count/center-distance actually
let them mesh" -- assembly_plan.py resolves WHERE a related part goes
(and llm_decompose._correct_gear_mesh_spacing even corrects gear center
distance AT PLACEMENT TIME), but nothing independently VALIDATES, after
the fact, that the relationship it placed is actually mechanically
sound.

Deliberately NOT a parallel assembly system: this reads the EXISTING
Relationship objects (assembly_plan.Relationship -- attached_to_face/
concentric_with/parallel_to/symmetric_about) as the authoritative record
of user intent (see llm_decompose.py's own system prompt: a relationship
IS how intent is captured and preserved through planning, exactly what
this module's own Part 2 calls for), and the EXISTING placed meshes
(parallel_builder.py's placed_meshes_by_id, built from the SAME
sandbox.py/program_model.py pipeline Problem 1/2 already validate) as
the actual, independently-measured geometry. The one new piece is
classifying a spatial Relationship into a MECHANICAL relationship type
(shaft-bore fit, gear mesh, bolt-hole fit, motor mount, bearing-housing
fit, or a generic face/axis relationship) from the two connected
components' own description text, then evaluating that type
deterministically against the real built geometry.

Reuses, rather than re-derives:
- llm_decompose._extract_gear_params for module/tooth-count (the SAME
  regex _correct_gear_mesh_spacing already trusts for gear-mesh
  arithmetic).
- evaluation.geometry_eval.evaluate_mesh's hole_diameters_mm/hole_axes/
  hole_through/hole_positions_mm (the last extended by this same effort
  -- see that module's own docstring -- specifically because comparing
  two parts' own hole LAYOUTS needed position data that existed
  internally but was never exposed before).
- dimension_requirements.extract_dimension_requirements for any
  explicit tolerance/fit-class/clearance number stated in a component's
  own description text (Problem 2's typed requirement system, per this
  brief's own Part 12 instruction to integrate with it rather than
  build a competing numeric parser).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum

import numpy as np
import trimesh

from dimension_requirements import extract_dimension_requirements
from llm_decompose import _extract_gear_params

_AXIS_INDEX = {"x": 0, "y": 1, "z": 2}


class CompatibilityStatus(Enum):
    COMPATIBLE = "COMPATIBLE"
    INCOMPATIBLE = "INCOMPATIBLE"
    UNDER_SPECIFIED = "UNDER_SPECIFIED"
    UNSUPPORTED_TO_VERIFY = "UNSUPPORTED_TO_VERIFY"
    GEOMETRY_VALIDATION_FAILED = "GEOMETRY_VALIDATION_FAILED"
    NOT_EVALUATED = "NOT_EVALUATED"


class MechanicalRelationshipType(Enum):
    SHAFT_BORE_FIT = "shaft_bore_fit"
    BEARING_HOUSING_FIT = "bearing_housing_fit"
    GEAR_MESH = "external_spur_gear_mesh"
    BOLT_HOLE_FIT = "bolt_hole_fit"
    MOTOR_MOUNT = "motor_mount"
    FACE_CONTACT = "face_contact"
    AXIS_ALIGNMENT = "axis_alignment"
    UNSUPPORTED = "unsupported"


@dataclass
class RelationshipEvaluation:
    relationship_id: str
    component_a: str
    component_b: str
    relationship_type: MechanicalRelationshipType
    source_phrase_a: str
    source_phrase_b: str
    requested: dict = field(default_factory=dict)
    actual: dict = field(default_factory=dict)
    calculated: dict = field(default_factory=dict)
    status: CompatibilityStatus = CompatibilityStatus.NOT_EVALUATED
    reason: str = ""
    suggested_action: "str | None" = None


@dataclass
class AssemblyCompatibilityReport:
    evaluations: list[RelationshipEvaluation] = field(default_factory=list)

    @property
    def overall_status(self) -> CompatibilityStatus:
        if not self.evaluations:
            return CompatibilityStatus.NOT_EVALUATED
        if any(e.status is CompatibilityStatus.INCOMPATIBLE for e in self.evaluations):
            return CompatibilityStatus.INCOMPATIBLE
        if any(e.status is CompatibilityStatus.GEOMETRY_VALIDATION_FAILED for e in self.evaluations):
            return CompatibilityStatus.GEOMETRY_VALIDATION_FAILED
        # Confirmed real bug (adversarial audit, live-reproduced): a
        # relationship whose own endpoint component never built at all
        # (excluded, omitted by decomposition, rejected) is evaluated as
        # NOT_EVALUATED by _evaluate_relationship_pair -- but this
        # rollup never checked for that status at all, falling through
        # every explicit branch to the default COMPATIBLE. A missing
        # relationship endpoint used to report the WHOLE assembly as
        # compatible, the opposite of truthful -- a stated relationship
        # that was never actually checked must never read the same as
        # one that passed.
        if any(e.status is CompatibilityStatus.NOT_EVALUATED for e in self.evaluations):
            return CompatibilityStatus.NOT_EVALUATED
        if any(e.status is CompatibilityStatus.UNDER_SPECIFIED for e in self.evaluations):
            return CompatibilityStatus.UNDER_SPECIFIED
        if any(e.status is CompatibilityStatus.UNSUPPORTED_TO_VERIFY for e in self.evaluations):
            return CompatibilityStatus.UNSUPPORTED_TO_VERIFY
        return CompatibilityStatus.COMPATIBLE

    def counts(self) -> dict:
        out = {s.value: 0 for s in CompatibilityStatus}
        for e in self.evaluations:
            out[e.status.value] += 1
        return out

    def render(self) -> str:
        lines = ["ASSEMBLY COMPATIBILITY REPORT", ""]
        for e in self.evaluations:
            lines.append(f"Component A: {e.component_a}")
            lines.append(f"Component B: {e.component_b}")
            lines.append("")
            lines.append(f"Relationship: {e.relationship_type.value}")
            lines.append("")
            if e.requested:
                lines.append("Requested:")
                for k, v in e.requested.items():
                    lines.append(f"  {k}: {v}")
                lines.append("")
            if e.actual:
                lines.append("Actual:")
                for k, v in e.actual.items():
                    lines.append(f"  {k}: {v}")
                lines.append("")
            if e.calculated:
                lines.append("Calculated:")
                for k, v in e.calculated.items():
                    lines.append(f"  {k}: {v}")
                lines.append("")
            lines.append("Status:")
            lines.append(f"  {e.status.value}")
            if e.reason:
                lines.append("")
                lines.append("Reason:")
                lines.append(f"  {e.reason}")
            if e.suggested_action:
                lines.append("")
                lines.append("Suggested corrective action:")
                lines.append(f"  {e.suggested_action}")
            lines.append("")
            lines.append("-" * 33)
            lines.append("")
        counts = self.counts()
        lines.append("OVERALL ASSEMBLY STATUS:")
        lines.append(f"  {self.overall_status.value}")
        lines.append("")
        lines.append(f"Compatible relationships: {counts[CompatibilityStatus.COMPATIBLE.value]}")
        lines.append(f"Incompatible relationships: {counts[CompatibilityStatus.INCOMPATIBLE.value]}")
        lines.append(f"Under-specified relationships: {counts[CompatibilityStatus.UNDER_SPECIFIED.value]}")
        lines.append(f"Unsupported relationships: {counts[CompatibilityStatus.UNSUPPORTED_TO_VERIFY.value]}")
        return "\n".join(lines)


# ---------------- classification: spatial Relationship -> mechanical type ----------------

_WORD = {
    "gear": r"\bgears?\b", "shaft": r"\bshafts?\b", "bearing": r"\bbearings?\b",
    "bore_or_housing": r"\bbores?\b|\bhousings?\b", "bolt": r"\bbolts?\b|\bscrews?\b|\bfasteners?\b",
    "hole": r"\bholes?\b", "motor": r"\bmotors?\b|\bsteppers?\b|\bnema\b",
    "mount": r"\bmounts?\b|\bbrackets?\b|\bplates?\b",
}


def _has(word: str, text: str) -> bool:
    return bool(re.search(_WORD[word], text, re.IGNORECASE))


_NON_SPUR_GEAR_TYPE_RE = re.compile(r"\b(helical|bevel|worm|internal|hypoid|miter)\s*gear", re.IGNORECASE)


def classify_relationship_type(description_a: str, description_b: str, relationship_kind: str) -> MechanicalRelationshipType:
    """Deterministic, keyword-driven -- never asks an LLM to decide what
    kind of mechanical relationship this is (Part 5's own rule: "Do not
    let the LLM decide whether a relationship is mechanically valid").
    Order matters: checked most-specific-first so a part mentioning both
    "bearing" and "housing" is never miscategorized as a looser, more
    generic relationship once a more specific one actually matches.
    Symmetric in (a, b) -- which side mentions which keyword doesn't
    matter, only that both sides together name a recognized pair."""
    gear_a, gear_b = _extract_gear_params(description_a), _extract_gear_params(description_b)
    if gear_a is not None and gear_b is not None:
        # llm_decompose._extract_gear_params matches "module N, M teeth"
        # regardless of gear TYPE -- a helical/bevel/worm/internal gear
        # phrased that way would otherwise be silently treated as a
        # standard external spur gear and run through the wrong formula
        # (Part 3B's own explicit rule: "do not apply the external spur
        # gear formula blindly" to one of these). Checked here, before
        # returning GEAR_MESH, so an unsupported gear type is classified
        # as UNSUPPORTED rather than reaching evaluate_gear_mesh at all.
        if _NON_SPUR_GEAR_TYPE_RE.search(description_a) or _NON_SPUR_GEAR_TYPE_RE.search(description_b):
            return MechanicalRelationshipType.UNSUPPORTED
        return MechanicalRelationshipType.GEAR_MESH

    bearing_a, bearing_b = _has("bearing", description_a), _has("bearing", description_b)
    bore_a, bore_b = _has("bore_or_housing", description_a), _has("bore_or_housing", description_b)
    shaft_a, shaft_b = _has("shaft", description_a), _has("shaft", description_b)
    bolt_a, bolt_b = _has("bolt", description_a), _has("bolt", description_b)
    hole_a, hole_b = _has("hole", description_a), _has("hole", description_b)
    motor_a, motor_b = _has("motor", description_a), _has("motor", description_b)
    mount_a, mount_b = _has("mount", description_a), _has("mount", description_b)

    # Every diametral-fit type below (shaft-bore, bearing-housing, bolt-
    # hole) is fundamentally a COAXIAL relationship -- it only means
    # anything evaluated against concentric_with/parallel_to, the only
    # two assembly_plan.Relationship kinds that even carry a meaningful
    # `axis`. Confirmed real gap this check closes (adversarial audit,
    # live-reproduced): descriptions mentioning "shaft" and "bearing"
    # used to classify as SHAFT_BORE_FIT regardless of the DECLARED
    # relationship kind -- an attached_to_face relationship (a flush-
    # face semantic, never concentric) between the same two components
    # still silently ran the full axis/concentricity/clearance check,
    # with evaluate_shaft_bore_fit falling back to inferring an axis
    # from the mesh alone rather than disclosing that the declared
    # relationship doesn't match what the descriptions imply. A
    # genuine mismatch here is reported as UNSUPPORTED (never silently
    # evaluated as if it were the OTHER, more generic type either --
    # FACE_CONTACT/AXIS_ALIGNMENT would be just as wrong a guess about
    # which of the two conflicting signals to trust).
    is_coaxial_kind = relationship_kind in ("concentric_with", "parallel_to")
    keyword_suggests_diametral_fit = (
        (bearing_a and bore_b) or (bearing_b and bore_a)
        or (shaft_a and (bearing_b or bore_b)) or (shaft_b and (bearing_a or bore_a))
        or (bolt_a and hole_b) or (bolt_b and hole_a)
    )
    if keyword_suggests_diametral_fit and not is_coaxial_kind:
        return MechanicalRelationshipType.UNSUPPORTED
    if is_coaxial_kind:
        if (bearing_a and bore_b) or (bearing_b and bore_a):
            return MechanicalRelationshipType.BEARING_HOUSING_FIT
        if (shaft_a and (bearing_b or bore_b)) or (shaft_b and (bearing_a or bore_a)):
            return MechanicalRelationshipType.SHAFT_BORE_FIT
        if (bolt_a and hole_b) or (bolt_b and hole_a):
            return MechanicalRelationshipType.BOLT_HOLE_FIT
        return MechanicalRelationshipType.AXIS_ALIGNMENT

    if relationship_kind == "attached_to_face" and ((motor_a and mount_b) or (motor_b and mount_a)):
        return MechanicalRelationshipType.MOTOR_MOUNT
    if relationship_kind == "attached_to_face":
        return MechanicalRelationshipType.FACE_CONTACT
    return MechanicalRelationshipType.UNSUPPORTED


# ---------------- measurement helpers (real built geometry, never ToolCall.args) ----------------

def _cylindrical_diameter_mm(mesh: trimesh.Trimesh, axis: str) -> float:
    """A plain shaft/boss/bolt's own diameter -- the two extents
    PERPENDICULAR to its running axis, averaged (a true cylinder has
    them equal; averaging is a cheap robustness margin against boolean-
    op triangulation noise, not a sign of ambiguity)."""
    perp = [i for i in range(3) if i != _AXIS_INDEX[axis]]
    return float((mesh.extents[perp[0]] + mesh.extents[perp[1]]) / 2.0)


def _infer_rotational_axis(mesh: trimesh.Trimesh, hint_axis: "str | None" = None) -> str:
    """Which world axis a cylindrical part's own rotational axis runs
    along, inferred from its OWN mesh extents -- the running axis is the
    one whose extent is NOT approximately equal to the other two (a
    cylinder/gear is isotropic in its two diameter directions, distinct
    from its own height/face-width). This is ALWAYS measured from the
    mesh's own already-placed (world/assembly-frame) geometry, never
    from a relationship's declared axis -- see this module's own E50
    fix note on `_evaluate_cylinder_in_bore`/`evaluate_axis_alignment`
    for why a declared axis must never substitute for this measurement.

    `hint_axis` is used ONLY as a tie-break for a shape whose own
    extents are genuinely ambiguous -- either a near-cube boss with no
    anisotropic extent at all, or (less obviously) a cylinder whose
    height happens to equal its own diameter, which looks EXACTLY like
    a cube in axis-aligned-bounding-box terms (D==D==H), so MORE than
    one axis passes the "other two extents agree" test below. A real,
    UNAMBIGUOUS measurement (exactly one axis qualifies) is never
    overridden by the hint -- "never trust declared over measured" still
    holds whenever there's something conclusive to measure. Callers pass
    the relationship's own declared axis here (when one exists) purely
    as a best-effort disambiguation hint for the genuinely-ambiguous
    case, falling back to this project's own standing "z" default when
    there's no usable hint either."""
    extents = mesh.extents
    candidates = [
        axis for axis, idx in _AXIS_INDEX.items()
        if abs(extents[(idx + 1) % 3] - extents[(idx + 2) % 3]) < 0.05 * max(extents[(idx + 1) % 3], extents[(idx + 2) % 3], 1e-6)
        and extents[idx] > 1e-6
    ]
    if len(candidates) == 1:
        return candidates[0]
    if hint_axis in candidates:
        return hint_axis
    if "z" in candidates:
        return "z"
    return candidates[0] if candidates else "z"


def _nearest_hole(geometry_report, axis: "str | None" = None, near_position: "tuple | None" = None) -> "dict | None":
    """The detected hole (evaluation.geometry_eval.GeometryReport's own
    row-aligned hole_* lists) best matching `axis` and/or closest to
    `near_position` -- never just "the first one," so a target with
    SEVERAL holes (a plate with many mounting holes) still resolves to
    the specific one a relationship actually points at."""
    indices = list(range(geometry_report.detected_hole_count))
    if axis is not None:
        indices = [i for i in indices if geometry_report.hole_axes[i] == axis]
    if not indices:
        return None
    if near_position is not None and len(indices) > 1:
        def _dist2(i: int) -> float:
            p = geometry_report.hole_positions_mm[i]
            return sum((p[k] - near_position[k]) ** 2 for k in range(3))
        indices.sort(key=_dist2)
    i = indices[0]
    return {
        "diameter_mm": geometry_report.hole_diameters_mm[i],
        "position_mm": geometry_report.hole_positions_mm[i],
        "through": geometry_report.hole_through[i],
        "axis": geometry_report.hole_axes[i],
    }


_CLEARANCE_RE = re.compile(r"(\d+(?:\.\d+)?)\s*mm\s*clearance|clearance\s*(?:of\s*)?(\d+(?:\.\d+)?)\s*mm", re.IGNORECASE)
_FIT_CLASS_RE = re.compile(r"\b(interference|press|line-to-line|slip|clearance)\s*fit\b", re.IGNORECASE)


def _stated_clearance_requirement_mm(description_a: str, description_b: str) -> "float | None":
    for text in (description_a, description_b):
        m = _CLEARANCE_RE.search(text)
        if m:
            return float(m.group(1) or m.group(2))
    return None


def _stated_fit_class(description_a: str, description_b: str) -> "str | None":
    for text in (description_a, description_b):
        m = _FIT_CLASS_RE.search(text)
        if m:
            return m.group(1).lower()
    return None


_ALIGNMENT_TOLERANCE_MM = 1.0  # how far off-center two supposedly-concentric features may be before this calls it a real misalignment, not measurement noise


def _evaluate_diametral_fit(
    inner_label: str, inner_diameter_mm: "float | None", outer_label: str, outer_diameter_mm: "float | None",
    description_a: str, description_b: str,
) -> tuple[CompatibilityStatus, dict, str, "str | None"]:
    """The shared clearance/interference calculation Part 3A/C/E all
    reduce to: diametral_clearance = outer - inner. Never classifies a
    positive clearance as a precision fit it was never told to check
    (Part 3A's own explicit rule) -- only validates against a fit class
    or numeric clearance if the text actually states one."""
    if inner_diameter_mm is None or outer_diameter_mm is None:
        missing = inner_label if inner_diameter_mm is None else outer_label
        return (
            CompatibilityStatus.UNDER_SPECIFIED, {},
            f"{missing}'s own diameter could not be determined from the built geometry -- insufficient information to establish fit",
            None,
        )

    clearance = outer_diameter_mm - inner_diameter_mm
    calculated = {"diametral_clearance_mm": round(clearance, 3)}
    stated_clearance = _stated_clearance_requirement_mm(description_a, description_b)
    fit_class = _stated_fit_class(description_a, description_b)

    # Checked BEFORE the blanket "negative clearance == incompatible"
    # rule below: an interference/press fit is NEGATIVE clearance BY
    # DEFINITION -- that rule would otherwise unconditionally reject
    # the exact geometry a correctly-designed interference fit produces
    # (confirmed real bug, adversarial audit: a stated "interference
    # fit" with the shaft genuinely larger than the bore used to be
    # reported INCOMPATIBLE for being an interference, never reaching
    # this check at all).
    if fit_class in ("interference", "press"):
        if clearance > 1e-6:
            return (
                CompatibilityStatus.INCOMPATIBLE, calculated,
                f"an {fit_class} fit was specified, but the actual diametral clearance is {clearance:.2f}mm (positive -- these parts would not need to be pressed together at all)",
                f"reduce {outer_label}'s own diameter so it is smaller than {inner_label}'s",
            )
        return CompatibilityStatus.COMPATIBLE, calculated, "", None

    if clearance < 0:
        reason = f"{inner_label} ({inner_diameter_mm:.2f}mm) is larger than {outer_label} ({outer_diameter_mm:.2f}mm) -- interference, not clearance"
        suggestion = f"increase {outer_label} to at least {inner_diameter_mm:.2f}mm, or reduce {inner_label} below that"
        return CompatibilityStatus.INCOMPATIBLE, calculated, reason, suggestion

    if stated_clearance is not None:
        tolerance = max(0.05, stated_clearance * 0.2)
        if abs(clearance - stated_clearance) > tolerance:
            return (
                CompatibilityStatus.INCOMPATIBLE, calculated,
                f"a clearance of {stated_clearance:g}mm was specified, but the actual built clearance is {clearance:.2f}mm",
                f"adjust {outer_label}'s own diameter to {inner_diameter_mm + stated_clearance:.2f}mm",
            )
        return CompatibilityStatus.COMPATIBLE, calculated, "", None

    # No stated tolerance/fit-class anywhere in either description --
    # report the actual clearance and classify it honestly, never
    # invent a precision fit it was never asked to meet.
    calculated["fit_classification"] = "line-to-line fit" if clearance < 1e-6 else "clearance fit (no tolerance or fit class specified in the request)"
    return CompatibilityStatus.COMPATIBLE, calculated, "", None


def _axial_alignment_check(mesh_a: trimesh.Trimesh, mesh_b: trimesh.Trimesh, axis: str) -> "tuple[bool, float]":
    """True (aligned) when the two meshes' own centers agree on BOTH
    coordinates perpendicular to `axis`, within _ALIGNMENT_TOLERANCE_MM
    -- measured from the ACTUAL placed meshes, never assumed from
    whatever relationship resolved their positions (Part 7's own "must
    use actual generated geometry... never blindly trust the planner"
    rule, generalized from dimensions to positions/alignment too)."""
    perp = [i for i in range(3) if i != _AXIS_INDEX[axis]]
    center_a, center_b = mesh_a.bounds.mean(axis=0), mesh_b.bounds.mean(axis=0)
    offset = float(np.linalg.norm([center_a[i] - center_b[i] for i in perp]))
    return offset <= _ALIGNMENT_TOLERANCE_MM, offset


def _evaluate_cylinder_in_bore(
    relationship_id: str, id_a: str, id_b: str, description_a: str, description_b: str,
    mesh_a: trimesh.Trimesh, mesh_b: trimesh.Trimesh, relationship, mechanical_type: MechanicalRelationshipType,
    cylinder_role_keyword: str, cylinder_label: str, bore_label: str,
) -> RelationshipEvaluation:
    """Shared by Parts 3A/C/E (shaft-bore, bolt-hole, bearing-housing are
    all the SAME shape of check: one component's own plain cylindrical
    feature against another's detected bore) -- parameterized only by
    which keyword decides the "cylinder" role and the labels used in the
    report. Classifies by keyword, then: (1) checks the bore exists in
    the built geometry at all, (2) checks the bore's own axis matches
    the cylinder's, (3) checks their centers are actually concentric,
    (4) calculates diametral clearance. Any earlier check failing
    short-circuits the later ones -- a misaligned pair is reported as
    misaligned, never diluted into a clearance number that was never
    really comparable in the first place."""
    from evaluation.geometry_eval import evaluate_mesh

    cyl_is_a = _has(cylinder_role_keyword, description_a) or not _has(cylinder_role_keyword, description_b)
    cyl_id, bore_id = (id_a, id_b) if cyl_is_a else (id_b, id_a)
    cyl_mesh, bore_mesh = (mesh_a, mesh_b) if cyl_is_a else (mesh_b, mesh_a)
    cyl_desc, bore_desc = (description_a, description_b) if cyl_is_a else (description_b, description_a)

    eva = RelationshipEvaluation(
        relationship_id=relationship_id, component_a=cyl_id, component_b=bore_id,
        relationship_type=mechanical_type, source_phrase_a=cyl_desc, source_phrase_b=bore_desc,
    )

    # E50 fix (coordinate-frame-correct measurement): the cylinder's own
    # axis is ALWAYS independently measured from its own actual built
    # geometry (already in world/assembly coordinates by the time it
    # reaches this module), never assumed to equal relationship.axis.
    # relationship.axis is the OWNER's DECLARED rotation target --
    # parallel_builder.resolve_rotation_deg only ever rotates the
    # relationship's OWNER to match it, never the TARGET, so trusting
    # it unconditionally silently measured the wrong pair of extents
    # whenever cyl_mesh was the relationship's target (or any component
    # whose build never actually reached that orientation): a shaft
    # genuinely built along world z had its diameter measured against
    # a declared "x", averaging its LENGTH and diameter together
    # instead of its two true diameter extents (confirmed, live-
    # reproduced: a radius-5/height-100 shaft measured 55.0mm instead
    # of its real 10.0mm). The declared axis is kept only as a
    # disambiguation HINT for a genuinely ambiguous shape (see
    # `_infer_rotational_axis`'s own docstring), and recorded separately
    # when it disagrees with the measured axis, purely for diagnosis.
    declared_axis = relationship.axis if relationship is not None else None
    cyl_axis = _infer_rotational_axis(cyl_mesh, hint_axis=declared_axis)
    cyl_diameter = _cylindrical_diameter_mm(cyl_mesh, cyl_axis)
    eva.requested[f"{cylinder_label}_axis"] = cyl_axis
    if declared_axis and declared_axis != cyl_axis:
        eva.requested[f"{cylinder_label}_declared_axis"] = declared_axis
    eva.actual[f"{cylinder_label}_diameter_mm"] = round(cyl_diameter, 3)

    bore_report = evaluate_mesh(bore_mesh)
    if bore_report.detected_hole_count == 0:
        eva.status = CompatibilityStatus.GEOMETRY_VALIDATION_FAILED
        eva.reason = f"{bore_id}'s own description implies a {bore_label}, but no hole was detected in its built geometry at all"
        return eva

    bore = _nearest_hole(bore_report, axis=cyl_axis, near_position=tuple(cyl_mesh.bounds.mean(axis=0)))
    if bore is None:
        detected_axes = sorted(set(bore_report.hole_axes))
        eva.status = CompatibilityStatus.INCOMPATIBLE
        note = (
            f" (the relationship declared axis {declared_axis!r}, but {cyl_id}'s own built geometry actually runs along {cyl_axis!r})"
            if declared_axis and declared_axis != cyl_axis else ""
        )
        eva.reason = f"{cylinder_label} runs along {cyl_axis!r}, but the only detected hole(s) on {bore_id} run along {detected_axes} -- axes are not aligned{note}"
        eva.suggested_action = f"rotate {cyl_id} (or {bore_id}) so their running axes match"
        return eva

    eva.actual[f"{bore_label}_diameter_mm"] = round(bore["diameter_mm"], 3)
    eva.actual[f"{bore_label}_through"] = bore["through"]

    aligned, offset = _axial_alignment_check(cyl_mesh, bore_mesh, cyl_axis)
    # _axial_alignment_check compares whole-mesh centers, a coarse proxy
    # when bore_mesh has only ONE hole roughly centered on it; refine
    # using the hole's OWN measured center when there's a real
    # discrepancy, since a plate's hole can sit far from the plate's
    # own bounding-box center by design.
    if not aligned:
        perp = [i for i in range(3) if i != _AXIS_INDEX[cyl_axis]]
        cyl_center = cyl_mesh.bounds.mean(axis=0)
        offset = float(np.linalg.norm([cyl_center[i] - bore["position_mm"][i] for i in perp]))
        aligned = offset <= _ALIGNMENT_TOLERANCE_MM
    eva.calculated["concentricity_offset_mm"] = round(offset, 3)
    if not aligned:
        eva.status = CompatibilityStatus.INCOMPATIBLE
        eva.reason = f"{cylinder_label} and {bore_label} centers are offset by {offset:.2f}mm perpendicular to the shared {cyl_axis!r} axis -- not concentric"
        eva.suggested_action = f"align {cyl_id}'s position with {bore_id}'s own bore center"
        return eva

    status, calculated, reason, suggestion = _evaluate_diametral_fit(
        f"{cyl_id}'s {cylinder_label}", cyl_diameter, f"{bore_id}'s {bore_label}", bore["diameter_mm"], cyl_desc, bore_desc,
    )
    eva.calculated.update(calculated)
    eva.status, eva.reason, eva.suggested_action = status, reason, suggestion
    return eva


def evaluate_shaft_bore_fit(
    relationship_id: str, id_a: str, id_b: str, description_a: str, description_b: str,
    mesh_a: trimesh.Trimesh, mesh_b: trimesh.Trimesh, relationship,
) -> RelationshipEvaluation:
    """Part 3A."""
    return _evaluate_cylinder_in_bore(
        relationship_id, id_a, id_b, description_a, description_b, mesh_a, mesh_b, relationship,
        MechanicalRelationshipType.SHAFT_BORE_FIT, cylinder_role_keyword="shaft", cylinder_label="shaft", bore_label="bore",
    )


def evaluate_bolt_hole_fit(
    relationship_id: str, id_a: str, id_b: str, description_a: str, description_b: str,
    mesh_a: trimesh.Trimesh, mesh_b: trimesh.Trimesh, relationship,
) -> RelationshipEvaluation:
    """Part 3C."""
    return _evaluate_cylinder_in_bore(
        relationship_id, id_a, id_b, description_a, description_b, mesh_a, mesh_b, relationship,
        MechanicalRelationshipType.BOLT_HOLE_FIT, cylinder_role_keyword="bolt", cylinder_label="bolt", bore_label="hole",
    )


def evaluate_bearing_housing_fit(
    relationship_id: str, id_a: str, id_b: str, description_a: str, description_b: str,
    mesh_a: trimesh.Trimesh, mesh_b: trimesh.Trimesh, relationship,
) -> RelationshipEvaluation:
    """Part 3E -- the bearing-outer-diameter-vs-housing-bore half of a
    bearing's own two mating relationships (the OTHER half, a shaft's
    diameter against the bearing's own INNER diameter, is a separate
    SHAFT_BORE_FIT relationship against the bearing component itself --
    see this module's own docstring: a 3-way bearing/shaft/housing chain
    is represented as two pairwise relationships, not a new 3-way
    relationship type, consistent with assembly_plan.Relationship's own
    pairwise design)."""
    return _evaluate_cylinder_in_bore(
        relationship_id, id_a, id_b, description_a, description_b, mesh_a, mesh_b, relationship,
        MechanicalRelationshipType.BEARING_HOUSING_FIT, cylinder_role_keyword="bearing", cylinder_label="bearing", bore_label="housing bore",
    )


_GEAR_CENTER_DISTANCE_TOLERANCE_FRACTION = 0.03  # a meshing pair's center distance is a precise geometric requirement, not a loose fit -- 3% or 0.5mm, whichever is larger
_PRESSURE_ANGLE_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(?:deg(?:ree)?s?)?\s*pressure\s*angle|pressure\s*angle\s*(?:of\s*)?(\d+(?:\.\d+)?)\s*deg", re.IGNORECASE)


def _extract_pressure_angle_deg(description: str) -> "float | None":
    """A separate, LOCAL extraction, not folded into llm_decompose.
    _extract_gear_params -- that function's own regex is reused as-is
    for module/tooth-count (the only two fields _correct_gear_mesh_
    spacing's own placement-time correction needs), and pressure angle
    is a distinct, optional field with no reason to touch that shared
    function's own scope. None when the text never states one -- a
    missing pressure angle is simply not checked (Part 3B's own rule:
    "compatible pressure angle, IF SPECIFIED"), never treated as a
    mismatch or an under-specified failure."""
    m = _PRESSURE_ANGLE_RE.search(description)
    return float(m.group(1) or m.group(2)) if m else None


def evaluate_gear_mesh(
    relationship_id: str, id_a: str, id_b: str, description_a: str, description_b: str,
    mesh_a: trimesh.Trimesh, mesh_b: trimesh.Trimesh,
) -> RelationshipEvaluation:
    """Part 3B, external spur gears only. Reuses llm_decompose.
    _extract_gear_params for module/tooth-count -- the SAME regex
    _correct_gear_mesh_spacing already trusts at PLACEMENT time; this
    is the matching VALIDATION half, checked against the actual placed
    meshes, never assuming the placement correction already ran (Test C/
    D's own "72mm actual vs 65mm nominal -> INCOMPATIBLE" case is
    exactly a plan whose placement correction did NOT apply cleanly).
    Deliberately never claims tooth-level meshing -- see this function's
    own COMPATIBLE reason string."""
    eva = RelationshipEvaluation(
        relationship_id=relationship_id, component_a=id_a, component_b=id_b,
        relationship_type=MechanicalRelationshipType.GEAR_MESH, source_phrase_a=description_a, source_phrase_b=description_b,
    )
    gear_a, gear_b = _extract_gear_params(description_a), _extract_gear_params(description_b)
    module_a, teeth_a = gear_a
    module_b, teeth_b = gear_b
    eva.requested.update({"module": module_a, "gear_a_teeth": teeth_a, "gear_b_teeth": teeth_b})

    if abs(module_a - module_b) > 1e-6:
        eva.status = CompatibilityStatus.INCOMPATIBLE
        eva.reason = f"gear modules differ ({module_a:g}mm vs {module_b:g}mm) -- different modules cannot mesh regardless of center distance"
        eva.suggested_action = "use the same module for both gears"
        return eva

    # Confirmed real gap (adversarial audit, live-reproduced): two gears
    # with explicitly stated, genuinely different pressure angles (e.g.
    # 20deg vs 14.5deg) used to report COMPATIBLE purely because module/
    # distance matched -- Part 3B's own requirement ("compatible
    # pressure angle, if specified") was simply never implemented.
    # Skipped entirely (never UNDER_SPECIFIED) when neither side states
    # one, per that same rule's own "if specified" qualifier.
    pressure_a, pressure_b = _extract_pressure_angle_deg(description_a), _extract_pressure_angle_deg(description_b)
    if pressure_a is not None and pressure_b is not None and abs(pressure_a - pressure_b) > 1e-6:
        eva.requested["pressure_angle_deg"] = f"{pressure_a:g} (gear A) vs {pressure_b:g} (gear B)"
        eva.status = CompatibilityStatus.INCOMPATIBLE
        eva.reason = f"gear pressure angles differ ({pressure_a:g} deg vs {pressure_b:g} deg) -- gears with different pressure angles cannot mesh correctly regardless of module or center distance"
        eva.suggested_action = "use the same pressure angle for both gears"
        return eva
    if pressure_a is not None:
        eva.requested["pressure_angle_deg"] = pressure_a

    axis_a, axis_b = _infer_rotational_axis(mesh_a), _infer_rotational_axis(mesh_b)
    if axis_a != axis_b:
        eva.status = CompatibilityStatus.INCOMPATIBLE
        eva.reason = f"gear shafts are not parallel -- {id_a}'s own rotational axis is {axis_a!r}, {id_b}'s is {axis_b!r}"
        eva.suggested_action = "rotate one gear so both run along the same axis"
        return eva
    eva.actual["shaft_axes"] = f"{axis_a} (parallel)"

    expected_distance = module_a * (teeth_a + teeth_b) / 2.0
    eva.requested["center_distance_mm"] = round(expected_distance, 3)

    perp = [i for i in range(3) if i != _AXIS_INDEX[axis_a]]
    center_a, center_b = mesh_a.bounds.mean(axis=0), mesh_b.bounds.mean(axis=0)
    actual_distance = float(np.linalg.norm([center_a[i] - center_b[i] for i in perp]))
    eva.actual["center_distance_mm"] = round(actual_distance, 3)

    tolerance = max(0.5, expected_distance * _GEAR_CENTER_DISTANCE_TOLERANCE_FRACTION)
    deviation = actual_distance - expected_distance
    eva.calculated["deviation_mm"] = round(deviation, 3)
    if abs(deviation) > tolerance:
        eva.status = CompatibilityStatus.INCOMPATIBLE
        direction = "too close (interference)" if deviation < 0 else "too far apart (separation -- teeth will not engage)"
        eva.reason = f"actual center distance ({actual_distance:.2f}mm) differs from the required nominal distance ({expected_distance:.2f}mm) -- {direction}"
        eva.suggested_action = f"reposition one gear so the center distance is {expected_distance:.2f}mm"
        return eva

    eva.status = CompatibilityStatus.COMPATIBLE
    eva.reason = "nominal pitch-circle center distance is correct -- this does NOT verify actual tooth-profile meshing, which is not checked"
    return eva


def evaluate_motor_mount(
    relationship_id: str, id_a: str, id_b: str, description_a: str, description_b: str,
    mesh_a: trimesh.Trimesh, mesh_b: trimesh.Trimesh, relationship,
) -> RelationshipEvaluation:
    """Part 3D, bounded to what's actually checkable from built geometry:
    hole COUNT match, hole LAYOUT match (each hole's own offset from its
    component's center, compared between motor and mount -- using
    evaluation.geometry_eval.GeometryReport.hole_positions_mm, extended
    by this same effort specifically because this check needed it), and
    hole diameter compatibility (the mount's own holes must not be
    smaller than the motor's). Never assumes a named motor standard's
    dimensions -- only ever measures the actual built geometry."""
    from evaluation.geometry_eval import evaluate_mesh

    motor_is_a = _has("motor", description_a) or not _has("motor", description_b)
    motor_id, mount_id = (id_a, id_b) if motor_is_a else (id_b, id_a)
    motor_mesh, mount_mesh = (mesh_a, mesh_b) if motor_is_a else (mesh_b, mesh_a)
    motor_desc, mount_desc = (description_a, description_b) if motor_is_a else (description_b, description_a)

    eva = RelationshipEvaluation(
        relationship_id=relationship_id, component_a=motor_id, component_b=mount_id,
        relationship_type=MechanicalRelationshipType.MOTOR_MOUNT, source_phrase_a=motor_desc, source_phrase_b=mount_desc,
    )

    face_axis = None
    if relationship is not None and relationship.kind == "attached_to_face" and relationship.face:
        face_axis = relationship.face[-1]
    if face_axis is None:
        face_axis = _infer_rotational_axis(motor_mesh)

    motor_report, mount_report = evaluate_mesh(motor_mesh), evaluate_mesh(mount_mesh)
    motor_holes = [i for i, a in enumerate(motor_report.hole_axes) if a == face_axis]
    mount_holes = [i for i, a in enumerate(mount_report.hole_axes) if a == face_axis]
    eva.requested["motor_mounting_holes"] = len(motor_holes)
    eva.actual["mount_holes"] = len(mount_holes)

    if not motor_holes or not mount_holes:
        eva.status = CompatibilityStatus.UNDER_SPECIFIED
        eva.reason = "could not detect a mounting-hole pattern on one or both components"
        return eva

    if len(motor_holes) != len(mount_holes):
        eva.status = CompatibilityStatus.INCOMPATIBLE
        eva.reason = f"motor has {len(motor_holes)} mounting hole(s), mount has {len(mount_holes)} -- hole counts do not match"
        return eva

    perp = [i for i in range(3) if i != _AXIS_INDEX[face_axis]]
    motor_center, mount_center = motor_mesh.bounds.mean(axis=0), mount_mesh.bounds.mean(axis=0)
    motor_offsets = [[motor_report.hole_positions_mm[i][k] - motor_center[k] for k in perp] for i in motor_holes]
    mount_offsets = [[mount_report.hole_positions_mm[i][k] - mount_center[k] for k in perp] for i in mount_holes]

    unmatched = list(range(len(mount_offsets)))
    max_mismatch = 0.0
    for mo in motor_offsets:
        best_j, best_d = None, None
        for j in unmatched:
            d = float(np.linalg.norm([mo[k] - mount_offsets[j][k] for k in range(len(mo))]))
            if best_d is None or d < best_d:
                best_d, best_j = d, j
        unmatched.remove(best_j)
        max_mismatch = max(max_mismatch, best_d)

    eva.calculated["max_hole_position_mismatch_mm"] = round(max_mismatch, 3)
    if max_mismatch > _ALIGNMENT_TOLERANCE_MM:
        eva.status = CompatibilityStatus.INCOMPATIBLE
        eva.reason = f"motor's mounting-hole pattern does not align with the mount's own hole pattern (largest mismatch: {max_mismatch:.2f}mm)"
        eva.suggested_action = "reposition the mount's holes to match the motor's own mounting pattern"
        return eva

    motor_diam = max(motor_report.hole_diameters_mm[i] for i in motor_holes)
    mount_diam = min(mount_report.hole_diameters_mm[i] for i in mount_holes)
    eva.actual["motor_hole_diameter_mm"] = round(motor_diam, 3)
    eva.actual["mount_hole_diameter_mm"] = round(mount_diam, 3)
    if mount_diam < motor_diam - 1e-6:
        eva.status = CompatibilityStatus.INCOMPATIBLE
        eva.reason = f"mount's own holes ({mount_diam:.2f}mm) are smaller than the motor's own mounting holes ({motor_diam:.2f}mm) -- a fastener through both would not fit"
        return eva

    eva.status = CompatibilityStatus.COMPATIBLE
    return eva


def evaluate_face_contact(
    relationship_id: str, id_a: str, id_b: str, description_a: str, description_b: str,
    mesh_a: trimesh.Trimesh, mesh_b: trimesh.Trimesh, relationship,
) -> RelationshipEvaluation:
    """Part 3F -- generic coincident-face contact, for an attached_to_face
    relationship that didn't classify into a more specific mechanical
    type. `component_a` is the ATTACHED part, `component_b` is the
    target it attaches to (matching assembly_plan.resolve_position's
    own convention: `relationship.face` names the TARGET's face)."""
    eva = RelationshipEvaluation(
        relationship_id=relationship_id, component_a=id_a, component_b=id_b,
        relationship_type=MechanicalRelationshipType.FACE_CONTACT, source_phrase_a=description_a, source_phrase_b=description_b,
    )
    face = relationship.face if relationship is not None else None
    eva.requested["face"] = face or "unspecified"
    if face is None:
        eva.status = CompatibilityStatus.UNDER_SPECIFIED
        eva.reason = "no face direction specified for this attachment"
        return eva
    axis, sign = face[-1], (1 if face[0] == "+" else -1)
    idx = _AXIS_INDEX[axis]
    target_extreme = mesh_b.bounds[1][idx] if sign > 0 else mesh_b.bounds[0][idx]
    own_extreme = mesh_a.bounds[0][idx] if sign > 0 else mesh_a.bounds[1][idx]
    gap = abs(own_extreme - target_extreme)
    eva.calculated["face_gap_mm"] = round(gap, 3)
    if gap > _ALIGNMENT_TOLERANCE_MM:
        eva.status = CompatibilityStatus.INCOMPATIBLE
        eva.reason = f"components are not actually in contact at the {face!r} face -- gap of {gap:.2f}mm"
        eva.suggested_action = f"move {id_a} so it sits flush against {id_b}'s {face!r} face"
        return eva
    eva.status = CompatibilityStatus.COMPATIBLE
    return eva


def evaluate_axis_alignment(
    relationship_id: str, id_a: str, id_b: str, description_a: str, description_b: str,
    mesh_a: trimesh.Trimesh, mesh_b: trimesh.Trimesh, relationship,
) -> RelationshipEvaluation:
    """Part 3F -- generic concentric/coaxial alignment, for a
    concentric_with/parallel_to relationship that didn't classify into a
    more specific mechanical type (no shaft/bore/bearing/gear keyword on
    either side)."""
    eva = RelationshipEvaluation(
        relationship_id=relationship_id, component_a=id_a, component_b=id_b,
        relationship_type=MechanicalRelationshipType.AXIS_ALIGNMENT, source_phrase_a=description_a, source_phrase_b=description_b,
    )
    # E50 fix: each component's own axis is independently measured from
    # its own actual built geometry -- never assumed from relationship.
    # axis (see `_evaluate_cylinder_in_bore`'s own fix note; the same
    # root cause applies here: mesh_a/mesh_b are not both guaranteed to
    # have actually been rotated to the relationship's declared axis).
    # A real disagreement between the two components' own measured axes
    # is reported directly as a misalignment, never silently resolved by
    # picking one side's axis and comparing perpendicular coordinates
    # that were never really comparable in the first place.
    declared_axis = relationship.axis if relationship is not None else None
    axis_a = _infer_rotational_axis(mesh_a, hint_axis=declared_axis)
    axis_b = _infer_rotational_axis(mesh_b, hint_axis=declared_axis)
    eva.requested["axis"] = axis_a
    if declared_axis and declared_axis != axis_a:
        eva.requested["declared_axis"] = declared_axis
    if axis_a != axis_b:
        eva.status = CompatibilityStatus.INCOMPATIBLE
        eva.reason = f"{id_a}'s own built geometry runs along {axis_a!r}, but {id_b}'s runs along {axis_b!r} -- these components are not actually coaxial"
        eva.suggested_action = f"rotate {id_a} (or {id_b}) so both run along the same axis"
        return eva
    aligned, offset = _axial_alignment_check(mesh_a, mesh_b, axis_a)
    eva.calculated["offset_mm"] = round(offset, 3)
    eva.status = CompatibilityStatus.COMPATIBLE if aligned else CompatibilityStatus.INCOMPATIBLE
    if not aligned:
        eva.reason = f"components are offset by {offset:.2f}mm perpendicular to the shared {axis_a!r} axis"
        eva.suggested_action = f"align {id_a} and {id_b} along the {axis_a!r} axis"
    return eva


# ---------------- main engine ----------------

_EVALUATOR_BY_TYPE = {
    MechanicalRelationshipType.SHAFT_BORE_FIT: evaluate_shaft_bore_fit,
    MechanicalRelationshipType.BOLT_HOLE_FIT: evaluate_bolt_hole_fit,
    MechanicalRelationshipType.BEARING_HOUSING_FIT: evaluate_bearing_housing_fit,
    MechanicalRelationshipType.FACE_CONTACT: evaluate_face_contact,
    MechanicalRelationshipType.AXIS_ALIGNMENT: evaluate_axis_alignment,
    MechanicalRelationshipType.MOTOR_MOUNT: evaluate_motor_mount,
}


def evaluate_assembly_compatibility(points: list, placed_meshes_by_id: dict, relationships: dict) -> AssemblyCompatibilityReport:
    """Part 5's `compatibility_engine.validate(...)`. `points` is the
    assembly's own list of llm_decompose.ActionablePoint (or anything
    duck-typed the same way: .id, .description, .relationship), `
    placed_meshes_by_id`/`relationships` are the SAME dicts parallel_
    builder.build_assembly_from_points already builds for its own
    collision check -- this reads them, never recomputes a parallel
    copy. Deterministic throughout: every evaluator above is plain
    arithmetic over real measured geometry, never an LLM call (Part 5's
    own "do not let the LLM decide" rule).

    Two independent discovery mechanisms feed the engine, matching how
    this project's own relationships actually show up in practice (see
    llm_decompose.py's own worked examples):
    1. Gear pairs are found directly by scanning for exactly 2 gear-
       bearing points (mirroring _correct_gear_mesh_spacing's own
       discovery -- gears normally have no DIRECT relationship to each
       other at all; each is concentric_with its OWN shaft instead).
    2. Every other mechanical relationship is found by walking each
       point's own explicit `relationship` -- the authoritative record
       of intent this engine never infers from final geometry alone
       (Part 2's own rule)."""
    by_id = {p.id: p for p in points if p.id}
    report = AssemblyCompatibilityReport()
    evaluated_pairs: set[frozenset] = set()
    next_id = [1]

    def _rel_id() -> str:
        rid = f"REL-{next_id[0]:03d}"
        next_id[0] += 1
        return rid

    gear_ids = [p.id for p in points if p.id and p.id in placed_meshes_by_id and _extract_gear_params(p.description) is not None]
    if len(gear_ids) == 2:
        a_id, b_id = gear_ids
        a_desc, b_desc = by_id[a_id].description, by_id[b_id].description
        if _NON_SPUR_GEAR_TYPE_RE.search(a_desc) or _NON_SPUR_GEAR_TYPE_RE.search(b_desc):
            report.evaluations.append(RelationshipEvaluation(
                relationship_id=_rel_id(), component_a=a_id, component_b=b_id, relationship_type=MechanicalRelationshipType.UNSUPPORTED,
                source_phrase_a=a_desc, source_phrase_b=b_desc, status=CompatibilityStatus.UNSUPPORTED_TO_VERIFY,
                reason="one or both gears are a non-standard-external-spur type (helical/bevel/worm/internal/hypoid/miter) -- the external spur gear center-distance formula does not apply and this engine does not model these types",
            ))
        else:
            report.evaluations.append(evaluate_gear_mesh(
                _rel_id(), a_id, b_id, a_desc, b_desc, placed_meshes_by_id[a_id], placed_meshes_by_id[b_id],
            ))
        evaluated_pairs.add(frozenset((a_id, b_id)))

    for pid, point in by_id.items():
        rel = relationships.get(pid) if relationships else point.relationship
        if rel is None:
            continue
        target_id = rel.target_id
        target = by_id.get(target_id)
        if target is None:
            continue
        pair = frozenset((pid, target_id))
        if pair in evaluated_pairs:
            continue

        eva = _evaluate_relationship_pair(_rel_id(), pid, target_id, point, target, placed_meshes_by_id, rel)
        report.evaluations.append(eva)
        evaluated_pairs.add(pair)

    return report


def _evaluate_relationship_pair(relationship_id: str, pid: str, target_id: str, point, target, placed_meshes_by_id: dict, rel) -> RelationshipEvaluation:
    """The single-pair classify-and-dispatch logic shared by
    evaluate_assembly_compatibility's own main loop and
    attempt_repair_with_revalidation below -- factored out so a repair's
    own side-effect check runs through EXACTLY the same evaluation path
    as the original report, never a second, slightly-different one."""
    if pid not in placed_meshes_by_id or target_id not in placed_meshes_by_id:
        return RelationshipEvaluation(
            relationship_id=relationship_id, component_a=pid, component_b=target_id,
            relationship_type=MechanicalRelationshipType.UNSUPPORTED,
            source_phrase_a=point.description, source_phrase_b=target.description,
            status=CompatibilityStatus.NOT_EVALUATED,
            reason="one or both components were not successfully built -- nothing to evaluate",
        )

    mechanical_type = classify_relationship_type(point.description, target.description, rel.kind)
    if mechanical_type == MechanicalRelationshipType.GEAR_MESH:
        try:
            return evaluate_gear_mesh(relationship_id, pid, target_id, point.description, target.description, placed_meshes_by_id[pid], placed_meshes_by_id[target_id])
        except Exception as exc:
            return RelationshipEvaluation(
                relationship_id=relationship_id, component_a=pid, component_b=target_id, relationship_type=mechanical_type,
                source_phrase_a=point.description, source_phrase_b=target.description,
                status=CompatibilityStatus.GEOMETRY_VALIDATION_FAILED, reason=f"could not evaluate: {exc}",
            )

    evaluator = _EVALUATOR_BY_TYPE.get(mechanical_type)
    if evaluator is None:
        return RelationshipEvaluation(
            relationship_id=relationship_id, component_a=pid, component_b=target_id,
            relationship_type=mechanical_type, source_phrase_a=point.description, source_phrase_b=target.description,
            status=CompatibilityStatus.UNSUPPORTED_TO_VERIFY,
            reason=f"relationship kind {rel.kind!r} between these two components' descriptions does not match any mechanical relationship type this engine currently supports",
        )

    try:
        return evaluator(relationship_id, pid, target_id, point.description, target.description, placed_meshes_by_id[pid], placed_meshes_by_id[target_id], rel)
    except Exception as exc:
        return RelationshipEvaluation(
            relationship_id=relationship_id, component_a=pid, component_b=target_id,
            relationship_type=mechanical_type, source_phrase_a=point.description, source_phrase_b=target.description,
            status=CompatibilityStatus.GEOMETRY_VALIDATION_FAILED,
            reason=f"could not evaluate this relationship against the actual built geometry: {exc}",
        )


# ---------------- Part 10: constraint-aware repair ----------------
#
# Deliberately NARROW, not a general constraint solver (Part 12's own
# "do not build a full industrial-grade... engine" rule): exactly the
# two repair shapes this engine can perform WITHOUT ever touching a
# hard dimension, module, tooth count, or clearance (Part 10's own
# forbidden list) -- repositioning one component along the axis a
# relationship already declares. A relationship this doesn't apply to
# (a module mismatch, an under-specified clearance, an unsupported gear
# type) is reported as unfixed, never silently papered over.


@dataclass
class RepairAttempt:
    strategy: str
    description: str
    applied: bool


def repair_gear_center_distance(moving_mesh: trimesh.Trimesh, fixed_mesh: trimesh.Trimesh, expected_distance_mm: float, axis: str) -> trimesh.Trimesh:
    """Translates a COPY of `moving_mesh` so its center ends up exactly
    `expected_distance_mm` from `fixed_mesh`'s own center, in the plane
    perpendicular to `axis`, preserving `moving_mesh`'s own shape, size,
    and axial coordinate entirely -- the same center-distance formula
    llm_decompose._correct_gear_mesh_spacing already applies at
    PLACEMENT time, reapplied here directly to built geometry for
    POST-hoc repair. Never changes module or tooth count -- those are
    baked into the mesh's own already-built geometry, untouched."""
    perp = [i for i in range(3) if i != _AXIS_INDEX[axis]]
    center_fixed, center_moving = fixed_mesh.bounds.mean(axis=0), moving_mesh.bounds.mean(axis=0)
    current_vec = np.array([center_moving[i] - center_fixed[i] for i in perp])
    current_dist = float(np.linalg.norm(current_vec))
    direction = current_vec / current_dist if current_dist > 1e-6 else np.array([1.0, 0.0])
    target_offset = direction * expected_distance_mm
    delta = np.zeros(3)
    for k, i in enumerate(perp):
        delta[i] = (center_fixed[i] + target_offset[k]) - center_moving[i]
    repaired = moving_mesh.copy()
    repaired.apply_translation(delta)
    return repaired


def repair_concentricity(moving_mesh: trimesh.Trimesh, target_center_mm: "tuple[float, float, float]", axis: str) -> trimesh.Trimesh:
    """Translates a COPY of `moving_mesh` so its own center aligns with
    `target_center_mm` on the two axes perpendicular to `axis`,
    preserving its own axial coordinate and its shape/size entirely --
    a shaft/bolt/bearing's own diameter is never touched to fix a
    position (Part 10's own "never modify hard dimensions to fix a
    positional relationship" rule)."""
    perp = [i for i in range(3) if i != _AXIS_INDEX[axis]]
    moving_center = moving_mesh.bounds.mean(axis=0)
    delta = np.zeros(3)
    for i in perp:
        delta[i] = target_center_mm[i] - moving_center[i]
    repaired = moving_mesh.copy()
    repaired.apply_translation(delta)
    return repaired


def attempt_repair(evaluation: RelationshipEvaluation, mesh_a: trimesh.Trimesh, mesh_b: trimesh.Trimesh) -> "tuple[trimesh.Trimesh, trimesh.Trimesh, RepairAttempt] | None":
    """Returns (possibly-repaired mesh_a, possibly-repaired mesh_b,
    RepairAttempt) for the two relationship/failure shapes this engine
    can safely auto-correct, or None when this evaluation's own failure
    isn't one of them (a module mismatch, a stated-clearance violation,
    an UNDER_SPECIFIED/UNSUPPORTED_TO_VERIFY result, or a fit that's
    already COMPATIBLE) -- callers must always re-run the relevant
    evaluator afterward (never assume success) and must not apply a
    repair that would move an explicitly author-fixed component (this
    function has no way to know that on its own -- see this module's
    own docstring on repair wiring for where that authorization check
    belongs)."""
    if evaluation.status is not CompatibilityStatus.INCOMPATIBLE:
        return None

    if evaluation.relationship_type == MechanicalRelationshipType.GEAR_MESH and "center_distance_mm" in evaluation.requested:
        axis = str(evaluation.actual.get("shaft_axes", "z")).split()[0]
        expected = float(evaluation.requested["center_distance_mm"])
        repaired_b = repair_gear_center_distance(mesh_b, mesh_a, expected, axis)
        attempt = RepairAttempt(
            "reposition_gear_for_center_distance",
            f"moved {evaluation.component_b} to the required {expected:g}mm center distance from {evaluation.component_a}, "
            "without changing either gear's own module or tooth count",
            True,
        )
        return mesh_a, repaired_b, attempt

    if (
        evaluation.relationship_type in (MechanicalRelationshipType.SHAFT_BORE_FIT, MechanicalRelationshipType.BOLT_HOLE_FIT, MechanicalRelationshipType.BEARING_HOUSING_FIT)
        and "concentricity_offset_mm" in evaluation.calculated and evaluation.calculated["concentricity_offset_mm"] > _ALIGNMENT_TOLERANCE_MM
    ):
        from evaluation.geometry_eval import evaluate_mesh

        axis = str(evaluation.requested.get(f"{evaluation.relationship_type.value.split('_')[0]}_axis", "")) or _infer_rotational_axis(mesh_a)
        bore_report = evaluate_mesh(mesh_b)
        if bore_report.detected_hole_count == 0:
            return None  # the earlier GEOMETRY_VALIDATION_FAILED case -- nothing to align to
        bore = _nearest_hole(bore_report, axis=axis, near_position=tuple(mesh_a.bounds.mean(axis=0)))
        if bore is None:
            return None  # axis mismatch, not a plain offset -- repositioning alone can't fix a wrong rotation
        repaired_a = repair_concentricity(mesh_a, bore["position_mm"], axis)
        attempt = RepairAttempt(
            "align_to_bore_center",
            f"moved {evaluation.component_a} so its own axis aligns with {evaluation.component_b}'s bore center, without changing either component's diameter",
            True,
        )
        return repaired_a, mesh_b, attempt

    return None


@dataclass
class RevalidatedRepairResult:
    meshes_by_id: dict  # the repaired meshes (or the ORIGINAL, unmodified ones if rejected)
    attempt: "RepairAttempt | None"
    accepted: bool
    broken_relationships: list = field(default_factory=list)  # RelationshipEvaluation entries that would newly break -- only when accepted=False


def attempt_repair_with_revalidation(
    evaluation: RelationshipEvaluation, placed_meshes_by_id: dict, points: list, relationships: dict,
) -> RevalidatedRepairResult:
    """Part 10's own "after every repair, revalidate all affected
    relationships, not only the original failure" rule, and "if no
    valid solution exists, report the conflict rather than producing a
    misleading assembly" -- confirmed real gap this closes (adversarial
    audit, live-reproduced): attempt_repair() alone has no knowledge of
    a component's OTHER relationships, so repairing a gear pair's own
    center distance by moving gear_B alone -- with no awareness of
    gear_B's own SEPARATE concentric_with relationship to its own shaft
    -- silently broke that other relationship (a real, measured 7mm
    offset) with nothing surfacing it; a caller trusting the repair in
    isolation would get a false-positive "fixed" report.

    Deliberately NOT a general constraint-dependency-graph solver (the
    original brief's own Part 12 "keep focused" rule) -- this checks
    exactly the relationships that actually TOUCH a repaired component
    (by scanning the same flat `relationships` dict every other part of
    this engine already reads, never a separate graph structure), and
    REJECTS the repair outright (returns the original, untouched
    meshes) if it would turn any previously-COMPATIBLE relationship
    into anything else. Never applies a repair that trades one fixed
    relationship for a newly broken one."""
    comp_a_id, comp_b_id = evaluation.component_a, evaluation.component_b
    mesh_a, mesh_b = placed_meshes_by_id.get(comp_a_id), placed_meshes_by_id.get(comp_b_id)
    if mesh_a is None or mesh_b is None:
        return RevalidatedRepairResult(placed_meshes_by_id, None, False)

    result = attempt_repair(evaluation, mesh_a, mesh_b)
    if result is None:
        return RevalidatedRepairResult(placed_meshes_by_id, None, False)
    repaired_a, repaired_b, attempt = result

    candidate_meshes = dict(placed_meshes_by_id)
    candidate_meshes[comp_a_id] = repaired_a
    candidate_meshes[comp_b_id] = repaired_b

    by_id = {p.id: p for p in points if p.id}
    affected_ids = {comp_a_id, comp_b_id}
    repaired_pair = frozenset((comp_a_id, comp_b_id))
    newly_broken: list[RelationshipEvaluation] = []

    for pid, rel in relationships.items():
        if rel is None or (pid not in affected_ids and rel.target_id not in affected_ids):
            continue
        if frozenset((pid, rel.target_id)) == repaired_pair:
            continue  # the relationship we just fixed -- already accounted for by the caller's own re-evaluation
        target = by_id.get(rel.target_id)
        point = by_id.get(pid)
        if target is None or point is None:
            continue
        if pid not in placed_meshes_by_id or rel.target_id not in placed_meshes_by_id:
            continue  # wasn't successfully built before the repair either -- nothing this repair could have broken

        before = _evaluate_relationship_pair("REL-CHECK", pid, rel.target_id, point, target, placed_meshes_by_id, rel)
        if before.status is not CompatibilityStatus.COMPATIBLE:
            continue  # already not compatible beforehand -- not something this repair newly broke
        after = _evaluate_relationship_pair("REL-CHECK", pid, rel.target_id, point, target, candidate_meshes, rel)
        if after.status is not CompatibilityStatus.COMPATIBLE:
            newly_broken.append(after)

    if newly_broken:
        return RevalidatedRepairResult(placed_meshes_by_id, attempt, False, newly_broken)

    return RevalidatedRepairResult(candidate_meshes, attempt, True)
