"""Part 6's own "dimension audit" reconciliation report -- for every
mandatory requirement, answers (in one place): where did this number
come from, which feature does it describe, which operation consumed
it, and does the ACTUAL built geometry satisfy it. Never trusts
ToolCall.args alone as proof of correctness -- every line that CAN be
independently re-measured from the real mesh (dimension_requirements.py's
base-shape and hole-feature parameter types, via trimesh extents and
evaluation.geometry_eval's own hole detection) is, and only falls back
to reporting the assigned value un-re-measured for the handful of
parameter types this report doesn't have a cheap independent geometric
check for (see _UNVERIFIABLE_PARAMETER_TYPES below) -- disclosed as
such, never silently presented with the same confidence as a verified
line."""

from __future__ import annotations

from dataclasses import dataclass, field

from dimension_requirements import DimensionRequirement, FeatureCountRequirement, RequirementStatus

_TOLERANCE_FRACTION = 0.05  # 5% -- matches this project's own established dimension-matching tolerance elsewhere (spec_validation.py)

# main_diameter/reduced_diameter/overall_length (a stepped shaft's own 3
# fields) don't have a single bounding-box extent that directly
# corresponds to them the way thickness/height/length/width/outer_
# diameter do -- re-measuring them precisely needs the radius-at-two-
# z-slices technique tests/test_dimension_regression.py's own Test 5
# already uses, not a cheap one-line extents lookup. Rather than fake a
# false-precision "VERIFIED" for these from the assigned value alone,
# this report honestly labels them ASSIGNED (not independently
# re-measured here) instead.
_UNVERIFIABLE_PARAMETER_TYPES = frozenset({"main_diameter", "reduced_diameter", "overall_length"})

_EXTENT_INDEX_BY_PARAMETER = {"length": 0, "width": 1, "thickness": 2, "height": 2, "outer_diameter": 0}


@dataclass
class AuditLine:
    label: str
    requested: str
    actual: str
    status: str  # "VERIFIED" | "MISMATCH" | "ASSIGNED" | "UNRESOLVED" | "UNASSIGNED" | "FAILED" | "NOT BUILT"


@dataclass
class DimensionAuditReport:
    lines: list[AuditLine] = field(default_factory=list)

    @property
    def mandatory_total(self) -> int:
        return len(self.lines)

    @property
    def mandatory_verified(self) -> int:
        return sum(1 for line in self.lines if line.status == "VERIFIED")

    def render(self) -> str:
        out = ["DIMENSION AUDIT", ""]
        for line in self.lines:
            out.append(f"{line.label}:")
            out.append(f"  Requested: {line.requested}")
            out.append(f"  Actual: {line.actual}")
            out.append(f"  Status: {line.status}")
            out.append("")
        out.append(f"Overall: {self.mandatory_verified}/{self.mandatory_total} mandatory requirements verified")
        return "\n".join(out)


def _dimension_line(req: DimensionRequirement, mesh, geometry_report: "dict | None" = None) -> AuditLine:
    requested = f"{req.normalized_value:g} {req.normalized_unit}"
    if req.status is RequirementStatus.UNRESOLVED:
        return AuditLine(req.source_phrase, requested, "N/A -- ambiguous, never assigned", "UNRESOLVED")
    if req.status is RequirementStatus.UNASSIGNED:
        return AuditLine(req.source_phrase, requested, "N/A -- no operation in the built program accepts this", "UNASSIGNED")
    if mesh is None:
        return AuditLine(req.source_phrase, requested, "N/A -- nothing was built", "NOT BUILT")
    if req.parameter in _UNVERIFIABLE_PARAMETER_TYPES:
        return AuditLine(req.source_phrase, requested, f"assigned to {req.assigned_arg_name} (not independently re-measured here)", "ASSIGNED")
    if req.parameter == "hole_diameter":
        diameters = (geometry_report or {}).get("hole_diameters_mm") or []
        if not diameters:
            return AuditLine(req.source_phrase, requested, "no hole detected in the built geometry at all", "MISMATCH")
        closest = min(diameters, key=lambda d: abs(d - req.normalized_value))
        matched = abs(closest - req.normalized_value) <= max(req.normalized_value * _TOLERANCE_FRACTION, 0.5)
        return AuditLine(req.source_phrase, requested, f"{closest:.1f} mm (closest detected hole)", "VERIFIED" if matched else "MISMATCH")
    if req.parameter == "hole_depth":
        # No independent "built depth" measurement exists in
        # evaluation.geometry_eval.GeometryReport today (only diameter,
        # circularity, and through/blind) -- honestly disclosed as
        # ASSIGNED-but-unverified rather than claiming a check this
        # report cannot actually perform.
        return AuditLine(req.source_phrase, requested, f"assigned to {req.assigned_arg_name} (no independent depth measurement available)", "ASSIGNED")
    extent_index = _EXTENT_INDEX_BY_PARAMETER.get(req.parameter)
    if extent_index is None:
        return AuditLine(req.source_phrase, requested, f"assigned to {req.assigned_arg_name} (no independent extent check for this parameter)", "ASSIGNED")
    actual_mm = float(mesh.extents[extent_index])
    matched = abs(actual_mm - req.normalized_value) <= max(req.normalized_value * _TOLERANCE_FRACTION, 0.5)
    return AuditLine(req.source_phrase, requested, f"{actual_mm:.1f} mm", "VERIFIED" if matched else "MISMATCH")


def _count_line(req: FeatureCountRequirement, geometry_report) -> AuditLine:
    requested = f"{req.count}"
    if req.status in (RequirementStatus.UNASSIGNED, RequirementStatus.FAILED):
        detected = geometry_report.get("detected_hole_count") if geometry_report else None
        return AuditLine(req.source_phrase, requested, f"{detected}" if detected is not None else "N/A", req.status.value)
    detected = geometry_report.get("detected_hole_count") if geometry_report else None
    if detected is None:
        return AuditLine(req.source_phrase, requested, "N/A -- nothing was built", "NOT BUILT")
    return AuditLine(req.source_phrase, requested, f"{detected}", "VERIFIED" if detected == req.count else "MISMATCH")


def build_dimension_audit_report(
    dimension_reqs: list[DimensionRequirement], count_reqs: list[FeatureCountRequirement],
    mesh, geometry_report: "dict | None",
) -> DimensionAuditReport:
    """Never raises. `mesh` and `geometry_report` are the REAL built
    result (trimesh.Trimesh and evaluation.geometry_eval.GeometryReport.
    as_dict(), respectively) -- this never infers "actual" from
    ToolCall.args, only from the geometry a caller independently
    measured, per this project's own "a mesh can be watertight while
    still being dimensionally incorrect" discipline."""
    report = DimensionAuditReport()
    for req in dimension_reqs:
        if not req.mandatory:
            continue
        report.lines.append(_dimension_line(req, mesh, geometry_report))
    for req in count_reqs:
        if not req.mandatory:
            continue
        report.lines.append(_count_line(req, geometry_report))
    return report
