"""Required Feature Ledger -- Part 6/7 of the brief: independent,
geometry-only verification of each extracted op_requirements.Requirement
against the ACTUAL BUILT mesh, never trusting the planner's own chosen
op or its args as proof a feature exists (this project's own recurring
"never trust the planner, always re-measure built geometry" principle,
already applied throughout dimension_repair.py and
assembly_compatibility.py).

Deliberately scoped, not a general feature verifier for every type
op_requirements.py can extract: a genuinely, independently re-measurable
feature type gets a real geometric check; every other extracted type is
honestly reported UNSUPPORTED_TO_VERIFY (its own op-sequence-level
compatible_ops check, already run by op_arbitration.evaluate_candidate,
remains the only signal for those -- never silently upgraded here to
look like a geometry-verified pass)."""

from __future__ import annotations

import dataclasses

import numpy as np
import trimesh

from dimension_requirements import RequirementStatus, extract_dimension_requirements
from evaluation.geometry_eval import GeometryReport, _fit_circle_xy, evaluate_mesh
from op_requirements import Requirement, bore_type

_AXIS_INDEX = {"x": 0, "y": 1, "z": 2}


def _extent_shape_signals(report: GeometryReport) -> tuple[bool, bool]:
    """(two_equal, all_equal) over report's own 3 bounding-box extents.
    Confirmed real bug (live-reproduced): a cylinder's diameter pair is
    the LARGEST two of the three sorted extents whenever its height is
    the smallest dimension (e.g. a 40mm-diameter, 30mm-tall cylinder
    sorts to [30, 40, 40] -- the equal pair is (mid, large), not
    (small, mid)) -- checking only (small, mid) for equality missed this
    entirely. Check all 3 pairs; "two_equal" means ANY pair matches.

    Extent equality ALONE is not sufficient to tell a cylinder from a
    box, though: a square-footprint box (e.g. 40x40x20) also has two
    equal extents by sheer coincidence of its own chosen dimensions --
    confirmed real bug this module's own curvature check (below) exists
    to resolve. Kept as a secondary signal (a cylinder is still expected
    to have two equal extents; a mismatch there is still informative),
    not the primary one."""
    small, mid, large = sorted(report.bounding_box_mm)
    if large < 1e-6:
        return False, False
    pairs_equal = (abs(mid - small) / large < 0.08, abs(large - mid) / large < 0.08, abs(large - small) / large < 0.08)
    two_equal = any(pairs_equal)
    all_equal = all(pairs_equal)
    return two_equal, all_equal


def _has_circular_cross_section(mesh: trimesh.Trimesh) -> bool:
    """Whether the mesh's own MID-HEIGHT cross-section (perpendicular to
    its longest axis) is well-fit by a circle -- the actual, reliable
    discriminator between a box-family shape and a cylinder/sphere.

    A face-adjacency-angle approach was tried first and rejected: it
    cannot tell a cylinder's own curved side wall apart from a
    chamfered box's own flat bevel faces, since BOTH produce the same
    signature (many edges at one single repeated discrete angle,
    confirmed directly: a chamfer's own 48 new edges cluster at ~35-45
    degrees exactly as tightly as a cylinder's 32 wall edges cluster at
    ~11.25 degrees -- "one repeated angle value" cannot distinguish
    "flat bevel" from "curved wall" at all). A chamfer or fillet only
    ever affects material near an EDGE/CORNER, though -- the cross-
    section taken at the shape's own MID-height is untouched by either
    and stays a clean rectangle; a cylinder's cross-section is circular
    at every height, including the middle. Confirmed directly: a plain
    box measures residual=0.17, a CHAMFERED box measures residual=0.11
    (still clearly non-circular, unaffected by the edge treatment), an
    L-shape extrusion measures 0.12, while a plain cylinder measures
    0.002 and a sphere (through its own equator) measures 0.002 -- a
    wide, reliable margin.

    Tried along all 3 world axes, not just the longest extent: a
    cylinder's own rotation axis is not necessarily the longest extent
    (confirmed real bug -- a 40mm-diameter, 30mm-tall cylinder has its
    TWO diameter extents, 40 and 40, both exceed its own 30mm height,
    so picking "the longest extent" via argmax breaks the tie toward a
    DIAMETER axis instead of the rotation axis, slicing perpendicular to
    a diameter produces a RECTANGULAR cross-section, not circular, and
    incorrectly reports no circular cross-section at all). A circular
    cross-section perpendicular to ANY axis is sufficient -- true for a
    cylinder along its own rotation axis, and true for a sphere along
    every axis.

    Uses only the LARGEST discrete closed loop in the cross-section, not
    every vertex combined -- confirmed real bug: a cylinder with a hole
    drilled through it along its own rotation axis produces an ANNULUS
    cross-section (two separate boundary loops, outer radius=cylinder,
    inner radius=hole, confirmed directly: trimesh's own section.discrete
    splits them cleanly). Fitting one circle to both loops' points mixed
    together is fitting a circle to an annulus, which is a high-residual
    fit for ANY hole size, incorrectly reporting "not circular" for a
    genuine cylinder for no reason other than it also has a hole -- a
    base-geometry check must not depend on whether a separate feature
    requirement happens to also be satisfied. The hole is always the
    SMALLER loop; the shape's own true outer boundary is always the
    largest."""
    for axis in range(3):
        lo, hi = mesh.bounds[0][axis], mesh.bounds[1][axis]
        if hi - lo < 1e-6:
            continue
        normal = np.zeros(3)
        normal[axis] = 1.0
        origin = mesh.bounds[0].copy()
        origin[axis] = (lo + hi) / 2.0
        try:
            section = mesh.section(plane_origin=origin, plane_normal=normal)
        except Exception:
            continue
        if section is None or not section.discrete:
            continue
        perp = [i for i in range(3) if i != axis]
        # The loop with the largest own bounding-box diagonal, not the
        # most POINTS -- confirmed real bug: a hole cutter and the base
        # shape can tessellate their own circular boundary with the
        # SAME point count (both measured 65 points in a direct test),
        # so picking by point count is an arbitrary tie, not a reliable
        # "outer vs inner" signal; spatial extent is unambiguous.
        outer_loop = max(section.discrete, key=lambda loop: np.ptp(loop, axis=0).max() if len(loop) >= 2 else 0.0)
        if len(outer_loop) < 6:
            continue
        uv = outer_loop[:, perp]
        _cx, _cy, r, residual = _fit_circle_xy(uv)
        if r > 1e-6 and residual < 0.05:
            return True
    return False


def _verify_base_geometry(req: Requirement, mesh: trimesh.Trimesh, report: GeometryReport, description: str) -> Requirement:
    method = "bounding_box_and_curvature_signature"
    if req.geometry_type == "gear":
        return dataclasses.replace(
            req, status=RequirementStatus.UNSUPPORTED_TO_VERIFY, verification_method=method,
            failure_reason="gear tooth geometry is not independently re-measured from built geometry by this check -- only the op-sequence-level compatibility check (op_arbitration.py) covers gears",
        )
    two_equal, all_equal = _extent_shape_signals(report)
    curved = _has_circular_cross_section(mesh)
    if req.geometry_type == "rectangular":
        ok = not curved
    elif req.geometry_type == "cylindrical":
        ok = curved and two_equal and not all_equal
    elif req.geometry_type == "spherical":
        ok = curved and all_equal
    elif req.geometry_type == "toroidal":
        return dataclasses.replace(
            req, status=RequirementStatus.UNSUPPORTED_TO_VERIFY, verification_method=method,
            failure_reason="toroidal base geometry is not independently re-measured from built geometry by this check",
        )
    else:
        return dataclasses.replace(
            req, status=RequirementStatus.UNSUPPORTED_TO_VERIFY, verification_method=method,
            failure_reason=f"geometry_type {req.geometry_type!r} is not independently re-measured from built geometry by this check",
        )
    status = RequirementStatus.GEOMETRY_VERIFIED if ok else RequirementStatus.FAILED
    reason = "" if ok else (
        f"requested base geometry {req.geometry_type!r} does not match the built mesh's own shape signature "
        f"(bounding box {report.bounding_box_mm}, has_circular_cross_section={curved})"
    )
    return dataclasses.replace(req, status=status, verification_method=method, failure_reason=reason)


def _verify_subtractive_hole(req: Requirement, mesh: trimesh.Trimesh, report: GeometryReport, description: str) -> Requirement:
    method = "hole_count_and_diameter"
    actual = report.detected_hole_count
    if req.count is not None:
        if actual != req.count:
            return dataclasses.replace(
                req, status=RequirementStatus.FAILED, actual_count=actual, verification_method=method,
                failure_reason=f"requested {req.count} hole(s), built geometry has {actual}",
            )
    elif actual == 0:
        return dataclasses.replace(
            req, status=RequirementStatus.FAILED, actual_count=actual, verification_method=method,
            failure_reason="no hole-like feature detected in the built geometry",
        )

    # Diameter check -- reuses dimension_requirements.py's own clause-
    # aware extraction rather than a second, weaker ad hoc parse here;
    # only checked when it actually resolved a hole_diameter requirement
    # (never guessed from an unrelated number).
    claimed_diameter = next(
        (d.normalized_value for d in extract_dimension_requirements(description) if d.parameter == "hole_diameter"),
        None,
    )
    if claimed_diameter is not None and report.hole_diameters_mm:
        tolerance = max(0.5, claimed_diameter * 0.15)
        if not any(abs(d - claimed_diameter) <= tolerance for d in report.hole_diameters_mm):
            return dataclasses.replace(
                req, status=RequirementStatus.FAILED, actual_count=actual, verification_method=method,
                failure_reason=f"no detected hole matches the claimed {claimed_diameter:g}mm diameter (detected: {[round(d, 1) for d in report.hole_diameters_mm]})",
            )

    # Through vs blind -- confirmed real gap (adversarial audit, brief's
    # own Test B: "test that a blind hole does not pass" when a through-
    # hole was requested): count/diameter alone can both match while the
    # hole simply never breaks through the far side. bore_type() is a
    # SOFT text signal (op_requirements.py's own docstring) -- only
    # checked when the text actually states one, never invented.
    stated_bore_type = bore_type(description)
    if stated_bore_type == "through" and report.hole_through and not any(report.hole_through):
        return dataclasses.replace(
            req, status=RequirementStatus.FAILED, actual_count=actual, verification_method=method,
            failure_reason="a through-hole was requested, but every detected hole stops short of the far side (blind)",
        )
    if stated_bore_type == "blind" and report.hole_through and all(report.hole_through):
        return dataclasses.replace(
            req, status=RequirementStatus.FAILED, actual_count=actual, verification_method=method,
            failure_reason="a blind hole was requested, but the detected hole runs fully through the part",
        )
    return dataclasses.replace(req, status=RequirementStatus.GEOMETRY_VERIFIED, actual_count=actual, verification_method=method, failure_reason="")


def _verify_subtractive_slot(req: Requirement, mesh: trimesh.Trimesh, report: GeometryReport, description: str) -> Requirement:
    method = "wall_opening_elongated_loop"
    actual = report.detected_slot_count
    if actual == 0:
        return dataclasses.replace(
            req, status=RequirementStatus.FAILED, actual_count=0, verification_method=method,
            failure_reason="no slot-shaped (elongated through-opening) feature detected in the built geometry",
        )
    if req.dimensions_mm.get("length") is not None and req.dimensions_mm.get("width") is not None:
        claimed_len, claimed_wid = req.dimensions_mm["length"], req.dimensions_mm["width"]
        best = min(report.slot_dimensions_mm, key=lambda d: abs(d[0] - claimed_len) + abs(d[1] - claimed_wid))
        len_tol, wid_tol = max(2.0, claimed_len * 0.2), max(1.0, claimed_wid * 0.2)
        if abs(best[0] - claimed_len) > len_tol or abs(best[1] - claimed_wid) > wid_tol:
            return dataclasses.replace(
                req, status=RequirementStatus.FAILED, actual_count=actual, verification_method=method,
                failure_reason=f"closest detected slot is {best[0]:.1f}x{best[1]:.1f}mm, requested {claimed_len:g}x{claimed_wid:g}mm",
            )
    return dataclasses.replace(req, status=RequirementStatus.GEOMETRY_VERIFIED, actual_count=actual, verification_method=method, failure_reason="")


def _has_edge_treatment_signature(mesh: trimesh.Trimesh) -> bool:
    """Coarse PRESENCE-only signal, never a precise radius/distance
    measurement (Part 6's own explicit instruction for fillets/chamfers:
    "do not claim exact verification if the backend cannot reliably
    measure it"). A sharp, untreated box edge's own face-adjacency angle
    is tightly clustered at ~90deg (plus ~0deg from in-face diagonal
    triangulation) with NOTHING in between -- confirmed directly, a
    plain box has zero face-adjacency angles in [5, 80) degrees. A
    chamfer introduces new edges in roughly [20, 70) (the bevel's own
    transition angles); a fillet introduces many small-angle edges
    concentrated in roughly [5, 30) (the round's own fine facets). Both
    are caught by the same single window; only called when the base
    shape is confirmed box-like (see this module's own gating in
    verify_requirements) -- a cylindrical/spherical base's own curved
    surface produces the same small-angle signature on its own, with
    nothing applied at all (confirmed directly: a plain cylinder has 32
    such edges), so this check is never run against one."""
    angles = np.degrees(mesh.face_adjacency_angles)
    return int(np.sum((angles >= 5.0) & (angles < 80.0))) >= 3


def _verify_edge_treatment(req: Requirement, mesh: trimesh.Trimesh, report: GeometryReport, description: str, base_geometry_type: "str | None") -> Requirement:
    method = "edge_dihedral_angle_presence"
    if base_geometry_type != "rectangular":
        return dataclasses.replace(
            req, status=RequirementStatus.UNSUPPORTED_TO_VERIFY, verification_method=method,
            failure_reason=(
                "edge-treatment presence is only independently checked against a confirmed rectangular base "
                "shape -- a curved (cylindrical/spherical) base's own surface tessellation produces the same "
                "small-angle face-adjacency signature on its own, with nothing applied, so this check cannot "
                "reliably distinguish a requested fillet/chamfer from the base shape's own curvature there"
            ),
        )
    if not _has_edge_treatment_signature(mesh):
        return dataclasses.replace(
            req, status=RequirementStatus.FAILED, verification_method=method,
            failure_reason="no fillet/chamfer-like edge signature detected on the built geometry (presence-only check -- exact radius/distance is not independently verified)",
        )
    return dataclasses.replace(req, status=RequirementStatus.GEOMETRY_VERIFIED, verification_method=method, failure_reason="")


def _verify_stepped_section(req: Requirement, mesh: trimesh.Trimesh, report: GeometryReport, description: str) -> Requirement:
    """A genuine step means the part's own cross-sectional footprint
    (perpendicular to its principal axis) measurably CHANGES somewhere
    along its length -- sampled directly via trimesh.section, never
    trusted from the chosen op's own name (add_shouldered_cylinder being
    PLANNED is not proof a step actually exists in the built mesh;
    dimension_assignment.py could, in principle, leave both radii
    equal)."""
    method = "axial_cross_section_radius_change"
    two_equal, _all_equal = _extent_shape_signals(report)
    axis_name = max(_AXIS_INDEX, key=lambda a: report.bounding_box_mm[_AXIS_INDEX[a]]) if not two_equal else "z"
    axis_idx = _AXIS_INDEX[axis_name]
    axis_vec = np.zeros(3)
    axis_vec[axis_idx] = 1.0
    lo, hi = mesh.bounds[0][axis_idx], mesh.bounds[1][axis_idx]
    span = hi - lo
    if span < 1e-6:
        return dataclasses.replace(
            req, status=RequirementStatus.UNSUPPORTED_TO_VERIFY, verification_method=method,
            failure_reason="could not determine the part's own principal axis to sample cross-sections along",
        )
    radii = []
    for frac in (0.1, 0.3, 0.5, 0.7, 0.9):
        origin = mesh.bounds[0].copy()
        origin[axis_idx] = lo + span * frac
        try:
            section = mesh.section(plane_origin=origin, plane_normal=axis_vec)
        except Exception:
            section = None
        if section is None:
            continue
        pts = section.vertices
        perp = [i for i in range(3) if i != axis_idx]
        center = pts[:, perp].mean(axis=0)
        radii.append(float(np.max(np.linalg.norm(pts[:, perp] - center, axis=1))))
    if len(radii) < 2:
        return dataclasses.replace(
            req, status=RequirementStatus.UNSUPPORTED_TO_VERIFY, verification_method=method,
            failure_reason="could not sample enough cross-sections along the part's own axis to detect a step",
        )
    max_r, min_r = max(radii), min(radii)
    has_step = max_r > 1e-6 and (max_r - min_r) / max_r > 0.1
    status = RequirementStatus.GEOMETRY_VERIFIED if has_step else RequirementStatus.FAILED
    reason = "" if has_step else f"no meaningful cross-sectional radius change detected along the part's own {axis_name!r} axis (sampled radii: {[round(r, 2) for r in radii]})"
    return dataclasses.replace(req, status=status, verification_method=method, failure_reason=reason)


_UNVERIFIABLE_REASON_TEMPLATE = (
    "feature_verification.py does not yet independently re-measure a {feature_type!r} feature from built "
    "geometry -- only op_arbitration.py's own op-sequence-level compatibility check (did the PLANNED program "
    "include a plausible operation, not whether the BUILT mesh actually has the feature) covers this type"
)


def verify_requirements(
    requirements: list[Requirement], mesh: "trimesh.Trimesh | None", description: str, geometry_report: "GeometryReport | None" = None,
) -> list[Requirement]:
    """Never raises; returns NEW Requirement objects (never mutates the
    input list in place, matching dimension_assignment.AssignmentResult's
    own "return a new result" convention). `mesh` is None exactly when
    there was no mesh to check at all (a rejected build) -- every
    requirement becomes NOT_EVALUATED-equivalent (UNRESOLVED) rather than
    silently skipped."""
    if mesh is None:
        return [
            dataclasses.replace(req, status=RequirementStatus.UNRESOLVED, verification_method="no_mesh_built", failure_reason="no mesh was built at all -- nothing to verify this requirement against")
            for req in requirements
        ]
    if geometry_report is None:
        geometry_report = evaluate_mesh(mesh)

    base_geometry_type = next((r.geometry_type for r in requirements if r.feature_type == "base_geometry"), None)
    out: list[Requirement] = []
    for req in requirements:
        try:
            if req.feature_type == "base_geometry":
                out.append(_verify_base_geometry(req, mesh, geometry_report, description))
            elif req.feature_type == "subtractive_hole":
                out.append(_verify_subtractive_hole(req, mesh, geometry_report, description))
            elif req.feature_type == "subtractive_slot":
                out.append(_verify_subtractive_slot(req, mesh, geometry_report, description))
            elif req.feature_type == "edge_treatment":
                out.append(_verify_edge_treatment(req, mesh, geometry_report, description, base_geometry_type))
            elif req.feature_type == "stepped_section":
                out.append(_verify_stepped_section(req, mesh, geometry_report, description))
            else:
                out.append(dataclasses.replace(
                    req, status=RequirementStatus.UNSUPPORTED_TO_VERIFY, verification_method="op_sequence_only_not_geometry_verified",
                    failure_reason=_UNVERIFIABLE_REASON_TEMPLATE.format(feature_type=req.feature_type),
                ))
        except Exception as exc:
            out.append(dataclasses.replace(
                req, status=RequirementStatus.FAILED, verification_method=f"{req.feature_type}_check_raised",
                failure_reason=f"could not verify this requirement against the built geometry: {exc}",
            ))
    return out
