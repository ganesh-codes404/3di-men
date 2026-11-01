"""The 10 required regression tests (Part 11) for the Assembly
Compatibility and Constraint Engine -- every test builds REAL geometry
via sandbox.Sandbox and inspects actual measured dimensions/positions,
never hand-written specification values alone, per this brief's own
"must inspect actual generated geometry and transforms" rule."""

from dataclasses import dataclass

import pytest

from assembly_compatibility import (
    CompatibilityStatus, MechanicalRelationshipType, attempt_repair, attempt_repair_with_revalidation,
    classify_relationship_type, evaluate_assembly_compatibility, evaluate_axis_alignment, evaluate_bearing_housing_fit,
    evaluate_bolt_hole_fit, evaluate_face_contact, evaluate_gear_mesh, evaluate_motor_mount, evaluate_shaft_bore_fit,
)
from assembly_plan import Relationship
from sandbox import Sandbox


@dataclass
class _Point:
    id: str
    description: str
    relationship: "Relationship | None" = None


# ---------------- Test A: compatible shaft and bore ----------------

def test_a_compatible_shaft_and_bore():
    sb_h = Sandbox()
    housing_h = sb_h.add_box(60, 60, 30, center=(0, 0, 0))
    housing_h = sb_h.drill_hole(housing_h, radius=6.1, depth=30, center=(0, 0, 0), axis="z")  # 12.2mm bore
    housing_mesh = sb_h._get(housing_h)

    sb_s = Sandbox()
    shaft_h = sb_s.add_cylinder(radius=6.0, height=40, center=(0, 0, 0), axis="z")  # 12mm shaft
    shaft_mesh = sb_s._get(shaft_h)

    rel = Relationship(kind="concentric_with", target_id="housing_01", axis="z")
    eva = evaluate_shaft_bore_fit("REL-A", "shaft_01", "housing_01", "a 12mm diameter drive shaft", "a bearing housing with a 12.2mm bore", shaft_mesh, housing_mesh, rel)
    assert eva.status is CompatibilityStatus.COMPATIBLE
    assert eva.actual["shaft_diameter_mm"] == pytest.approx(12.0, abs=0.2)
    assert eva.actual["bore_diameter_mm"] == pytest.approx(12.2, abs=0.2)
    assert eva.calculated["diametral_clearance_mm"] == pytest.approx(0.2, abs=0.3)


# ---------------- Test B: incompatible shaft and bore (interference) ----------------

def test_b_incompatible_shaft_and_bore():
    sb_h = Sandbox()
    housing_h = sb_h.add_box(60, 60, 30, center=(0, 0, 0))
    housing_h = sb_h.drill_hole(housing_h, radius=5.0, depth=30, center=(0, 0, 0), axis="z")  # 10mm bore
    housing_mesh = sb_h._get(housing_h)

    sb_s = Sandbox()
    shaft_h = sb_s.add_cylinder(radius=6.0, height=40, center=(0, 0, 0), axis="z")  # 12mm shaft
    shaft_mesh = sb_s._get(shaft_h)

    rel = Relationship(kind="concentric_with", target_id="housing_01", axis="z")
    eva = evaluate_shaft_bore_fit("REL-B", "shaft_01", "housing_01", "a 12mm diameter shaft", "a housing with a 10mm bore", shaft_mesh, housing_mesh, rel)
    assert eva.status is CompatibilityStatus.INCOMPATIBLE
    assert eva.calculated["diametral_clearance_mm"] < 0


# ---------------- Adversarial audit: tolerance/fit-class matrix (Part 5) ----------------

def test_b2_explicitly_stated_interference_fit_with_genuine_interference_is_compatible():
    """A real interference/press fit is, BY DEFINITION, negative
    diametral clearance (the shaft is machined slightly larger than the
    bore so it must be pressed in). Confirmed real bug (adversarial
    audit, live-reproduced): the engine's own blanket "negative
    clearance == INCOMPATIBLE" rule used to fire before the fit-class
    check ever ran, so a CORRECTLY designed interference fit was always
    reported incompatible for being exactly what it was asked to be."""
    sb_h = Sandbox()
    housing_h = sb_h.add_box(60, 60, 30, center=(0, 0, 0))
    housing_h = sb_h.drill_hole(housing_h, radius=6.0, depth=30, center=(0, 0, 0), axis="z")  # 12.0mm bore
    housing_mesh = sb_h._get(housing_h)

    sb_s = Sandbox()
    shaft_h = sb_s.add_cylinder(radius=6.05, height=40, center=(0, 0, 0), axis="z")  # 12.1mm shaft -- genuine interference
    shaft_mesh = sb_s._get(shaft_h)

    rel = Relationship(kind="concentric_with", target_id="housing_01", axis="z")
    eva = evaluate_shaft_bore_fit("REL-B2", "shaft_01", "housing_01", "a 12.1mm diameter shaft, interference fit", "a housing with a 12.0mm bore", shaft_mesh, housing_mesh, rel)
    assert eva.status is CompatibilityStatus.COMPATIBLE
    assert eva.calculated["diametral_clearance_mm"] < 0


def test_b3_stated_interference_fit_but_actual_positive_clearance_is_incompatible():
    """The converse of the above: an interference fit was REQUESTED, but
    the built geometry has a shaft smaller than the bore -- it would
    simply fall out, never needing to be pressed. Must still be flagged,
    not waved through just because clearance happens to be positive."""
    sb_h = Sandbox()
    housing_h = sb_h.add_box(60, 60, 30, center=(0, 0, 0))
    housing_h = sb_h.drill_hole(housing_h, radius=6.5, depth=30, center=(0, 0, 0), axis="z")  # 13mm bore
    housing_mesh = sb_h._get(housing_h)

    sb_s = Sandbox()
    shaft_h = sb_s.add_cylinder(radius=6.0, height=40, center=(0, 0, 0), axis="z")  # 12mm shaft
    shaft_mesh = sb_s._get(shaft_h)

    rel = Relationship(kind="concentric_with", target_id="housing_01", axis="z")
    eva = evaluate_shaft_bore_fit("REL-B3", "shaft_01", "housing_01", "a 12mm diameter shaft, interference fit", "a housing with a 13mm bore", shaft_mesh, housing_mesh, rel)
    assert eva.status is CompatibilityStatus.INCOMPATIBLE


def test_b4_zero_clearance_line_to_line_fit_is_compatible_and_labeled_as_such():
    """Build the shaft to the bore's own ACTUALLY MEASURED diameter
    (never the nominal drill radius -- a drilled bore's measured
    diameter has a small, pre-existing tessellation bias relative to its
    nominal radius, a mesh-measurement artifact unrelated to this
    engine's own fit logic) so the two are genuinely line-to-line."""
    from evaluation.geometry_eval import evaluate_mesh

    sb_h = Sandbox()
    housing_h = sb_h.add_box(60, 60, 30, center=(0, 0, 0))
    housing_h = sb_h.drill_hole(housing_h, radius=6.0, depth=30, center=(0, 0, 0), axis="z")
    housing_mesh = sb_h._get(housing_h)
    measured_bore_diameter = evaluate_mesh(housing_mesh).hole_diameters_mm[0]

    sb_s = Sandbox()
    shaft_h = sb_s.add_cylinder(radius=measured_bore_diameter / 2.0, height=40, center=(0, 0, 0), axis="z")
    shaft_mesh = sb_s._get(shaft_h)

    rel = Relationship(kind="concentric_with", target_id="housing_01", axis="z")
    eva = evaluate_shaft_bore_fit("REL-B4", "shaft_01", "housing_01", "a shaft", "a housing with a bore", shaft_mesh, housing_mesh, rel)
    assert eva.status is CompatibilityStatus.COMPATIBLE
    assert eva.calculated["diametral_clearance_mm"] == pytest.approx(0.0, abs=0.01)
    assert eva.calculated["fit_classification"] == "line-to-line fit"


def test_b5_explicit_numeric_clearance_requirement_violated_is_incompatible():
    """An explicit numeric clearance requirement (not just a fit-class
    name) must be checked against the ACTUAL built clearance, not just
    verified to be positive."""
    sb_h = Sandbox()
    housing_h = sb_h.add_box(60, 60, 30, center=(0, 0, 0))
    housing_h = sb_h.drill_hole(housing_h, radius=7.0, depth=30, center=(0, 0, 0), axis="z")  # 14mm bore -- 2mm actual clearance
    housing_mesh = sb_h._get(housing_h)

    sb_s = Sandbox()
    shaft_h = sb_s.add_cylinder(radius=6.0, height=40, center=(0, 0, 0), axis="z")  # 12mm shaft
    shaft_mesh = sb_s._get(shaft_h)

    rel = Relationship(kind="concentric_with", target_id="housing_01", axis="z")
    eva = evaluate_shaft_bore_fit("REL-B5", "shaft_01", "housing_01", "a 12mm diameter shaft", "a housing with a bore, clearance 0.2mm", shaft_mesh, housing_mesh, rel)
    assert eva.status is CompatibilityStatus.INCOMPATIBLE  # 2mm actual vs 0.2mm required


def test_b6_missing_tolerance_information_reports_honest_clearance_not_invented_precision():
    """No fit class, no numeric clearance stated anywhere -- the engine
    must report the actual measured clearance honestly rather than
    inventing a precision classification it was never asked to verify
    (Part 3A's own explicit rule)."""
    sb_h = Sandbox()
    housing_h = sb_h.add_box(60, 60, 30, center=(0, 0, 0))
    housing_h = sb_h.drill_hole(housing_h, radius=6.5, depth=30, center=(0, 0, 0), axis="z")  # 13mm bore
    housing_mesh = sb_h._get(housing_h)

    sb_s = Sandbox()
    shaft_h = sb_s.add_cylinder(radius=6.0, height=40, center=(0, 0, 0), axis="z")  # 12mm shaft, 1mm clearance
    shaft_mesh = sb_s._get(shaft_h)

    rel = Relationship(kind="concentric_with", target_id="housing_01", axis="z")
    eva = evaluate_shaft_bore_fit("REL-B6", "shaft_01", "housing_01", "a 12mm diameter shaft", "a housing with a bore", shaft_mesh, housing_mesh, rel)
    assert eva.status is CompatibilityStatus.COMPATIBLE
    assert "no tolerance or fit class specified" in eva.calculated["fit_classification"]


# ---------------- Test C: gear center distance ----------------

def test_c_gear_center_distance_nominal_is_compatible():
    expected = 2.5 * (20 + 32) / 2.0  # 65mm
    sb_a = Sandbox()
    gear_a_h = sb_a.add_gear_profile(module_mm=2.5, tooth_count=20, face_width_mm=10, bore_diameter_mm=5, center=(0, 0, 0))
    mesh_a = sb_a._get(gear_a_h)

    sb_b = Sandbox()
    gear_b_h = sb_b.add_gear_profile(module_mm=2.5, tooth_count=32, face_width_mm=10, bore_diameter_mm=5, center=(expected, 0, 0))
    mesh_b = sb_b._get(gear_b_h)

    eva = evaluate_gear_mesh("REL-C", "gear_A", "gear_B", "a gear, module 2.5, 20 teeth", "a gear, module 2.5, 32 teeth", mesh_a, mesh_b)
    assert eva.status is CompatibilityStatus.COMPATIBLE
    assert eva.requested["center_distance_mm"] == pytest.approx(65.0, abs=0.01)
    assert eva.actual["center_distance_mm"] == pytest.approx(65.0, abs=1.0)


def test_c_gear_center_distance_wrong_is_incompatible():
    sb_a = Sandbox()
    gear_a_h = sb_a.add_gear_profile(module_mm=2.5, tooth_count=20, face_width_mm=10, bore_diameter_mm=5, center=(0, 0, 0))
    mesh_a = sb_a._get(gear_a_h)

    sb_b = Sandbox()
    gear_b_h = sb_b.add_gear_profile(module_mm=2.5, tooth_count=32, face_width_mm=10, bore_diameter_mm=5, center=(72.0, 0, 0))
    mesh_b = sb_b._get(gear_b_h)

    eva = evaluate_gear_mesh("REL-C2", "gear_A", "gear_B", "a gear, module 2.5, 20 teeth", "a gear, module 2.5, 32 teeth", mesh_a, mesh_b)
    assert eva.status is CompatibilityStatus.INCOMPATIBLE
    assert eva.actual["center_distance_mm"] == pytest.approx(72.0, abs=1.0)


# ---------------- Test D: gear module mismatch ----------------

def test_d_gear_module_mismatch_is_incompatible_regardless_of_distance():
    expected = 2.5 * (20 + 32) / 2.0
    sb_a = Sandbox()
    gear_a_h = sb_a.add_gear_profile(module_mm=2.5, tooth_count=20, face_width_mm=10, bore_diameter_mm=5, center=(0, 0, 0))
    mesh_a = sb_a._get(gear_a_h)

    sb_b = Sandbox()
    gear_b_h = sb_b.add_gear_profile(module_mm=3.0, tooth_count=32, face_width_mm=10, bore_diameter_mm=5, center=(expected, 0, 0))
    mesh_b = sb_b._get(gear_b_h)

    eva = evaluate_gear_mesh("REL-D", "gear_A", "gear_B", "a gear, module 2.5, 20 teeth", "a gear, module 3.0, 32 teeth", mesh_a, mesh_b)
    assert eva.status is CompatibilityStatus.INCOMPATIBLE
    assert "module" in eva.reason.lower()


def test_d2_gear_pressure_angle_mismatch_is_incompatible_even_with_correct_module_and_distance():
    """Confirmed real gap found by adversarial audit: correct module and
    correct center distance used to report COMPATIBLE even when the two
    gears' own stated pressure angles genuinely differ (20deg vs
    14.5deg) -- gears at different pressure angles cannot mesh
    correctly regardless of module/distance. Fixed by adding a
    pressure-angle check (only when the text actually states one on
    both sides, per Part 3B's own "if specified" qualifier)."""
    expected = 2.5 * (20 + 32) / 2.0
    sb_a = Sandbox()
    mesh_a = sb_a._get(sb_a.add_gear_profile(module_mm=2.5, tooth_count=20, face_width_mm=10, bore_diameter_mm=5, pressure_angle_deg=20.0, center=(0, 0, 0)))
    sb_b = Sandbox()
    mesh_b = sb_b._get(sb_b.add_gear_profile(module_mm=2.5, tooth_count=32, face_width_mm=10, bore_diameter_mm=5, pressure_angle_deg=14.5, center=(expected, 0, 0)))

    eva = evaluate_gear_mesh(
        "REL-D2", "gear_A", "gear_B", "a gear, module 2.5, 20 teeth, 20 degree pressure angle",
        "a gear, module 2.5, 32 teeth, 14.5 degree pressure angle", mesh_a, mesh_b,
    )
    assert eva.status is CompatibilityStatus.INCOMPATIBLE
    assert "pressure angle" in eva.reason.lower()

    # same scenario with no pressure angle stated at all must NOT regress to UNDER_SPECIFIED
    eva_unstated = evaluate_gear_mesh("REL-D3", "gear_A", "gear_B", "a gear, module 2.5, 20 teeth", "a gear, module 2.5, 32 teeth", mesh_a, mesh_b)
    assert eva_unstated.status is CompatibilityStatus.COMPATIBLE


def test_d4_gear_with_nonparallel_rotational_axes_is_incompatible_even_with_correct_module_and_distance():
    """Two spur gears whose own rotational axes are not parallel (one
    rotated 90 degrees relative to the other) can never actually mesh,
    no matter how correct their module/tooth-count/center-distance
    numbers are -- confirmed via direct measurement of real built,
    rotated geometry (not merely trusting a planned rotation value)."""
    expected = 2.5 * (20 + 32) / 2.0
    sb_a = Sandbox()
    mesh_a = sb_a._get(sb_a.add_gear_profile(module_mm=2.5, tooth_count=20, face_width_mm=10, bore_diameter_mm=5, center=(0, 0, 0)))
    sb_b = Sandbox()
    gear_b_h = sb_b.add_gear_profile(module_mm=2.5, tooth_count=32, face_width_mm=10, bore_diameter_mm=5, center=(0, 0, 0))
    gear_b_h = sb_b.rotate(gear_b_h, axis="x", angle_deg=90)
    mesh_b = sb_b._get(gear_b_h)
    mesh_b.apply_translation((expected, 0, 0))  # even at the otherwise-correct nominal center distance

    eva = evaluate_gear_mesh("REL-D4", "gear_A", "gear_B", "a gear, module 2.5, 20 teeth", "a gear, module 2.5, 32 teeth", mesh_a, mesh_b)
    assert eva.status is CompatibilityStatus.INCOMPATIBLE
    assert "not parallel" in eva.reason.lower()


# ---------------- Test E: misaligned shaft and bore ----------------

def test_e_misaligned_shaft_and_bore_is_incompatible():
    sb_h = Sandbox()
    housing_h = sb_h.add_box(60, 60, 30, center=(0, 0, 0))
    housing_h = sb_h.drill_hole(housing_h, radius=6.1, depth=30, center=(0, 0, 0), axis="z")
    housing_mesh = sb_h._get(housing_h)

    sb_s = Sandbox()
    # correct diameter, but offset 15mm sideways from the bore's own center
    shaft_h = sb_s.add_cylinder(radius=6.0, height=40, center=(15.0, 0, 0), axis="z")
    shaft_mesh = sb_s._get(shaft_h)

    rel = Relationship(kind="concentric_with", target_id="housing_01", axis="z")
    eva = evaluate_shaft_bore_fit("REL-E", "shaft_01", "housing_01", "a 12mm diameter shaft", "a housing with a 12.2mm bore", shaft_mesh, housing_mesh, rel)
    assert eva.status is CompatibilityStatus.INCOMPATIBLE
    assert eva.calculated["concentricity_offset_mm"] > 1.0


# ---------------- Test F: motor mounting pattern ----------------

def _nema_pattern_box(length, width, height, hole_radius, offset, bad_hole_offset=None):
    sb = Sandbox()
    h = sb.add_box(length, width, height, center=(0, 0, 0))
    positions = [(offset, offset), (offset, -offset), (-offset, offset), (-offset, -offset)]
    if bad_hole_offset is not None:
        positions[0] = bad_hole_offset
    for x, y in positions:
        h = sb.drill_hole(h, radius=hole_radius, depth=height, center=(x, y, 0), axis="z")
    return sb._get(h)


def test_f_motor_mounting_pattern_compatible():
    motor_mesh = _nema_pattern_box(42, 42, 10, 1.5, 15.5)
    mount_mesh = _nema_pattern_box(80, 80, 6, 1.6, 15.5)
    rel = Relationship(kind="attached_to_face", target_id="mount_01", face="+z")
    eva = evaluate_motor_mount("REL-F", "motor_01", "mount_01", "a NEMA17 stepper motor", "a mounting bracket plate", motor_mesh, mount_mesh, rel)
    assert eva.status is CompatibilityStatus.COMPATIBLE


def test_f_motor_mounting_pattern_one_hole_misplaced_is_incompatible():
    motor_mesh = _nema_pattern_box(42, 42, 10, 1.5, 15.5)
    mount_mesh = _nema_pattern_box(80, 80, 6, 1.6, 15.5, bad_hole_offset=(15.5, 30.0))
    rel = Relationship(kind="attached_to_face", target_id="mount_01", face="+z")
    eva = evaluate_motor_mount("REL-F2", "motor_01", "mount_01", "a NEMA17 stepper motor", "a mounting bracket plate", motor_mesh, mount_mesh, rel)
    assert eva.status is CompatibilityStatus.INCOMPATIBLE
    assert eva.requested["motor_mounting_holes"] == eva.actual["mount_holes"] == 4  # same COUNT -- the failure is the LAYOUT


def _asymmetric_pattern_box(length, width, height, hole_radius, positions):
    sb = Sandbox()
    h = sb.add_box(length, width, height, center=(0, 0, 0))
    for x, y in positions:
        h = sb.drill_hole(h, radius=hole_radius, depth=height, center=(x, y, 0), axis="z")
    return sb._get(h)


def test_f3_mirrored_asymmetric_hole_pattern_same_count_and_diameter_is_incompatible():
    """A pattern that is the MIRROR IMAGE of another is the classic
    counterexample for a naive "nearest distance from center" hole-
    pattern check -- same hole count, same diameters, even the same set
    of per-hole distances from the center, but the holes are NOT where
    the motor needs them. Only caught correctly if the comparison uses
    full 2D offset VECTORS (dx, dy) per hole, not just scalar distance.
    Must use an ASYMMETRIC pattern -- a symmetric one would be
    unaffected by mirroring and wouldn't actually test this."""
    motor_positions = [(10, 0), (-5, 15), (-5, -15)]
    mirror_positions = [(-10, 0), (5, 15), (5, -15)]
    motor_mesh = _asymmetric_pattern_box(42, 42, 10, 1.5, motor_positions)
    mount_mesh = _asymmetric_pattern_box(80, 80, 6, 1.6, mirror_positions)

    rel = Relationship(kind="attached_to_face", target_id="mount_01", face="+z")
    eva = evaluate_motor_mount("REL-F3", "motor_01", "mount_01", "a motor", "a mount", motor_mesh, mount_mesh, rel)
    assert eva.status is CompatibilityStatus.INCOMPATIBLE
    assert eva.requested["motor_mounting_holes"] == eva.actual["mount_holes"] == 3  # same count -- the failure is genuinely the layout


# ---------------- Test G: intended insertion is not a collision ----------------

def test_g_intended_insertion_is_compatible_not_a_collision():
    from parallel_builder import _relationship_connected

    sb_h = Sandbox()
    bearing_h = sb_h.add_box(40, 40, 20, center=(0, 0, 0))
    bearing_h = sb_h.drill_hole(bearing_h, radius=6.1, depth=20, center=(0, 0, 0), axis="z")
    bearing_mesh = sb_h._get(bearing_h)

    sb_s = Sandbox()
    shaft_h = sb_s.add_cylinder(radius=6.0, height=60, center=(0, 0, 0), axis="z")
    shaft_mesh = sb_s._get(shaft_h)

    relationships = {"shaft_01": Relationship(kind="concentric_with", target_id="bearing_01", axis="z"), "bearing_01": None}
    assert _relationship_connected("shaft_01", "bearing_01", relationships)  # the EXISTING collision detector must treat this as intended, not a collision

    rel = Relationship(kind="concentric_with", target_id="bearing_01", axis="z")
    eva = evaluate_shaft_bore_fit("REL-G", "shaft_01", "bearing_01", "a 12mm shaft", "a bearing with a 12.2mm bore", shaft_mesh, bearing_mesh, rel)
    assert eva.status is CompatibilityStatus.COMPATIBLE


# ---------------- Test H: intended face contact ----------------

def test_h_intended_face_contact_is_compatible():
    sb_base = Sandbox()
    base_h = sb_base.add_box(60, 60, 10, center=(0, 0, 0))
    base_mesh = sb_base._get(base_h)

    sb_lid = Sandbox()
    lid_h = sb_lid.add_box(60, 60, 5, center=(0, 0, 7.5))  # flush on top: base top at z=5, lid bottom at z=5
    lid_mesh = sb_lid._get(lid_h)

    rel = Relationship(kind="attached_to_face", target_id="base_01", face="+z")
    eva = evaluate_face_contact("REL-H", "lid_01", "base_01", "a lid plate", "a base box", lid_mesh, base_mesh, rel)
    assert eva.status is CompatibilityStatus.COMPATIBLE
    assert eva.calculated["face_gap_mm"] < 1.0


# ---------------- Adversarial audit: generic AXIS_ALIGNMENT fallback (Part 4, missing keywords) ----------------

def test_axis_alignment_fallback_compatible_when_no_mechanical_keywords_present():
    """Two components with a concentric_with relationship but NEITHER
    description mentioning any shaft/bore/bearing/bolt/gear keyword must
    still be evaluated -- as the generic AXIS_ALIGNMENT type, not
    silently skipped or reported UNSUPPORTED just for lacking a more
    specific keyword match."""
    mech_type = classify_relationship_type("a rod", "a tube", "concentric_with")
    assert mech_type is MechanicalRelationshipType.AXIS_ALIGNMENT

    sb_a = Sandbox()
    mesh_a = sb_a._get(sb_a.add_cylinder(radius=5.0, height=40, center=(0, 0, 0), axis="z"))
    sb_b = Sandbox()
    mesh_b = sb_b._get(sb_b.add_cylinder(radius=10.0, height=20, center=(0, 0, 0), axis="z"))

    rel = Relationship(kind="concentric_with", target_id="tube_01", axis="z")
    eva = evaluate_axis_alignment("REL-AXIS-1", "rod_01", "tube_01", "a rod", "a tube", mesh_a, mesh_b, rel)
    assert eva.status is CompatibilityStatus.COMPATIBLE
    assert eva.calculated["offset_mm"] < 1.0


def test_axis_alignment_fallback_incompatible_when_actually_offset():
    mech_type = classify_relationship_type("a rod", "a tube", "concentric_with")
    assert mech_type is MechanicalRelationshipType.AXIS_ALIGNMENT

    sb_a = Sandbox()
    mesh_a = sb_a._get(sb_a.add_cylinder(radius=5.0, height=40, center=(20.0, 0, 0), axis="z"))  # offset 20mm off the tube's own axis
    sb_b = Sandbox()
    mesh_b = sb_b._get(sb_b.add_cylinder(radius=10.0, height=20, center=(0, 0, 0), axis="z"))

    rel = Relationship(kind="concentric_with", target_id="tube_01", axis="z")
    eva = evaluate_axis_alignment("REL-AXIS-2", "rod_01", "tube_01", "a rod", "a tube", mesh_a, mesh_b, rel)
    assert eva.status is CompatibilityStatus.INCOMPATIBLE
    assert eva.calculated["offset_mm"] == pytest.approx(20.0, abs=0.5)


def test_axis_alignment_reached_end_to_end_through_the_main_engine():
    """Confirms the fallback is actually WIRED into evaluate_assembly_
    compatibility's own dispatch, not just reachable by calling
    evaluate_axis_alignment directly."""
    sb_a = Sandbox()
    mesh_a = sb_a._get(sb_a.add_cylinder(radius=5.0, height=40, center=(20.0, 0, 0), axis="z"))
    sb_b = Sandbox()
    mesh_b = sb_b._get(sb_b.add_cylinder(radius=10.0, height=20, center=(0, 0, 0), axis="z"))

    rel = Relationship(kind="concentric_with", target_id="tube_01", axis="z")
    points = [_Point("rod_01", "a rod"), _Point("tube_01", "a tube")]
    report = evaluate_assembly_compatibility(points, {"rod_01": mesh_a, "tube_01": mesh_b}, {"rod_01": rel, "tube_01": None})
    assert len(report.evaluations) == 1
    assert report.evaluations[0].relationship_type is MechanicalRelationshipType.AXIS_ALIGNMENT
    assert report.evaluations[0].status is CompatibilityStatus.INCOMPATIBLE


# ---------------- Test I: unsupported relationship ----------------

def test_i_unsupported_gear_type_is_never_compatible():
    mech_type = classify_relationship_type("a helical gear, module 2.5, 20 teeth", "a helical gear, module 2.5, 32 teeth", "concentric_with")
    assert mech_type is MechanicalRelationshipType.UNSUPPORTED

    points = [
        _Point("gear_A", "a helical gear, module 2.5, 20 teeth"),
        _Point("gear_B", "a helical gear, module 2.5, 32 teeth"),
    ]
    sb_a = Sandbox()
    mesh_a = sb_a._get(sb_a.add_gear_profile(module_mm=2.5, tooth_count=20, face_width_mm=10, bore_diameter_mm=5, center=(0, 0, 0)))
    sb_b = Sandbox()
    mesh_b = sb_b._get(sb_b.add_gear_profile(module_mm=2.5, tooth_count=32, face_width_mm=10, bore_diameter_mm=5, center=(65, 0, 0)))
    report = evaluate_assembly_compatibility(points, {"gear_A": mesh_a, "gear_B": mesh_b}, {"gear_A": None, "gear_B": None})
    assert len(report.evaluations) == 1
    assert report.evaluations[0].status is CompatibilityStatus.UNSUPPORTED_TO_VERIFY
    assert report.evaluations[0].status is not CompatibilityStatus.COMPATIBLE


# ---------------- Adversarial audit: declared relationship kind conflicting with description keywords ----------------

def test_relationship_kind_conflicting_with_description_keywords_is_disclosed_not_silently_misapplied():
    """A description implying a coaxial diametral fit (shaft+bearing
    keywords) but a DECLARED relationship kind that is not actually
    coaxial (attached_to_face is a flush-face semantic, never
    concentric) must never be silently classified and evaluated as
    SHAFT_BORE_FIT anyway -- doing so would run a concentricity/
    clearance check using an axis the relationship never declared,
    and could report COMPATIBLE for two components that were never
    actually asserted to be coaxial at all. Confirmed real bug (found
    during the adversarial audit) rather than a hypothetical: before
    this fix, classify_relationship_type ignored relationship_kind
    entirely for its shaft/bore/bearing/bolt keyword-based branches."""
    mech_type = classify_relationship_type("a 12mm diameter shaft", "a bearing housing with a bore", "attached_to_face")
    assert mech_type is MechanicalRelationshipType.UNSUPPORTED

    sb_h = Sandbox()
    housing_h = sb_h.add_box(60, 60, 30, center=(0, 0, 0))
    housing_h = sb_h.drill_hole(housing_h, radius=6.25, depth=30, center=(0, 0, 0), axis="z")  # 12.5mm bore
    housing_mesh = sb_h._get(housing_h)

    sb_s = Sandbox()
    shaft_h = sb_s.add_cylinder(radius=6.0, height=40, center=(0, 0, 0), axis="z")  # 12mm shaft, perfectly aligned
    shaft_mesh = sb_s._get(shaft_h)

    points = [
        _Point("shaft_01", "a 12mm diameter shaft"),
        _Point("housing_01", "a bearing housing with a bore"),
    ]
    rel = Relationship(kind="attached_to_face", target_id="housing_01", face="+z")
    report = evaluate_assembly_compatibility(
        points, {"shaft_01": shaft_mesh, "housing_01": housing_mesh}, {"shaft_01": rel, "housing_01": None},
    )
    assert len(report.evaluations) == 1
    eva = report.evaluations[0]
    assert eva.status is CompatibilityStatus.UNSUPPORTED_TO_VERIFY
    assert eva.status is not CompatibilityStatus.COMPATIBLE  # the real risk this guards against: a false positive from using the wrong evaluator


def test_relationship_kind_actually_coaxial_still_classifies_normally():
    """The new relationship_kind gate must not be overbroad: concentric_with
    and parallel_to are the two kinds a diametral fit check is actually
    meaningful for, and both must still classify exactly as before."""
    assert classify_relationship_type("a 12mm diameter shaft", "a bearing housing with a bore", "concentric_with") is MechanicalRelationshipType.SHAFT_BORE_FIT
    assert classify_relationship_type("a 12mm diameter shaft", "a bearing housing with a bore", "parallel_to") is MechanicalRelationshipType.SHAFT_BORE_FIT
    assert classify_relationship_type("a bolt", "a plate with a hole", "concentric_with") is MechanicalRelationshipType.BOLT_HOLE_FIT
    assert classify_relationship_type("a bearing", "a housing with a bore", "symmetric_about") is MechanicalRelationshipType.UNSUPPORTED


def test_unknown_relationship_kind_never_silently_passes_even_with_matching_keywords():
    """A relationship kind this engine has never heard of (a typo, or a
    kind introduced elsewhere in the pipeline this module doesn't yet
    support) must fall through to UNSUPPORTED deterministically -- never
    crash, never guess a type from keywords alone, and never slip into
    MOTOR_MOUNT/FACE_CONTACT just because motor+mount keywords happen to
    be present (those two types are only ever inferred together with
    relationship_kind == "attached_to_face", not from keywords alone)."""
    assert classify_relationship_type("a plate", "a bracket", "some_unrecognized_kind") is MechanicalRelationshipType.UNSUPPORTED
    assert classify_relationship_type("a motor", "a mount bracket", "some_unrecognized_kind") is MechanicalRelationshipType.UNSUPPORTED


def test_component_playing_two_roles_still_resolves_deterministically_not_randomly():
    """A component whose own description mentions BOTH "shaft" and
    "bearing" (an ambiguous dual role, e.g. "a shaft with an integrated
    bearing collar") must still resolve to exactly one type, every call,
    not an arbitrary/unstable choice -- and whichever label wins
    (BEARING_HOUSING_FIT takes priority over SHAFT_BORE_FIT as the more
    specific match), the UNDERLYING geometric check performed is the
    same diametral-fit measurement either way (_evaluate_cylinder_in_bore
    is shared by all three evaluate_*_fit functions), so the label choice
    alone can never turn a real interference into a false COMPATIBLE."""
    first = classify_relationship_type("a shaft with an integrated bearing collar", "a housing with a bore", "concentric_with")
    second = classify_relationship_type("a shaft with an integrated bearing collar", "a housing with a bore", "concentric_with")
    assert first is second is MechanicalRelationshipType.BEARING_HOUSING_FIT

    sb_h = Sandbox()
    housing_h = sb_h.add_box(60, 60, 30, center=(0, 0, 0))
    housing_h = sb_h.drill_hole(housing_h, radius=5.0, depth=30, center=(0, 0, 0), axis="z")  # 10mm bore
    housing_mesh = sb_h._get(housing_h)
    sb_s = Sandbox()
    shaft_h = sb_s.add_cylinder(radius=6.0, height=40, center=(0, 0, 0), axis="z")  # 12mm -- genuine interference
    shaft_mesh = sb_s._get(shaft_h)
    rel = Relationship(kind="concentric_with", target_id="housing_01", axis="z")
    eva = evaluate_bearing_housing_fit("REL-DUALROLE", "shaft_01", "housing_01", "a shaft with an integrated bearing collar", "a housing with a bore", shaft_mesh, housing_mesh, rel)
    assert eva.status is CompatibilityStatus.INCOMPATIBLE  # the real interference is still caught regardless of which label was chosen


# ---------------- Test J: multiple constraints, one conflict ----------------

def test_j_one_conflicting_relationship_does_not_affect_the_others():
    """A shaft must (1) fit into a bearing [introduced conflict: too
    large] and (2) sit flush against a mounting plate. Both
    relationships are evaluated independently -- the conflict in (1)
    must not change the correct COMPATIBLE result of (2)."""
    sb_bearing = Sandbox()
    bearing_h = sb_bearing.add_box(40, 40, 20, center=(0, 0, 0))
    bearing_h = sb_bearing.drill_hole(bearing_h, radius=5.0, depth=20, center=(0, 0, 0), axis="z")  # 10mm bore -- too small for a 12mm shaft
    bearing_mesh = sb_bearing._get(bearing_h)

    sb_shaft = Sandbox()
    shaft_h = sb_shaft.add_cylinder(radius=6.0, height=60, center=(0, 0, 0), axis="z")  # 12mm shaft
    shaft_mesh = sb_shaft._get(shaft_h)

    # shaft spans z=[-30, 30]; plate (6mm thick) placed so its own top face is flush with the shaft's bottom face (z=-30)
    sb_plate = Sandbox()
    plate_h = sb_plate.add_box(80, 80, 6, center=(0, 0, -33.0))
    plate_mesh = sb_plate._get(plate_h)

    bearing_rel = Relationship(kind="concentric_with", target_id="bearing_01", axis="z")
    eva_bearing = evaluate_shaft_bore_fit("REL-J1", "shaft_01", "bearing_01", "a 12mm shaft", "a bearing with a 10mm bore", shaft_mesh, bearing_mesh, bearing_rel)
    assert eva_bearing.status is CompatibilityStatus.INCOMPATIBLE  # the introduced conflict

    plate_rel = Relationship(kind="attached_to_face", target_id="plate_01", face="+z")  # the plate's own TOP face (z=-30), which the shaft's bottom end (z=-30) rests on
    eva_plate = evaluate_face_contact("REL-J2", "shaft_01", "plate_01", "a 12mm shaft", "a mounting plate", shaft_mesh, plate_mesh, plate_rel)
    assert eva_plate.status is CompatibilityStatus.COMPATIBLE  # unaffected by the OTHER relationship's own conflict


# ---------------- Part 10: bounded repair ----------------

def test_repair_fixes_gear_center_distance_without_changing_module_or_teeth():
    sb_a = Sandbox()
    mesh_a = sb_a._get(sb_a.add_gear_profile(module_mm=2.5, tooth_count=20, face_width_mm=10, bore_diameter_mm=5, center=(0, 0, 0)))
    sb_b = Sandbox()
    mesh_b = sb_b._get(sb_b.add_gear_profile(module_mm=2.5, tooth_count=32, face_width_mm=10, bore_diameter_mm=5, center=(72.0, 0, 0)))

    eva = evaluate_gear_mesh("REL-REPAIR-1", "gear_A", "gear_B", "a gear, module 2.5, 20 teeth", "a gear, module 2.5, 32 teeth", mesh_a, mesh_b)
    assert eva.status is CompatibilityStatus.INCOMPATIBLE

    result = attempt_repair(eva, mesh_a, mesh_b)
    assert result is not None
    repaired_a, repaired_b, attempt = result
    assert attempt.applied

    reeval = evaluate_gear_mesh("REL-REPAIR-1b", "gear_A", "gear_B", "a gear, module 2.5, 20 teeth", "a gear, module 2.5, 32 teeth", repaired_a, repaired_b)
    assert reeval.status is CompatibilityStatus.COMPATIBLE
    assert reeval.actual["center_distance_mm"] == pytest.approx(65.0, abs=1.0)


def test_repair_aligns_shaft_to_bore_without_changing_diameter():
    sb_h = Sandbox()
    housing_h = sb_h.add_box(60, 60, 30, center=(0, 0, 0))
    housing_h = sb_h.drill_hole(housing_h, radius=6.1, depth=30, center=(0, 0, 0), axis="z")
    housing_mesh = sb_h._get(housing_h)

    sb_s = Sandbox()
    shaft_h = sb_s.add_cylinder(radius=6.0, height=40, center=(15.0, 0, 0), axis="z")
    shaft_mesh = sb_s._get(shaft_h)

    rel = Relationship(kind="concentric_with", target_id="housing_01", axis="z")
    eva = evaluate_shaft_bore_fit("REL-REPAIR-2", "shaft_01", "housing_01", "a 12mm shaft", "a housing with a 12.2mm bore", shaft_mesh, housing_mesh, rel)
    assert eva.status is CompatibilityStatus.INCOMPATIBLE

    result = attempt_repair(eva, shaft_mesh, housing_mesh)
    assert result is not None
    repaired_shaft, repaired_housing, attempt = result
    original_diameter = eva.actual["shaft_diameter_mm"]

    reeval = evaluate_shaft_bore_fit("REL-REPAIR-2b", "shaft_01", "housing_01", "a 12mm shaft", "a housing with a 12.2mm bore", repaired_shaft, repaired_housing, rel)
    assert reeval.status is CompatibilityStatus.COMPATIBLE
    assert reeval.actual["shaft_diameter_mm"] == pytest.approx(original_diameter, abs=0.01)  # diameter never touched to fix a position


def test_repair_is_rejected_when_it_would_break_a_components_own_other_relationship():
    """Confirmed real gap found by adversarial audit: a gear pair's own
    center-distance repair moves gear_B's mesh alone -- with no
    awareness that gear_B is ALSO concentric_with its own separate
    shaft (shaft_B). Applying that repair in isolation silently broke
    shaft_B's own concentricity with gear_B (a real, measured 7mm
    offset) with nothing surfacing it. attempt_repair_with_revalidation
    must detect this and REJECT the repair outright -- never trade one
    fixed relationship for a newly broken one."""
    expected = 2.5 * (20 + 32) / 2.0
    sb_a = Sandbox()
    mesh_a = sb_a._get(sb_a.add_gear_profile(module_mm=2.5, tooth_count=20, face_width_mm=10, bore_diameter_mm=5, center=(0, 0, 0)))
    sb_b = Sandbox()
    mesh_b_wrong = sb_b._get(sb_b.add_gear_profile(module_mm=2.5, tooth_count=32, face_width_mm=10, bore_diameter_mm=5, center=(72, 0, 0)))  # wrong center distance
    sb_shaft_b = Sandbox()
    shaft_b_mesh = sb_shaft_b._get(sb_shaft_b.add_cylinder(radius=5.0, height=60, center=(72, 0, 0), axis="z"))  # concentric with gear_B's CURRENT (wrong) position

    points = [
        _Point("gear_A", "a gear, module 2.5, 20 teeth"),
        _Point("gear_B", "a gear, module 2.5, 32 teeth"),
        _Point("shaft_B", "a shaft", relationship=Relationship(kind="concentric_with", target_id="gear_B", axis="z")),
    ]
    placed_meshes_by_id = {"gear_A": mesh_a, "gear_B": mesh_b_wrong, "shaft_B": shaft_b_mesh}
    relationships = {"gear_A": None, "gear_B": None, "shaft_B": Relationship(kind="concentric_with", target_id="gear_B", axis="z")}

    eva_gear = evaluate_gear_mesh("REL-REPAIR-3", "gear_A", "gear_B", "a gear, module 2.5, 20 teeth", "a gear, module 2.5, 32 teeth", mesh_a, mesh_b_wrong)
    assert eva_gear.status is CompatibilityStatus.INCOMPATIBLE

    result = attempt_repair_with_revalidation(eva_gear, placed_meshes_by_id, points, relationships)
    assert result.accepted is False
    assert result.meshes_by_id is placed_meshes_by_id  # rejected -- original meshes untouched
    assert len(result.broken_relationships) == 1
    assert result.broken_relationships[0].component_a == "shaft_B"


def test_repair_with_revalidation_is_accepted_when_nothing_else_is_affected():
    """The other half of the same guard: a repair with no OTHER
    relationship to break must still be applied -- the revalidation
    check must not reject every repair indiscriminately."""
    expected = 2.5 * (20 + 32) / 2.0
    sb_a = Sandbox()
    mesh_a = sb_a._get(sb_a.add_gear_profile(module_mm=2.5, tooth_count=20, face_width_mm=10, bore_diameter_mm=5, center=(0, 0, 0)))
    sb_b = Sandbox()
    mesh_b_wrong = sb_b._get(sb_b.add_gear_profile(module_mm=2.5, tooth_count=32, face_width_mm=10, bore_diameter_mm=5, center=(72, 0, 0)))

    points = [_Point("gear_A", "a gear, module 2.5, 20 teeth"), _Point("gear_B", "a gear, module 2.5, 32 teeth")]
    placed_meshes_by_id = {"gear_A": mesh_a, "gear_B": mesh_b_wrong}
    relationships = {"gear_A": None, "gear_B": None}

    eva_gear = evaluate_gear_mesh("REL-REPAIR-4", "gear_A", "gear_B", "a gear, module 2.5, 20 teeth", "a gear, module 2.5, 32 teeth", mesh_a, mesh_b_wrong)
    result = attempt_repair_with_revalidation(eva_gear, placed_meshes_by_id, points, relationships)
    assert result.accepted is True
    assert result.broken_relationships == []

    reeval = evaluate_gear_mesh("REL-REPAIR-4b", "gear_A", "gear_B", "a gear, module 2.5, 20 teeth", "a gear, module 2.5, 32 teeth", result.meshes_by_id["gear_A"], result.meshes_by_id["gear_B"])
    assert reeval.status is CompatibilityStatus.COMPATIBLE


# ---------------- Adversarial audit: repair regression matrix (Part 6) ----------------

def test_repair_never_touches_the_fixed_components_own_mesh():
    """attempt_repair's own forbidden-action rule ("never modify a hard
    dimension... reposition only") taken literally: the component NOT
    being moved must come back as the exact same mesh object, not merely
    an equivalent copy -- the strongest possible guarantee that nothing
    about it (position, shape, or any hard dimension) changed."""
    sb_a = Sandbox()
    mesh_a = sb_a._get(sb_a.add_gear_profile(module_mm=2.5, tooth_count=20, face_width_mm=10, bore_diameter_mm=5, center=(0, 0, 0)))
    sb_b = Sandbox()
    mesh_b = sb_b._get(sb_b.add_gear_profile(module_mm=2.5, tooth_count=32, face_width_mm=10, bore_diameter_mm=5, center=(72.0, 0, 0)))

    eva = evaluate_gear_mesh("REL-FIXED-1", "gear_A", "gear_B", "a gear, module 2.5, 20 teeth", "a gear, module 2.5, 32 teeth", mesh_a, mesh_b)
    result = attempt_repair(eva, mesh_a, mesh_b)
    assert result is not None
    repaired_a, repaired_b, attempt = result
    assert repaired_a is mesh_a  # gear_A was never the one moved -- comes back as the literal same object
    assert len(repaired_b.faces) == len(mesh_b.faces)  # gear_B's own shape/tooth geometry is untouched, only its position moved


def test_repair_is_infeasible_and_rejected_for_a_module_mismatch():
    """A module mismatch can never be fixed by repositioning alone --
    the teeth are the wrong size regardless of where the gears sit.
    attempt_repair must report this as unfixed (return None), never
    silently reposition the gears to some distance that doesn't
    actually make the teeth mesh correctly."""
    sb_a = Sandbox()
    mesh_a = sb_a._get(sb_a.add_gear_profile(module_mm=2.0, tooth_count=20, face_width_mm=10, bore_diameter_mm=5, center=(0, 0, 0)))
    sb_b = Sandbox()
    mesh_b = sb_b._get(sb_b.add_gear_profile(module_mm=3.0, tooth_count=20, face_width_mm=10, bore_diameter_mm=5, center=(60, 0, 0)))

    eva = evaluate_gear_mesh("REL-INFEASIBLE-1", "gear_A", "gear_B", "a gear, module 2.0, 20 teeth", "a gear, module 3.0, 20 teeth", mesh_a, mesh_b)
    assert eva.status is CompatibilityStatus.INCOMPATIBLE
    assert attempt_repair(eva, mesh_a, mesh_b) is None


def test_repair_is_infeasible_and_rejected_for_a_pure_clearance_violation():
    """A shaft that is perfectly centered in its bore but simply the
    wrong SIZE for the stated clearance requirement has no positional
    fix at all -- repositioning a perfectly-centered shaft changes
    nothing. attempt_repair must recognize this isn't a concentricity
    problem and report it as unfixed rather than moving the shaft to an
    arbitrary off-center position that wouldn't help."""
    sb_h = Sandbox()
    housing_h = sb_h.add_box(60, 60, 30, center=(0, 0, 0))
    housing_h = sb_h.drill_hole(housing_h, radius=7.0, depth=30, center=(0, 0, 0), axis="z")  # 14mm bore
    housing_mesh = sb_h._get(housing_h)
    sb_s = Sandbox()
    shaft_h = sb_s.add_cylinder(radius=6.0, height=40, center=(0, 0, 0), axis="z")  # 12mm shaft, perfectly centered, 2mm clearance
    shaft_mesh = sb_s._get(shaft_h)

    rel = Relationship(kind="concentric_with", target_id="housing_01", axis="z")
    eva = evaluate_shaft_bore_fit("REL-INFEASIBLE-2", "shaft_01", "housing_01", "a 12mm shaft", "a bore, clearance 0.2mm", shaft_mesh, housing_mesh, rel)
    assert eva.status is CompatibilityStatus.INCOMPATIBLE
    assert eva.calculated["concentricity_offset_mm"] == pytest.approx(0.0, abs=0.01)  # confirms this really is NOT a positional problem
    assert attempt_repair(eva, shaft_mesh, housing_mesh) is None


def test_repair_is_infeasible_and_rejected_for_an_unsupported_relationship():
    """An UNSUPPORTED_TO_VERIFY result (a non-spur gear type here) is
    never INCOMPATIBLE in the first place -- attempt_repair's own first
    check (status must be INCOMPATIBLE) must reject it outright, since
    there is nothing concrete to repair against and no business
    fabricating one."""
    mech_type = classify_relationship_type("a helical gear, module 2.5, 20 teeth", "a helical gear, module 2.5, 32 teeth", "concentric_with")
    assert mech_type is MechanicalRelationshipType.UNSUPPORTED

    from assembly_compatibility import RelationshipEvaluation

    fabricated_eva = RelationshipEvaluation(
        relationship_id="REL-INFEASIBLE-3", component_a="gear_A", component_b="gear_B",
        relationship_type=MechanicalRelationshipType.UNSUPPORTED,
        source_phrase_a="a helical gear, module 2.5, 20 teeth", source_phrase_b="a helical gear, module 2.5, 32 teeth",
        status=CompatibilityStatus.UNSUPPORTED_TO_VERIFY, reason="non-spur gear type",
    )
    sb_a = Sandbox()
    mesh_a = sb_a._get(sb_a.add_gear_profile(module_mm=2.5, tooth_count=20, face_width_mm=10, bore_diameter_mm=5, center=(0, 0, 0)))
    sb_b = Sandbox()
    mesh_b = sb_b._get(sb_b.add_gear_profile(module_mm=2.5, tooth_count=32, face_width_mm=10, bore_diameter_mm=5, center=(60, 0, 0)))
    assert attempt_repair(fabricated_eva, mesh_a, mesh_b) is None
