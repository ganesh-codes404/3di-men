"""Phase 8 Part A -- the 10 required coordinate-frame regression tests
for the E50 fix. E50's confirmed root cause (Phase 7's own lock-in test,
test_shared_unit_dimension_parsing.test_e50_root_cause_is_axis_
measurement_not_dimension_parsing): assembly_compatibility.py measured a
component's own diameter/alignment axis from the RELATIONSHIP's declared
`axis` field instead of the component's own actual built geometry.
parallel_builder.resolve_rotation_deg only ever rotates a relationship's
OWNER to match its declared axis -- the TARGET is never rotated on the
owner's behalf and keeps whatever orientation it actually has. Every
test here builds REAL geometry via sandbox.Sandbox (never hand-written
spec numbers) and inspects the actual evaluation result, matching this
project's own standing "must inspect actual generated geometry" rule.

The fix (assembly_compatibility._infer_rotational_axis / _evaluate_
cylinder_in_bore / evaluate_axis_alignment) does NOT special-case X/Z,
swap dimensions, or touch a tolerance -- it always independently
measures each mesh's own built axis, using the relationship's declared
axis ONLY as a tie-break hint for a shape whose own extents are
genuinely ambiguous (never to override a real measurement)."""

import trimesh

from assembly_compatibility import (
    CompatibilityStatus, _infer_rotational_axis, evaluate_axis_alignment, evaluate_shaft_bore_fit,
)
from assembly_plan import Relationship
from sandbox import Sandbox


def _shaft(radius: float, height: float, axis: str = "z", rotate: "tuple[str, float] | None" = None) -> trimesh.Trimesh:
    sb = Sandbox()
    h = sb.add_cylinder(radius=radius, height=height, center=(0, 0, 0), axis=axis)
    if rotate is not None:
        rot_axis, angle = rotate
        h = sb.rotate(h, axis=rot_axis, angle_deg=angle)
    return sb._get(h)


def _housing_with_bore(
    outer: float, bore_radius: float, depth: float, axis: str = "z", rotate: "tuple[str, float] | None" = None,
) -> trimesh.Trimesh:
    sb = Sandbox()
    h = sb.add_box(outer, outer, depth, center=(0, 0, 0))
    h = sb.drill_hole(h, radius=bore_radius, depth=depth, center=(0, 0, 0), axis=axis)
    if rotate is not None:
        rot_axis, angle = rotate
        h = sb.rotate(h, axis=rot_axis, angle_deg=angle)
    return sb._get(h)


# 1. unrotated -- shaft and housing both built along their default axis (z), declared axis matches.
def test_1_unrotated_compatible_and_diameter_correct():
    shaft = _shaft(radius=6.0, height=40)
    housing = _housing_with_bore(outer=60, bore_radius=6.1, depth=30)
    rel = Relationship(kind="concentric_with", target_id="housing_01", axis="z")
    eva = evaluate_shaft_bore_fit("REL-1", "shaft_01", "housing_01", "a 12mm shaft", "a bearing housing", shaft, housing, rel)
    assert eva.status is CompatibilityStatus.COMPATIBLE
    assert eva.actual["shaft_diameter_mm"] == 12.0
    assert eva.requested["shaft_axis"] == "z"


# 2. X-rotated -- both shaft and housing genuinely rotated so their shared axis is world x.
def test_2_both_rotated_to_x_compatible_and_diameter_correct():
    shaft = _shaft(radius=6.0, height=40, rotate=("y", 90.0))
    housing = _housing_with_bore(outer=60, bore_radius=6.1, depth=30, rotate=("y", 90.0))
    rel = Relationship(kind="concentric_with", target_id="housing_01", axis="x")
    eva = evaluate_shaft_bore_fit("REL-2", "shaft_01", "housing_01", "a 12mm shaft", "a bearing housing", shaft, housing, rel)
    assert eva.status is CompatibilityStatus.COMPATIBLE
    assert eva.actual["shaft_diameter_mm"] == 12.0
    assert eva.requested["shaft_axis"] == "x"


# 3. Y-rotated -- same, along world y.
def test_3_both_rotated_to_y_compatible_and_diameter_correct():
    shaft = _shaft(radius=6.0, height=40, rotate=("x", 90.0))
    housing = _housing_with_bore(outer=60, bore_radius=6.1, depth=30, rotate=("x", 90.0))
    rel = Relationship(kind="concentric_with", target_id="housing_01", axis="y")
    eva = evaluate_shaft_bore_fit("REL-3", "shaft_01", "housing_01", "a 12mm shaft", "a bearing housing", shaft, housing, rel)
    assert eva.status is CompatibilityStatus.COMPATIBLE
    assert eva.actual["shaft_diameter_mm"] == 12.0
    assert eva.requested["shaft_axis"] == "y"


# 4. Z-rotated (explicit, non-trivial transform that still nets out to z) -- a 180-degree
# flip about x leaves the cylinder still running along z, just end-for-end -- confirms the
# fix measures the mesh's ACTUAL final orientation, not merely "did any rotation happen."
def test_4_explicit_z_rotation_still_resolves_correctly():
    shaft = _shaft(radius=6.0, height=40, rotate=("x", 180.0))
    housing = _housing_with_bore(outer=60, bore_radius=6.1, depth=30, rotate=("x", 180.0))
    rel = Relationship(kind="concentric_with", target_id="housing_01", axis="z")
    eva = evaluate_shaft_bore_fit("REL-4", "shaft_01", "housing_01", "a 12mm shaft", "a bearing housing", shaft, housing, rel)
    assert eva.status is CompatibilityStatus.COMPATIBLE
    assert eva.actual["shaft_diameter_mm"] == 12.0
    assert eva.requested["shaft_axis"] == "z"


# 5. arbitrary-axis -- a composed, non-trivial ROTATION SEQUENCE (two sequential rotations,
# not a single simple turn) that nonetheless nets out to a clean world axis (x) -- confirms
# the fix is correct for any transform path reaching that final orientation, not just a
# single-step rotation.
def test_5_arbitrary_rotation_sequence_resolving_to_x_compatible():
    sb_s = Sandbox()
    h = sb_s.add_cylinder(radius=6.0, height=40, center=(0, 0, 0), axis="z")
    h = sb_s.rotate(h, axis="y", angle_deg=45.0)
    h = sb_s.rotate(h, axis="y", angle_deg=45.0)  # two 45s compose to the same 90 as test 2
    shaft = sb_s._get(h)

    sb_h = Sandbox()
    hh = sb_h.add_box(60, 60, 30, center=(0, 0, 0))
    hh = sb_h.drill_hole(hh, radius=6.1, depth=30, center=(0, 0, 0), axis="z")
    hh = sb_h.rotate(hh, axis="y", angle_deg=90.0)
    housing = sb_h._get(hh)

    rel = Relationship(kind="concentric_with", target_id="housing_01", axis="x")
    eva = evaluate_shaft_bore_fit("REL-5", "shaft_01", "housing_01", "a 12mm shaft", "a bearing housing", shaft, housing, rel)
    assert eva.status is CompatibilityStatus.COMPATIBLE
    assert eva.actual["shaft_diameter_mm"] == 12.0


# 6. owner-rotated-target-not -- the CONFIRMED E50 bug scenario itself: the housing (the
# relationship's OWNER) is genuinely rotated to run along x; the shaft (the TARGET) is
# never rotated on the owner's behalf and stays built along its own true z. These two
# components are NOT actually coaxial -- the fix must report this honestly as INCOMPATIBLE
# (a real geometric mismatch), and must measure the shaft's own diameter correctly (12mm)
# rather than the old bug's wrong 55mm-style average of length and diameter.
def test_6_owner_rotated_target_not_is_correctly_incompatible():
    shaft = _shaft(radius=6.0, height=100.0)  # TARGET: never rotated, stays along true z
    housing = _housing_with_bore(outer=60, bore_radius=6.1, depth=30, rotate=("y", 90.0))  # OWNER: rotated to x
    rel = Relationship(kind="concentric_with", target_id="shaft_01", axis="x")
    eva = evaluate_shaft_bore_fit("REL-6", "housing_01", "shaft_01", "a bearing housing", "a shaft", housing, shaft, rel)
    assert eva.status is CompatibilityStatus.INCOMPATIBLE
    assert eva.actual["shaft_diameter_mm"] == 12.0  # correctly measured along the shaft's TRUE axis (z), never 55mm
    assert eva.requested["shaft_axis"] == "z"
    assert eva.requested["shaft_declared_axis"] == "x"
    assert "not aligned" in eva.reason


# 7. both-rotated -- owner AND target independently rotated (by different transform
# PATHS) but landing on the SAME true axis -- must be COMPATIBLE, confirming the fix
# doesn't assume "rotated at all" implies misalignment, only genuinely different axes do.
def test_7_both_independently_rotated_to_same_axis_compatible():
    shaft = _shaft(radius=6.0, height=40, rotate=("y", -90.0))  # rotated the OTHER direction...
    housing = _housing_with_bore(outer=60, bore_radius=6.1, depth=30, rotate=("y", 90.0))  # ...from the owner
    rel = Relationship(kind="concentric_with", target_id="housing_01", axis="x")
    eva = evaluate_shaft_bore_fit("REL-7", "shaft_01", "housing_01", "a 12mm shaft", "a bearing housing", shaft, housing, rel)
    assert eva.status is CompatibilityStatus.COMPATIBLE
    assert eva.actual["shaft_diameter_mm"] == 12.0


# 8. non-cylindrical -- two boxes routed through the generic axis-alignment path (no
# shaft/bore/gear keyword). Neither component has a determinable rotational axis at all
# (fully anisotropic boxes still pick SOME candidate per-axis, but genuinely ambiguous
# shapes must not crash) -- must not raise, and must produce SOME typed status.
def test_8_non_cylindrical_components_do_not_crash():
    sb_a = Sandbox()
    mesh_a = sb_a._get(sb_a.add_box(40, 40, 40, center=(0, 0, 0)))  # a perfect cube: fully axis-ambiguous
    sb_b = Sandbox()
    mesh_b = sb_b._get(sb_b.add_box(40, 40, 40, center=(0, 0, 0)))
    rel = Relationship(kind="concentric_with", target_id="block_01", axis="y")
    eva = evaluate_axis_alignment("REL-8", "block_a", "block_01", "a block", "another block", mesh_a, mesh_b, rel)
    assert eva.status in (CompatibilityStatus.COMPATIBLE, CompatibilityStatus.INCOMPATIBLE)
    assert eva.requested["axis"] == "y"  # the declared axis is honored as the hint for a genuinely ambiguous cube


# 9. missing-transform -- relationship is None entirely (no declared axis to hint with at
# all) -- must fall back purely to measurement/default, never crash on a missing hint.
def test_9_missing_relationship_falls_back_safely():
    shaft = _shaft(radius=6.0, height=40)
    housing = _housing_with_bore(outer=60, bore_radius=6.1, depth=30)
    eva = evaluate_shaft_bore_fit("REL-9", "shaft_01", "housing_01", "a 12mm shaft", "a bearing housing", shaft, housing, None)
    assert eva.status is CompatibilityStatus.COMPATIBLE
    assert eva.actual["shaft_diameter_mm"] == 12.0
    assert "shaft_declared_axis" not in eva.requested


# 10. inconsistent-transform -- the generic axis-alignment path's own version of test 6:
# relationship declares one axis, but the two components' own measured axes genuinely
# disagree -- must be reported directly as "not actually coaxial," never silently resolved
# by picking one side's axis and comparing incomparable perpendicular coordinates.
def test_10_inconsistent_declared_axis_reported_incompatible():
    sb_a = Sandbox()
    mesh_a = sb_a._get(sb_a.add_cylinder(radius=5.0, height=40, center=(0, 0, 0), axis="z"))  # true axis: z
    sb_b = Sandbox()
    h = sb_b.add_cylinder(radius=10.0, height=35, center=(0, 0, 0), axis="z")
    h = sb_b.rotate(h, axis="y", angle_deg=90.0)  # true axis: x
    mesh_b = sb_b._get(h)
    rel = Relationship(kind="concentric_with", target_id="tube_01", axis="z")  # declared axis matches NEITHER side's true x, contradicts b
    eva = evaluate_axis_alignment("REL-10", "rod_01", "tube_01", "a rod", "a tube", mesh_a, mesh_b, rel)
    assert eva.status is CompatibilityStatus.INCOMPATIBLE
    assert "not actually coaxial" in eva.reason


# -- direct unit coverage of the ambiguity-hint fix in _infer_rotational_axis itself --

def test_11_infer_rotational_axis_prefers_hint_only_when_genuinely_ambiguous():
    # Unambiguous: one axis clearly anisotropic -- hint is ignored even when it disagrees.
    unambiguous = trimesh.creation.cylinder(radius=5.0, height=40.0, sections=32)
    assert _infer_rotational_axis(unambiguous, hint_axis="x") == "z"

    # Genuinely ambiguous: height equals diameter, extents are (20, 20, 20) -- indistinguishable
    # from a cube by bounding box alone. The hint resolves the tie; no hint falls back to "z".
    ambiguous = trimesh.creation.cylinder(radius=10.0, height=20.0, sections=32)
    assert _infer_rotational_axis(ambiguous, hint_axis="x") == "x"
    assert _infer_rotational_axis(ambiguous, hint_axis=None) == "z"
