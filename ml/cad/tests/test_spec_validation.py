"""Tests for spec_validation.py -- the correctness-to-spec check
ACCURACY_AUDIT.md section 3 identified as missing: nothing in this
pipeline previously checked whether a BUILT mesh's actual dimensions
match what its own description text claimed, only whether it was a
structurally valid mesh."""

from sandbox import Sandbox
from spec_validation import (
    DimensionValidationReport,
    HoleCountValidationReport,
    extract_claimed_dimensions_mm,
    extract_claimed_hole_count,
    validate_dimensions_against_description,
    validate_hole_count_against_description,
)


def test_extract_claimed_dimensions_mm_finds_every_number():
    claims = extract_claimed_dimensions_mm("a cylinder 30mm diameter, 80mm tall")
    assert claims == [30.0, 80.0]


def test_extract_claimed_dimensions_mm_handles_decimals():
    claims = extract_claimed_dimensions_mm("a gear profile, module 2.5mm, 14 teeth")
    assert claims == [2.5]


def test_extract_claimed_dimensions_mm_returns_empty_for_no_numbers():
    assert extract_claimed_dimensions_mm("an L-shape bracket") == []


def test_validate_dimensions_matches_a_correctly_built_cylinder():
    sb = Sandbox()
    h = sb.add_cylinder(radius=15, height=80)
    mesh = sb._get(h)
    report = validate_dimensions_against_description("a cylinder 30mm diameter, 80mm tall", mesh)
    assert isinstance(report, DimensionValidationReport)
    assert report.all_matched is True
    assert report.match_rate == 1.0


def test_validate_dimensions_flags_a_wrongly_built_cylinder():
    sb = Sandbox()
    h = sb.add_cylinder(radius=40, height=200)  # built 80mm diameter, 200mm tall -- NOT what's claimed
    mesh = sb._get(h)
    report = validate_dimensions_against_description("a cylinder 30mm diameter, 80mm tall", mesh)
    assert report.all_matched is False
    claim_30 = next(c for c in report.claims if c.claimed_mm == 30.0)
    assert claim_30.matched is False
    assert claim_30.deviation_fraction > 0.15


def test_validate_dimensions_reports_an_empty_report_for_no_claims():
    sb = Sandbox()
    h = sb.add_box(50, 50, 50)
    mesh = sb._get(h)
    report = validate_dimensions_against_description("an L-shape bracket", mesh)
    assert report.claims == []
    assert report.all_matched is True  # vacuously true -- nothing to check, not a confirmed match
    assert report.match_rate == 1.0


def test_validate_dimensions_honestly_scoped_to_overall_extents_not_feature_details():
    """A hole-pattern margin number is NOT an overall extent -- this is a
    disclosed limitation of this scoped first version (see
    spec_validation.py's own module docstring), not a bug: the check
    should honestly report it as unmatched rather than silently ignoring
    it or false-matching it against an unrelated extent."""
    sb = Sandbox()
    h = sb.add_box(100, 60, 8)
    h2 = sb.drill_hole_pattern(h, pattern="corner_array", count=4, radius=4, depth=8, axis="z", margin=20.0, circle_diameter=0, z=0)
    mesh = sb._get(h2)
    report = validate_dimensions_against_description(
        "a plate 100mm x 60mm x 8mm with a corner array of four holes, each 20mm from a corner", mesh
    )
    matched_values = {c.claimed_mm for c in report.claims if c.matched}
    assert {100.0, 60.0, 8.0}.issubset(matched_values)
    assert 20.0 in report.unmatched_claims_mm
    assert 0.0 < report.match_rate < 1.0


def test_validate_dimensions_never_raises_on_degenerate_mesh():
    import trimesh

    empty_mesh = trimesh.Trimesh(vertices=[], faces=[])
    report = validate_dimensions_against_description("a box 50mm x 40mm x 10mm", empty_mesh)
    assert report.claims  # still produces claims, just all unmatched
    assert report.all_matched is False


# Feature validation (hole count) -- wires GeometryReport.detected_hole_count
# (evaluation/geometry_eval.py), already computed by every build but
# previously discarded by every caller, into an actual check.

def test_extract_claimed_hole_count_handles_word_and_digit_counts():
    assert extract_claimed_hole_count("a cylinder 20mm diameter, 15mm tall with a 6mm diameter hole through it") == 1
    assert extract_claimed_hole_count("a box with a corner array of four holes, each 8mm diameter") == 4
    assert extract_claimed_hole_count("a plate with 6 holes") == 6


def test_extract_claimed_hole_count_sums_multiple_hole_groups():
    text = "a flange with a corner array of four holes, each 5mm diameter and a circular array of three holes, each 3mm diameter"
    assert extract_claimed_hole_count(text) == 7


def test_extract_claimed_hole_count_returns_none_when_no_holes_mentioned():
    assert extract_claimed_hole_count("a box 50mm x 40mm x 10mm") is None
    assert extract_claimed_hole_count("a plain box with no holes at all") is None  # "no" isn't a valid count word


def test_validate_hole_count_matches_when_counts_agree():
    report = validate_hole_count_against_description(
        "a box with a corner array of four holes, each 8mm diameter", {"detected_hole_count": 4},
    )
    assert isinstance(report, HoleCountValidationReport)
    assert report.matched is True
    assert report.claimed_count == 4
    assert report.detected_count == 4


def test_validate_hole_count_flags_a_real_mismatch():
    report = validate_hole_count_against_description(
        "a box with a corner array of four holes, each 8mm diameter", {"detected_hole_count": 3},
    )
    assert report.matched is False
    assert report.claimed_count == 4
    assert report.detected_count == 3
    assert report.detector_limitation  # the honesty disclosure is never empty


def test_validate_hole_count_is_vacuously_true_with_nothing_to_check():
    # no hole claim in the text at all
    assert validate_hole_count_against_description("a box 50mm x 40mm x 10mm", {"detected_hole_count": 2}).matched is True
    # no geometry_report available (e.g. the build produced no mesh)
    assert validate_hole_count_against_description("a box with four holes", None).matched is True
