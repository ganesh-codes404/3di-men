"""Tests for spec_validation.py -- the correctness-to-spec check
ACCURACY_AUDIT.md section 3 identified as missing: nothing in this
pipeline previously checked whether a BUILT mesh's actual dimensions
match what its own description text claimed, only whether it was a
structurally valid mesh."""

from sandbox import Sandbox
from spec_validation import (
    DimensionValidationReport,
    HoleCountValidationReport,
    HoleDiameterValidationReport,
    extract_claimed_dimensions_mm,
    extract_claimed_envelope_mm,
    extract_claimed_hole_count,
    extract_claimed_hole_diameters_mm,
    detect_semantic_size_conflicts,
    validate_dimensions_against_description,
    validate_hole_count_against_description,
    validate_hole_diameters_against_description,
)


# ---------------- extract_claimed_envelope_mm -- confirmed real gap this closes
# (mission-brief regression: "final bounding box 251.2x124.0x108.6mm, versus an
# approximate requested footprint of 150x100mm" -- there was previously no way
# to even recover that 150x100mm claim from the original prompt text at all). ----

def test_extract_claimed_envelope_mm_recognizes_close_to_phrasing():
    claim = extract_claimed_envelope_mm("a motor mount with the overall footprint close to 150mm x 100mm")
    assert claim == (150.0, 100.0)


def test_extract_claimed_envelope_mm_recognizes_of_roughly_phrasing():
    claim = extract_claimed_envelope_mm("keep a footprint of roughly 200 x 80mm")
    assert claim == (200.0, 80.0)


def test_extract_claimed_envelope_mm_recognizes_bare_footprint_phrasing():
    claim = extract_claimed_envelope_mm("a bracket assembly, footprint 60mm x 50mm")
    assert claim == (60.0, 50.0)


def test_extract_claimed_envelope_mm_returns_none_when_prompt_makes_no_claim():
    """A per-part dimension claim (extract_claimed_dimensions_mm's own
    concern) is NOT an assembly-level footprint claim -- this must not
    fire just because the prompt happens to contain "WxH mm"-shaped
    numbers for an individual part."""
    assert extract_claimed_envelope_mm("a box 60mm x 50mm x 10mm with a gear next to it") is None
    assert extract_claimed_envelope_mm("a water bottle") is None


def test_extract_claimed_envelope_mm_only_uses_the_first_match():
    """A footprint claim is a property of the whole assembly, not a
    per-component list (see the function's own docstring) -- a second
    "WxH" mention later in the prompt must not be treated as a second,
    additive claim."""
    claim = extract_claimed_envelope_mm(
        "keep the overall footprint close to 150mm x 100mm, with a bracket footprint of 40mm x 30mm on top"
    )
    assert claim == (150.0, 100.0)


def test_extract_claimed_dimensions_mm_finds_every_number():
    claims = extract_claimed_dimensions_mm("a cylinder 30mm diameter, 80mm tall")
    assert claims == [30.0, 80.0]


def test_extract_claimed_dimensions_mm_handles_decimals():
    claims = extract_claimed_dimensions_mm("a gear profile, module 2.5mm, 14 teeth")
    assert claims == [2.5]


def test_extract_claimed_dimensions_mm_returns_empty_for_no_numbers():
    assert extract_claimed_dimensions_mm("an L-shape bracket") == []


def test_extract_claimed_dimensions_mm_excludes_edge_and_corner_offsets():
    """Regression test for the mission brief's own exact reproduction:
    "a slot positioned 10mm from the left edge" must NOT produce a
    [10.0] size claim -- that 10mm is a placement offset, not a built
    dimension. A real size claim earlier in the same sentence must still
    be caught."""
    assert extract_claimed_dimensions_mm("a slot positioned 10mm from the left edge of the plate") == []
    assert extract_claimed_dimensions_mm("a corner array of four holes, each 15mm from a corner") == []
    assert extract_claimed_dimensions_mm("centered at 12mm from adjacent edges") == []
    assert extract_claimed_dimensions_mm(
        "a plate 100mm x 60mm x 8mm with a corner array of four holes, each 20mm from a corner"
    ) == [100.0, 60.0, 8.0]


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


def test_validate_dimensions_excludes_an_edge_offset_from_an_overall_extent_check():
    """Regression test for a confirmed real bug: "20mm from a corner" is a
    PLACEMENT offset (how far a hole sits from the plate's own edge), not
    a built SIZE claim -- it used to get checked against the mesh's
    overall extents exactly like a stated dimension, producing a false
    "75% matched, unmatched: 20mm"-style warning on a perfectly correct
    build. extract_claimed_dimensions_mm must exclude it entirely (not
    include it as a failed/unmatched claim), while still catching the
    real 100/60/8mm size claims in the same sentence."""
    sb = Sandbox()
    h = sb.add_box(100, 60, 8)
    h2 = sb.drill_hole_pattern(h, pattern="corner_array", count=4, radius=4, depth=8, axis="z", margin=20.0, circle_diameter=0, z=0)
    mesh = sb._get(h2)
    report = validate_dimensions_against_description(
        "a plate 100mm x 60mm x 8mm with a corner array of four holes, each 20mm from a corner", mesh
    )
    claimed_values = {c.claimed_mm for c in report.claims}
    assert {100.0, 60.0, 8.0}.issubset(claimed_values)
    assert 20.0 not in claimed_values  # the edge-offset value never becomes a claim at all
    assert report.all_matched is True  # with the margin excluded, every REAL claim here matches
    assert report.match_rate == 1.0


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


# Feature validation (hole/bore DIAMETER) -- wires GeometryReport.hole_diameters_mm
# (evaluation/geometry_eval.py), already measurable from each detected hole's own
# fitted-circle radius, into an actual check -- more precise than the general
# dimension check above, which can only compare a claim against the mesh's overall
# extents, not a specific feature.

def test_extract_claimed_hole_diameters_mm_recognizes_hole_and_bore():
    assert extract_claimed_hole_diameters_mm("a cylinder 30mm diameter, 20mm tall with a 12mm diameter hole through it") == [12.0]
    assert extract_claimed_hole_diameters_mm("a cylinder 30mm diameter, 20mm tall with a 12mm diameter bore through it") == [12.0]
    assert extract_claimed_hole_diameters_mm("a corner array of four 6.6mm diameter holes, each 15mm from a corner") == [6.6]


def test_extract_claimed_hole_diameters_mm_returns_empty_when_none_mentioned():
    assert extract_claimed_hole_diameters_mm("a box 50mm x 40mm x 10mm") == []


def test_validate_hole_diameters_matches_a_correctly_built_bore():
    report = validate_hole_diameters_against_description(
        "a cylinder 30mm diameter, 20mm tall with a 12mm diameter bore through it",
        {"hole_diameters_mm": [11.95]},
    )
    assert isinstance(report, HoleDiameterValidationReport)
    assert report.all_matched is True
    assert report.unmatched_claims_mm == []


def test_validate_hole_diameters_flags_a_real_mismatch():
    """The actual confirmed mission-brief case: a claimed 12mm bore that
    doesn't match what was actually built (e.g. a radius/diameter mix-up
    building a 24mm-diameter hole instead)."""
    report = validate_hole_diameters_against_description(
        "a cylinder 30mm diameter, 20mm tall with a 12mm diameter bore through it",
        {"hole_diameters_mm": [24.0]},
    )
    assert report.all_matched is False
    assert report.unmatched_claims_mm == [12.0]
    assert report.detector_limitation  # the honesty disclosure is never empty


def test_validate_hole_diameters_is_vacuously_true_with_nothing_to_check():
    # no diameter claim in the text at all
    assert validate_hole_diameters_against_description("a box with four holes", {"hole_diameters_mm": [5.0]}).all_matched is True
    # no geometry_report available
    assert validate_hole_diameters_against_description("a 12mm diameter hole through it", None).all_matched is False


# Semantic-name-vs-explicit-dimension conflict detection -- the project's
# own non-negotiable principle: explicit dimensions override a semantic
# default implied by a name like "NEMA 17". Scoped to NEMA stepper-motor
# mounting faces specifically (real, standard, documented sizes), not a
# general lookup for informal names.

def test_detect_semantic_size_conflicts_flags_a_real_disagreement():
    """The exact mission-brief scenario: a NEMA 17 (standard 42.3mm
    square face) named alongside an explicit, disagreeing 50mm x 50mm
    claim."""
    conflicts = detect_semantic_size_conflicts("a NEMA 17 stepper motor mount, 50mm x 50mm x 10mm thick")
    assert len(conflicts) == 1
    c = conflicts[0]
    assert c.semantic_name == "NEMA 17"
    assert c.standard_value_mm == 42.3
    assert c.explicit_value_mm == 50.0
    assert c.resolved_value_mm == 50.0  # explicit always wins


def test_detect_semantic_size_conflicts_silent_when_explicit_matches_the_standard():
    assert detect_semantic_size_conflicts("a NEMA 17 stepper motor mount, 42.3mm x 42.3mm x 10mm thick") == []


def test_detect_semantic_size_conflicts_ignores_unrelated_single_dimensions():
    """Confirmed real false-positive this guards against: an unrelated
    dimension elsewhere in the same sentence (here, a 10mm thickness)
    must never be compared against the NEMA standard's own face size --
    only a REPEATED value (the actual signal of a square-face claim,
    since a NEMA mounting face is always square) is considered."""
    assert detect_semantic_size_conflicts("a NEMA 17 stepper motor, 48mm long") == []


def test_detect_semantic_size_conflicts_returns_empty_without_a_nema_name():
    assert detect_semantic_size_conflicts("a box 50mm x 50mm x 10mm") == []


def test_detect_semantic_size_conflicts_returns_empty_with_no_explicit_dimensions_at_all():
    assert detect_semantic_size_conflicts("a NEMA 17 stepper motor mount") == []
