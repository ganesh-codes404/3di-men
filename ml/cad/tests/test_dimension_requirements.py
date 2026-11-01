"""Direct unit tests of dimension_requirements.py's extraction logic --
fast, deterministic, no model loading. See tests/test_dimension_
regression.py for the end-to-end production-pipeline tests these feed
into."""

from dimension_requirements import RequirementStatus, extract_dimension_requirements, extract_feature_count_requirements


def _by_parameter(text: str, parameter: str) -> list:
    return [r for r in extract_dimension_requirements(text) if r.parameter == parameter]


def test_thickness_and_hole_diameter_do_not_cross_contaminate():
    """Failure A's own exact text."""
    text = "A bracket with a 5 mm thick base and two 6 mm mounting holes."
    thickness = _by_parameter(text, "thickness")
    hole = _by_parameter(text, "hole_diameter")
    assert [r.normalized_value for r in thickness] == [5.0]
    assert [r.normalized_value for r in hole] == [6.0]


def test_bore_diameter_never_leaks_into_base_height():
    """Failure C's own exact text."""
    text = "A cylindrical component with 40 mm outer diameter, 12 mm through-bore, and 30 mm height."
    assert [r.normalized_value for r in _by_parameter(text, "outer_diameter")] == [40.0]
    assert [r.normalized_value for r in _by_parameter(text, "hole_diameter")] == [12.0]
    assert [r.normalized_value for r in _by_parameter(text, "height")] == [30.0]


def test_blind_and_through_bore_type_is_attached_to_both_diameter_and_depth():
    text = "A 50 mm diameter, 20 mm tall cylinder with a 10 mm diameter blind hole, 8 mm deep."
    diameter_reqs = _by_parameter(text, "hole_diameter")
    depth_reqs = _by_parameter(text, "hole_depth")
    assert diameter_reqs[0].normalized_value == 10.0
    assert "bore_type=blind" in diameter_reqs[0].notes
    assert depth_reqs[0].normalized_value == 8.0
    assert "bore_type=blind" in depth_reqs[0].notes


def test_main_and_reduced_diameter_distinguish_a_stepped_shaft():
    text = "A shaft with a 20 mm diameter main section, 12 mm diameter stepped section, and 100 mm overall length."
    assert [r.normalized_value for r in _by_parameter(text, "main_diameter")] == [20.0]
    assert [r.normalized_value for r in _by_parameter(text, "reduced_diameter")] == [12.0]
    assert [r.normalized_value for r in _by_parameter(text, "overall_length")] == [100.0]


def test_ambiguous_dimension_with_no_nearby_keyword_is_unresolved_not_guessed():
    text = "A 20 mm bracket with a 6 mm hole."
    reqs = extract_dimension_requirements(text)
    ambiguous = next(r for r in reqs if r.value == 20.0)
    assert ambiguous.status is RequirementStatus.UNRESOLVED
    assert ambiguous.parameter is None
    unambiguous = next(r for r in reqs if r.value == 6.0)
    assert unambiguous.parameter == "hole_diameter"
    assert unambiguous.status is RequirementStatus.EXTRACTED


def test_ordered_box_dimensions_are_length_then_width_then_height():
    text = "A 70 mm by 40 mm plate with a 3 mm thickness and three 8 mm holes."
    assert [r.normalized_value for r in _by_parameter(text, "length")] == [70.0]
    assert [r.normalized_value for r in _by_parameter(text, "width")] == [40.0]
    assert [r.normalized_value for r in _by_parameter(text, "thickness")] == [3.0]


def test_unit_conversion_cm_to_mm():
    text = "A plate 10 cm long, 50 mm wide and 5 mm thick."
    assert [r.normalized_value for r in _by_parameter(text, "length")] == [100.0]
    length_req = _by_parameter(text, "length")[0]
    assert length_req.original_unit == "cm"
    assert length_req.value == 10.0


def test_edge_offset_is_distinguished_from_hole_diameter_across_a_clause_boundary():
    text = "A mounting plate, 5 mm thick, with four holes of 6 mm diameter and 20 mm edge offsets."
    assert [r.normalized_value for r in _by_parameter(text, "hole_diameter")] == [6.0]
    assert [r.normalized_value for r in _by_parameter(text, "edge_offset")] == [20.0]


def test_feature_count_requirement_carries_its_own_diameter_and_pattern_hint():
    text = "A rectangular mounting plate measuring 120 mm by 80 mm and 5 mm thick, with four 6 mm holes, one near each corner."
    counts = extract_feature_count_requirements(text)
    assert len(counts) == 1
    assert counts[0].count == 4
    assert counts[0].diameter_mm == 6.0
    assert counts[0].pattern_hint == "corner_array"


def test_feature_count_pattern_hint_detects_linear_layout():
    text = "A 100 mm long plate with five 4 mm diameter holes equally spaced along its centreline."
    counts = extract_feature_count_requirements(text)
    assert counts[0].count == 5
    assert counts[0].pattern_hint == "linear_array"


def test_no_hole_mention_produces_no_count_requirement():
    assert extract_feature_count_requirements("A plate 10 cm long, 50 mm wide and 5 mm thick.") == []


def test_no_dimension_or_feature_text_extracts_nothing():
    assert extract_dimension_requirements("make something nice") == []
    assert extract_feature_count_requirements("make something nice") == []
