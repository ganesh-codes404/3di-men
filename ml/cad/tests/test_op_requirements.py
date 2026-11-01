"""Direct tests of op_requirements.extract_requirements() -- the shared
structured-requirement layer op_arbitration.py checks both the
classifier's and RAG's op-sequence proposals against. Deliberately no
model training/loading here (that's tests/test_op_arbitration.py and
the predict_program_hybrid tests in tests/test_program_model.py) --
this file is pure text-in, Requirement-list-out, so it stays fast."""

from op_requirements import bore_type, extract_requirements


def _types(text: str) -> set[str]:
    return {r.feature_type for r in extract_requirements(text)}


def _base_geometry(text: str) -> "str | None":
    for r in extract_requirements(text):
        if r.feature_type == "base_geometry":
            return r.geometry_type
    return None


def test_cylinder_with_bore_extracts_cylindrical_base_and_subtractive_hole():
    text = "a cylinder, 40mm diameter, 30mm tall, with a 12mm diameter bore through it"
    reqs = extract_requirements(text)
    assert _base_geometry(text) == "cylindrical"
    hole = next(r for r in reqs if r.feature_type == "subtractive_hole")
    assert "drill_hole" in hole.compatible_ops
    assert "add_boss_with_bore" in hole.compatible_ops
    assert bore_type(text) == "through"


def test_plate_with_four_holes_extracts_rectangular_base_and_hole_count():
    text = "a rectangular plate 100mm x 80mm x 6mm with four 8mm diameter through-holes near the corners"
    assert _base_geometry(text) == "rectangular"
    hole = next(r for r in extract_requirements(text) if r.feature_type == "subtractive_hole")
    assert hole.count == 4


def test_bracket_with_two_holes_extracts_rectangular_base_and_hole_count():
    text = "a bracket 80mm x 50mm x 5mm with two 6mm diameter mounting holes"
    assert _base_geometry(text) == "rectangular"
    hole = next(r for r in extract_requirements(text) if r.feature_type == "subtractive_hole")
    assert hole.count == 2


def test_stepped_shaft_extracts_cylindrical_base_and_stepped_section():
    text = "a stepped shaft with a 20mm main diameter, 15mm reduced section and 100mm overall length"
    assert _base_geometry(text) == "cylindrical"
    assert "stepped_section" in _types(text)


def test_cylindrical_housing_with_blind_bore_extracts_blind_bore_type():
    text = "a cylindrical housing with a 40mm outside diameter, 12mm blind bore, 25mm depth and 50mm total height"
    assert _base_geometry(text) == "cylindrical"
    assert "subtractive_hole" in _types(text)
    assert bore_type(text) == "blind"


def test_hollow_rectangular_enclosure_extracts_hollow_shell():
    text = "a rectangular enclosure with a hollow interior and 3mm walls"
    assert _base_geometry(text) == "rectangular"
    assert "hollow_shell" in _types(text)


def test_bare_cylindrical_housing_extracts_only_base_geometry_no_invented_features():
    text = "a cylindrical housing"
    reqs = extract_requirements(text)
    assert len(reqs) == 1
    assert reqs[0].feature_type == "base_geometry"
    assert reqs[0].geometry_type == "cylindrical"


def test_diameter_plus_height_phrasing_implies_cylindrical_without_a_shape_keyword():
    text = "a part with a 40mm diameter and 30mm tall profile"
    assert _base_geometry(text) == "cylindrical"


def test_three_dimension_phrasing_implies_rectangular_without_a_shape_keyword():
    text = "a part measuring 40mm x 30mm x 10mm"
    assert _base_geometry(text) == "rectangular"


def test_explicit_shape_keyword_wins_over_numeric_fallback():
    # "cylinder" is explicit; even though the text also has an NxNxN-shaped
    # number string buried in it, the explicit keyword must win -- same
    # "explicit beats inferred" precedence used elsewhere in this project.
    text = "a cylinder sized roughly like a 40mm x 40mm x 40mm block, 40mm diameter, 30mm tall"
    assert _base_geometry(text) == "cylindrical"


def test_gear_and_sphere_keywords_map_to_their_own_geometry_kind():
    assert _base_geometry("a gear with 24 teeth and a 5mm module") == "gear"
    assert _base_geometry("a sphere, 30mm diameter") == "spherical"


def test_no_recognizable_geometry_or_feature_text_extracts_nothing():
    assert extract_requirements("make something nice") == []


def test_rib_and_knurl_and_edge_treatment_keywords_map_to_their_feature_types():
    assert "rib" in _types("a bracket with three ribs for stiffness")
    assert "knurled_surface" in _types("a cylindrical knob with a knurled grip surface")
    assert "edge_treatment" in _types("a plate with chamfered edges")


def test_slot_and_pocket_keywords_map_to_their_feature_types():
    assert "subtractive_slot" in _types("a plate with a slot cut through the middle")
    assert "subtractive_pocket" in _types("a block with a shallow pocket milled into the top")
