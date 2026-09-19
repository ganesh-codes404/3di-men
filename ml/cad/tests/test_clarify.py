from nlp.clarify import build_plan

_GEARBOX_PROMPT = """Design a complete industrial gearbox mounting assembly consisting of a rectangular base plate, two vertical support brackets, a cylindrical horizontal shaft, and a protective top housing.

The base plate should be 300 mm long, 220 mm wide, and 15 mm thick. Add four mounting holes, each 14 mm in diameter, positioned symmetrically 25 mm from the respective edges. Apply a 5 mm fillet to the outer edges of the base plate.

Construct two vertical support brackets, each 100 mm tall, 80 mm wide, and 12 mm thick. Each bracket should have a circular shaft hole with a diameter of 40 mm.

Create a horizontal cylindrical shaft with a diameter of 35 mm and a length of 200 mm.

Add a protective rectangular top housing measuring 240 mm long, 180 mm wide, and 100 mm tall.
"""


def test_fully_specified_single_part_prompt_needs_no_clarification():
    plan = build_plan("circular steel flange, 120mm outer diameter, 40mm bore, 8mm thick, four mounting holes 8mm in diameter, 90mm bolt circle")
    assert len(plan.components) == 1
    assert plan.components[0].family == "flange"
    assert not plan.needs_clarification, [q.field for q in plan.questions]


def test_vague_prompt_asks_about_every_defaulted_field():
    plan = build_plan("a small aluminum bracket")
    assert plan.needs_clarification
    fields = {q.field for q in plan.questions if q.component_id == "component_0"}
    # material was parsed ("aluminum"); everything else was defaulted and must be asked about
    assert "material" not in fields
    assert {"leg1_length_mm", "leg2_length_mm", "thickness_mm", "length_mm", "hole_diameter_mm", "hole_edge_margin_mm", "holes_per_arm"} <= fields


def test_gearbox_prompt_decomposes_into_four_real_components():
    plan = build_plan(_GEARBOX_PROMPT)
    assert [c.family for c in plan.components] == ["plate", "bracket", "shaft", "flange"]
    assert [c.count for c in plan.components] == [1, 2, 1, 1]


def test_gearbox_plate_component_is_not_asked_about_fields_it_already_specified():
    """Regression test for a real bug: _parse_plate's postpositive-fallback
    branch used to mark length/width/thickness as "defaulted" even when it
    successfully parsed them from text like "300 mm long, 220 mm wide, and
    15 mm thick" -- which would have made this feature ask about fields the
    user had already answered."""
    plan = build_plan(_GEARBOX_PROMPT)
    plate = plan.components[0]
    assert {"length_mm", "width_mm", "thickness_mm", "hole_diameter_mm"} <= set(plate.parsed_fields)
    plate_question_fields = {q.field for q in plan.questions if q.component_id == plate.id}
    assert plate_question_fields.isdisjoint({"length_mm", "width_mm", "thickness_mm", "hole_diameter_mm"})


def test_ambiguous_housing_gets_a_family_choice_question_not_a_silent_default():
    plan = build_plan(_GEARBOX_PROMPT)
    housing = plan.components[3]
    assert housing.family == "flange"  # today's silent default, per nlp/prompt_parser.py's documented behavior
    assert housing.family_options == ["flange", "enclosure"]
    family_questions = [q for q in plan.questions if q.component_id == housing.id and q.kind == "family_choice"]
    assert len(family_questions) == 1
    assert family_questions[0].choices == ["flange", "enclosure"]
    # the ambiguous component's ordinary field questions must NOT also be
    # asked in the same pass -- they'd be for the wrong family until the
    # family question is answered
    assert not any(q.component_id == housing.id and q.kind != "family_choice" for q in plan.questions)


def test_answering_the_housing_family_choice_reveals_correct_enclosure_dimensions():
    plan = build_plan(_GEARBOX_PROMPT, family_overrides={"component_3": "enclosure"})
    housing = plan.components[3]
    assert housing.family == "enclosure"
    assert housing.params["length_mm"] == 240.0
    assert housing.params["width_mm"] == 180.0
    assert housing.params["height_mm"] == 100.0
    assert {"length_mm", "width_mm", "height_mm"} <= set(housing.parsed_fields)
    assert housing.family_options is None  # no override needed a second time


def test_component_ids_are_positional_and_stable_across_overrides():
    plan_a = build_plan(_GEARBOX_PROMPT)
    plan_b = build_plan(_GEARBOX_PROMPT, family_overrides={"component_3": "enclosure"})
    assert [c.id for c in plan_a.components] == [c.id for c in plan_b.components]
