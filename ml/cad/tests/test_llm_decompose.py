"""Tests the JSON-extraction/fallback logic with canned model outputs --
no real model load needed for these (fast, portable, no dependency on the
specific .gguf file being present). One real-model smoke test is gated
behind the model file actually existing, so the rest of the suite stays
fast and doesn't require this exact machine's setup."""

import pytest

from llm_decompose import (
    DEFAULT_MODEL_PATH,
    ActionablePoint,
    _correct_gear_mesh_spacing,
    _extract_gear_params,
    _extract_json_array,
    _parse_points,
    ask_clarifying_questions,
    decompose_prompt,
    fold_answers_into_prompt,
)


def test_extract_json_array_from_clean_output():
    raw = '[{"description": "a plate", "position_mm": {"x": 0, "y": 0, "z": 0}}]'
    items = _extract_json_array(raw)
    assert items == [{"description": "a plate", "position_mm": {"x": 0, "y": 0, "z": 0}}]


def test_extract_json_array_ignores_surrounding_prose():
    raw = 'Sure, here is the breakdown:\n[{"description": "a bracket", "position_mm": {"x": 1, "y": 2, "z": 3}}]\nHope this helps!'
    items = _extract_json_array(raw)
    assert items[0]["description"] == "a bracket"


def test_extract_json_array_ignores_thinking_tags():
    """Qwen3-style <think>...</think> reasoning tokens before the actual
    answer -- must not be mistaken for the answer itself."""
    raw = '<think>the user wants [1, 2, 3] things, let me consider...</think>\n[{"description": "a shaft", "position_mm": {"x": 0, "y": 0, "z": 0}}]'
    items = _extract_json_array(raw)
    assert items == [{"description": "a shaft", "position_mm": {"x": 0, "y": 0, "z": 0}}]


def test_extract_json_array_raises_on_no_array():
    with pytest.raises(ValueError):
        _extract_json_array("I cannot help with that.")


def test_extract_json_array_raises_on_malformed_json():
    with pytest.raises(Exception):
        _extract_json_array("[{\"description\": \"unterminated")


def test_extract_json_array_concatenates_multiple_final_answer_arrays():
    """Regression test for a real, confirmed failure: Qwen3-1.7B reliably
    enumerates every part correctly in its <think> reasoning, but frequently
    formats its FINAL answer as several separate single-item arrays back to
    back instead of one combined array. Taking only the last one (the
    original implementation) silently dropped every part but the final one
    on every multi-part prompt tested against the real model. This is the
    exact raw output captured from a real run of
    decompose_prompt("a base plate with a shaft on top and a gear next to it")."""
    raw = (
        "<think>\nOkay, let's tackle this query. The user provided a description: "
        '"a base plate with a shaft on top and a gear next to it". I need to break this into JSON parts.\n'
        "First, the base plate... the user wants maybe [1, 2, 3] variants to consider...\n"
        "</think>\n\n"
        '[{"description": "Base plate (flat surface, no dimensions specified)", "position_mm": {"x": 0, "y": 0, "z": 0}}], '
        '[{"description": "Shaft (vertical, 100mm length)", "position_mm": {"x": 0, "y": 0, "z": 100}}], '
        '[{"description": "Gear (circular, 20mm diameter)", "position_mm": {"x": 0, "y": 0, "z": 150}}]]'
    )
    points = _parse_points(raw)
    assert [p.description for p in points] == [
        "Base plate (flat surface, no dimensions specified)",
        "Shaft (vertical, 100mm length)",
        "Gear (circular, 20mm diameter)",
    ]
    assert points[1].position_mm == (0.0, 0.0, 100.0)
    assert points[2].position_mm == (0.0, 0.0, 150.0)


def test_extract_json_array_repairs_stray_quote_after_position_number():
    """Regression test for a real, confirmed failure: on an otherwise
    perfectly-correct 5-part decomposition, the model emitted a stray extra
    quote immediately after two numeric position_mm values ('"z": 0"}}'
    instead of '"z": 0}}'), which broke strict JSON parsing outright even
    though the intended value is completely unambiguous."""
    raw = (
        '[{"description": "Rectangular base plate, 300mm long, 220mm wide, 15mm thick", "position_mm": {"x": 0, "y": 0, "z": 0"}}, '
        '{"description": "Vertical support bracket, 100mm tall, 80mm wide, 12mm thick", "position_mm": {"x": 0, "y": 0, "z": 15"}}, '
        '{"description": "Cylindrical horizontal shaft, 35mm diameter, 200mm long", "position_mm": {"x": 0, "y": 0, "z": 15}}]'
    )
    points = _parse_points(raw)
    assert len(points) == 3
    assert points[0].position_mm == (0.0, 0.0, 0.0)
    assert points[1].position_mm == (0.0, 0.0, 15.0)


def test_repair_does_not_touch_a_legitimate_string_ending_in_a_digit():
    """The repair regex is scoped to the "x"/"y"/"z" KEY specifically so it
    can never misfire on an ordinary string field that happens to end in a
    digit followed by its own real closing quote."""
    raw = '[{"description": "engine model v123", "position_mm": {"x": 0, "y": 0, "z": 0}}]'
    points = _parse_points(raw)
    assert points[0].description == "engine model v123"


def test_parse_points_defaults_missing_position_to_origin():
    points = _parse_points('[{"description": "a gear"}]')
    assert points[0].description == "a gear"
    assert points[0].position_mm == (0.0, 0.0, 0.0)


def test_parse_points_raises_on_empty_list():
    with pytest.raises(ValueError):
        _parse_points("[]")


def test_decompose_prompt_falls_back_to_one_point_on_malformed_llm_output(monkeypatch):
    """The core safety-net property: a malformed/unparseable LLM response
    must never crash the pipeline or silently drop the request -- it
    degrades to treating the whole prompt as one actionable point, and
    says so via fell_back=True."""
    import llm_decompose

    class _FakeLlm:
        def create_chat_completion(self, **kwargs):
            return {"choices": [{"message": {"content": "I'm not sure how to help with that."}}]}

    monkeypatch.setattr(llm_decompose, "_get_llm", lambda model_path=None: _FakeLlm())
    result = decompose_prompt("a mysterious object")
    assert result.fell_back is True
    assert len(result.points) == 1
    assert result.points[0].description == "a mysterious object"
    assert result.points[0].position_mm == (0.0, 0.0, 0.0)


def test_decompose_prompt_parses_well_formed_multi_point_response(monkeypatch):
    import llm_decompose

    canned = (
        '[{"description": "a base plate, 100mm x 80mm x 8mm", "position_mm": {"x": 0, "y": 0, "z": 0}}, '
        '{"description": "a shaft, 20mm diameter, 100mm long", "position_mm": {"x": 0, "y": 0, "z": 50}}]'
    )

    class _FakeLlm:
        def create_chat_completion(self, **kwargs):
            return {"choices": [{"message": {"content": canned}}]}

    monkeypatch.setattr(llm_decompose, "_get_llm", lambda model_path=None: _FakeLlm())
    result = decompose_prompt("a plate with a shaft on top")
    assert result.fell_back is False
    assert len(result.points) == 2
    assert result.points[1].position_mm == (0.0, 0.0, 50.0)


@pytest.mark.skipif(not DEFAULT_MODEL_PATH.exists(), reason="requires the real Qwen3 .gguf model file on disk")
def test_real_model_decomposes_a_multi_part_prompt():
    result = decompose_prompt("a base plate with a shaft on top and a gear next to it")
    assert len(result.points) >= 1  # honest floor: a real small local model may not always split cleanly


def test_ask_clarifying_questions_parses_well_formed_response(monkeypatch):
    import llm_decompose

    canned = '{"questions": ["What size should the bracket be?", "How many holes?"]}'

    class _FakeLlm:
        def create_chat_completion(self, **kwargs):
            return {"choices": [{"message": {"content": canned}}]}

    monkeypatch.setattr(llm_decompose, "_get_llm", lambda model_path=None: _FakeLlm())
    result = ask_clarifying_questions("a bracket with some holes")
    assert result.questions == ["What size should the bracket be?", "How many holes?"]


def test_ask_clarifying_questions_returns_empty_when_prompt_is_already_complete(monkeypatch):
    import llm_decompose

    class _FakeLlm:
        def create_chat_completion(self, **kwargs):
            return {"choices": [{"message": {"content": '{"questions": []}'}}]}

    monkeypatch.setattr(llm_decompose, "_get_llm", lambda model_path=None: _FakeLlm())
    result = ask_clarifying_questions("a box 50mm x 40mm x 10mm")
    assert result.questions == []


def test_ask_clarifying_questions_ignores_thinking_tags(monkeypatch):
    import llm_decompose

    raw = '<think>the shape seems clear enough, {"a": 1} maybe not...</think>\n{"questions": ["How tall should it be?"]}'

    class _FakeLlm:
        def create_chat_completion(self, **kwargs):
            return {"choices": [{"message": {"content": raw}}]}

    monkeypatch.setattr(llm_decompose, "_get_llm", lambda model_path=None: _FakeLlm())
    result = ask_clarifying_questions("a tall box")
    assert result.questions == ["How tall should it be?"]


def test_ask_clarifying_questions_never_raises_on_malformed_output(monkeypatch):
    import llm_decompose

    class _FakeLlm:
        def create_chat_completion(self, **kwargs):
            return {"choices": [{"message": {"content": "I'm not sure."}}]}

    monkeypatch.setattr(llm_decompose, "_get_llm", lambda model_path=None: _FakeLlm())
    result = ask_clarifying_questions("a mysterious object")
    assert result.questions == []


def test_ask_clarifying_questions_caps_at_max_questions(monkeypatch):
    import llm_decompose

    canned = '{"questions": ["q1?", "q2?", "q3?", "q4?", "q5?"]}'

    class _FakeLlm:
        def create_chat_completion(self, **kwargs):
            return {"choices": [{"message": {"content": canned}}]}

    monkeypatch.setattr(llm_decompose, "_get_llm", lambda model_path=None: _FakeLlm())
    result = ask_clarifying_questions("something vague", max_questions=3)
    assert result.questions == ["q1?", "q2?", "q3?"]


def test_fold_answers_into_prompt_appends_answered_questions_only():
    enriched = fold_answers_into_prompt(
        "a bracket with some holes",
        [("What size?", "80x40x6mm"), ("Anything else?", "   ")],
    )
    assert "80x40x6mm" in enriched
    assert "Anything else?" not in enriched
    assert enriched.startswith("a bracket with some holes")


def test_fold_answers_into_prompt_returns_original_when_nothing_answered():
    assert fold_answers_into_prompt("a plain box", [("size?", "")]) == "a plain box"


@pytest.mark.skipif(not DEFAULT_MODEL_PATH.exists(), reason="requires the real Qwen3 .gguf model file on disk")
def test_real_model_asks_or_skips_sensibly():
    vague = ask_clarifying_questions("a bracket with some holes in it")
    assert isinstance(vague.questions, list)
    complete = ask_clarifying_questions("a box 50mm x 40mm x 10mm with a 16mm diameter, 20mm tall cylinder centered on top")
    assert isinstance(complete.questions, list)


def test_extract_gear_params_parses_module_and_tooth_count():
    assert _extract_gear_params("a gear profile, module 2, 16 teeth") == (2.0, 16)
    assert _extract_gear_params("a gear profile, module 2mm, 16 teeth") == (2.0, 16)
    assert _extract_gear_params("a gear profile, mod 2.5mm, 20 teeth") == (2.5, 20)
    assert _extract_gear_params("a box 40mm x 30mm x 20mm") is None


def test_correct_gear_mesh_spacing_fixes_a_wrong_distance():
    """Regression test for a real, confirmed model failure: tested live
    against a prompt with module=2mm, 18 and 30 teeth (correct center
    distance = 2*(18+30)/2 = 48mm), the real 1.7B model output 45mm --
    copied straight from the system prompt's own worked example instead of
    computing fresh arithmetic for the request's actual numbers. This is
    the exact shape of that failure: a wrong distance that this function
    must correct to the true formula result."""
    points = [
        ActionablePoint("a cylinder 10mm diameter, 60mm long", (0.0, 0.0, 0.0)),
        ActionablePoint("a gear profile, module 2mm, 18 teeth", (0.0, 0.0, 0.0)),
        ActionablePoint("a cylinder 10mm diameter, 60mm long", (45.0, 0.0, 0.0)),
        ActionablePoint("a gear profile, module 2mm, 30 teeth", (45.0, 0.0, 0.0)),
    ]
    corrected = _correct_gear_mesh_spacing(points)
    assert corrected[0].position_mm == (0.0, 0.0, 0.0)
    assert corrected[1].position_mm == (0.0, 0.0, 0.0)
    # both the gear AND its co-located mounting shaft move together
    assert corrected[2].position_mm == (48.0, 0.0, 0.0)
    assert corrected[3].position_mm == (48.0, 0.0, 0.0)


def test_correct_gear_mesh_spacing_preserves_the_llms_chosen_axis():
    points = [
        ActionablePoint("a gear profile, module 1.5, 10 teeth", (0.0, 0.0, 0.0)),
        ActionablePoint("a gear profile, module 1.5, 20 teeth", (0.0, 100.0, 0.0)),  # LLM chose the y axis
    ]
    corrected = _correct_gear_mesh_spacing(points)
    expected = 1.5 * (10 + 20) / 2
    assert corrected[0].position_mm == (0.0, 0.0, 0.0)
    assert corrected[1].position_mm == (0.0, expected, 0.0)


def test_correct_gear_mesh_spacing_defaults_to_x_axis_when_gears_coincide():
    points = [
        ActionablePoint("a gear profile, module 2, 16 teeth", (5.0, 5.0, 5.0)),
        ActionablePoint("a gear profile, module 2, 16 teeth", (5.0, 5.0, 5.0)),  # LLM placed them on top of each other
    ]
    corrected = _correct_gear_mesh_spacing(points)
    expected = 2 * (16 + 16) / 2
    assert corrected[1].position_mm == (5.0 + expected, 5.0, 5.0)


def test_correct_gear_mesh_spacing_leaves_gear_trains_untouched():
    """3+ gears have ambiguous meshing topology -- not attempted, rather
    than guessed at incorrectly."""
    points = [
        ActionablePoint("a gear profile, module 2, 16 teeth", (0.0, 0.0, 0.0)),
        ActionablePoint("a gear profile, module 2, 20 teeth", (30.0, 0.0, 0.0)),
        ActionablePoint("a gear profile, module 2, 24 teeth", (60.0, 0.0, 0.0)),
    ]
    corrected = _correct_gear_mesh_spacing(points)
    assert [p.position_mm for p in corrected] == [p.position_mm for p in points]


def test_correct_gear_mesh_spacing_leaves_mismatched_modules_untouched():
    points = [
        ActionablePoint("a gear profile, module 1.5, 16 teeth", (0.0, 0.0, 0.0)),
        ActionablePoint("a gear profile, module 2, 20 teeth", (30.0, 0.0, 0.0)),
    ]
    corrected = _correct_gear_mesh_spacing(points)
    assert [p.position_mm for p in corrected] == [p.position_mm for p in points]


def test_correct_gear_mesh_spacing_is_a_no_op_without_exactly_two_gears():
    points = [ActionablePoint("a gear profile, module 2, 16 teeth", (0.0, 0.0, 0.0))]
    assert _correct_gear_mesh_spacing(points) == points


def test_decompose_prompt_applies_gear_mesh_correction(monkeypatch):
    import llm_decompose

    # canned output deliberately has the WRONG distance (45mm instead of the correct 48mm)
    canned = (
        '[{"description": "a cylinder 10mm diameter, 60mm long", "position_mm": {"x": 0, "y": 0, "z": 0}}, '
        '{"description": "a gear profile, module 2mm, 18 teeth", "position_mm": {"x": 0, "y": 0, "z": 0}}, '
        '{"description": "a cylinder 10mm diameter, 60mm long", "position_mm": {"x": 45, "y": 0, "z": 0}}, '
        '{"description": "a gear profile, module 2mm, 30 teeth", "position_mm": {"x": 45, "y": 0, "z": 0}}]'
    )

    class _FakeLlm:
        def create_chat_completion(self, **kwargs):
            return {"choices": [{"message": {"content": canned}}]}

    monkeypatch.setattr(llm_decompose, "_get_llm", lambda model_path=None: _FakeLlm())
    result = decompose_prompt("gearbox with module 2mm gears, 18 and 30 teeth")
    assert result.fell_back is False
    assert result.points[2].position_mm == (48.0, 0.0, 0.0)
    assert result.points[3].position_mm == (48.0, 0.0, 0.0)
