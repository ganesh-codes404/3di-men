"""Tests the JSON-extraction/fallback logic with canned model outputs --
no real model load needed for these (fast, portable, no dependency on the
specific .gguf file being present). One real-model smoke test is gated
behind the model file actually existing, so the rest of the suite stays
fast and doesn't require this exact machine's setup."""

import json

import pytest

from assembly_plan import Relationship
from llm_decompose import (
    DEFAULT_MODEL_PATH,
    _POINTS_JSON_SCHEMA,
    _SYSTEM_PROMPT,
    ActionablePoint,
    _correct_gear_mesh_spacing,
    _extract_gear_params,
    _extract_json_array,
    _get_clarify_grammar,
    _get_points_grammar,
    _parse_points,
    ask_clarifying_questions,
    decompose_prompt,
    fold_answers_into_prompt,
)


def test_system_prompt_instructs_overlap_avoidance_and_a_final_self_check():
    """Explicit user request: the LLM must understand overlaps between
    parts and actively avoid them, understand how parts relate to each
    other, and never hand off a half-built decomposition --
    parallel_builder.py's collision exclusion (_find_and_resolve_collisions)
    and repair.py's truncation strategy are reactive repair nets for when
    something still goes wrong downstream, not a substitute for the LLM
    getting the layout right the first time. These phrases anchor that
    guidance in the prompt -- if a future edit drops them, this fails."""
    assert "REAL clearance" in _SYSTEM_PROMPT  # overlap avoidance for unrelated parts (STEP 2's position_mm fallback)
    assert "STEP 4" in _SYSTEM_PROMPT and "review your own array" in _SYSTEM_PROMPT  # the closing self-check pass
    assert "half-built assembly" in _SYSTEM_PROMPT


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


def test_points_grammar_builds_successfully():
    """Grammar construction itself (LlamaGrammar.from_json_schema against
    _POINTS_JSON_SCHEMA) must not raise -- this is the thing that makes
    malformed JSON structurally impossible to emit, so if it can't even be
    built, decompose_prompt silently loses its main defense (it still falls
    back to unconstrained generation, see
    test_decompose_prompt_still_works_if_grammar_construction_fails, but
    that's the degraded path, not the intended one)."""
    from llama_cpp.llama_grammar import LlamaGrammar

    assert isinstance(_get_points_grammar(), LlamaGrammar)


def test_clarify_grammar_builds_successfully():
    from llama_cpp.llama_grammar import LlamaGrammar

    assert isinstance(_get_clarify_grammar(), LlamaGrammar)


def test_points_json_schema_requires_id_on_every_point():
    """Regression test for a real failure confirmed via build_log.jsonl on
    2026-10-15: two separate real decompositions ("a water bottle", "a
    simple bracket...") each had the model emit a later point's
    relationship.target_id referencing a semantic name ("body"/"bracket")
    that NO point in the same response actually carried as its own "id" --
    because "id" was optional, so the model sometimes skipped stating it on
    exactly the point its own later relationship went on to name.
    assembly_plan.validate_relationships correctly rejected the whole
    decomposition as a dangling reference in both real cases. "id" must be
    required so the grammar-constrained decoder can no longer produce that
    shape at all -- this test locks that in via the same jsonschema the
    model's actual GBNF grammar is built from (LlamaGrammar.from_json_schema
    uses this exact schema), without needing a real model call."""
    import jsonschema

    schema = json.loads(_POINTS_JSON_SCHEMA)

    # the exact real-world failure shape: second point's relationship names
    # "body", but no point anywhere in the list has "id": "body"
    reproduces_the_real_bug = [
        {"description": "a hollow vessel body", "position_mm": {"x": 0, "y": 0, "z": 0}},
        {
            "description": "a cap",
            "position_mm": {"x": 0, "y": 0, "z": 0},
            "relationship": {"kind": "attached_to_face", "target_id": "body", "face": "+z"},
        },
    ]
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(reproduces_the_real_bug, schema)

    # same shape, but every point states its own "id" -- must validate
    fixed = [
        {"id": "body", "description": "a hollow vessel body", "position_mm": {"x": 0, "y": 0, "z": 0}},
        {
            "id": "cap",
            "description": "a cap",
            "position_mm": {"x": 0, "y": 0, "z": 0},
            "relationship": {"kind": "attached_to_face", "target_id": "body", "face": "+z"},
        },
    ]
    jsonschema.validate(fixed, schema)


def test_decompose_prompt_passes_a_grammar_to_the_llm(monkeypatch):
    """The actual wiring, not just that the grammar object can be built in
    isolation: create_chat_completion must receive a non-None `grammar`
    kwarg on every call, so the sampler is actually constrained."""
    import llm_decompose

    canned = '[{"description": "a bracket", "position_mm": {"x": 0, "y": 0, "z": 0}}]'
    captured = {}

    class _FakeLlm:
        def create_chat_completion(self, **kwargs):
            captured.update(kwargs)
            return {"choices": [{"message": {"content": canned}}]}

    monkeypatch.setattr(llm_decompose, "_get_llm", lambda model_path=None: _FakeLlm())
    decompose_prompt("a bracket")
    assert captured.get("grammar") is not None


def test_ask_clarifying_questions_passes_a_grammar_to_the_llm(monkeypatch):
    import llm_decompose

    canned = '{"questions": []}'
    captured = {}

    class _FakeLlm:
        def create_chat_completion(self, **kwargs):
            captured.update(kwargs)
            return {"choices": [{"message": {"content": canned}}]}

    monkeypatch.setattr(llm_decompose, "_get_llm", lambda model_path=None: _FakeLlm())
    ask_clarifying_questions("a fully specified box 10x10x10mm")
    assert captured.get("grammar") is not None


def test_decompose_prompt_still_works_if_grammar_construction_fails(monkeypatch):
    """Grammar construction is defense in depth, not a hard requirement --
    if it ever fails (bad schema after a future edit, a llama-cpp-python
    upgrade that changes behavior, ...), decompose_prompt must degrade to
    unconstrained generation rather than crashing outright."""
    import llm_decompose

    canned = '[{"description": "a bracket", "position_mm": {"x": 0, "y": 0, "z": 0}}]'
    captured = {}

    class _FakeLlm:
        def create_chat_completion(self, **kwargs):
            captured.update(kwargs)
            return {"choices": [{"message": {"content": canned}}]}

    def _broken_grammar():
        raise RuntimeError("grammar construction broke")

    monkeypatch.setattr(llm_decompose, "_get_llm", lambda model_path=None: _FakeLlm())
    monkeypatch.setattr(llm_decompose, "_get_points_grammar", _broken_grammar)
    result = decompose_prompt("a bracket")
    assert captured.get("grammar") is None
    assert result.fell_back is False
    assert result.points[0].description == "a bracket"


def test_decompose_prompt_injects_retrieved_knowledge_context_into_the_user_message(monkeypatch):
    """The actual integration point (section 9's Stage B): when
    knowledge/pipeline.py's retrieve_context_for_prompt returns something,
    decompose_prompt must put it in the USER message (never the system
    prompt -- see the comment in decompose_prompt about why) and must
    thread it through onto the returned DecomposeResult, never silently
    drop it."""
    import llm_decompose

    canned = '[{"description": "a bracket", "position_mm": {"x": 0, "y": 0, "z": 0}}]'
    captured = {}

    class _FakeLlm:
        def create_chat_completion(self, **kwargs):
            captured.update(kwargs)
            return {"choices": [{"message": {"content": canned}}]}

    monkeypatch.setattr(llm_decompose, "_get_llm", lambda model_path=None: _FakeLlm())
    monkeypatch.setattr(
        "knowledge.pipeline.retrieve_context_for_prompt",
        lambda prompt, config=None: "Reference notes: - [doc] a fillet radius must not exceed...",
    )
    result = decompose_prompt("a bracket with a fillet")
    messages = captured["messages"]
    system_msg = next(m["content"] for m in messages if m["role"] == "system")
    user_msg = next(m["content"] for m in messages if m["role"] == "user")
    assert "Reference notes" not in system_msg
    assert "Reference notes" in user_msg
    assert "Request: a bracket with a fillet" in user_msg
    assert result.knowledge_context == "Reference notes: - [doc] a fillet radius must not exceed..."


def test_decompose_prompt_uses_the_plain_prompt_when_no_knowledge_context_is_found(monkeypatch):
    import llm_decompose

    canned = '[{"description": "a bracket", "position_mm": {"x": 0, "y": 0, "z": 0}}]'
    captured = {}

    class _FakeLlm:
        def create_chat_completion(self, **kwargs):
            captured.update(kwargs)
            return {"choices": [{"message": {"content": canned}}]}

    monkeypatch.setattr(llm_decompose, "_get_llm", lambda model_path=None: _FakeLlm())
    monkeypatch.setattr("knowledge.pipeline.retrieve_context_for_prompt", lambda prompt, config=None: "")
    result = decompose_prompt("a plain bracket")
    user_msg = next(m["content"] for m in captured["messages"] if m["role"] == "user")
    assert user_msg == "a plain bracket"  # unchanged from pre-knowledge-subsystem behavior
    assert result.knowledge_context == ""


def test_decompose_prompt_tolerates_the_knowledge_subsystem_raising(monkeypatch):
    """Section 3's graceful-degradation contract: a knowledge-subsystem
    failure must never break decomposition."""
    import llm_decompose

    canned = '[{"description": "a bracket", "position_mm": {"x": 0, "y": 0, "z": 0}}]'

    class _FakeLlm:
        def create_chat_completion(self, **kwargs):
            return {"choices": [{"message": {"content": canned}}]}

    def _broken(prompt, config=None):
        raise RuntimeError("index is corrupt")

    monkeypatch.setattr(llm_decompose, "_get_llm", lambda model_path=None: _FakeLlm())
    monkeypatch.setattr("knowledge.pipeline.retrieve_context_for_prompt", _broken)
    result = decompose_prompt("a plain bracket")
    assert result.fell_back is False
    assert result.knowledge_context == ""


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


def test_correct_gear_mesh_spacing_still_fires_when_gears_use_relationships():
    """Regression test for a confirmed real bug: once a housing is also
    present, _SYSTEM_PROMPT's own Example 5 teaches each gear to be
    concentric_with its own shaft, which used to disable this corrector
    entirely (relationship is not None for both gears) -- meaning the exact
    assemblies this project most wants right (a housing-mounted gear train)
    were the ones left uncorrected. Each gear's real position_mm is only a
    placeholder when a relationship is set (see ActionablePoint.relationship's
    own docstring) -- correcting the chain's free anchor point instead of the
    gear's own placeholder is safe and should still apply the formula."""
    points = [
        ActionablePoint("a box 120mm x 80mm x 60mm", (0.0, 0.0, 0.0), id="housing"),
        ActionablePoint(
            "a cylinder 20mm diameter, 10mm tall", (0.0, 0.0, 0.0), id="boss_A",
            relationship=Relationship(kind="attached_to_face", target_id="housing", face="+y"),
        ),
        ActionablePoint(
            "a cylinder 10mm diameter, 70mm long", (0.0, 0.0, 0.0), id="shaft_A",
            relationship=Relationship(kind="concentric_with", target_id="boss_A", axis="y"),
        ),
        ActionablePoint(
            "a gear profile, module 2.5, 24 teeth", (0.0, 0.0, 0.0), id="gear_A",
            relationship=Relationship(kind="concentric_with", target_id="shaft_A", axis="y"),
        ),
        # the second, parallel shaft -- plain position_mm, no relationship,
        # deliberately placed at the WRONG distance (40mm) to confirm the
        # corrector actually moves it
        ActionablePoint("a cylinder 10mm diameter, 70mm long", (40.0, 0.0, 0.0), id="shaft_B"),
        ActionablePoint(
            "a gear profile, module 2.5, 36 teeth", (40.0, 0.0, 0.0), id="gear_B",
            relationship=Relationship(kind="concentric_with", target_id="shaft_B", axis="y"),
        ),
    ]
    corrected = _correct_gear_mesh_spacing(points)
    expected = 2.5 * (24 + 36) / 2  # 75mm

    by_id = {p.id: p for p in corrected}
    assert by_id["shaft_B"].position_mm == (expected, 0.0, 0.0)
    # the housing/boss/shaft/gear chain anchoring gear_A is untouched
    assert by_id["housing"].position_mm == (0.0, 0.0, 0.0)
    assert by_id["boss_A"].position_mm == (0.0, 0.0, 0.0)
    assert by_id["shaft_A"].position_mm == (0.0, 0.0, 0.0)
    assert by_id["gear_A"].position_mm == (0.0, 0.0, 0.0)
    # id/relationship must survive the rebuild on every point, not just the
    # moved ones (a confirmed separate real bug: the old rebuild dropped
    # both fields unconditionally, risking dangling target_id references)
    for p in corrected:
        assert p.id == by_id[p.id].id
    assert by_id["gear_B"].relationship.target_id == "shaft_B"
    assert by_id["shaft_A"].relationship.target_id == "boss_A"


def test_correct_gear_mesh_spacing_leaves_a_shared_anchor_untouched():
    """Both gears attached directly to the SAME housing (no per-gear shaft)
    have no unambiguous single point to shift -- this must be left
    uncorrected rather than guessed at, exactly like the pre-existing
    3+-gears and mismatched-modules bail-outs."""
    points = [
        ActionablePoint("a box 100mm x 100mm x 100mm", (0.0, 0.0, 0.0), id="housing"),
        ActionablePoint(
            "a gear profile, module 2.5, 14 teeth", (0.0, 0.0, 0.0), id="gear_A",
            relationship=Relationship(kind="attached_to_face", target_id="housing", face="+x"),
        ),
        ActionablePoint(
            "a gear profile, module 2.5, 22 teeth", (0.0, 0.0, 0.0), id="gear_B",
            relationship=Relationship(kind="attached_to_face", target_id="housing", face="-x"),
        ),
    ]
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
