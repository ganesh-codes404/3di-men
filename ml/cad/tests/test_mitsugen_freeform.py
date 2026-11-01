"""Tests for Mitsugen's free-form primitive-composition path -- the
second, independent construction strategy (alongside the original
radial-sweep vessel path) added so Mitsugen can attempt objects that
aren't a simple body of revolution (a mug with a handle, a toy car, a
bracket). See mitsugen/freeform.py and mitsugen/freeform_builder.py.

Most tests here use CANNED LLM output (no real model load) to test the
parsing/building logic deterministically and fast -- mirroring
tests/test_llm_decompose.py's own established pattern for the same
reason. One real-model smoke test is gated behind a check that the
on-disk .gguf model actually exists, so the rest of this file -- and
the rest of the suite -- stays fast and portable."""

import pytest

from mitsugen.freeform import (
    MODEL_PATH,
    BooleanStep,
    FreeformSpec,
    PrimitiveSpec,
    _fallback_spec,
    _parse_freeform,
    decompose_freeform_prompt,
)
from mitsugen.freeform_builder import build_freeform_solid
from mitsugen.pipeline import _requires_freeform, generate_mitsugen_model
from mitsugen.spec import compose_spec_from_prompt

_WELL_FORMED_CAR_JSON = """{"primitives": [
  {"id": "body", "type": "box", "size_mm": [80, 40, 20], "position_mm": [0, 0, 15]},
  {"id": "wheel_fl", "type": "cylinder", "radius_mm": 8, "height_mm": 6, "position_mm": [25, 22, 8], "rotation_deg": [90, 0, 0]},
  {"id": "wheel_fr", "type": "cylinder", "radius_mm": 8, "height_mm": 6, "position_mm": [25, -22, 8], "rotation_deg": [90, 0, 0]},
  {"id": "wheel_bl", "type": "cylinder", "radius_mm": 8, "height_mm": 6, "position_mm": [-25, 22, 8], "rotation_deg": [90, 0, 0]},
  {"id": "wheel_br", "type": "cylinder", "radius_mm": 8, "height_mm": 6, "position_mm": [-25, -22, 8], "rotation_deg": [90, 0, 0]}
], "operations": [
  {"op": "union", "target_id": "wheel_fl"}, {"op": "union", "target_id": "wheel_fr"},
  {"op": "union", "target_id": "wheel_bl"}, {"op": "union", "target_id": "wheel_br"}
]}"""


# ---------------- parsing: well-formed, reasoning-wrapped, malformed ----------------

def test_parses_a_well_formed_decomposition():
    spec = _parse_freeform(_WELL_FORMED_CAR_JSON, "toy car")
    assert len(spec.primitives) == 5
    assert len(spec.operations) == 4
    assert spec.dropped_primitives == []
    assert spec.auto_unioned_ids == []
    assert spec.fell_back is False


def test_strips_qwen3_style_reasoning_before_the_json_answer():
    raw = "<think>the user wants [1, 2, 3] parts, let me plan this out...</think>" + _WELL_FORMED_CAR_JSON
    spec = _parse_freeform(raw, "toy car")
    assert len(spec.primitives) == 5


def test_a_primitive_never_referenced_by_an_operation_is_auto_unioned_not_dropped():
    """A confirmed real failure mode this guards against: the model lists
    a feature but forgets to wire it into an operation -- that must not
    silently vanish from the final geometry."""
    raw = """{"primitives": [
      {"id": "body", "type": "cylinder", "radius_mm": 40, "height_mm": 95, "position_mm": [0, 0, 47]},
      {"id": "handle", "type": "cylinder", "radius_mm": 5, "height_mm": 60, "position_mm": [0, 45, 50], "rotation_deg": [90, 0, 0]}
    ], "operations": []}"""
    spec = _parse_freeform(raw, "mug")
    assert spec.auto_unioned_ids == ["handle"]
    assert len(spec.operations) == 1
    assert spec.operations[0].target_id == "handle"
    assert spec.operations[0].op == "union"


def test_a_malformed_primitive_is_dropped_and_disclosed_not_silently_built_wrong():
    raw = """{"primitives": [
      {"id": "body", "type": "box", "size_mm": [80, 40, 20], "position_mm": [0, 0, 15]},
      {"id": "mystery", "type": "box", "position_mm": [0, 0, 0]}
    ], "operations": []}"""
    spec = _parse_freeform(raw, "something")
    assert len(spec.primitives) == 1
    assert "mystery" in spec.dropped_primitives


def test_an_operation_referencing_an_unknown_id_is_silently_excluded_from_the_op_list():
    raw = """{"primitives": [
      {"id": "body", "type": "box", "size_mm": [80, 40, 20], "position_mm": [0, 0, 15]}
    ], "operations": [{"op": "union", "target_id": "does_not_exist"}]}"""
    spec = _parse_freeform(raw, "something")
    assert spec.operations == []


def test_totally_unparseable_output_raises_so_the_caller_can_fall_back():
    with pytest.raises(ValueError):
        _parse_freeform("I'm sorry, I cannot help with that request.", "something")


def test_fallback_spec_is_a_single_disclosed_box_never_claims_full_fidelity():
    spec = _fallback_spec("a 50mm wide gadget", "<unparseable raw output>")
    assert spec.fell_back is True
    assert len(spec.primitives) == 1
    assert spec.primitives[0].kind == "box"


# ---------------- building: real CSG construction ----------------

def test_build_freeform_solid_produces_one_connected_watertight_body():
    spec = _parse_freeform(_WELL_FORMED_CAR_JSON, "toy car")
    build = build_freeform_solid(spec)
    assert build.error is None
    assert build.mesh.is_watertight
    assert build.mesh.body_count == 1


def test_build_freeform_solid_repairs_a_near_miss_wheel_gap_into_one_connected_body():
    """Confirmed real root cause of a live failure (a real LLM
    decomposition of "a toy car" produced body_count=4, not 1): a
    predicted position close to right but a few mm short of actual
    contact still produces an individually-watertight, but DISCONNECTED,
    boolean union -- trimesh's own is_watertight is a per-component
    property and never catches this. Reuses _WELL_FORMED_CAR_JSON's own
    body/wheel sizing but pulls every wheel 4mm further out along y than
    the reference (which already only just touches, by ~1mm) -- a
    deliberate, consistent near-miss, the exact shape of error a small
    quantized model's own numeric estimate is prone to. The builder's
    own _nudge_into_contact repair (freeform_builder.py) must close this
    gap automatically; this must still come back as ONE connected,
    watertight body, not four separate ones."""
    raw = """{"primitives": [
      {"id": "body", "type": "box", "size_mm": [80, 40, 20], "position_mm": [0, 0, 15]},
      {"id": "wheel_fl", "type": "cylinder", "radius_mm": 8, "height_mm": 6, "position_mm": [25, 26, 8], "rotation_deg": [90, 0, 0]},
      {"id": "wheel_fr", "type": "cylinder", "radius_mm": 8, "height_mm": 6, "position_mm": [25, -26, 8], "rotation_deg": [90, 0, 0]},
      {"id": "wheel_bl", "type": "cylinder", "radius_mm": 8, "height_mm": 6, "position_mm": [-25, 26, 8], "rotation_deg": [90, 0, 0]},
      {"id": "wheel_br", "type": "cylinder", "radius_mm": 8, "height_mm": 6, "position_mm": [-25, -26, 8], "rotation_deg": [90, 0, 0]}
    ], "operations": [
      {"op": "union", "target_id": "wheel_fl"}, {"op": "union", "target_id": "wheel_fr"},
      {"op": "union", "target_id": "wheel_bl"}, {"op": "union", "target_id": "wheel_br"}
    ]}"""
    spec = _parse_freeform(raw, "toy car")
    build = build_freeform_solid(spec)
    assert build.error is None
    assert build.mesh.is_watertight
    assert build.mesh.body_count == 1


def test_build_freeform_solid_refuses_to_silently_relocate_a_wildly_misplaced_primitive():
    """The other half of the same guard: a primitive placed far beyond a
    plausible near-miss (here, 500mm from a 90mm-tall mug -- not "a few
    mm short," a wholesale different location) must NOT be silently
    teleported into contact -- that would invent a position the LLM
    never intended rather than correct a real near-miss. It stays
    disconnected, and the existing body_count check discloses that
    rather than this repair papering over it."""
    raw = """{"primitives": [
      {"id": "body", "type": "cylinder", "radius_mm": 37.5, "height_mm": 90, "position_mm": [0, 0, 0]},
      {"id": "handle", "type": "torus", "major_radius_mm": 22, "minor_radius_mm": 6, "position_mm": [0, 500, 0], "rotation_deg": [0, 90, 0]}
    ], "operations": [{"op": "union", "target_id": "handle"}]}"""
    spec = _parse_freeform(raw, "a mug with a handle")
    build = build_freeform_solid(spec)
    assert build.error is None  # still a valid, individually-watertight mesh
    assert build.mesh.is_watertight
    assert build.mesh.body_count == 2  # never silently fused into 1


def test_build_freeform_solid_supports_a_difference_operation_cutting_a_cavity():
    raw = """{"primitives": [
      {"id": "block", "type": "box", "size_mm": [40, 40, 40], "position_mm": [0, 0, 20]},
      {"id": "bore", "type": "cylinder", "radius_mm": 8, "height_mm": 50, "position_mm": [0, 0, 20]}
    ], "operations": [{"op": "difference", "target_id": "bore"}]}"""
    spec = _parse_freeform(raw, "a block with a hole through it")
    build = build_freeform_solid(spec)
    assert build.error is None
    assert build.mesh.is_watertight
    solid_block_volume = 40 * 40 * 40
    assert build.mesh.volume < solid_block_volume  # real material was actually removed


def test_build_freeform_solid_fails_loudly_on_an_unbuildable_primitive_rather_than_silently_omitting_it():
    spec = FreeformSpec(
        object_description="x",
        primitives=[PrimitiveSpec(id="body", kind="box", size_mm=(10, 10, 10)), PrimitiveSpec(id="bad", kind="sphere", radius_mm=None)],
        operations=[],
    )
    build = build_freeform_solid(spec)
    assert build.mesh is None
    assert build.error is not None
    assert "bad" in build.error


def test_build_freeform_solid_requires_at_least_one_primitive():
    build = build_freeform_solid(FreeformSpec(object_description="x", primitives=[], operations=[]))
    assert build.mesh is None
    assert build.error is not None


# ---------------- routing: vessel path vs. freeform path ----------------

@pytest.mark.parametrize("prompt", [
    "a vase, 200mm tall, 110mm wide at the base, narrowing to 40mm at the top",
    "a bottle, 180mm tall",
    "a bowl",
])
def test_plain_vessel_prompts_stay_on_the_radial_sweep_path(prompt):
    vessel_spec = compose_spec_from_prompt(prompt)
    assert _requires_freeform(prompt, vessel_spec) is False


@pytest.mark.parametrize("prompt", [
    "a mug, 95mm tall",  # a real mug has a handle -- the radial-sweep path cannot represent it
    "a vase with a handle",
    "a toy car",
    "a square vase, 100mm tall",  # "vase" matched, but "square" means it isn't a circular revolve
    "a bracket with four mounting holes",
])
def test_featured_or_unrecognized_prompts_route_to_freeform(prompt):
    vessel_spec = compose_spec_from_prompt(prompt)
    assert _requires_freeform(prompt, vessel_spec) is True


def test_generate_mitsugen_model_reports_construction_mode_for_the_vessel_path():
    result = generate_mitsugen_model("a vase, 150mm tall, 60mm wide at the base, narrowing to 30mm at the top", max_slices=8)
    assert result.construction_mode == "radial_sweep"
    assert result.spec is not None
    assert result.freeform_spec is None


def test_freeform_path_always_discloses_the_approximation_even_on_pass():
    """Non-negotiable: a freeform result must never look indistinguishable
    from a precise radial-sweep PASS -- the approximation disclosure
    finding is always present, regardless of overall_status."""
    import mitsugen.pipeline as pipeline_mod

    def fake_decompose(prompt, model_path=None):
        from mitsugen.freeform import FreeformSpec, PrimitiveSpec

        return FreeformSpec(object_description=prompt, primitives=[PrimitiveSpec(id="body", kind="sphere", radius_mm=30.0)], operations=[])

    original = pipeline_mod.decompose_freeform_prompt
    pipeline_mod.decompose_freeform_prompt = fake_decompose
    try:
        result = generate_mitsugen_model("a toy car")
        assert result.construction_mode == "primitive_composition"
        disclosure = next(f for f in result.findings if "never reproduced exactly" in f.requirement)
        assert disclosure.passed is True
    finally:
        pipeline_mod.decompose_freeform_prompt = original


def test_freeform_path_is_fail_not_pass_when_the_build_itself_fails():
    import mitsugen.pipeline as pipeline_mod

    def fake_decompose_empty(prompt, model_path=None):
        from mitsugen.freeform import FreeformSpec

        return FreeformSpec(object_description=prompt, primitives=[], operations=[])

    original = pipeline_mod.decompose_freeform_prompt
    pipeline_mod.decompose_freeform_prompt = fake_decompose_empty
    try:
        result = generate_mitsugen_model("a toy car")
        assert result.overall_status == "FAIL"
        assert result.error is not None
    finally:
        pipeline_mod.decompose_freeform_prompt = original


# ---------------- real end-to-end LLM decomposition (gated) ----------------

@pytest.mark.skipif(not MODEL_PATH.exists(), reason="local .gguf model not present on this machine")
def test_real_llm_decomposes_a_toy_car_into_a_connected_printable_solid():
    spec = decompose_freeform_prompt("a simple toy car: a rectangular body on four cylindrical wheels")
    assert len(spec.primitives) >= 1
    build = build_freeform_solid(spec)
    # Never assert the LLM got the "right" shape -- only that whatever it
    # proposed was actually realized as valid, connected, printable geometry.
    if build.mesh is not None:
        assert build.mesh.is_watertight
        assert build.mesh.body_count == 1


# ---------------- torus: the curved-ring primitive (handles, hooks) ----------------

def test_torus_primitive_parses_and_builds_a_mug_handle():
    """Confirmed real vocabulary gap (mission-brief audit): a straight
    cylinder/cone cannot represent a curved handle or hook at all -- a
    live test of "a wall hook" produced a cone standing in for the
    hook's curve, visibly wrong. torus is the one primitive here that
    can actually represent a ring/loop/curve."""
    raw = """{"primitives": [
      {"id": "body", "type": "cylinder", "radius_mm": 40, "height_mm": 95, "position_mm": [0, 0, 0]},
      {"id": "handle", "type": "torus", "major_radius_mm": 25, "minor_radius_mm": 6, "position_mm": [0, 45, 10], "rotation_deg": [0, 90, 0]}
    ], "operations": [{"op": "union", "target_id": "handle"}]}"""
    spec = _parse_freeform(raw, "a mug with a handle")
    assert len(spec.primitives) == 2
    build = build_freeform_solid(spec)
    assert build.error is None
    assert build.mesh.is_watertight
    assert build.mesh.body_count == 1


def test_torus_primitive_rejects_a_tube_thicker_than_the_ring_itself():
    raw = """{"primitives": [
      {"id": "body", "type": "box", "size_mm": [40, 40, 40], "position_mm": [0, 0, 0]},
      {"id": "bad_ring", "type": "torus", "major_radius_mm": 10, "minor_radius_mm": 15, "position_mm": [0, 0, 0]}
    ], "operations": [{"op": "union", "target_id": "bad_ring"}]}"""
    spec = _parse_freeform(raw, "something")
    assert len(spec.primitives) == 1
    assert "bad_ring" in spec.dropped_primitives


# ---------------- MODEL_PATH resolution (the real "generates cubes" root cause) ----------------

def test_model_path_falls_back_to_the_shared_model_path_env_var(monkeypatch):
    """Confirmed real, severe bug this guards against (live-reported
    "Mitsugen generates cubes for nearly every prompt"): this module
    used to read ONLY its own MITSUGEN_FREEFORM_MODEL_PATH, never the
    shared MODEL_PATH env var llm_decompose.py and this project's own
    .env.rtx4060.example profile already use. On that machine, .env
    sets MODEL_PATH to a deliberately bigger model (Qwen3-8B) at a path
    that doesn't match this module's Mac-specific hardcoded default --
    every freeform decomposition call failed to even load the model,
    silently falling back to the disclosed default box for every
    non-vase prompt (the vase/bottle/bowl/cup path is pure regex and
    never calls an LLM, which is why only THAT path kept working).
    Calls _resolve_model_path() directly under a patched environment --
    NOT importlib.reload(), which recreates every class in the module as
    a new object and breaks == for any name (like BooleanStep) another
    test file already imported before the reload; confirmed real, not
    hypothetical: an earlier version of this test intermittently broke
    a pattern-operation test's BooleanStep equality check this way."""
    from mitsugen.freeform import _resolve_model_path

    monkeypatch.setenv("MODEL_PATH", "models/Qwen3-8B-Q4_K_M.gguf")
    monkeypatch.delenv("MITSUGEN_FREEFORM_MODEL_PATH", raising=False)
    assert str(_resolve_model_path()) == "models/Qwen3-8B-Q4_K_M.gguf"


def test_model_path_own_override_still_wins_over_the_shared_one(monkeypatch):
    """MITSUGEN_FREEFORM_MODEL_PATH stays a real override for the rare
    case someone deliberately wants Mitsugen on a DIFFERENT model than
    3di-men's own CAD decomposition -- the shared MODEL_PATH fallback
    above must never shadow it."""
    from mitsugen.freeform import _resolve_model_path

    monkeypatch.setenv("MODEL_PATH", "models/Qwen3-8B-Q4_K_M.gguf")
    monkeypatch.setenv("MITSUGEN_FREEFORM_MODEL_PATH", "models/some-other-model.gguf")
    assert str(_resolve_model_path()) == "models/some-other-model.gguf"


# ---------------- clarify stage: Mitsugen's own independent analogue of llm_decompose.py's ask_clarifying_questions ----------------

def test_find_clarify_shaped_objects_parses_a_well_formed_questions_object():
    from mitsugen.freeform import _find_clarify_shaped_objects

    candidates = _find_clarify_shaped_objects('{"questions": ["Roughly how big should this be?", "How many legs?"]}')
    assert len(candidates) == 1
    assert candidates[0]["questions"] == ["Roughly how big should this be?", "How many legs?"]


def test_find_clarify_shaped_objects_strips_qwen3_reasoning_and_ignores_unrelated_objects():
    from mitsugen.freeform import _find_clarify_shaped_objects

    text = '<think>the user wants {"primitives": [1]} something</think>{"questions": []}'
    # _find_clarify_shaped_objects itself doesn't strip </think> (that's ask_clarifying_questions'
    # own job, mirroring _extract_json_object) -- but it must still ignore an object shaped like a
    # freeform decomposition (no "questions" key) even when scanning the whole string.
    candidates = _find_clarify_shaped_objects(text)
    assert len(candidates) == 1
    assert candidates[0]["questions"] == []


def test_fold_answers_into_prompt_appends_only_answered_questions():
    from mitsugen.freeform import fold_answers_into_prompt

    enriched = fold_answers_into_prompt("a stand with some legs", [("How many legs?", "4"), ("Color?", "")])
    assert "a stand with some legs" in enriched
    assert "How many legs? 4" in enriched
    assert "Color?" not in enriched  # unanswered -- never folded in


def test_fold_answers_into_prompt_is_a_noop_with_no_real_answers():
    from mitsugen.freeform import fold_answers_into_prompt

    assert fold_answers_into_prompt("a toy car", []) == "a toy car"
    assert fold_answers_into_prompt("a toy car", [("Q?", "")]) == "a toy car"


def test_ask_clarifying_questions_never_raises_on_llm_failure(monkeypatch):
    """Same never-raises discipline as decompose_freeform_prompt -- an
    optional step that could itself crash would defeat the point of
    being optional."""
    import mitsugen.freeform as ff

    def broken_get_llm(model_path=None):
        raise RuntimeError("model not found")

    monkeypatch.setattr(ff, "_get_llm", broken_get_llm)
    result = ff.ask_clarifying_questions("a stand with some legs")
    assert result.questions == []


# ---------------- pattern operations: declare a feature once, repeat it, instead of enumerating N primitives ----------------

def test_circular_pattern_parses_and_builds_a_four_legged_stool():
    """The exact scenario pattern ops exist for: ONE leg primitive +
    ONE circular_pattern operation, instead of 4 separately-declared
    leg primitives (the enumeration approach that's most likely to
    silently end up with the wrong count)."""
    raw = """{"primitives": [
      {"id": "seat", "type": "cylinder", "radius_mm": 150, "height_mm": 20, "position_mm": [0, 0, 440]},
      {"id": "leg", "type": "cylinder", "radius_mm": 12, "height_mm": 430, "position_mm": [0, 0, 215]}
    ], "operations": [{"op": "circular_pattern", "target_id": "leg", "count": 4, "pattern_radius_mm": 120, "axis": "z"}]}"""
    spec = _parse_freeform(raw, "a round stool with four legs")
    assert len(spec.primitives) == 2
    assert len(spec.operations) == 1
    assert spec.operations[0].op == "circular_pattern"
    assert spec.operations[0].count == 4
    build = build_freeform_solid(spec)
    assert build.error is None
    assert build.mesh.is_watertight
    assert build.mesh.body_count == 1
    assert build.mesh.extents[2] == pytest.approx(450, abs=1)  # seat top (450) -- full stool height


def test_linear_pattern_parses_and_builds_a_ribbed_plate():
    raw = """{"primitives": [
      {"id": "plate", "type": "box", "size_mm": [100, 40, 5], "position_mm": [0, 0, 0]},
      {"id": "rib", "type": "box", "size_mm": [4, 40, 10], "position_mm": [0, 0, 7.5]}
    ], "operations": [{"op": "linear_pattern", "target_id": "rib", "count": 5, "spacing_mm": 18, "axis": "x"}]}"""
    spec = _parse_freeform(raw, "a plate with five ribs")
    build = build_freeform_solid(spec)
    assert build.error is None
    assert build.mesh.is_watertight
    assert build.mesh.body_count == 1


def test_pattern_operation_with_malformed_fields_is_dropped_not_fatal():
    """A malformed pattern op (missing count) is dropped, same discipline
    as a malformed primitive -- the target then falls through to
    auto_unioned (a single, unpatterned instance) rather than losing
    the feature or failing the whole spec."""
    raw = """{"primitives": [
      {"id": "seat", "type": "cylinder", "radius_mm": 150, "height_mm": 20, "position_mm": [0, 0, 440]},
      {"id": "leg", "type": "cylinder", "radius_mm": 12, "height_mm": 430, "position_mm": [0, 0, 215]}
    ], "operations": [{"op": "circular_pattern", "target_id": "leg", "pattern_radius_mm": 120, "axis": "z"}]}"""
    spec = _parse_freeform(raw, "a stool")
    assert spec.operations == [BooleanStep(op="union", target_id="leg")]
    assert spec.auto_unioned_ids == ["leg"]


def test_pattern_operation_axis_defaults_to_z_when_missing_or_invalid():
    raw = """{"primitives": [
      {"id": "plate", "type": "box", "size_mm": [100, 40, 5], "position_mm": [0, 0, 0]},
      {"id": "rib", "type": "box", "size_mm": [4, 40, 10], "position_mm": [0, 0, 7.5]}
    ], "operations": [{"op": "linear_pattern", "target_id": "rib", "count": 3, "spacing_mm": 18, "axis": "diagonal"}]}"""
    spec = _parse_freeform(raw, "a plate with ribs")
    assert spec.operations[0].axis == "z"


# ---------------- Route B routing: neural generation, with an honest fallback when unavailable ----------------

def test_organic_prompt_falls_back_to_primitive_composition_with_a_disclosed_reason(monkeypatch):
    """On a machine with no neural backend configured (true of every
    machine by default, and genuinely true of the one these tests run
    on -- no CUDA GPU), an organic/figurine-style request must still
    produce a REAL result via the existing, proven primitive-
    composition path -- never a bare failure just because the ideal
    path isn't available -- with a disclosed finding explaining
    specifically why Route B wasn't used."""
    import mitsugen.pipeline as pipeline_mod

    def fake_decompose(prompt, model_path=None):
        from mitsugen.freeform import FreeformSpec, PrimitiveSpec

        return FreeformSpec(object_description=prompt, primitives=[PrimitiveSpec(id="body", kind="sphere", radius_mm=30.0)], operations=[])

    monkeypatch.setenv("MITSUGEN_NEURAL_ENABLED", "0")
    original = pipeline_mod.decompose_freeform_prompt
    pipeline_mod.decompose_freeform_prompt = fake_decompose
    try:
        result = generate_mitsugen_model("a cat figurine")
        assert result.construction_mode == "primitive_composition"
        route_b_finding = next(f for f in result.findings if "Route B" in f.requirement)
        assert route_b_finding.passed is True  # unavailability is disclosed, not treated as this result's own failure
        assert "unavailable" in route_b_finding.measured
    finally:
        pipeline_mod.decompose_freeform_prompt = original


def test_non_organic_prompt_never_attempts_route_b_at_all():
    """A plain Route A request (no organic/figurine keyword) must not
    even check neural availability -- confirmed by the absence of any
    Route B finding, not just by the construction_mode."""
    import mitsugen.pipeline as pipeline_mod

    def fake_decompose(prompt, model_path=None):
        from mitsugen.freeform import FreeformSpec, PrimitiveSpec

        return FreeformSpec(object_description=prompt, primitives=[PrimitiveSpec(id="body", kind="box", size_mm=(40.0, 40.0, 40.0))], operations=[])

    original = pipeline_mod.decompose_freeform_prompt
    pipeline_mod.decompose_freeform_prompt = fake_decompose
    try:
        result = generate_mitsugen_model("a wall hook with a mounting plate")
        assert result.construction_mode == "primitive_composition"
        assert not any("Route B" in f.requirement for f in result.findings)
    finally:
        pipeline_mod.decompose_freeform_prompt = original
