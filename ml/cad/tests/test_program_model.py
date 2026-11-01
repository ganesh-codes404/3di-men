from pathlib import Path
from unittest.mock import patch

import pytest

pytest.importorskip("sklearn")

from data.program_dataset import sample_program_dataset, write_program_dataset
from sandbox import Sandbox


@pytest.fixture(scope="module")
def tiny_program_dataset(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("tiny_program_dataset")
    samples = sample_program_dataset(
        {"plate": 60, "flange": 60, "bracket": 60, "shaft": 60, "enclosure": 60, "gear": 60},
        n_generic=60, seed=0,
    )
    write_program_dataset(root, samples, seed=0)
    return root


def test_train_produces_a_bundle(tiny_program_dataset):
    from program_model import ProgramModelConfig, train_program_model

    bundle, metrics = train_program_model(tiny_program_dataset, ProgramModelConfig(seed=0))
    assert len(bundle.slot_op_classifiers) == bundle.config.k_slots
    assert metrics["n_train"] > 0


def test_bundle_save_load_round_trips_to_identical_predictions(tiny_program_dataset, tmp_path):
    from program_model import ProgramModelBundle, ProgramModelConfig, predict_program, train_program_model

    bundle, _ = train_program_model(tiny_program_dataset, ProgramModelConfig(seed=0))
    checkpoint = tmp_path / "bundle.joblib"
    bundle.save(checkpoint)
    loaded = ProgramModelBundle.load(checkpoint)

    text = "rectangular aluminum mounting plate, 120 mm x 80 mm, with four 10 mm mounting holes"
    before = predict_program(bundle, text)
    after = predict_program(loaded, text)
    assert [c.op for c in before] == [c.op for c in after]


def test_predicted_programs_for_each_family_are_structurally_valid(tiny_program_dataset):
    """Invariant: whatever the model predicts must at least pass
    Sandbox.validate()'s zero-cost structural check -- a program with a
    dangling handle reference should never be silently handed to the
    executor. This does NOT assert execution always succeeds (a Ridge
    regressor can still predict geometrically silly numbers) -- that
    honesty lives in program_eval.py's execution-success rate, not here."""
    from program_model import ProgramModelConfig, predict_program, train_program_model

    bundle, _ = train_program_model(tiny_program_dataset, ProgramModelConfig(seed=0))
    fixture_texts = [
        "rectangular aluminum mounting plate, 120 mm x 80 mm, with four 10 mm mounting holes",
        "circular steel flange, 100mm outer diameter, 40mm bore, six 8mm bolt holes",
        "steel angle bracket, 60mm x 40mm legs",
        "aluminum shaft, 20mm diameter, 100mm long",
        "aluminum enclosure, 100mm x 80mm x 40mm",
        "steel spur gear, module 2, 20 teeth",
        "",  # degenerate input -- must still produce a structurally valid (possibly empty) program
    ]
    for text in fixture_texts:
        program = predict_program(bundle, text)
        issues = Sandbox.validate(program)
        assert issues == [], f"{text!r} -> structurally invalid program: {issues}"


def test_predict_program_terminates_within_k_slots(tiny_program_dataset):
    from program_model import ProgramModelConfig, predict_program, train_program_model

    bundle, _ = train_program_model(tiny_program_dataset, ProgramModelConfig(seed=0))
    program = predict_program(bundle, "a gear")
    assert len(program) <= bundle.config.k_slots


def test_in_distribution_text_scores_above_threshold(tiny_program_dataset):
    from program_model import ProgramModelConfig, predict_program_checked, train_program_model

    bundle, _ = train_program_model(tiny_program_dataset, ProgramModelConfig(seed=0))
    prediction = predict_program_checked(bundle, "rectangular aluminum mounting plate, 120 mm x 80 mm, with four 10 mm mounting holes")
    assert prediction.in_distribution
    assert prediction.nearest_neighbor_similarity >= bundle.config.ood_similarity_threshold


def test_predict_program_hybrid_does_not_force_a_weakly_matched_rag_sequence(tiny_program_dataset):
    """Confirmed real gap (mission-brief audit): predict_program_hybrid used
    to force RAG's retrieved op-sequence onto ANY disagreement with the
    classifier, with no floor on how similar that "winning" neighbor
    actually was -- forcing an unrelated training example's op-sequence
    (which can include ops like add_cylinder or round_box_edges that share
    no real connection to the actual request) onto a text with no genuinely
    close neighbor at all is exactly how an unrequested cylinder or rounded
    block could reach a built model. Below bundle.config.
    rag_override_similarity_threshold, the classifier's own (unforced)
    program must be used instead, even when it disagrees with RAG's
    retrieved sequence."""
    from program_model import ProgramModelConfig, predict_program, predict_program_hybrid, train_program_model

    bundle, _ = train_program_model(tiny_program_dataset, ProgramModelConfig(seed=0))
    text = "a spaceship"  # confirmed elsewhere in this file: zero real training overlap, low similarity
    classifier_program = predict_program(bundle, text)
    disagreeing_ops = tuple(c.op for c in classifier_program) + ("add_cylinder",)  # force a real disagreement

    with patch("program_model.retrieve_nearest_op_sequence", return_value=(disagreeing_ops, 0.05)):
        program, used_rag = predict_program_hybrid(bundle, text)

    assert used_rag is False
    assert [c.op for c in program] == [c.op for c in classifier_program]


def test_predict_program_hybrid_still_uses_rag_when_the_match_is_strong(tiny_program_dataset):
    """The other half of the same gate: a genuinely close match (similarity
    at/above rag_override_similarity_threshold) that disagrees with the
    classifier must still win -- this is the hybrid's own documented
    value-add (measured gains on hand_written/generic_composition
    prompts), and the floor must not quietly disable it for every
    disagreement. Uses rag_override_similarity_threshold specifically
    (NOT ood_similarity_threshold -- see ProgramModelConfig's own
    docstring for why these are now two separate floors, not one reused
    number)."""
    from program_model import ProgramModelConfig, predict_program, predict_program_hybrid, train_program_model

    bundle, _ = train_program_model(tiny_program_dataset, ProgramModelConfig(seed=0))
    text = "rectangular aluminum mounting plate, 120 mm x 80 mm, with four 10 mm mounting holes"
    classifier_program = predict_program(bundle, text)
    disagreeing_ops = ("add_cylinder", "finalize")  # short, valid, and different from the plate's own prediction
    assert tuple(c.op for c in classifier_program) != disagreeing_ops
    strong_sim = bundle.config.rag_override_similarity_threshold + 0.1

    with patch("program_model.retrieve_nearest_op_sequence", return_value=(disagreeing_ops, strong_sim)):
        program, used_rag = predict_program_hybrid(bundle, text)

    assert used_rag is True
    assert tuple(c.op for c in program) == disagreeing_ops


def test_wildly_novel_prompt_is_flagged_out_of_distribution_not_silently_trusted(tiny_program_dataset):
    """The concrete failure this guards against: "a spaceship" and "a coffee
    mug" -- zero vocabulary overlap with any trained family -- used to
    produce a confident, byte-identical, meaningless flange-shaped guess
    with no signal to the caller that anything was wrong. This test doesn't
    assert the executed geometry is bad (it may still be status="ok" -- see
    the module docstring); it asserts the caller gets told not to trust it."""
    from program_model import ProgramModelConfig, predict_program_checked, train_program_model

    bundle, _ = train_program_model(tiny_program_dataset, ProgramModelConfig(seed=0))
    for text in ["a spaceship", "a coffee mug", "a human skull"]:
        prediction = predict_program_checked(bundle, text)
        assert not prediction.in_distribution, f"{text!r} should have been flagged OOD (similarity={prediction.nearest_neighbor_similarity:.3f})"


def test_nearest_neighbor_similarity_requires_stored_train_vectors(tiny_program_dataset):
    import dataclasses

    import pytest as _pytest

    from program_model import ProgramModelConfig, nearest_neighbor_similarity, train_program_model

    bundle, _ = train_program_model(tiny_program_dataset, ProgramModelConfig(seed=0))
    stripped = dataclasses.replace(bundle, train_text_vectors=None)
    with _pytest.raises(ValueError):
        nearest_neighbor_similarity(stripped, "a gear")


def test_malformed_handcrafted_program_is_rejected_not_silently_repaired():
    """Per the design's execution-side validation rule: a program with a
    dangling reference must be REJECTED before any geometry runs, never
    silently patched into something plausible."""
    from sandbox import ToolCall

    program = [ToolCall(op="drill_hole", args={"target": 42, "radius": 4, "depth": 10, "center": [0, 0, 0], "axis": "z"})]
    result = Sandbox.execute(program)
    assert result.status == "rejected"
    assert result.mesh is None


def test_override_bore_hole_dimensions_uses_the_literal_diameter_near_bore_or_hole():
    """Feature-level analogue of the base-shape override: confirmed real
    gap (mission-brief audit, live-reproduced against the actual trained
    bundle) -- a feature op's own explicit bore/hole diameter had no
    override mechanism at all, unlike the base shape's own primary
    dimensions. "12mm diameter bore" -> radius 6.0, regardless of what
    the (possibly very wrong) regressor predicted."""
    from program_model import _override_bore_hole_dimensions

    text = "a cylinder, 40mm diameter, 30mm tall, with a 12mm diameter bore through it"
    args = {"target": 0, "radius": 99.0, "depth": 5.0, "center": [0, 0, 0], "axis": "z"}
    overridden = _override_bore_hole_dimensions("drill_hole", args, text)
    assert overridden["radius"] == 6.0
    assert overridden["depth"] == 5.0  # untouched -- this override is scoped to the radius field only


def test_override_bore_hole_dimensions_handles_bore_before_the_number_too():
    from program_model import _override_bore_hole_dimensions

    text = "a boss with a bore of 8mm through the plate"
    args = {"target": 0, "direction": "+z", "offset_u": 0.0, "offset_v": 0.0, "boss_radius": 20.0, "boss_height": 10.0, "bore_radius": 99.0}
    overridden = _override_bore_hole_dimensions("add_boss_with_bore", args, text)
    assert overridden["bore_radius"] == 4.0


def test_override_bore_hole_dimensions_does_nothing_without_an_explicit_bore_claim():
    """Never a guessed mapping: with no "Nmm bore/hole" phrasing at all,
    the regressor's own prediction is left untouched."""
    from program_model import _override_bore_hole_dimensions

    args = {"target": 0, "radius": 7.5, "depth": 5.0, "center": [0, 0, 0], "axis": "z"}
    overridden = _override_bore_hole_dimensions("drill_hole", args, "a cylinder 40mm diameter, 30mm tall")
    assert overridden is args or overridden["radius"] == 7.5


def test_override_bore_hole_dimensions_does_not_apply_to_unrelated_ops():
    """Scoped to the small, explicit _BORE_HOLE_RADIUS_FIELD map -- an op
    with no corresponding entry (e.g. the base shape itself) is passed
    through unchanged, never guessed at."""
    from program_model import _override_bore_hole_dimensions

    args = {"radius": 20.0, "height": 30.0, "axis": "z", "center": [0, 0, 0]}
    overridden = _override_bore_hole_dimensions("add_cylinder", args, "a cylinder with a 12mm diameter bore")
    assert overridden is args


def test_override_gear_dimensions_matches_every_field_by_its_own_label():
    """Confirmed real, severe gap (mission-brief audit, live-reproduced
    against the actual trained bundle): _override_primary_shape_dimensions
    never fires for a gear at all (module/tooth_count have no "mm" suffix,
    so len(literals) < len(numeric_fields)), and training's own
    _render_gear (data/prompt_dataset.py) shuffles clause order, so
    position wouldn't be safe even if it did fire. Every one of the 4
    stated numbers must land exactly, regardless of order."""
    from program_model import _override_gear_dimensions

    text = "a gear, module 2, 20 teeth, 10mm face width, with a 6mm bore"
    args = {"module_mm": 2.54, "tooth_count": 21.0, "face_width_mm": 12.29, "bore_diameter_mm": 7.46, "pressure_angle_deg": 20.0, "center": [0, 0, 0]}
    overridden = _override_gear_dimensions("add_gear_profile", args, text)
    assert overridden["module_mm"] == 2.0
    assert overridden["tooth_count"] == 20
    assert overridden["face_width_mm"] == 10.0
    assert overridden["bore_diameter_mm"] == 6.0
    assert overridden["pressure_angle_deg"] == 20.0  # no claim in the text -- regressor's own value kept


def test_override_gear_dimensions_is_order_independent():
    """The exact scenario _override_primary_shape_dimensions cannot
    handle: training shuffles clause order, so "20 teeth" can appear
    before OR after "module 2" in real text -- label-based matching
    must get both right regardless."""
    from program_model import _override_gear_dimensions

    text = "spur gear, 6mm bore, 20 teeth, 10mm face width, module 2"
    args = {"module_mm": 1.0, "tooth_count": 1.0, "face_width_mm": 1.0, "bore_diameter_mm": 1.0, "pressure_angle_deg": 20.0, "center": [0, 0, 0]}
    overridden = _override_gear_dimensions("add_gear_profile", args, text)
    assert overridden["module_mm"] == 2.0
    assert overridden["tooth_count"] == 20
    assert overridden["face_width_mm"] == 10.0
    assert overridden["bore_diameter_mm"] == 6.0


def test_override_gear_dimensions_does_not_apply_to_other_ops():
    from program_model import _override_gear_dimensions

    args = {"radius": 20.0, "height": 30.0, "axis": "z", "center": [0, 0, 0]}
    overridden = _override_gear_dimensions("add_cylinder", args, "module 2, 20 teeth")
    assert overridden is args
