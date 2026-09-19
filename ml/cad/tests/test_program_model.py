from pathlib import Path

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
