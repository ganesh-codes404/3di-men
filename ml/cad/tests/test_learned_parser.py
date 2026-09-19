from pathlib import Path

import pytest

pytest.importorskip("sklearn")

from data.prompt_dataset import sample_prompt_dataset
from data.prompt_store import write_dataset
from part_families import get_part_family


@pytest.fixture
def tiny_dataset(tmp_path: Path) -> Path:
    samples = sample_prompt_dataset({"plate": 60, "flange": 60}, seed=0)
    write_dataset(tmp_path, samples, seed=0)
    return tmp_path


def test_train_produces_a_bundle_with_reasonable_train_fit(tiny_dataset):
    from parser_model import ParserModelConfig, train_parser_model

    bundle, train_metrics = train_parser_model(tiny_dataset, ParserModelConfig(seed=0))
    assert set(bundle.field_estimators.keys()) == {"plate", "flange"}
    # this is a WIRING smoke test on a tiny fixture, not a real-world accuracy
    # claim -- see parser_model.py's docstring and parser_eval.py for the
    # honest evaluation methodology
    assert train_metrics["n_train"] > 0
    for family_metrics in train_metrics["families"].values():
        assert all(score > 0.3 for score in family_metrics.values())


def test_bundle_save_load_round_trips_to_identical_predictions(tiny_dataset, tmp_path):
    from parser_model import ParserModelConfig, predict, train_parser_model, ParserModelBundle

    bundle, _ = train_parser_model(tiny_dataset, ParserModelConfig(seed=0))
    checkpoint = tmp_path / "bundle.joblib"
    bundle.save(checkpoint)
    loaded = ParserModelBundle.load(checkpoint)

    text = "steel mounting plate 100mm x 60mm with four 8mm holes"
    before = predict(bundle, text)
    after = predict(loaded, text)
    assert before.part_type == after.part_type
    assert before.params == after.params


def test_predictions_always_build_successfully(tiny_dataset):
    """Invariant: whatever the model predicts, the deterministic builder must
    accept it -- a bad regression prediction should never produce an
    unbuildable part."""
    from parser_model import ParserModelConfig, predict, train_parser_model

    bundle, _ = train_parser_model(tiny_dataset, ParserModelConfig(seed=0))
    fixture_texts = [
        "steel mounting plate 100mm x 60mm with four 8mm holes",
        "circular flange, 120mm outer diameter, 40mm bore",
        "a small aluminum bracket",
        "titanium flange with six bolt holes",
        "",  # degenerate input -- must still produce something buildable
    ]
    for text in fixture_texts:
        result = predict(bundle, text)
        spec = get_part_family(result.part_type)
        mesh = spec.builder(result.params)
        assert len(mesh.vertices) > 0
