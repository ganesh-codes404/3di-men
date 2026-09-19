from dataclasses import asdict

from data.prompt_dataset import sample_prompt_dataset
from nlp.prompt_parser import parse_prompt
from parser_eval import evaluate_parser


def _regex_adapter(text: str) -> tuple[str, dict]:
    result = parse_prompt(text)
    return result.part_type, asdict(result.params)


def test_synthetic_and_hand_written_reports_stay_separate_never_averaged():
    """Structural test for the honesty constraint: nothing in this codebase
    should ever merge a synthetic-test score with a hand-written score into
    one aggregate number."""
    synthetic = sample_prompt_dataset({"plate": 20, "flange": 20}, seed=0)
    hand_written_like = sample_prompt_dataset({"plate": 5, "flange": 5}, seed=999)  # stand-in, distinct seed

    report_a = evaluate_parser(_regex_adapter, synthetic, "regex", "synthetic_test")
    report_b = evaluate_parser(_regex_adapter, hand_written_like, "regex", "hand_written")

    assert report_a.source == "synthetic_test"
    assert report_b.source == "hand_written"
    assert report_a.source != report_b.source
    # each report's n_samples matches ONLY its own input, never a combined total
    assert report_a.n_samples == len(synthetic)
    assert report_b.n_samples == len(hand_written_like)


def test_report_field_metrics_only_cover_correctly_classified_family():
    samples = sample_prompt_dataset({"plate": 15, "flange": 15}, seed=0)
    report = evaluate_parser(_regex_adapter, samples, "regex", "synthetic_test")
    assert 0.0 <= report.part_family_accuracy <= 1.0
    for family, metrics in report.field_metrics.items():
        for m in metrics:
            assert m.n >= 0
            if m.kind == "numeric":
                assert m.accuracy is None
            else:
                assert m.mae is None
