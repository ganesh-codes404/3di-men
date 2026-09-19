from pathlib import Path

from data.program_dataset import sample_program_dataset
from program_eval import evaluate_program_model, load_hand_written_program_samples
from sandbox import ToolCall


def _identity_predict_fn(samples_by_text: dict):
    def predict(text: str):
        return list(samples_by_text[text].program)

    return predict


def test_three_buckets_stay_structurally_separate_never_averaged():
    synthetic = sample_program_dataset({"plate": 10, "flange": 10}, n_generic=0, seed=0)
    generic = sample_program_dataset({}, n_generic=10, seed=1)

    by_text_synth = {s.text: s for s in synthetic}
    by_text_generic = {s.text: s for s in generic}

    report_a = evaluate_program_model(_identity_predict_fn(by_text_synth), synthetic, "synthetic_test", "test_model")
    report_b = evaluate_program_model(_identity_predict_fn(by_text_generic), generic, "generic_composition_synthetic", "test_model")

    assert report_a.source != report_b.source
    assert report_a.n_samples == len(synthetic)
    assert report_b.n_samples == len(generic)
    # a perfect (identity) predictor must show 100% exact match and all-ok execution
    assert report_a.exact_op_sequence_accuracy == 1.0
    assert report_a.execution_status_counts["ok"] == len(synthetic)


def test_hand_written_program_samples_load_and_are_traceable():
    path = Path(__file__).resolve().parents[1] / "data" / "eval_hand_written.jsonl"
    samples = load_hand_written_program_samples(path)
    assert len(samples) == 24
    for s in samples:
        assert len(s.program) > 0
        assert s.program[-1].op == "finalize"


def test_report_execution_status_counts_sum_to_n_samples():
    synthetic = sample_program_dataset({"plate": 5}, n_generic=0, seed=0)
    by_text = {s.text: s for s in synthetic}
    report = evaluate_program_model(_identity_predict_fn(by_text), synthetic, "synthetic_test", "identity")
    assert sum(report.execution_status_counts.values()) == report.n_samples


def test_malformed_predictor_shows_up_as_rejected_not_hidden():
    synthetic = sample_program_dataset({"plate": 5}, n_generic=0, seed=0)

    def bad_predict(text: str):
        return [ToolCall(op="drill_hole", args={"target": 99, "radius": 4, "depth": 10, "center": [0, 0, 0], "axis": "z"})]

    report = evaluate_program_model(bad_predict, synthetic, "synthetic_test", "bad_model")
    assert report.execution_status_counts["rejected"] == len(synthetic)
    assert report.execution_success_rate == 0.0
