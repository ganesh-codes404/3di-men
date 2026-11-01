#!/usr/bin/env python3
"""Evaluates an ALREADY-TRAINED program_model.joblib bundle against a fresh
test split, without retraining -- same 3-bucket honest eval
scripts/train_program_model.py runs at the end of training (classifier-only
/ RAG-only / hybrid), just split out so a baseline can be captured before
and after changing predict_program_hybrid's own arbitration logic, with no
retraining cost in between.

python scripts/eval_existing_program_model.py [--bundle ml/cad/models/program_model.joblib] [--seed 0]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.program_dataset import read_program_split, sample_program_dataset, write_program_dataset
from program_eval import evaluate_program_model, load_hand_written_program_samples
from program_model import ProgramModelBundle, predict_program, predict_program_hybrid, predict_program_rag

DEFAULT_BUNDLE = Path(__file__).resolve().parent.parent / "models" / "program_model.joblib"
FAMILY_COUNTS = {"plate": 400, "flange": 400, "bracket": 400, "shaft": 400, "enclosure": 400, "gear": 400}
N_GENERIC = 1080


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dataset-dir", type=Path, default=None)
    args = ap.parse_args()

    bundle = ProgramModelBundle.load(args.bundle)

    dataset_root = args.dataset_dir or Path("/tmp/program_dataset_for_eval_only")
    if args.dataset_dir is None:
        samples = sample_program_dataset(FAMILY_COUNTS, n_generic=N_GENERIC, seed=args.seed)
        write_program_dataset(dataset_root, samples, seed=args.seed)

    test_samples = read_program_split(dataset_root, "test")
    families_test = [s for s in test_samples if s.source_kind == "family"]
    generic_test = [s for s in test_samples if s.source_kind == "generic_composition"]
    hand_written = load_hand_written_program_samples(Path(__file__).resolve().parent.parent / "data" / "eval_hand_written.jsonl")

    def predict_fn_classifier(text: str):
        return predict_program(bundle, text)

    def predict_fn_rag(text: str):
        return predict_program_rag(bundle, text)

    def predict_fn_hybrid(text: str):
        program, _used_rag, _arbitration = predict_program_hybrid(bundle, text)
        return program

    print("\n--- honest 3-bucket eval: classifier-only (legacy path) ---")
    evaluate_program_model(predict_fn_classifier, families_test, "synthetic_test", "program_model").print_summary()
    evaluate_program_model(predict_fn_classifier, hand_written, "hand_written", "program_model").print_summary()
    evaluate_program_model(predict_fn_classifier, generic_test, "generic_composition_synthetic", "program_model").print_summary()

    print("\n--- honest 3-bucket eval: RAG / retrieval, pure k=5 ---")
    evaluate_program_model(predict_fn_rag, families_test, "synthetic_test", "program_model").print_summary()
    evaluate_program_model(predict_fn_rag, hand_written, "hand_written", "program_model").print_summary()
    evaluate_program_model(predict_fn_rag, generic_test, "generic_composition_synthetic", "program_model").print_summary()

    print("\n--- honest 3-bucket eval: HYBRID (what predict_program_checked uses) ---")
    evaluate_program_model(predict_fn_hybrid, families_test, "synthetic_test", "program_model").print_summary()
    evaluate_program_model(predict_fn_hybrid, hand_written, "hand_written", "program_model").print_summary()
    evaluate_program_model(predict_fn_hybrid, generic_test, "generic_composition_synthetic", "program_model").print_summary()


if __name__ == "__main__":
    main()
