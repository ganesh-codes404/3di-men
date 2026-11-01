#!/usr/bin/env python3
"""HISTORICAL / DIAGNOSTIC ONLY as of op_arbitration.py's requirement-aware
arbitration engine: predict_program_hybrid no longer consults
ProgramModelConfig.rag_override_similarity_threshold at all (see that
field's own now-updated docstring), so sweeping it here will print 5
IDENTICAL result blocks -- that is itself the confirmation that similarity-
threshold tuning is dead, not a bug in this script. Kept because it's the
real measurement that originally proved a single global similarity floor
could never cleanly separate a harmful RAG override from a helpful RAG
rescue (see ProgramModelConfig.rag_override_similarity_threshold's
docstring for the actual numbers), which is exactly the finding that
motivated building a requirement-aware arbitration engine instead.

Sweeps ProgramModelConfig.rag_override_similarity_threshold over a few
candidate values against the SAME 3-bucket honest eval (classifier/RAG/
hybrid, never averaged), on the ALREADY-TRAINED program_model.joblib
bundle -- no retraining needed.

python scripts/sweep_rag_override_threshold.py [--bundle ...] [--seed 0]
"""
import argparse
import dataclasses
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.program_dataset import read_program_split, sample_program_dataset, write_program_dataset
from program_eval import evaluate_program_model, load_hand_written_program_samples
from program_model import ProgramModelBundle, predict_program_hybrid

DEFAULT_BUNDLE = Path(__file__).resolve().parent.parent / "models" / "program_model.joblib"
FAMILY_COUNTS = {"plate": 400, "flange": 400, "bracket": 400, "shaft": 400, "enclosure": 400, "gear": 400}
N_GENERIC = 1080
CANDIDATE_THRESHOLDS = [0.20, 0.30, 0.40, 0.50, 0.60]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dataset-dir", type=Path, default=None)
    args = ap.parse_args()

    base_bundle = ProgramModelBundle.load(args.bundle)

    dataset_root = args.dataset_dir or Path("/tmp/program_dataset_for_eval_only")
    if args.dataset_dir is None and not dataset_root.exists():
        samples = sample_program_dataset(FAMILY_COUNTS, n_generic=N_GENERIC, seed=args.seed)
        write_program_dataset(dataset_root, samples, seed=args.seed)

    test_samples = read_program_split(dataset_root, "test")
    families_test = [s for s in test_samples if s.source_kind == "family"]
    generic_test = [s for s in test_samples if s.source_kind == "generic_composition"]
    hand_written = load_hand_written_program_samples(Path(__file__).resolve().parent.parent / "data" / "eval_hand_written.jsonl")

    for threshold in CANDIDATE_THRESHOLDS:
        bundle = dataclasses.replace(base_bundle, config=dataclasses.replace(base_bundle.config, rag_override_similarity_threshold=threshold))

        def predict_fn_hybrid(text: str, _bundle=bundle):
            program, _used_rag, _arbitration = predict_program_hybrid(_bundle, text)
            return program

        print(f"\n=== rag_override_similarity_threshold = {threshold:.2f} ===")
        evaluate_program_model(predict_fn_hybrid, families_test, "synthetic_test", "program_model").print_summary()
        evaluate_program_model(predict_fn_hybrid, hand_written, "hand_written", "program_model").print_summary()
        evaluate_program_model(predict_fn_hybrid, generic_test, "generic_composition_synthetic", "program_model").print_summary()


if __name__ == "__main__":
    main()
