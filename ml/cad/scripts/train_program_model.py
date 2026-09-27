#!/usr/bin/env python3
"""python scripts/train_program_model.py [--out ml/cad/models/program_model.joblib]

Regenerates the (text, program) training corpus (data/program_dataset.py --
family-shaped samples traced through data/program_builders.py, plus the
generic-composition corpus covering tools with no family home) and trains
program_model.py's tool-sequence predictor on it, saving to a stable,
checked-in path instead of the ephemeral /tmp paths used during development.

Run this again whenever sandbox.py's tool roster or the generic-composition
patterns in data/program_dataset.py change -- the bundle does not update
itself.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.program_dataset import sample_program_dataset, write_program_dataset
from program_eval import evaluate_program_model, load_hand_written_program_samples
from program_model import ProgramModelBundle, ProgramModelConfig, predict_program, train_program_model

DEFAULT_OUT = Path(__file__).resolve().parent.parent / "models" / "program_model.joblib"

FAMILY_COUNTS = {"plate": 400, "flange": 400, "bracket": 400, "shaft": 400, "enclosure": 400, "gear": 400}
N_GENERIC = 460  # was 440 before _pattern_shouldered_cylinder (and reworking _pattern_hollow_vessel) added another generic pattern needing coverage


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dataset-dir", type=Path, default=None, help="reuse an existing dataset dir instead of regenerating")
    args = ap.parse_args()

    dataset_root = args.dataset_dir or Path("/tmp/program_dataset_for_training")
    if args.dataset_dir is None:
        print(f"generating dataset at {dataset_root} ...")
        samples = sample_program_dataset(FAMILY_COUNTS, n_generic=N_GENERIC, seed=args.seed)
        manifest = write_program_dataset(dataset_root, samples, seed=args.seed)
        print("counts:", manifest["counts"])

    print("training ...")
    bundle, metrics = train_program_model(dataset_root, ProgramModelConfig(seed=args.seed))
    print(f"trained on {metrics['n_train']} samples")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    bundle.save(args.out)
    print(f"saved to {args.out}")

    print("\n--- honest 3-bucket eval (never averaged) ---")

    def predict_fn(text: str):
        return predict_program(bundle, text)

    from data.program_dataset import read_program_split

    test_samples = read_program_split(dataset_root, "test")
    families_test = [s for s in test_samples if s.source_kind == "family"]
    generic_test = [s for s in test_samples if s.source_kind == "generic_composition"]

    evaluate_program_model(predict_fn, families_test, "synthetic_test", "program_model").print_summary()
    hand_written = load_hand_written_program_samples(Path(__file__).resolve().parent.parent / "data" / "eval_hand_written.jsonl")
    evaluate_program_model(predict_fn, hand_written, "hand_written", "program_model").print_summary()
    if generic_test:
        evaluate_program_model(predict_fn, generic_test, "generic_composition_synthetic", "program_model").print_summary()


if __name__ == "__main__":
    main()
