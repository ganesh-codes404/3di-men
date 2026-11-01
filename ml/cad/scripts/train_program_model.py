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
from program_model import ProgramModelBundle, ProgramModelConfig, predict_program, predict_program_hybrid, predict_program_rag, train_program_model

DEFAULT_OUT = Path(__file__).resolve().parent.parent / "models" / "program_model.joblib"

FAMILY_COUNTS = {"plate": 400, "flange": 400, "bracket": 400, "shaft": 400, "enclosure": 400, "gear": 400}
N_GENERIC = 1080  # was 920; bumped for 8 new generic patterns (add_loft/add_sweep-arc/add_sweep-helix/
                 # add_tube-straight/add_tube-arc/smooth_subdivide/add_wireframe_from_edges/weld_vertices)
                 # added after confirming (by grep, not assumption) these 6 tools had ZERO generic-
                 # composition training examples despite being registered in sandbox.py/ARG_SCHEMA --
                 # scaled proportionally to _GENERIC_PATTERNS' new slot count (49 -> 57) to keep roughly
                 # the same per-pattern sample density as before, same reasoning as the 900->920 bump above


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

    from data.program_dataset import read_program_split

    test_samples = read_program_split(dataset_root, "test")
    families_test = [s for s in test_samples if s.source_kind == "family"]
    generic_test = [s for s in test_samples if s.source_kind == "generic_composition"]
    hand_written = load_hand_written_program_samples(Path(__file__).resolve().parent.parent / "data" / "eval_hand_written.jsonl")

    # Hybrid (agreement-gated classifier/RAG arbitration) is what
    # predict_program_checked() -- and so server.py/parallel_builder.py --
    # actually uses by default now; the other two paths are kept and
    # reported SEPARATELY, never averaged in, as a standing before/after
    # comparison, not nostalgia.
    def predict_fn_classifier(text: str):
        return predict_program(bundle, text)

    def predict_fn_rag(text: str):
        return predict_program_rag(bundle, text)

    def predict_fn_hybrid(text: str):
        program, _used_rag = predict_program_hybrid(bundle, text)
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
