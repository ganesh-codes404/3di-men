#!/usr/bin/env python3
"""Train the learned (scikit-learn) prompt parser -- Phase 4.

Usage:
    python ml/cad/scripts/train_parser_model.py --dataset data/cad_prompts_v1 \\
        --out models/parser_v1.joblib
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # ml/cad/


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    try:
        from parser_model import ParserModelConfig, train_parser_model
    except ImportError as exc:
        print(f"the learned parser needs scikit-learn + joblib: pip install scikit-learn joblib ({exc})", file=sys.stderr)
        return 2

    bundle, train_metrics = train_parser_model(args.dataset, ParserModelConfig(seed=args.seed))
    bundle.save(args.out)
    print(f"saved model to {args.out}")
    print(json.dumps(train_metrics, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
