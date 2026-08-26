#!/usr/bin/env python3
"""Generate the v1 procedural mounting-plate dataset.

Usage:
    python ml/cad/scripts/generate_dataset.py --out data/cad_v1 --n 200
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # ml/cad/

from data.procedural_generator import generate_dataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True, help="output directory")
    ap.add_argument("--n", type=int, default=100, help="number of samples")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    manifest = generate_dataset(args.out, args.n, seed=args.seed)
    print(f"wrote {len(manifest)} samples to {args.out}")
    print(f"example description: {manifest[0]['description']}")


if __name__ == "__main__":
    main()
