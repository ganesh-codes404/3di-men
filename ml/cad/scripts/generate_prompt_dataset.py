#!/usr/bin/env python3
"""Generate the synthetic prompt-understanding dataset (Phase 3).

Usage:
    python ml/cad/scripts/generate_prompt_dataset.py --out data/cad_prompts_v1 \\
        --n-per-family 4000 --seed 0
    # or per-family counts:
    python ml/cad/scripts/generate_prompt_dataset.py --out data/cad_prompts_v1 \\
        --n plate=4000 --n flange=4000 --n gear=1000 --seed 0
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # ml/cad/

from data.prompt_dataset import sample_prompt_dataset
from data.prompt_store import write_dataset
from part_families import list_part_families


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n-per-family", type=int, default=4000, help="default count for every registered family")
    ap.add_argument(
        "--n", action="append", default=[], metavar="FAMILY=COUNT",
        help="override the count for one family, e.g. --n gear=1000 (repeatable)",
    )
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    counts = {family: args.n_per_family for family in list_part_families()}
    for entry in args.n:
        family, _, count = entry.partition("=")
        counts[family] = int(count)

    samples = sample_prompt_dataset(counts, seed=args.seed)
    manifest = write_dataset(args.out, samples, seed=args.seed)

    print(f"wrote {len(samples)} samples ({counts}) to {args.out}")
    print(json.dumps(manifest.to_dict(), indent=2))


if __name__ == "__main__":
    main()
