#!/usr/bin/env python3
"""Generate the v1 procedural mesh dataset -- now covers every registered
part family (plate AND flange), not just mounting plates.

Usage:
    python ml/cad/scripts/generate_dataset.py --out data/cad_v1 --n 200
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # ml/cad/

from part_families import list_part_families, get_part_family
from data.procedural_generator import generate_dataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True, help="output directory")
    ap.add_argument("--n", type=int, default=100, help="number of samples PER part family")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    # generate_dataset() writes its own per-call manifest.json into --out;
    # since we call it once per family into the same directory, combine into
    # one manifest ourselves afterward rather than letting the last call's
    # manifest silently overwrite the earlier families' entries.
    combined_manifest = []
    for i, family in enumerate(list_part_families()):
        spec = get_part_family(family)
        manifest = generate_dataset(
            args.out, args.n, seed=args.seed + i,
            sampler=spec.sampler, builder=spec.builder, id_prefix=family,
        )
        combined_manifest.extend(manifest)

    (args.out / "manifest.json").write_text(json.dumps(combined_manifest, indent=2))
    print(f"wrote {len(combined_manifest)} samples across {list_part_families()} to {args.out}")
    print(f"example description: {combined_manifest[0]['description']}")


if __name__ == "__main__":
    main()
