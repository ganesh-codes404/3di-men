#!/usr/bin/env python3
"""python inference.py --prompt "..." --out out.stl

Local, GPU-free prototype pipeline: prompt -> rule-based CAD-intent parser ->
parametric mesh generator -> mesh file. No Hunyuan3D, no cloud, no CUDA --
per the project owner's direction, this prototype must run entirely on a
laptop with no GPU and stay light. See inference/prompt_parser.py for why this
is rule-based rather than a fine-tuned model, and
docs/cad-finetuning-technical-report.md Section 10 for the decision record.
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from data.procedural_generator import build_flange, build_mounting_plate
from nlp.prompt_parser import parse_prompt

_BUILDERS = {"plate": build_mounting_plate, "flange": build_flange}


def generate_from_prompt(prompt: str):
    """Returns (trimesh.Trimesh, info_dict). info_dict records what was parsed
    from the prompt vs. defaulted, and the wall-clock time taken."""
    t0 = time.perf_counter()
    result = parse_prompt(prompt)
    mesh = _BUILDERS[result.part_type](result.params)
    elapsed = time.perf_counter() - t0

    info = {
        "prompt": prompt,
        "part_type": result.part_type,
        "part_type_inferred": result.part_type_inferred,
        "resolved_description": result.params.to_description(),
        "parsed_fields": sorted(result.parsed_fields),
        "defaulted_fields": sorted(result.defaulted_fields),
        "generation_time_seconds": elapsed,
    }
    return mesh, info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    mesh, info = generate_from_prompt(args.prompt)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(args.out)
    info["mesh_path"] = str(args.out)
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
