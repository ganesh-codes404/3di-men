#!/usr/bin/env python3
"""Build a multi-part assembly from a compound prompt.

Usage:
    python ml/cad/scripts/generate_assembly.py \\
        --prompt "a base plate with four mounting brackets, a shaft in the center, and a top cover with ventilation holes" \\
        --out assembly.stl
"""
import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # ml/cad/

from assembly import run_assembly
from nlp.prompt_parser import parse_prompt


def _regex_adapter(text: str) -> tuple[str, dict]:
    result = parse_prompt(text)
    return result.part_type, asdict(result.params)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    result = run_assembly(args.prompt, _regex_adapter)
    if result.mesh is None:
        print("no components decomposed from that prompt", file=sys.stderr)
        return 2

    args.out.parent.mkdir(parents=True, exist_ok=True)
    result.mesh.export(args.out)

    summary = {
        "prompt": args.prompt,
        "components": [
            {"component_id": p.spec.component_id, "instance": p.instance_index, "family": p.spec.family, "clause": p.spec.clause_text}
            for p in result.placed
        ],
        "geometry_reports": {k: {"watertight": v.watertight, "detected_hole_count": v.detected_hole_count} for k, v in result.geometry_reports.items()},
        "overlapping_pairs": list(result.overlapping_pairs),
        "mesh_path": str(args.out),
        "vertex_count": len(result.mesh.vertices),
        "face_count": len(result.mesh.faces),
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
