#!/usr/bin/env python3
"""python benchmark.py [--n 20]

Measures end-to-end generation latency for the local, GPU-free prototype
pipeline (prompt -> parser -> parametric mesh -> exported file) across a mix of
structured and unstructured prompts, plus mesh stats. Runs entirely on CPU --
there is no separate "baseline vs. fine-tuned" comparison to make yet, because
per the project owner's direction (docs/cad-finetuning-technical-report.md
Section 10) this prototype doesn't touch Hunyuan3D at all. That comparison
becomes meaningful again once/if the project moves to GPU compute and the v2
neural pipeline in Section 4 gets built.
"""
import argparse
import json
import statistics
import tempfile
from pathlib import Path

from inference import generate_from_prompt

_PROMPTS = [
    "rectangular aluminum mounting plate, 120 mm x 80 mm x 8 mm, with four 10 mm mounting holes, 20 mm from each edge, 5 mm corner fillets",
    "Create a compact mounting bracket with four mounting holes",
    "steel mounting plate 150mm x 100mm x 10mm with two 12mm holes",
    "Generate a cylindrical pump housing with a 40 mm central bore, 8 mm wall thickness, four M8 mounting holes, 120 mm outer diameter",
    "a small aluminum bracket",
    "circular flange, 100 mm outer diameter, with a 25 mm bore and six 8mm bolt holes on a 70mm bolt circle",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=len(_PROMPTS), help="number of prompts to run (cycles through the list)")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    with tempfile.TemporaryDirectory() as tmp:
        times = []
        results = []
        for i in range(args.n):
            prompt = _PROMPTS[i % len(_PROMPTS)]
            mesh, info = generate_from_prompt(prompt)
            mesh.export(Path(tmp) / f"bench_{i}.stl")
            times.append(info["generation_time_seconds"])
            results.append({**info, "face_count": len(mesh.faces), "watertight": bool(mesh.is_watertight)})

        report = {
            "n": args.n,
            "mean_generation_time_seconds": statistics.mean(times),
            "p95_generation_time_seconds": sorted(times)[int(0.95 * (len(times) - 1))],
            "max_generation_time_seconds": max(times),
            "vram_usage": "n/a (CPU-only pipeline, no GPU used)",
            "results": results,
        }
        print(json.dumps({k: v for k, v in report.items() if k != "results"}, indent=2))
        if args.out:
            args.out.write_text(json.dumps(report, indent=2))
            print(f"wrote full report to {args.out}")


if __name__ == "__main__":
    main()
