#!/usr/bin/env python3
"""Run the CAD geometry evaluator over a directory of meshes and print/save a report.

Usage:
    python ml/cad/scripts/run_eval.py --meshes data/cad_v1 --out data/cad_v1/eval_report.json

If a matching <stem>.json params file exists next to a mesh (as produced by
generate_dataset.py), this also cross-checks detected hole count against the
known ground truth, which is the self-consistency check described in
docs/cad-finetuning-technical-report.md Section 8.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # ml/cad/

import trimesh
from evaluation.geometry_eval import evaluate_mesh


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meshes", type=Path, required=True, help="directory of .stl/.obj/.glb files")
    ap.add_argument("--out", type=Path, default=None, help="where to write the JSON report")
    args = ap.parse_args()

    mesh_paths = sorted(
        p for p in args.meshes.iterdir() if p.suffix.lower() in (".stl", ".obj", ".glb", ".ply")
    )
    if not mesh_paths:
        print(f"no meshes found in {args.meshes}")
        return

    reports = []
    ground_truth_matches = 0
    ground_truth_checked = 0
    for mesh_path in mesh_paths:
        mesh = trimesh.load(mesh_path, force="mesh")
        report = evaluate_mesh(mesh)
        entry = {"mesh": mesh_path.name, **report.as_dict()}

        params_path = mesh_path.with_suffix(".json")
        if params_path.exists():
            params = json.loads(params_path.read_text())["params"]
            expected_holes = params["hole_count"]
            ground_truth_checked += 1
            ok = report.detected_hole_count == expected_holes
            ground_truth_matches += int(ok)
            entry["expected_hole_count"] = expected_holes
            entry["hole_count_match"] = ok

        reports.append(entry)
        print(
            f"{mesh_path.name}: watertight={report.watertight} "
            f"holes_detected={report.detected_hole_count} "
            f"planar_frac={report.planar_face_fraction:.2f} "
            f"symmetry={report.mirror_symmetry_score:.2f}"
        )

    if ground_truth_checked:
        print(
            f"\nhole-count ground-truth match: {ground_truth_matches}/{ground_truth_checked} "
            f"({100 * ground_truth_matches / ground_truth_checked:.0f}%)"
        )

    if args.out:
        args.out.write_text(json.dumps(reports, indent=2))
        print(f"wrote report to {args.out}")


if __name__ == "__main__":
    main()
