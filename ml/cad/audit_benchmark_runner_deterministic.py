"""Part 8 of the audit, deterministic half -- runs every hand-authored
decomposition in audit_benchmark_deterministic.py through the REAL,
unmodified parallel_builder.build_assembly_from_points (same function
server.py's /api/build calls for the manual-points path), and
independently re-measures the resulting geometry via evaluation.
geometry_eval.evaluate_mesh. No LLM anywhere in this path.

New audit-only artifact. Safe to run directly (single process, no
subprocess isolation needed -- there is no native-crash-prone library in
this path).
"""

from __future__ import annotations

import json
import time
import traceback
from pathlib import Path

BUNDLE_PATH = Path(__file__).resolve().parent / "models" / "program_model.joblib"
OUT_DIR = Path(__file__).resolve().parent / "audit_results" / "deterministic"


def _mesh_summary(mesh) -> dict | None:
    if mesh is None:
        return None
    return {
        "watertight": bool(mesh.is_watertight),
        "volume_mm3": float(mesh.volume) if mesh.is_watertight else None,
        "bounds_mm": [[float(v) for v in row] for row in mesh.bounds],
        "extents_mm": [float(v) for v in mesh.extents],
        "center_mm": [float(v) for v in mesh.bounds.mean(axis=0)],
    }


def run_one(prompt_id: str, points) -> dict:
    from evaluation.geometry_eval import evaluate_mesh
    from parallel_builder import build_assembly_from_points

    t0 = time.perf_counter()
    try:
        result = build_assembly_from_points(list(points), BUNDLE_PATH, max_workers=4)
        elapsed = time.perf_counter() - t0
        points_out = []
        for p in result.points:
            geom = evaluate_mesh(p.mesh) if p.mesh is not None else None
            points_out.append({
                "id": p.point.id, "description": p.point.description,
                "position_mm": list(p.point.position_mm), "status": p.status, "error": p.error,
                "resolution_method": p.resolution_method, "program_ops": [c["op"] for c in p.program],
                "mesh": _mesh_summary(p.mesh),
                "dimension_match_rate": p.dimension_match_rate, "unmatched_dimensions_mm": p.unmatched_dimensions_mm,
                "claimed_hole_count": p.claimed_hole_count, "detected_hole_count": p.detected_hole_count,
                "unmatched_hole_diameters_mm": p.unmatched_hole_diameters_mm,
                "excluded_from_assembly": p.excluded_from_assembly,
                "detected_hole_diameters_mm": geom.hole_diameters_mm if geom else [],
                "detected_hole_axes": geom.hole_axes if geom else [],
                "detected_hole_through": geom.hole_through if geom else [],
                "detected_cylindrical_patch_count": geom.cylindrical_patch_count if geom else None,
            })
        return {
            "id": prompt_id, "mode": "deterministic", "completed": True, "elapsed_seconds": elapsed,
            "overall_status": result.overall_status,
            "collision_warnings": result.collision_warnings, "envelope_warnings": result.envelope_warnings,
            "n_points": len(result.points), "points": points_out,
            "combined_mesh": _mesh_summary(result.mesh),
        }
    except Exception as exc:
        return {
            "id": prompt_id, "mode": "deterministic", "completed": False,
            "elapsed_seconds": time.perf_counter() - t0,
            "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc(),
        }


def main() -> None:
    from audit_benchmark_deterministic import DECOMPOSITIONS

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results = []
    for pid, points in DECOMPOSITIONS.items():
        print(f"=== {pid} ===", flush=True)
        out = run_one(pid, points)
        results.append(out)
        (OUT_DIR / f"{pid}.json").write_text(json.dumps(out, indent=2, default=str))
        print(f"{pid}: completed={out['completed']} status={out.get('overall_status')} elapsed={out['elapsed_seconds']:.1f}s")

    (OUT_DIR / "_summary.json").write_text(json.dumps(results, indent=2, default=str))
    completed = sum(1 for r in results if r["completed"])
    print(f"\n=== {completed}/{len(results)} completed ===")


if __name__ == "__main__":
    main()
