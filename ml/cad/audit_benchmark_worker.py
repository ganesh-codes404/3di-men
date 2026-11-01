"""Part 8 of the audit -- runs ONE prompt through the REAL production
pipeline (parallel_builder.build_assembly_from_prompt -> llm_decompose.
decompose_prompt -> program_model/sandbox, the exact path server.py's
/api/build hits) and writes a JSON result file.

Invoked as its OWN fresh subprocess per prompt (see audit_benchmark_
driver.py), never imported directly into a long-lived process -- this
codebase's current decompose_prompt() has no crash isolation (that was
reverted along with everything else from this session), and a native
GGML crash inside llama-cpp-python kills the whole process outright.
Running one prompt per subprocess means a crash only costs that one
prompt's own result, never the rest of the batch.

New audit-only artifact. Never imported by server.py or any production
module; this file imports the production pipeline, not the reverse.
"""

from __future__ import annotations

import json
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

BUNDLE_PATH = Path(__file__).resolve().parent / "models" / "program_model.joblib"
OUT_DIR = Path(__file__).resolve().parent / "audit_results" / "full_llm"


def _mesh_summary(mesh) -> dict | None:
    if mesh is None:
        return None
    return {
        "watertight": bool(mesh.is_watertight),
        "volume_mm3": float(mesh.volume) if mesh.is_watertight else None,
        "bounds_mm": [[float(v) for v in row] for row in mesh.bounds],
        "extents_mm": [float(v) for v in mesh.extents],
        "face_count": int(len(mesh.faces)),
    }


def run(prompt_id: str) -> None:
    from audit_benchmark_prompts import PROMPTS

    bp = next((p for p in PROMPTS if p.id == prompt_id), None)
    if bp is None:
        raise SystemExit(f"no such prompt id {prompt_id!r}")

    from evaluation.geometry_eval import evaluate_mesh
    from parallel_builder import build_assembly_from_prompt

    t0 = time.perf_counter()
    try:
        result = build_assembly_from_prompt(bp.prompt, BUNDLE_PATH, max_workers=2)
        elapsed = time.perf_counter() - t0

        points_out = []
        for p in result.points:
            geom = evaluate_mesh(p.mesh) if p.mesh is not None else None
            points_out.append({
                "description": p.point.description,
                "position_mm": list(p.point.position_mm),
                "status": p.status,
                "error": p.error,
                "resolution_method": p.resolution_method,
                "nearest_neighbor_similarity": p.nearest_neighbor_similarity,
                "program_ops": [c["op"] for c in p.program],
                "mesh": _mesh_summary(p.mesh),
                "dimension_match_rate": p.dimension_match_rate,
                "unmatched_dimensions_mm": p.unmatched_dimensions_mm,
                "claimed_hole_count": p.claimed_hole_count,
                "detected_hole_count": p.detected_hole_count,
                "unmatched_hole_diameters_mm": p.unmatched_hole_diameters_mm,
                "excluded_from_assembly": p.excluded_from_assembly,
                "detected_hole_diameters_mm": geom.hole_diameters_mm if geom else [],
                "detected_hole_axes": geom.hole_axes if geom else [],
                "detected_hole_through": geom.hole_through if geom else [],
            })

        out = {
            "id": bp.id, "category": bp.category, "prompt": bp.prompt, "mode": "full_llm",
            "completed": True, "elapsed_seconds": elapsed,
            "overall_status": result.overall_status,
            "fell_back_to_single_point": result.fell_back_to_single_point,
            "n_points": len(result.points),
            "collision_warnings": result.collision_warnings,
            "envelope_warnings": result.envelope_warnings,
            "points": points_out,
            "combined_mesh": _mesh_summary(result.mesh),
            "raw_llm_output": result.raw_llm_output,
        }
    except Exception as exc:
        elapsed = time.perf_counter() - t0
        out = {
            "id": bp.id, "category": bp.category, "prompt": bp.prompt, "mode": "full_llm",
            "completed": False, "elapsed_seconds": elapsed,
            "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc(),
        }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / f"{bp.id}.json").write_text(json.dumps(out, indent=2, default=str))
    print(f"wrote {bp.id}.json completed={out['completed']} elapsed={out['elapsed_seconds']:.1f}s")


if __name__ == "__main__":
    run(sys.argv[1])
