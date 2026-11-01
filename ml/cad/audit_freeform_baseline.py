"""Baseline/regression harness for the 3 targeted fixes (through-hole
phrasing, stepped-shaft precedence, annular inner-diameter validation).

Tests the 5 user-specified "free-form generation quality" prompts PLUS a
few extra regression probes for each fix area, run as single-component
descriptions directly through the real, unmodified build_assembly_from_
points (bypassing llm_decompose.decompose_prompt -- confirmed, in the
prior audit, impractical to run at this scale on this machine: a trivial
16-token completion took 6.6s, and a real decompose_prompt() call did not
complete within 50+ minutes). This still exercises exactly the code the
3 fixes touch (compositional_decomposer.py, spec_validation.py) plus
everything downstream (program_model.py, sandbox.py), using the SAME
dimension-first phrasing style llm_decompose.py's own system prompt
documents producing for each of these requests.

New audit-only artifact, not production code.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from llm_decompose import ActionablePoint

BUNDLE_PATH = Path(__file__).resolve().parent / "models" / "program_model.joblib"

# -- the 5 user-specified prompts, translated into the single-component
# description(s) llm_decompose.py's own system prompt documents it would
# produce for each (see that file's STEP 0/Example 2 for the water bottle
# case specifically) --
FREEFORM_CASES: dict[str, list[ActionablePoint]] = {
    "water_bottle": [
        ActionablePoint(id="body", description="a hollow vessel: a 65mm diameter, 190mm tall body smoothly narrowing to a 25mm diameter, 30mm tall neck over a 25mm shoulder, 2mm walls", position_mm=(0, 0, 0)),
        ActionablePoint(id="cap", description="a cylinder 27mm diameter, 15mm tall", position_mm=(0, 0, 0),
                         relationship=__import__("assembly_plan").Relationship(kind="attached_to_face", target_id="body", face="+z")),
    ],
    "cup": [
        ActionablePoint(id="body", description="a hollow vessel: a 80mm diameter, 95mm tall body smoothly narrowing to a 78mm diameter, 5mm tall neck over a 5mm shoulder, 2mm walls", position_mm=(0, 0, 0)),
    ],
    "mounting_plate_100x60x10": [
        ActionablePoint(id="plate", description="a box 100mm x 60mm x 10mm", position_mm=(0, 0, 0)),
    ],
    "ring_26od_8id": [
        ActionablePoint(id="ring", description="a flat ring 26mm outer diameter, 8mm inner diameter, 10mm tall", position_mm=(0, 0, 0)),
    ],
    "stepped_shaft_two_sections": [
        ActionablePoint(id="shaft", description="a cylinder 20mm diameter, 40mm tall, stepping down to a 14mm diameter, 60mm tall section", position_mm=(0, 0, 0)),
    ],
    # -- extra regression probes for fix 1 (through-hole phrasing variants) --
    "extra_hole_through_it": [
        ActionablePoint(id="box", description="a box 50mm x 50mm x 20mm with a 12.2mm diameter hole through it", position_mm=(0, 0, 0)),
    ],
    "extra_hole_through_its_center": [
        ActionablePoint(id="box", description="a box 50mm x 50mm x 20mm with a 12.2mm diameter hole through its center", position_mm=(0, 0, 0)),
    ],
    "extra_hole_through_the_middle": [
        ActionablePoint(id="box", description="a box 50mm x 50mm x 20mm with a 12.2mm diameter hole through the middle", position_mm=(0, 0, 0)),
    ],
    "extra_central_through_hole": [
        ActionablePoint(id="box", description="a box 50mm x 50mm x 20mm with a central 12.2mm diameter through-hole", position_mm=(0, 0, 0)),
    ],
    "extra_legitimate_hole_pattern_not_broken": [
        ActionablePoint(id="plate", description="a box 120mm x 80mm x 8mm with a corner array of four 6mm diameter holes, each 15mm from a corner", position_mm=(0, 0, 0)),
    ],
    # -- extra regression probes for fix 2 (stepped-shaft variants) --
    "extra_stepped_shaft_old_syntax": [
        ActionablePoint(id="shaft", description="a cylinder 20mm diameter, 40mm long with a concentric stepped section of 14mm diameter and 60mm length", position_mm=(0, 0, 0)),
    ],
    "extra_shoulder_phrasing": [
        ActionablePoint(id="shaft", description="a cylinder 20mm diameter, 40mm tall with a shoulder down to a 14mm diameter, 60mm tall section", position_mm=(0, 0, 0)),
    ],
    "extra_plain_cylinder_not_broken": [
        ActionablePoint(id="cyl", description="a cylinder 20mm diameter, 100mm tall", position_mm=(0, 0, 0)),
    ],
    # -- extra regression probes for fix 3 (annular inner-diameter variants) --
    "extra_bearing_inner_diameter": [
        ActionablePoint(id="bearing", description="a flat ring 35mm outer diameter, 15mm inner diameter, 10mm tall", position_mm=(0, 0, 0)),
    ],
    "extra_washer_inner_diameter_wrong": [
        # Deliberately a GENUINE mismatch (built ring will NOT have a 99mm inner diameter) --
        # must still be correctly flagged as a real failure, never silently passed.
        ActionablePoint(id="washer", description="a flat ring 26mm outer diameter, 99mm inner diameter, 4mm tall", position_mm=(0, 0, 0)),
    ],
}


def _mesh_summary(mesh) -> dict | None:
    if mesh is None:
        return None
    return {
        "watertight": bool(mesh.is_watertight),
        "extents_mm": [float(v) for v in mesh.extents],
    }


def run_one(case_id: str, points: list[ActionablePoint]) -> dict:
    from evaluation.geometry_eval import evaluate_mesh
    from parallel_builder import build_assembly_from_points

    t0 = time.perf_counter()
    result = build_assembly_from_points(list(points), BUNDLE_PATH, max_workers=2)
    elapsed = time.perf_counter() - t0
    points_out = []
    for p in result.points:
        geom = evaluate_mesh(p.mesh) if p.mesh is not None else None
        points_out.append({
            "id": p.point.id, "description": p.point.description, "status": p.status, "error": p.error,
            "resolution_method": p.resolution_method, "program_ops": [c["op"] for c in p.program],
            "mesh": _mesh_summary(p.mesh),
            "dimension_match_rate": p.dimension_match_rate, "unmatched_dimensions_mm": p.unmatched_dimensions_mm,
            "claimed_hole_count": p.claimed_hole_count, "detected_hole_count": p.detected_hole_count,
            "unmatched_hole_diameters_mm": p.unmatched_hole_diameters_mm,
            "detected_hole_diameters_mm": geom.hole_diameters_mm if geom else [],
        })
    return {
        "id": case_id, "completed": True, "elapsed_seconds": elapsed,
        "overall_status": result.overall_status, "points": points_out,
    }


def main(out_path: Path) -> None:
    results = {}
    for case_id, points in FREEFORM_CASES.items():
        print(f"=== {case_id} ===", flush=True)
        try:
            out = run_one(case_id, points)
        except Exception as exc:
            out = {"id": case_id, "completed": False, "error": f"{type(exc).__name__}: {exc}"}
        results[case_id] = out
        print(f"{case_id}: overall_status={out.get('overall_status')}")
    out_path.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    import sys
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else Path("audit_results/freeform_baseline.json"))
