#!/usr/bin/env python3
"""python scripts/benchmark_pipeline.py [--limit N] [--conditions baseline,rag] [--out report.json]

Section 12's evaluation benchmark: measures actual generated geometry
(not reasoning text) across a leveled prompt suite (Level 1 primitives
through Level 5 unfamiliar tool combinations, per section 12), comparing
the pipeline with the knowledge/RAG subsystem disabled ("baseline") against
enabled ("rag") -- items 1 and 2 of section 12's 4-way comparison. Items 3
(structured engineering rules) and 4 (feedback-driven repair) are NOT
implemented this session and are honestly left out of the comparison
rather than faked as "no difference."

Runs each (prompt, condition) pair in its OWN subprocess: KnowledgeConfig's
`enabled` field reads the KNOWLEDGE_ENABLED environment variable once at
class-definition time (matching llm_decompose.py's own MODEL_NAME/
MAX_CONTEXT convention -- env vars are read once at process start
throughout this project, not polled per-call), so a fresh interpreter per
condition is the correct way to actually get a different value, rather
than adding a config-override parameter to decompose_prompt purely for
this script's benefit.

HONEST COST NOTE, confirmed this session, not theoretical: this project's
local LLM decomposition step runs at roughly 0.1-0.4 tokens/sec on this
machine's hardware (2015 dual-core 1.8GHz CPU, no GPU offload configured
in the shipped default -- see llm_decompose.py's MAX_CONTEXT comment and
this session's own hardware diagnosis). A full run of this benchmark's
default ~20-prompt suite across 2 conditions is 40 real LLM calls, each
potentially 1-3+ minutes even with GPU offload enabled manually (not the
shipped default) -- a full sweep can run from tens of minutes to several
hours depending on hardware. Use --limit for a quick smoke run; run the
full suite in the background or on faster hardware when you actually need
the complete comparison.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

CAD_ROOT = Path(__file__).resolve().parent.parent
BUNDLE_PATH = CAD_ROOT / "models" / "program_model.joblib"

# Section 12's 5 difficulty levels. At least 20 prompts (the spec's stated
# minimum); deliberately includes Level 5 prompts that combine operations
# in ways no single hardcoded part-family recipe produces, so generalization
# isn't measured only on the examples that trained the tool rules.
PROMPT_SUITE: list[tuple[str, str]] = [
    # Level 1 -- basic primitives
    ("L1", "a cylinder 40mm diameter, 100mm tall"),
    ("L1", "a rectangular box 80mm x 50mm x 20mm with rounded edges"),
    ("L1", "a hollow cylindrical container, 60mm diameter, 100mm tall, 3mm walls"),
    # Level 2 -- product forms
    ("L2", "a reusable water bottle with a screw cap"),
    ("L2", "a mug with a handle"),
    ("L2", "a simple enclosure with a lid"),
    # Level 3 -- mechanical components
    ("L3", "a mounting plate with four holes"),
    ("L3", "a flange with evenly spaced bolt holes"),
    ("L3", "a shaft with a stepped diameter"),
    ("L3", "a bracket with a base and an upright section"),
    # Level 4 -- multi-component assemblies
    ("L4", "a simplified pump housing"),
    ("L4", "a box enclosure with mounting bosses"),
    ("L4", "a base plate with a shaft on top and a gear next to it"),
    # Level 5 -- unfamiliar combinations (not a dedicated hardcoded recipe)
    ("L5", "a tube swept along a helix with a flange on one end"),
    ("L5", "a lofted shape from a rectangle to an ellipse with four mounting holes in the rectangular base"),
    ("L5", "an L-shape bracket with a counterbore hole pattern and a knurled grip ring on its top face"),
    ("L5", "a shelled hexagonal prism with a boss on one face and a hole through the boss"),
    ("L5", "two gears meshing on parallel shafts, each shaft seated in its own bracket"),
    ("L5", "a ring of grip ridges around a tapered cone with a hex socket cut into the top"),
    ("L5", "a bracket with a rib bridging its inside corner and a bolt circle of six holes"),
]

_WORKER_SCRIPT = r"""
import json, sys, time
sys.path.insert(0, {cad_root!r})
from parallel_builder import build_assembly_from_prompt

t0 = time.time()
try:
    result = build_assembly_from_prompt({prompt!r}, {bundle_path!r}, max_workers=2)
    elapsed = time.time() - t0
    points = []
    for p in result.points:
        points.append({{
            "status": p.status,
            "watertight": (p.topology or {{}}).get("is_watertight") if p.topology else None,
            "component_count": (p.topology or {{}}).get("component_count") if p.topology else None,
        }})
    out = {{
        "ok": True,
        "elapsed_seconds": elapsed,
        "fell_back_to_single_point": result.fell_back_to_single_point,
        "knowledge_context_chars": len(result.knowledge_context or ""),
        "n_points": len(result.points),
        "points": points,
        "mesh_produced": result.mesh is not None,
    }}
except Exception as exc:
    out = {{"ok": False, "elapsed_seconds": time.time() - t0, "error": f"{{type(exc).__name__}}: {{exc}}"}}
print(json.dumps(out))
"""


@dataclass
class ConditionResult:
    level: str
    prompt: str
    condition: str
    ok: bool
    elapsed_seconds: float
    fell_back_to_single_point: bool | None = None
    knowledge_context_chars: int = 0
    n_points: int = 0
    all_points_ok: bool = False
    all_watertight: bool = False
    mesh_produced: bool = False
    error: str | None = None


def _run_one(prompt: str, condition: str, model_path_env: dict[str, str]) -> dict:
    script = _WORKER_SCRIPT.format(cad_root=str(CAD_ROOT), prompt=prompt, bundle_path=str(BUNDLE_PATH))
    env = dict(os.environ)
    env.update(model_path_env)
    env["KNOWLEDGE_ENABLED"] = "1" if condition == "rag" else "0"
    proc = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, cwd=str(CAD_ROOT), env=env)
    if proc.returncode != 0 or not proc.stdout.strip():
        return {"ok": False, "elapsed_seconds": 0.0, "error": f"subprocess failed: {proc.stderr[-2000:]}"}
    try:
        return json.loads(proc.stdout.strip().splitlines()[-1])
    except json.JSONDecodeError:
        return {"ok": False, "elapsed_seconds": 0.0, "error": f"could not parse worker output: {proc.stdout[-500:]}"}


def run_benchmark(limit: int | None, conditions: list[str], gpu_offload: bool) -> list[ConditionResult]:
    model_path_env: dict[str, str] = {}
    if gpu_offload:
        # NOT the shipped default (see llm_decompose.py's N_GPU_LAYERS,
        # which defaults to 0) -- an explicit opt-in purely so this
        # benchmark completes in reasonable time on hardware with a usable
        # GPU. N_GPU_LAYERS/N_THREADS are read directly by
        # llm_decompose.py's _get_llm, same as every other env-var config
        # in that file.
        model_path_env["N_GPU_LAYERS"] = "-1"
        model_path_env["N_THREADS"] = "2"

    suite = PROMPT_SUITE[:limit] if limit else PROMPT_SUITE
    results: list[ConditionResult] = []
    total = len(suite) * len(conditions)
    done = 0
    for level, prompt in suite:
        for condition in conditions:
            done += 1
            print(f"[{done}/{total}] ({condition}, {level}) {prompt!r} ...", flush=True)
            raw = _run_one(prompt, condition, model_path_env)
            if raw.get("ok"):
                points = raw.get("points", [])
                all_ok = bool(points) and all(p["status"] == "ok" for p in points)
                all_wt = bool(points) and all(p.get("watertight") for p in points)
                results.append(ConditionResult(
                    level=level, prompt=prompt, condition=condition, ok=True,
                    elapsed_seconds=raw["elapsed_seconds"],
                    fell_back_to_single_point=raw.get("fell_back_to_single_point"),
                    knowledge_context_chars=raw.get("knowledge_context_chars", 0),
                    n_points=raw.get("n_points", 0), all_points_ok=all_ok, all_watertight=all_wt,
                    mesh_produced=raw.get("mesh_produced", False),
                ))
            else:
                results.append(ConditionResult(
                    level=level, prompt=prompt, condition=condition, ok=False,
                    elapsed_seconds=raw.get("elapsed_seconds", 0.0), error=raw.get("error"),
                ))
            r = results[-1]
            print(f"    -> ok={r.ok} elapsed={r.elapsed_seconds:.1f}s points={r.n_points} "
                  f"all_ok={r.all_points_ok} watertight={r.all_watertight} "
                  f"knowledge_chars={r.knowledge_context_chars}", flush=True)
    return results


def summarize(results: list[ConditionResult]) -> dict:
    by_condition: dict[str, list[ConditionResult]] = {}
    for r in results:
        by_condition.setdefault(r.condition, []).append(r)

    summary = {}
    for condition, rows in by_condition.items():
        n = len(rows)
        summary[condition] = {
            "n_prompts": n,
            "subprocess_success_rate": sum(1 for r in rows if r.ok) / n if n else 0.0,
            "all_points_ok_rate": sum(1 for r in rows if r.all_points_ok) / n if n else 0.0,
            "all_watertight_rate": sum(1 for r in rows if r.all_watertight) / n if n else 0.0,
            "fallback_rate": sum(1 for r in rows if r.fell_back_to_single_point) / n if n else 0.0,
            "mean_latency_seconds": sum(r.elapsed_seconds for r in rows) / n if n else 0.0,
            "mean_knowledge_context_chars": sum(r.knowledge_context_chars for r in rows) / n if n else 0.0,
            "prompts_with_knowledge_context_used": sum(1 for r in rows if r.knowledge_context_chars > 0),
        }
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="only run the first N prompts from the suite")
    ap.add_argument("--conditions", default="baseline,rag", help="comma-separated: baseline,rag")
    ap.add_argument("--gpu-offload", action="store_true",
                     help="set N_GPU_LAYERS=-1/N_THREADS=2 in worker subprocesses -- NOT the project default, "
                          "only safe on hardware with a usable GPU for this model's size (see llm_decompose.py)")
    ap.add_argument("--out", type=Path, default=CAD_ROOT / "eval_report_pipeline_benchmark.json")
    args = ap.parse_args()

    if not BUNDLE_PATH.exists():
        print(f"No trained model at {BUNDLE_PATH} -- run scripts/train_program_model.py first.")
        raise SystemExit(1)

    conditions = [c.strip() for c in args.conditions.split(",") if c.strip()]
    t0 = time.time()
    results = run_benchmark(args.limit, conditions, gpu_offload=args.gpu_offload)
    elapsed = time.time() - t0

    summary = summarize(results)
    print("\n=== Summary ===")
    print(json.dumps(summary, indent=2))
    print(f"\nTotal benchmark wall time: {elapsed:.1f}s")

    report = {"summary": summary, "results": [asdict(r) for r in results], "total_wall_seconds": elapsed}
    args.out.write_text(json.dumps(report, indent=2))
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
