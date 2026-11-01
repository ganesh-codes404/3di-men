"""Reads a benchmark_runner.py run's saved artifacts (benchmark_runs/
<run_id>/_summary.json) and produces Part 14's own required summary
statistics + Part 11's per-axis scoring breakdown + Part 12's failure-
classification counts, broken down by category. Never recomputes
anything from the raw mesh/validators itself -- purely an aggregation
over what run_benchmark() already measured and saved, so re-running
this script against the same run_id is free and instant."""

from __future__ import annotations

import json
import statistics
from collections import Counter
from pathlib import Path


def load_run(run_dir: Path) -> list[dict]:
    return json.loads((run_dir / "_summary.json").read_text())


def summarize(results: list[dict]) -> dict:
    total = len(results)
    completed = [r for r in results if r["completed"]]
    not_completed = [r for r in results if not r["completed"]]

    def axis_rate(axis: str) -> "float | None":
        vals = [r["scores"].get(axis) for r in completed if r["scores"].get(axis) is not None]
        return (sum(vals) / len(vals)) if vals else None

    times = [r["elapsed_seconds"] for r in results]
    label_counts = Counter(label for r in results for label in r["failure_labels"])

    by_category: dict[str, dict] = {}
    for cat in sorted({r["category"] for r in results}):
        cat_results = [r for r in results if r["category"] == cat]
        cat_completed = [r for r in cat_results if r["completed"]]
        by_category[cat] = {
            "total": len(cat_results),
            "completed": len(cat_completed),
            "overall_pass": sum(1 for r in cat_completed if r.get("overall_status") == "PASS"),
            "avg_component_completeness": axis_rate_for(cat_completed, "component_completeness"),
            "avg_dimensional_accuracy": axis_rate_for(cat_completed, "dimensional_accuracy"),
            "avg_feature_completeness": axis_rate_for(cat_completed, "feature_completeness"),
            "avg_assembly_compatibility": axis_rate_for(cat_completed, "assembly_compatibility"),
            "avg_geometry_validity": axis_rate_for(cat_completed, "geometry_validity"),
        }

    slowest = sorted(results, key=lambda r: r["elapsed_seconds"], reverse=True)[:5]

    # Phase 8 Part D's own new clarification metrics -- directly OBSERVED
    # counts from clarification.assess_build()'s own already-computed
    # result for each prompt, never a second analysis pass. Deliberately
    # does NOT report precision/recall of asking against independent
    # ground truth for "was a clarification necessary" -- that requires a
    # dedicated, honest hand-labeling pass per prompt this run does not
    # include (see the Phase 8 final report's own disclosure on this).
    clar_reports = [r["clarification_report"] for r in completed if r.get("clarification_report")]
    build_decision_counts = dict(Counter(cr["build_decision"] for cr in clar_reports))
    total_clarification_requests = sum(len(cr["clarification_requests"]) for cr in clar_reports)
    blocking_clarification_requests = sum(
        sum(1 for req in cr["clarification_requests"] if req["blocking"]) for cr in clar_reports
    )
    total_assumptions_disclosed = sum(len(cr["assumptions"]) for cr in clar_reports)
    total_unsupported_requirements = sum(len(cr["unsupported_requirements"]) for cr in clar_reports)
    prompts_marked_unsupported = sum(1 for cr in clar_reports if cr["build_decision"] == "UNSUPPORTED")
    prompts_needing_clarification = sum(1 for cr in clar_reports if cr["build_decision"] == "NEEDS_CLARIFICATION")

    return {
        "total_prompts": total,
        "completed_generations": len(completed),
        "failed_generations": len(not_completed),
        "generation_success_rate": len(completed) / total if total else 0.0,
        "overall_status_counts": dict(Counter(r.get("overall_status") for r in completed)),
        "component_completeness_rate": axis_rate("component_completeness"),
        "dimensional_compliance_rate": axis_rate("dimensional_accuracy"),
        "feature_completeness_rate": axis_rate("feature_completeness"),
        "assembly_compatibility_rate": axis_rate("assembly_compatibility"),
        "geometry_validity_rate": axis_rate("geometry_validity"),
        "average_generation_time_seconds": statistics.mean(times) if times else 0.0,
        "median_generation_time_seconds": statistics.median(times) if times else 0.0,
        "slowest_prompts": [{"id": r["id"], "elapsed_seconds": r["elapsed_seconds"]} for r in slowest],
        "most_common_failure_labels": label_counts.most_common(),
        "by_category": by_category,
        "not_completed_ids": [r["id"] for r in not_completed],
        "clarification_metrics": {
            "build_decision_counts": build_decision_counts,
            "total_clarification_requests": total_clarification_requests,
            "blocking_clarification_requests": blocking_clarification_requests,
            "total_assumptions_disclosed": total_assumptions_disclosed,
            "total_unsupported_requirements": total_unsupported_requirements,
            "prompts_marked_unsupported": prompts_marked_unsupported,
            "prompts_needing_clarification": prompts_needing_clarification,
            "note": (
                "These are directly observed counts, not precision/recall against independent "
                "ground truth for whether a clarification was actually necessary -- that requires "
                "a dedicated hand-labeling pass per prompt, not computed in this run (see the Phase "
                "8 final report's own disclosure)."
            ),
        },
    }


def axis_rate_for(results: list[dict], axis: str) -> "float | None":
    vals = [r["scores"].get(axis) for r in results if r["scores"].get(axis) is not None]
    return (sum(vals) / len(vals)) if vals else None


if __name__ == "__main__":
    import sys

    run_dir = Path(sys.argv[1])
    results = load_run(run_dir)
    summary = summarize(results)
    print(json.dumps(summary, indent=2))
    (run_dir / "_aggregate_summary.json").write_text(json.dumps(summary, indent=2))
