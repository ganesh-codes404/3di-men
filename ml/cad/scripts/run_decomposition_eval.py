#!/usr/bin/env python3
"""python scripts/run_decomposition_eval.py [--limit N] [--out report.json]

Runs decompose_eval.py's real-model benchmark against data/eval_decomposition.jsonl
-- the missing counterpart to scripts/train_program_model.py's own eval
section, which only ever measured program_model.py, never the LLM
decomposition step that feeds it.

HONEST COST NOTE, matching scripts/benchmark_pipeline.py's own disclosure:
each sample is one real local LLM call, confirmed this session to take on
the order of 10-20+ minutes on this project's actual hardware (less with
manual GPU offload, see llm_decompose.py's N_GPU_LAYERS). Use --limit for
a quick, partial run; the full label set (10 samples as of this writing)
is a multi-hour commitment, not a quick check.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from decompose_eval import load_decomposition_labels, run_decomposition_eval

DEFAULT_LABELS = Path(__file__).resolve().parent.parent / "data" / "eval_decomposition.jsonl"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    ap.add_argument("--limit", type=int, default=None, help="only run the first N labels")
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parent.parent / "eval_report_decomposition.json")
    args = ap.parse_args()

    labels = load_decomposition_labels(args.labels)
    if args.limit:
        labels = labels[: args.limit]

    print(f"Running decomposition eval on {len(labels)} label(s) -- each is a real local LLM call, "
          f"expect this to take a while on this hardware.")
    t0 = time.time()
    summary = run_decomposition_eval(labels)
    elapsed = time.time() - t0

    summary.print_summary()
    print(f"\nTotal wall time: {elapsed:.1f}s")

    report = {
        "pass_rate": summary.pass_rate,
        "fallback_rate": summary.fallback_rate,
        "component_count_accuracy": summary.component_count_accuracy,
        "relationship_accuracy": summary.relationship_accuracy,
        "mean_latency_seconds": summary.mean_latency_seconds,
        "total_wall_seconds": elapsed,
        "results": [
            {
                "id": r.label.id, "prompt": r.label.prompt, "overall_pass": r.overall_pass,
                "actual_component_count": r.actual_component_count,
                "expected_component_count": r.label.expected_component_count,
                "component_count_correct": r.component_count_correct,
                "relationships_found": list(r.relationships_found),
                "relationship_kinds_satisfied": r.relationship_kinds_satisfied,
                "missing_required_substrings": list(r.missing_required_substrings),
                "has_invented_dimensions": r.has_invented_dimensions,
                "fell_back": r.fell_back,
                "elapsed_seconds": r.elapsed_seconds,
                "descriptions": [p.description for p in r.points],
            }
            for r in summary.results
        ],
    }
    args.out.write_text(json.dumps(report, indent=2))
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
