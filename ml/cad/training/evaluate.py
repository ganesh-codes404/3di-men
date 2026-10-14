#!/usr/bin/env python3
"""python training/evaluate.py --split test --limit 10 [--model-path ...]

Measurable evaluation for the decomposition/clarification LLM task -- never
"does the generated model look cool," per this project's standing rule (see
program_model.py's own module docstring for the same principle applied to
the tool-classifier). Runs against the REAL local llama.cpp/.gguf model via
llm_decompose.py's own decompose_prompt/ask_clarifying_questions -- works
today, on this CPU-only machine, against the base (not yet fine-tuned)
Qwen3-4B, and works identically against a fine-tuned model later (merge the
LoRA adapter, re-quantize to .gguf, point --model-path at it, re-run this
same script for a real before/after comparison -- Phase 9 of the roadmap).

Each test example costs ~10-15 minutes of real inference on this hardware
(see llm_decompose.py's own disclosed latency), so --limit defaults small; a
full pass over the test split is a deliberate, separate, long-running
decision, not this script's default behavior.

Metrics (all structural/executable, not subjective):
- json_valid_rate: did the LLM's raw output parse as a decomposition JSON
  array (decompose) or {"questions": [...]} object (clarify) at all.
- point_count_reasonable: for decompose, did it produce at least 1 point
  and not obviously explode into an ungrounded number of them.
- execution_success_rate: for decompose, each predicted description is fed
  through THIS project's own program_model.py + sandbox.py (the real,
  separate tool-composing model) -- does the resulting program actually
  execute to a valid mesh? This is the closest thing to "is the decomposition
  actually usable," independent of exact-text matching against the
  synthetic ground truth (which the LLM was never expected to reproduce
  verbatim for a paraphrased/novel input).
- clarify_direction_correct: for clarify examples, does "asks >=1 question"
  vs "asks 0 questions" match the ground truth's same boolean -- exact
  question-wording match is not a realistic or even desirable target.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from llm_decompose import DEFAULT_MODEL_PATH, decompose_prompt, ask_clarifying_questions
from program_model import ProgramModelBundle, predict_program_checked
from sandbox import Sandbox

DATA_ROOT = Path(__file__).resolve().parent / "data"


def load_split(split: str) -> list[dict]:
    path = DATA_ROOT / split / "sft.jsonl"
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _is_decompose_record(record: dict) -> bool:
    # the system prompt used at training/data time distinguishes the two tasks
    return "decomposing a request into the separate physical components" in record["messages"][0]["content"]


def evaluate(split: str, limit: int, model_path: Path, bundle_path: Path) -> dict:
    records = load_split(split)
    decompose_records = [r for r in records if _is_decompose_record(r)][:limit]
    clarify_records = [r for r in records if not _is_decompose_record(r)][:limit]

    bundle = ProgramModelBundle.load(bundle_path) if decompose_records else None

    decompose_results = []
    for r in decompose_records:
        user_text = r["messages"][1]["content"]
        t0 = time.time()
        result = decompose_prompt(user_text, model_path=model_path)
        elapsed = time.time() - t0

        json_valid = not result.fell_back
        point_count_reasonable = 1 <= len(result.points) <= 8

        exec_oks = 0
        for point in result.points:
            try:
                pred = predict_program_checked(bundle, point.description)
                exec_result = Sandbox.execute(pred.program)
                if exec_result.status == "ok":
                    exec_oks += 1
            except Exception:
                pass
        execution_success_rate = exec_oks / len(result.points) if result.points else 0.0

        decompose_results.append({
            "id": r["id"], "elapsed_s": round(elapsed, 1), "json_valid": json_valid,
            "point_count": len(result.points), "point_count_reasonable": point_count_reasonable,
            "execution_success_rate": round(execution_success_rate, 3),
        })
        print(json.dumps(decompose_results[-1]))

    clarify_results = []
    for r in clarify_records:
        user_text = r["messages"][1]["content"]
        expected = json.loads(r["messages"][2]["content"])
        expected_asks = len(expected["questions"]) > 0
        t0 = time.time()
        result = ask_clarifying_questions(user_text, model_path=model_path)
        elapsed = time.time() - t0
        actual_asks = len(result.questions) > 0
        clarify_results.append({
            "id": r["id"], "elapsed_s": round(elapsed, 1),
            "expected_asks": expected_asks, "actual_asks": actual_asks,
            "direction_correct": expected_asks == actual_asks,
        })
        print(json.dumps(clarify_results[-1]))

    summary = {
        "split": split, "model_path": str(model_path),
        "n_decompose": len(decompose_results), "n_clarify": len(clarify_results),
    }
    if decompose_results:
        summary["json_valid_rate"] = sum(r["json_valid"] for r in decompose_results) / len(decompose_results)
        summary["point_count_reasonable_rate"] = sum(r["point_count_reasonable"] for r in decompose_results) / len(decompose_results)
        summary["mean_execution_success_rate"] = sum(r["execution_success_rate"] for r in decompose_results) / len(decompose_results)
    if clarify_results:
        summary["clarify_direction_accuracy"] = sum(r["direction_correct"] for r in clarify_results) / len(clarify_results)
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["validation", "test"], default="test")
    ap.add_argument("--limit", type=int, default=5, help="max examples PER TASK (decompose/clarify) -- each costs ~10-15 min")
    ap.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    ap.add_argument("--bundle-path", type=Path, default=Path(__file__).resolve().parent.parent / "models" / "program_model.joblib")
    args = ap.parse_args()

    summary = evaluate(args.split, args.limit, args.model_path, args.bundle_path)
    print("\n--- summary (never averaged with any other model's numbers -- see program_eval.py's own rule) ---")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
