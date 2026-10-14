#!/usr/bin/env python3
"""python training/inference.py "a water bottle" [--task decompose|clarify] [--model-path ...]

Quick single-prompt manual test harness -- for eyeballing one result while
iterating, not for measurement (that's evaluate.py's job). Works against the
local .gguf model via llm_decompose.py, same as production; --model-path
lets you point at a merged+re-quantized fine-tuned model to compare against
the base model on one prompt at a time.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from llm_decompose import DEFAULT_MODEL_PATH, decompose_prompt, ask_clarifying_questions


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("prompt")
    ap.add_argument("--task", choices=["decompose", "clarify"], default="decompose")
    ap.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    args = ap.parse_args()

    print(f"model: {args.model_path}")
    t0 = time.time()
    if args.task == "decompose":
        result = decompose_prompt(args.prompt, model_path=args.model_path)
        elapsed = time.time() - t0
        print(f"elapsed: {elapsed:.1f}s  fell_back: {result.fell_back}")
        for i, point in enumerate(result.points):
            print(f"  [{i}] {point.description}  @ {point.position_mm}")
    else:
        result = ask_clarifying_questions(args.prompt, model_path=args.model_path)
        elapsed = time.time() - t0
        print(f"elapsed: {elapsed:.1f}s")
        if not result.questions:
            print("  (no questions -- request judged complete)")
        for q in result.questions:
            print(f"  - {q}")


if __name__ == "__main__":
    main()
