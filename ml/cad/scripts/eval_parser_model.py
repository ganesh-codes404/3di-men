#!/usr/bin/env python3
"""Evaluate BOTH the regex parser and the learned parser, on BOTH the
synthetic test split and a hand-written eval set -- 4 separate reports,
never merged. See parser_eval.py's docstring for why that separation matters.

Usage:
    python ml/cad/scripts/eval_parser_model.py --dataset data/cad_prompts_v1 \\
        --checkpoint models/parser_v1.joblib --hand-written data/eval_hand_written.jsonl
"""
import argparse
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # ml/cad/


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--hand-written", type=Path, required=True)
    args = ap.parse_args()

    try:
        from parser_model import ParserModelBundle, predict as learned_predict
    except ImportError as exc:
        print(f"the learned parser needs scikit-learn + joblib: pip install scikit-learn joblib ({exc})", file=sys.stderr)
        return 2

    from data.prompt_store import read_split
    from data.prompt_dataset import PromptSample
    from nlp.prompt_parser import parse_prompt
    from parser_eval import evaluate_parser

    bundle = ParserModelBundle.load(args.checkpoint)

    def regex_adapter(text: str) -> tuple[str, dict]:
        result = parse_prompt(text)
        return result.part_type, asdict(result.params)

    def learned_adapter(text: str) -> tuple[str, dict]:
        result = learned_predict(bundle, text)
        return result.part_type, asdict(result.params)

    synthetic_test = read_split(args.dataset, "test")
    hand_written = [
        PromptSample.from_dict(__import__("json").loads(line))
        for line in args.hand_written.read_text().splitlines() if line.strip()
    ]

    reports = [
        evaluate_parser(regex_adapter, synthetic_test, "regex", "synthetic_test"),
        evaluate_parser(regex_adapter, hand_written, "regex", "hand_written"),
        evaluate_parser(learned_adapter, synthetic_test, "learned", "synthetic_test"),
        evaluate_parser(learned_adapter, hand_written, "learned", "hand_written"),
    ]
    for report in reports:
        report.print_summary()
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
