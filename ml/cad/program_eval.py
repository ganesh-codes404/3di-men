"""Evaluation for program_model.py -- THREE separate, never-merged report
buckets, extending parser_eval.py's "never merge synthetic and hand-written
scores" principle to a third source:

- synthetic_test: held-out synthetic (text, program) pairs traced from the
  6 known part families (data/program_dataset.py). Expect this to score
  well almost by construction -- each family's program has a fixed
  op-sequence shape, so this mostly tests "did argument regression land
  close enough to still execute," not real generalization.
- hand_written: the SAME 24 hand-authored (text, family, params) examples
  data/eval_hand_written.jsonl already used for parser_eval.py, additionally
  traced into ground-truth programs here. Tests generalization to real
  phrasing, same as parser_eval.py's hand_written bucket.
- generic_composition_synthetic: the modest, separately-tagged corpus of
  hand-designed composable patterns with mechanically-rendered (telegraphic,
  not fluent) text. The only bucket that says anything about composition
  beyond the 6 fixed families -- and even this is a modest, honestly-scoped
  signal, not proof of open-ended generalization (see data/program_dataset.py's
  module docstring for the full honesty statement).

Never average these three into one number. A single "accuracy" figure here
would hide exactly the distinction the whole exercise exists to surface.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from data.program_builders import traced_build
from part_families import get_part_family
from sandbox import ProgramExecutionResult, Sandbox, ToolCall

ProgramPredictFn = Callable[[str], list[ToolCall]]


@dataclass
class ProgramEvalReport:
    source: str  # "synthetic_test" | "hand_written" | "generic_composition_synthetic"
    model_name: str
    n_samples: int
    exact_op_sequence_accuracy: float
    per_slot_op_accuracy: list[float]  # position-wise, over samples with a true op at that position
    execution_status_counts: dict[str, int]
    execution_success_rate: float  # fraction with status == "ok"

    def print_summary(self) -> None:
        print(f"=== {self.model_name} on {self.source} (n={self.n_samples}) ===")
        print(f"  exact_op_sequence_accuracy: {self.exact_op_sequence_accuracy:.3f}")
        print(f"  per_slot_op_accuracy: {[f'{a:.2f}' for a in self.per_slot_op_accuracy]}")
        print(f"  execution_status_counts: {self.execution_status_counts}")
        print(f"  execution_success_rate (status==ok): {self.execution_success_rate:.3f}")


def evaluate_program_model(
    predict_fn: ProgramPredictFn, samples: list, source: str, model_name: str,
) -> ProgramEvalReport:
    if not samples:
        raise ValueError("evaluate_program_model called with zero samples")

    exact_matches = 0
    slot_correct: dict[int, int] = {}
    slot_total: dict[int, int] = {}
    status_counts: dict[str, int] = {"ok": 0, "rejected": 0, "partial": 0, "suspect": 0}

    for sample in samples:
        true_ops = [c.op for c in sample.program]
        predicted_program = predict_fn(sample.text)
        predicted_ops = [c.op for c in predicted_program]

        if predicted_ops == true_ops:
            exact_matches += 1

        for i in range(max(len(true_ops), len(predicted_ops))):
            slot_total[i] = slot_total.get(i, 0) + (1 if i < len(true_ops) else 0)
            if i < len(true_ops) and i < len(predicted_ops) and predicted_ops[i] == true_ops[i]:
                slot_correct[i] = slot_correct.get(i, 0) + 1

        issues = Sandbox.validate(predicted_program)
        if issues:
            status_counts["rejected"] += 1
        else:
            result = Sandbox.execute(predicted_program)
            status_counts[result.status] += 1

    max_slot = max(slot_total.keys(), default=-1) + 1
    per_slot_accuracy = [
        (slot_correct.get(i, 0) / slot_total[i]) if slot_total.get(i, 0) > 0 else 0.0 for i in range(max_slot)
    ]

    return ProgramEvalReport(
        source=source, model_name=model_name, n_samples=len(samples),
        exact_op_sequence_accuracy=exact_matches / len(samples),
        per_slot_op_accuracy=per_slot_accuracy,
        execution_status_counts=status_counts,
        execution_success_rate=status_counts["ok"] / len(samples),
    )


def load_hand_written_program_samples(path: Path) -> list:
    """Traces the SAME hand-authored (text, family, params) examples
    parser_eval.py already uses into ground-truth programs -- reuses
    existing hand-written ground truth rather than hand-authoring a second,
    separate set of examples."""
    from data.program_dataset import ProgramSample, _program_canonical_key

    samples = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        family = d["part_family"]
        params_cls = get_part_family(family).params_cls
        params = params_cls(**d["params"])
        _, program = traced_build(family, params)
        samples.append(
            ProgramSample(
                id=f"handwritten_program_{d['id']}", text=d["text"], program=tuple(program),
                source_kind="family", family=family, canonical_key=_program_canonical_key(d["text"], program),
            )
        )
    return samples
