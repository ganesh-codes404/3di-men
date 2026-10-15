"""Evaluation for llm_decompose.py -- the missing counterpart to
program_eval.py. Confirmed, not assumed, that this gap existed: program_model.py
has had a real accuracy harness (program_eval.py, data/eval_hand_written.jsonl)
for a long time, but decompose_prompt() -- the LLM-decomposition step that
determines WHAT gets sent to program_model.py in the first place -- had no
equivalent. Every improvement to llm_decompose.py this session (vocabulary
expansion, grammar-constrained decoding, few-shot examples, knowledge
grounding) was judged by spot-checking a handful of real calls, never by a
repeatable, scored benchmark. This file is that benchmark.

Scoring is intentionally coarse and structural, not a fluent-text-quality
judgment call: component count, relationship-kind usage, required/forbidden
substring presence. These are the same properties _SYSTEM_PROMPT's own
STEP 0-3 rules are written to produce, so they're checkable without a
second LLM acting as judge (this project's own standing principle: use
deterministic geometric/structural tests wherever possible, never ask a
model to judge correctness from a text description -- see sandbox.py's
module docstring for the same principle applied to geometry).

A sample can leave expected_component_count unset (None) when the "right"
decomposition is genuinely ambiguous (e.g. a symmetric pair of features
could reasonably be 2 or 3 points) -- that field is simply excluded from
scoring for that sample rather than forcing an arbitrary ground truth.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

from llm_decompose import ActionablePoint, DecomposeResult, decompose_prompt


@dataclass
class DecompositionLabel:
    id: str
    prompt: str
    expected_component_count: int | None
    expected_relationship_kinds: tuple[str, ...] = ()
    required_substrings: tuple[str, ...] = ()
    expect_no_dimensions: bool = False
    notes: str = ""


def load_decomposition_labels(path: Path) -> list[DecompositionLabel]:
    labels = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        raw = json.loads(line)
        labels.append(DecompositionLabel(
            id=raw["id"], prompt=raw["prompt"],
            expected_component_count=raw.get("expected_component_count"),
            expected_relationship_kinds=tuple(raw.get("expected_relationship_kinds", [])),
            required_substrings=tuple(raw.get("required_substrings", [])),
            expect_no_dimensions=bool(raw.get("expect_no_dimensions", False)),
            notes=raw.get("notes", ""),
        ))
    return labels


_DIMENSION_RE = re.compile(r"\d")


@dataclass
class DecompositionEvalResult:
    label: DecompositionLabel
    points: list[ActionablePoint] = field(repr=False)
    fell_back: bool
    elapsed_seconds: float

    actual_component_count: int
    component_count_correct: bool | None  # None when the label leaves this unscored
    relationships_found: tuple[str, ...]
    relationship_kinds_satisfied: bool
    missing_required_substrings: tuple[str, ...]
    required_substrings_present: bool
    has_invented_dimensions: bool | None  # None when the label doesn't check this

    @property
    def overall_pass(self) -> bool:
        if self.fell_back:
            return False
        checks = [self.relationship_kinds_satisfied, self.required_substrings_present]
        if self.component_count_correct is not None:
            checks.append(self.component_count_correct)
        if self.has_invented_dimensions is not None:
            checks.append(not self.has_invented_dimensions)
        return all(checks)


def evaluate_one(label: DecompositionLabel) -> DecompositionEvalResult:
    """Never raises: a decompose_prompt() exception would itself be the
    clearest possible failure signal, but decompose_prompt is already
    documented as never raising (it falls back to fell_back=True on any
    internal failure) -- so this function doesn't need its own
    exception-to-failure translation, it just trusts that contract."""
    t0 = time.time()
    result: DecomposeResult = decompose_prompt(label.prompt)
    elapsed = time.time() - t0

    combined_text = " ".join(p.description for p in result.points).lower()
    relationships_found = tuple(sorted({p.relationship.kind for p in result.points if p.relationship}))
    missing = tuple(s for s in label.required_substrings if s.lower() not in combined_text)

    has_invented_dimensions = None
    if label.expect_no_dimensions:
        has_invented_dimensions = bool(_DIMENSION_RE.search(combined_text))

    component_count_correct = None
    if label.expected_component_count is not None:
        component_count_correct = (len(result.points) == label.expected_component_count)

    return DecompositionEvalResult(
        label=label, points=result.points, fell_back=result.fell_back, elapsed_seconds=elapsed,
        actual_component_count=len(result.points), component_count_correct=component_count_correct,
        relationships_found=relationships_found,
        relationship_kinds_satisfied=all(k in relationships_found for k in label.expected_relationship_kinds),
        missing_required_substrings=missing, required_substrings_present=(len(missing) == 0),
        has_invented_dimensions=has_invented_dimensions,
    )


@dataclass
class DecompositionEvalSummary:
    results: list[DecompositionEvalResult] = field(default_factory=list)

    @property
    def pass_rate(self) -> float:
        return sum(1 for r in self.results if r.overall_pass) / len(self.results) if self.results else 0.0

    @property
    def fallback_rate(self) -> float:
        return sum(1 for r in self.results if r.fell_back) / len(self.results) if self.results else 0.0

    @property
    def component_count_accuracy(self) -> float:
        scored = [r for r in self.results if r.component_count_correct is not None]
        return sum(1 for r in scored if r.component_count_correct) / len(scored) if scored else 0.0

    @property
    def relationship_accuracy(self) -> float:
        scored = [r for r in self.results if r.label.expected_relationship_kinds]
        return sum(1 for r in scored if r.relationship_kinds_satisfied) / len(scored) if scored else 0.0

    @property
    def mean_latency_seconds(self) -> float:
        return sum(r.elapsed_seconds for r in self.results) / len(self.results) if self.results else 0.0

    def print_summary(self) -> None:
        print(f"=== decompose_prompt() eval (n={len(self.results)}) ===")
        print(f"  overall_pass_rate: {self.pass_rate:.3f}")
        print(f"  fallback_rate: {self.fallback_rate:.3f}")
        print(f"  component_count_accuracy: {self.component_count_accuracy:.3f} "
              f"(n={sum(1 for r in self.results if r.component_count_correct is not None)})")
        print(f"  relationship_accuracy: {self.relationship_accuracy:.3f} "
              f"(n={sum(1 for r in self.results if r.label.expected_relationship_kinds)})")
        print(f"  mean_latency_seconds: {self.mean_latency_seconds:.1f}")
        for r in self.results:
            status = "PASS" if r.overall_pass else "FAIL"
            print(f"  [{status}] {r.label.id}: count={r.actual_component_count}"
                  f"(expected {r.label.expected_component_count}) "
                  f"relationships={list(r.relationships_found)} fell_back={r.fell_back} "
                  f"missing={list(r.missing_required_substrings)} "
                  f"invented_dims={r.has_invented_dimensions} elapsed={r.elapsed_seconds:.1f}s")


def run_decomposition_eval(labels: list[DecompositionLabel]) -> DecompositionEvalSummary:
    summary = DecompositionEvalSummary()
    for label in labels:
        summary.results.append(evaluate_one(label))
    return summary
