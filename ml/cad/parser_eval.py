"""Parser-agnostic evaluation harness -- the same code path scores both
nlp/prompt_parser.py (regex) and the learned model (parser_model.py), via one
adapter shape, so the comparison is apples-to-apples instead of two
hand-rolled scripts that could silently diverge.

Deliberately produces FOUR separate reports (regex x synthetic, regex x
hand-written, learned x synthetic, learned x hand-written) that are never
merged or averaged into one number. That's not incidental -- collapsing them
would be exactly the "claiming synthetic training proves real-world
generalization" mistake this project was told to avoid. The hand-written
numbers are expected to be visibly worse than the synthetic ones, especially
for numeric fields (see parser_model.py's docstring for why); that gap is
the point of measuring both.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal

from data.prompt_dataset import PromptSample
from part_families import FieldKind, get_part_family

ParseFn = Callable[[str], tuple[str, dict]]  # text -> (part_family, params_dict)


@dataclass
class FieldMetric:
    field: str
    kind: FieldKind
    n: int
    accuracy: float | None  # categorical/count fields
    mae: float | None  # numeric fields


@dataclass
class ParserEvalReport:
    source: Literal["synthetic_test", "hand_written"]
    parser_name: str
    part_family_accuracy: float
    field_metrics: dict[str, list[FieldMetric]]  # family -> per-field metrics
    n_samples: int

    def print_summary(self) -> None:
        print(f"=== {self.parser_name} on {self.source} (n={self.n_samples}) ===")
        print(f"  part_family_accuracy: {self.part_family_accuracy:.3f}")
        for family, metrics in self.field_metrics.items():
            print(f"  [{family}]")
            for m in metrics:
                if m.kind == "numeric":
                    print(f"    {m.field:28s} mae={m.mae:.3f} (n={m.n})" if m.mae is not None else f"    {m.field:28s} n=0")
                else:
                    print(f"    {m.field:28s} accuracy={m.accuracy:.3f} (n={m.n})" if m.accuracy is not None else f"    {m.field:28s} n=0")


def evaluate_parser(parse_fn: ParseFn, samples: list[PromptSample], parser_name: str, source: str) -> ParserEvalReport:
    if not samples:
        raise ValueError("evaluate_parser called with zero samples")

    correct_family = 0
    # per-family, per-field: lists of (predicted, true) pairs, only from
    # samples where the family prediction was ALSO correct -- otherwise a
    # numeric field's "predicted" value would be for the wrong part entirely
    per_family_field_pairs: dict[str, dict[str, list[tuple]]] = {}

    for sample in samples:
        pred_family, pred_params = parse_fn(sample.text)
        if pred_family == sample.part_family:
            correct_family += 1
            bucket = per_family_field_pairs.setdefault(sample.part_family, {})
            for field_name, true_value in sample.params.items():
                bucket.setdefault(field_name, []).append((pred_params.get(field_name), true_value))

    field_metrics: dict[str, list[FieldMetric]] = {}
    for family, field_pairs in per_family_field_pairs.items():
        spec = get_part_family(family)
        metrics = []
        for f in spec.fields:
            pairs = field_pairs.get(f.name, [])
            if not pairs:
                metrics.append(FieldMetric(field=f.name, kind=f.kind, n=0, accuracy=None, mae=None))
                continue
            if f.kind == "numeric":
                mae = sum(abs(float(p) - float(t)) for p, t in pairs) / len(pairs)
                metrics.append(FieldMetric(field=f.name, kind=f.kind, n=len(pairs), accuracy=None, mae=mae))
            else:
                acc = sum(1 for p, t in pairs if str(p) == str(t)) / len(pairs)
                metrics.append(FieldMetric(field=f.name, kind=f.kind, n=len(pairs), accuracy=acc, mae=None))
        field_metrics[family] = metrics

    return ParserEvalReport(
        source=source,
        parser_name=parser_name,
        part_family_accuracy=correct_family / len(samples),
        field_metrics=field_metrics,
        n_samples=len(samples),
    )
