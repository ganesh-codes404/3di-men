"""(text, program) training data for program_model.py -- two sources, kept
permanently separate (never merged for training OR eval, same principle as
parser_eval.py's "never merge synthetic and hand-written scores"):

1. "family" samples: every (text, params) pair data/prompt_dataset.py already
   generates for the 6 known part families, additionally traced through
   data/program_builders.py to get a (text, program) label for free. Because
   every family's program has a FIXED op-sequence shape (only arguments
   vary), this teaches the model to reproduce the 6 known shapes through the
   tool vocabulary -- it is NOT, by itself, evidence of compositional
   generalization. See module-level HONEST_LIMITATIONS below.

2. "generic_composition" samples: a modest (few hundred), separately-tagged
   corpus of hand-designed composable PATTERNS (not free-form random tool
   graphs, which mostly produce geometric garbage) -- the only source of any
   real exposure to primitive composition beyond the 6 fixed families. Text
   for these is mechanically rendered from the program itself, one clause
   per tool call -- there is no natural vocabulary for a random composition
   the way "gear" is a real word, so this reads as telegraphic, not fluent
   English. That is a known, deliberate limitation, not an oversight.

HONEST_LIMITATIONS: true open-ended compositional generalization (correctly
building an arbitrary novel object) is not realistically achievable under
this project's constraints (CPU-only, scikit-learn, no LLM API, a few-
hundred-example generic corpus) -- that needs large pretrained sequence
models trained on huge diverse CAD corpora, a different project. The honest
target here is: reliable reproduction of the 6 known shapes through a
uniform tool-call representation, modest and honestly-measured competence on
simple 2-3-primitive novel combinations resembling the generic-composition
grammar, and visible, typed failure beyond that (see sandbox.py's
ProgramExecutionResult) -- never silent degradation.
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from typing import Literal

from data.program_builders import traced_build
from data.prompt_dataset import sample_prompt_dataset
from part_families import list_part_families
from sandbox import ToolCall

SourceKind = Literal["family", "generic_composition"]


@dataclass(frozen=True)
class ProgramSample:
    id: str
    text: str
    program: tuple[ToolCall, ...]
    source_kind: SourceKind
    family: str | None  # only set for source_kind == "family"
    canonical_key: str

    def to_dict(self) -> dict:
        return {
            "id": self.id, "text": self.text,
            "program": [{"op": c.op, "args": c.args} for c in self.program],
            "source_kind": self.source_kind, "family": self.family, "canonical_key": self.canonical_key,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ProgramSample":
        return cls(
            id=d["id"], text=d["text"],
            program=tuple(ToolCall(op=c["op"], args=c["args"]) for c in d["program"]),
            source_kind=d["source_kind"], family=d["family"], canonical_key=d["canonical_key"],
        )


def _program_canonical_key(text: str, program: list[ToolCall]) -> str:
    payload = json.dumps({"text": text, "program": [{"op": c.op, "args": c.args} for c in program]}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()


def sample_family_program_dataset(counts: dict[str, int], seed: int) -> list[ProgramSample]:
    """Free (text, program) pairs: reuses sample_prompt_dataset() unmodified,
    then traces each sample's already-resolved params through the matching
    program builder."""
    base_samples = sample_prompt_dataset(counts, seed=seed)
    out = []
    for s in base_samples:
        spec_params_cls = _params_cls_for(s.part_family)
        params = spec_params_cls(**s.params)
        _, program = traced_build(s.part_family, params)
        out.append(
            ProgramSample(
                id=f"family_{s.id}", text=s.text, program=tuple(program), source_kind="family",
                family=s.part_family, canonical_key=_program_canonical_key(s.text, program),
            )
        )
    return out


def _params_cls_for(family: str):
    from part_families import get_part_family

    return get_part_family(family).params_cls


# ---------------------------------------------------------------------------
# Generic composition corpus: hand-designed composable patterns, not
# free-form random tool graphs (most random sequences are geometric
# garbage -- same reasoning as random_gear flooring tooth_count at 12).
# ---------------------------------------------------------------------------


def _pattern_union(rng: random.Random) -> tuple[str, list[ToolCall]]:
    l, w, h = rng.choice([30, 40, 50, 60]), rng.choice([20, 30, 40]), rng.choice([8, 10, 15])
    r, ch = rng.choice([5, 8, 10]), rng.choice([15, 20, 25])
    program = [
        ToolCall("add_box", {"length": l, "width": w, "height": h, "center": [0, 0, 0]}),
        ToolCall("add_cylinder", {"radius": r, "height": ch, "center": [0, 0, h / 2 + ch / 2], "axis": "z"}),
        ToolCall("boolean_union", {"a": 0, "b": 1}),
        ToolCall("finalize", {"target": 2}),
    ]
    text = f"a box {l:g}mm x {w:g}mm x {h:g}mm with a cylinder {2*r:g}mm diameter x {ch:g}mm tall on top, unioned together"
    return text, program


def _pattern_cavity(rng: random.Random) -> tuple[str, list[ToolCall]]:
    L, W, H = rng.choice([60, 80, 100]), rng.choice([40, 60, 80]), rng.choice([20, 30, 40])
    wall = rng.choice([2, 3, 4])
    program = [
        ToolCall("add_box", {"length": L, "width": W, "height": H, "center": [0, 0, 0]}),
        ToolCall("add_box", {"length": L - 2 * wall, "width": W - 2 * wall, "height": H - wall, "center": [0, 0, wall / 2]}),
        ToolCall("boolean_difference", {"a": 0, "b": 1}),
        ToolCall("finalize", {"target": 2}),
    ]
    text = f"a box {L:g}mm x {W:g}mm x {H:g}mm with an inset cavity leaving {wall:g}mm walls"
    return text, program


def _pattern_ring(rng: random.Random) -> tuple[str, list[ToolCall]]:
    R, H = rng.choice([30, 40, 50]), rng.choice([5, 8, 10])
    r = R * rng.choice([0.4, 0.5, 0.6])
    program = [
        ToolCall("add_cylinder", {"radius": R, "height": H, "center": [0, 0, 0], "axis": "z"}),
        ToolCall("add_cylinder", {"radius": r, "height": H + 10, "center": [0, 0, 0], "axis": "z"}),
        ToolCall("boolean_difference", {"a": 0, "b": 1}),
        ToolCall("finalize", {"target": 2}),
    ]
    text = f"a ring, {2*R:g}mm outer diameter, {2*r:g}mm inner diameter, {H:g}mm thick"
    return text, program


def _pattern_holes(rng: random.Random) -> tuple[str, list[ToolCall]]:
    shape = rng.choice(["box", "cylinder"])
    count = rng.choice([2, 3, 4])
    hole_r = rng.choice([2, 3, 4])
    if shape == "box":
        l, w, h = rng.choice([40, 60, 80]), rng.choice([30, 40, 60]), rng.choice([6, 8, 10])
        program = [ToolCall("add_box", {"length": l, "width": w, "height": h, "center": [0, 0, 0]})]
        shape_desc = f"a box {l:g}mm x {w:g}mm x {h:g}mm"
        depth = h
    else:
        r, h = rng.choice([20, 30, 40]), rng.choice([6, 8, 10])
        program = [ToolCall("add_cylinder", {"radius": r, "height": h, "center": [0, 0, 0], "axis": "z"})]
        shape_desc = f"a cylinder {2*r:g}mm diameter x {h:g}mm tall"
        depth = h
    program.append(
        ToolCall("drill_hole_pattern", {"target": 0, "pattern": "corner_array" if shape == "box" else "bolt_circle",
                                         "count": count, "radius": hole_r, "depth": depth, "axis": "z",
                                         "margin": 8.0, "circle_diameter": 0.0, "z": 0.0})
    )
    program.append(ToolCall("finalize", {"target": 1}))
    text = f"{shape_desc} with {count} {2*hole_r:g}mm holes"
    return text, program


_GENERIC_PATTERNS = [_pattern_union, _pattern_cavity, _pattern_ring, _pattern_holes]


def sample_generic_composition_dataset(n: int, seed: int) -> list[ProgramSample]:
    rng = random.Random(seed)
    out = []
    for i in range(n):
        pattern_fn = rng.choice(_GENERIC_PATTERNS)
        text, program = pattern_fn(rng)
        out.append(
            ProgramSample(
                id=f"generic_{i:06d}", text=text, program=tuple(program), source_kind="generic_composition",
                family=None, canonical_key=_program_canonical_key(text, program),
            )
        )
    return out


def sample_program_dataset(family_counts: dict[str, int], n_generic: int, seed: int) -> list[ProgramSample]:
    samples = sample_family_program_dataset(family_counts, seed=seed)
    samples += sample_generic_composition_dataset(n_generic, seed=seed + 1)
    return samples


# ---------------------------------------------------------------------------
# Storage: same manifest+jsonl+atomic-write spirit as data/prompt_store.py,
# split via the SAME leakage-safe canonical-key hashing as dataset_split.py
# (imported directly, not reimplemented) -- one manifest.json + train/val/test
# .jsonl, no per-sample directories (no binary payload here either).
# ---------------------------------------------------------------------------

from pathlib import Path

from data.dataset_split import Split, assign_split

_SPLITS: tuple[Split, ...] = ("train", "val", "test")


def _atomic_write(path: Path, content: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(content)
    tmp.replace(path)


def write_program_dataset(root: Path, samples: list[ProgramSample], seed: int) -> dict:
    root.mkdir(parents=True, exist_ok=True)
    by_split: dict[Split, list[ProgramSample]] = {s: [] for s in _SPLITS}
    for sample in samples:
        by_split[assign_split(sample.canonical_key, seed=seed)].append(sample)

    counts: dict[str, dict[str, int]] = {}
    for split in _SPLITS:
        rows = by_split[split]
        content = "\n".join(json.dumps(s.to_dict()) for s in rows)
        _atomic_write(root / f"{split}.jsonl", content + ("\n" if content else ""))
        split_counts: dict[str, int] = {}
        for s in rows:
            key = s.family if s.source_kind == "family" else "generic_composition"
            split_counts[key] = split_counts.get(key, 0) + 1
        counts[split] = split_counts

    manifest = {"version": "1.0.0", "seed": seed, "counts": counts}
    _atomic_write(root / "manifest.json", json.dumps(manifest, indent=2))
    return manifest


def read_program_split(root: Path, split: Split) -> list[ProgramSample]:
    path = root / f"{split}.jsonl"
    if not path.exists():
        return []
    return [ProgramSample.from_dict(json.loads(line)) for line in path.read_text().splitlines() if line.strip()]
