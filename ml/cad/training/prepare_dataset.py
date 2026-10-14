#!/usr/bin/env python3
"""python training/prepare_dataset.py [--n-per-pattern 12] [--seed 0]

Builds the supervised fine-tuning (SFT) dataset for llm_decompose.py's two
LLM tasks -- decomposition (_SYSTEM_PROMPT) and clarification
(_CLARIFY_SYSTEM_PROMPT) -- as chat-format JSONL, ready for a LoRA/QLoRA run
via train.py on a rented GPU. This runs entirely on CPU, no GPU, no network,
no LLM call: it is pure Python generation reusing this project's own
existing ground-truth text generators, not a new content source.

WHERE THE CONTENT COMES FROM (three sources, clearly separated so nobody
mistakes one for another):

1. Generic-composition patterns (data/program_dataset.py's _GENERIC_PATTERNS,
   43 patterns) and the 6 part families (data/prompt_dataset.py) already
   produce a precise, engineer-style text description for a wide range of
   single-object shapes -- exactly the TARGET phrasing _SYSTEM_PROMPT's
   "Available operations" vocabulary teaches. Reused here as the JSON
   decomposition's "description" field, wrapped in a couple of genuinely
   varied INPUT phrasings (see _vary_phrasing) so the model sees more than
   one way a person might ask for the same thing.

2. A small, hand-authored set of multi-part ASSEMBLY examples (gearbox,
   shaft+bearing+housing, bracket+cover) -- genuine multi-part spatial
   reasoning (STEP 1-3 of _SYSTEM_PROMPT) cannot come from a single-object
   pattern generator, so these are written directly, mirroring the system
   prompt's own worked examples.

3. A small, hand-authored set of CLARIFICATION examples (underspecified ->
   real questions, complete -> empty list), mirroring _CLARIFY_SYSTEM_PROMPT's
   own worked examples.

HONEST LIMITATION, disclosed not hidden: _vary_phrasing's "natural language
variety" is template-wrapped, not real user language or LLM-paraphrased --
generating genuine paraphrase diversity via the LLM itself would cost
~10+ minutes PER EXAMPLE on this project's CPU-only hardware, which does
not scale to a few-thousand-example dataset. This is real, if modest, input
diversity beyond pure identity-mapping, not a claim of natural-language
richness. The `curated/`, `failures/` directories exist for exactly the
higher-quality sources (real user requests, expert demonstrations, corrected
failures) this generator cannot produce -- see training/README.md.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.program_dataset import _GENERIC_PATTERNS
from data.prompt_dataset import sample_prompt_dataset
from llm_decompose import _CLARIFY_SYSTEM_PROMPT, _SYSTEM_PROMPT

OUT_ROOT = Path(__file__).resolve().parent / "data"

_PHRASING_WRAPPERS = [
    "{text}",
    "Can you make {text_lower}?",
    "I need {text_lower}.",
    "Make me {text_lower}.",
    "Build {text_lower} for me.",
    "{text}, please.",
    "Please design {text_lower}.",
]


def _vary_phrasing(text: str, rng: random.Random) -> str:
    wrapper = rng.choice(_PHRASING_WRAPPERS)
    # only lowercase the leading article/noun, not units/numbers embedded later
    text_lower = text[0].lower() + text[1:] if text else text
    return wrapper.format(text=text, text_lower=text_lower)


@dataclass
class SftExample:
    id: str
    system_prompt: str
    user_text: str
    assistant_json: str
    source: str  # "generic_pattern" | "family" | "assembly_handwritten" | "clarify_handwritten"


def _single_point_json(description: str) -> str:
    return json.dumps([{"description": description, "position_mm": {"x": 0, "y": 0, "z": 0}}])


def build_decompose_examples(n_per_pattern: int, seed: int) -> list[SftExample]:
    rng = random.Random(seed)
    out: list[SftExample] = []

    for fn in _GENERIC_PATTERNS:
        for i in range(n_per_pattern):
            pattern_rng = random.Random(rng.randint(0, 2**31))
            text, _program = fn(pattern_rng)
            user_text = _vary_phrasing(text, rng)
            out.append(SftExample(
                id=f"generic_{fn.__name__}_{i:03d}", system_prompt=_SYSTEM_PROMPT,
                user_text=user_text, assistant_json=_single_point_json(text),
                source="generic_pattern",
            ))

    family_counts = {"plate": n_per_pattern, "flange": n_per_pattern, "bracket": n_per_pattern,
                      "shaft": n_per_pattern, "enclosure": n_per_pattern, "gear": n_per_pattern}
    for sample in sample_prompt_dataset(family_counts, seed=seed):
        user_text = _vary_phrasing(sample.text, rng)
        out.append(SftExample(
            id=f"family_{sample.id}", system_prompt=_SYSTEM_PROMPT,
            user_text=user_text, assistant_json=_single_point_json(sample.text),
            source="family",
        ))

    out.extend(_HANDWRITTEN_ASSEMBLIES)
    rng.shuffle(out)
    return out


# ---------------------------------------------------------------------------
# Hand-authored multi-part assembly examples -- the one thing the pattern
# generators above genuinely cannot produce (real spatial/mating reasoning
# across DISTINCT parts). Mirrors _SYSTEM_PROMPT's own two worked examples in
# style, adds enough additional variety to matter for training, not just eval.
# ---------------------------------------------------------------------------

_HANDWRITTEN_ASSEMBLIES: list[SftExample] = [
    SftExample(
        id="assembly_gearbox_1", system_prompt=_SYSTEM_PROMPT,
        user_text="two gears meshing on parallel shafts, module 2mm, 14 and 22 teeth, each on a 10mm diameter, 60mm shaft",
        assistant_json=json.dumps([
            {"description": "a cylinder 10mm diameter, 60mm long", "position_mm": {"x": 0, "y": 0, "z": 0}},
            {"description": "a gear profile, module 2, 14 teeth", "position_mm": {"x": 0, "y": 0, "z": 0}},
            {"description": "a cylinder 10mm diameter, 60mm long", "position_mm": {"x": 40, "y": 0, "z": 0}},
            {"description": "a gear profile, module 2, 22 teeth", "position_mm": {"x": 40, "y": 0, "z": 0}},
        ]),
        source="assembly_handwritten",
    ),
    SftExample(
        id="assembly_shaft_housing_1", system_prompt=_SYSTEM_PROMPT,
        user_text="a 12mm diameter, 80mm long shaft running through a bearing bore in a 60mm x 60mm x 20mm housing block, centered",
        assistant_json=json.dumps([
            {"description": "a box 60mm x 60mm x 20mm with a 12mm diameter hole through the center", "position_mm": {"x": 0, "y": 0, "z": 0}},
            {"description": "a cylinder 12mm diameter, 80mm long", "position_mm": {"x": 0, "y": 0, "z": 0}},
        ]),
        source="assembly_handwritten",
    ),
    SftExample(
        id="assembly_bracket_cover_1", system_prompt=_SYSTEM_PROMPT,
        user_text="an L-bracket 50mm and 50mm legs, 6mm thick, 30mm deep, with a flat cover plate bolted flush on top",
        assistant_json=json.dumps([
            {"description": "an L-shape 50mm x 50mm legs, 6mm thick, extruded 30mm", "position_mm": {"x": 0, "y": 0, "z": 0}},
            {"description": "a box 50mm x 30mm x 3mm", "position_mm": {"x": 0, "y": 0, "z": 6}},
        ]),
        source="assembly_handwritten",
    ),
    SftExample(
        id="assembly_standoff_pair_1", system_prompt=_SYSTEM_PROMPT,
        user_text="a 100mm x 80mm x 4mm base plate with two 15mm tall standoff cylinders, 8mm diameter, near opposite corners",
        assistant_json=json.dumps([
            {"description": "a box 100mm x 80mm x 4mm", "position_mm": {"x": 0, "y": 0, "z": 0}},
            {"description": "a cylinder 8mm diameter, 15mm tall", "position_mm": {"x": -40, "y": -30, "z": 4}},
            {"description": "a cylinder 8mm diameter, 15mm tall", "position_mm": {"x": 40, "y": 30, "z": 4}},
        ]),
        source="assembly_handwritten",
    ),
]

_HANDWRITTEN_CLARIFICATIONS: list[SftExample] = [
    SftExample(
        id="clarify_bracket_vague", system_prompt=_CLARIFY_SYSTEM_PROMPT,
        user_text="a bracket with some holes in it",
        assistant_json=json.dumps({"questions": [
            "What size should the bracket be (length x width x thickness, in mm)?",
            "How many holes, and roughly where should they go?",
        ]}),
        source="clarify_handwritten",
    ),
    SftExample(
        id="clarify_motor_bracket_vague", system_prompt=_CLARIFY_SYSTEM_PROMPT,
        user_text="make a bracket for this motor",
        assistant_json=json.dumps({"questions": [
            "What are the motor's mounting dimensions (bolt-hole pattern and diameter, or the motor's own body diameter)?",
            "How thick should the bracket be, and what's it mounting to on the other side?",
        ]}),
        source="clarify_handwritten",
    ),
    SftExample(
        id="clarify_water_bottle_complete", system_prompt=_CLARIFY_SYSTEM_PROMPT,
        user_text="a water bottle",
        assistant_json=json.dumps({"questions": []}),
        source="clarify_handwritten",
    ),
    SftExample(
        id="clarify_full_box_complete", system_prompt=_CLARIFY_SYSTEM_PROMPT,
        user_text="a box 50mm x 40mm x 10mm with a 16mm diameter, 20mm tall cylinder centered on top",
        assistant_json=json.dumps({"questions": []}),
        source="clarify_handwritten",
    ),
    SftExample(
        id="clarify_gear_vague", system_prompt=_CLARIFY_SYSTEM_PROMPT,
        user_text="a gear for my project",
        assistant_json=json.dumps({"questions": [
            "What module and tooth count does the gear need (or what diameter and how many teeth)?",
            "What size bore does it need in the middle, to fit on its shaft?",
        ]}),
        source="clarify_handwritten",
    ),
]


def to_chat_record(ex: SftExample) -> dict:
    return {
        "id": ex.id, "source": ex.source,
        "messages": [
            {"role": "system", "content": ex.system_prompt},
            {"role": "user", "content": ex.user_text},
            {"role": "assistant", "content": ex.assistant_json},
        ],
    }


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def split_records(records: list[dict], seed: int, val_frac: float = 0.1, test_frac: float = 0.1) -> dict[str, list[dict]]:
    rng = random.Random(seed)
    shuffled = records[:]
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_val = int(n * val_frac)
    n_test = int(n * test_frac)
    return {
        "validation": shuffled[:n_val],
        "test": shuffled[n_val:n_val + n_test],
        "training": shuffled[n_val + n_test:],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-pattern", type=int, default=12,
                     help="raw examples per generic-composition pattern / per part family (before split)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    decompose_examples = build_decompose_examples(args.n_per_pattern, args.seed)
    clarify_examples = list(_HANDWRITTEN_CLARIFICATIONS)

    decompose_records = [to_chat_record(ex) for ex in decompose_examples]
    clarify_records = [to_chat_record(ex) for ex in clarify_examples]

    _write_jsonl(OUT_ROOT / "raw" / "decompose_raw.jsonl", decompose_records)
    _write_jsonl(OUT_ROOT / "raw" / "clarify_raw.jsonl", clarify_records)

    decompose_splits = split_records(decompose_records, args.seed)
    # clarify set is small and 100% hand-authored -- keep it all in training
    # plus a fixed 2-example holdout, rather than a proportional split that
    # would leave test/validation with almost nothing.
    clarify_splits = {
        "training": clarify_records[:-2] if len(clarify_records) > 2 else clarify_records,
        "validation": clarify_records[-2:-1] if len(clarify_records) > 2 else [],
        "test": clarify_records[-1:] if len(clarify_records) > 2 else [],
    }

    for split in ("training", "validation", "test"):
        combined = decompose_splits[split] + clarify_splits[split]
        random.Random(args.seed + 1).shuffle(combined)
        _write_jsonl(OUT_ROOT / split / "sft.jsonl", combined)

    manifest = {
        "seed": args.seed, "n_per_pattern": args.n_per_pattern,
        "counts": {
            "decompose_total": len(decompose_records), "clarify_total": len(clarify_records),
            "training": len(decompose_splits["training"]) + len(clarify_splits["training"]),
            "validation": len(decompose_splits["validation"]) + len(clarify_splits["validation"]),
            "test": len(decompose_splits["test"]) + len(clarify_splits["test"]),
        },
        "sources": sorted({ex.source for ex in decompose_examples + clarify_examples}),
    }
    (OUT_ROOT / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
