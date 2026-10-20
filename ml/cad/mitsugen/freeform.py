"""Mitsugen's OWN free-form object decomposition -- a second, independent
spec stage alongside spec.py's radial-sweep vessel grammar, for objects
that are not a simple body of revolution (a toy car, a bracket, a mug
with a handle, a figure).

This is a CONSTRUCTIVE APPROXIMATION technique, not a claim of exact
reproduction: a local LLM decomposes the object description into a small
set of primitives (box/cylinder/sphere/cone) combined with boolean
operations, which freeform_builder.py then actually constructs via real
CSG. Organic/sculptural detail beyond what box/cylinder/sphere/cone and
boolean union/difference/intersection can express is NOT captured --
every result built this way carries that disclosure forward (see
pipeline.py's Stage F), never silently presented as a perfect replica.

Reuses the llama-cpp-python LIBRARY and the same on-disk .gguf model file
3di-men's own llm_decompose.py uses -- "LLM inference" is explicitly
shared infrastructure per this project's architecture brief. Does NOT
import llm_decompose.py itself: the system prompt, JSON schema, grammar,
and parsing logic below are fresh, independent code for a completely
different decomposition target (primitives+booleans, not mechanical
parts+relationships) -- see mitsugen/__init__.py."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

PRIMITIVE_KINDS = {"box", "cylinder", "sphere", "cone"}
BOOLEAN_OPS = {"union", "difference", "intersection"}

MODEL_PATH = Path(os.environ.get(
    "MITSUGEN_FREEFORM_MODEL_PATH",
    str(Path(__file__).resolve().parent.parent / "models" / "qwen" / "quantized" / "Qwen3-4B-Instruct-2507-Q4_K_M.gguf"),
))
# This module's own context budget -- kept separately namespaced from
# llm_decompose.py's MAX_CONTEXT (not shared) because the right VALUE
# genuinely differs: that file's system prompt alone runs ~6850 tokens
# and has already broken decomposition once when context ran too tight
# (see its own MAX_CONTEXT comment), while this module's much shorter
# prompt (~500-900 tokens) never needs that much headroom.
MAX_CONTEXT = int(os.environ.get("MITSUGEN_FREEFORM_MAX_CONTEXT", "8192"))
# N_GPU_LAYERS/N_THREADS, by contrast, DELIBERATELY use the SAME env var
# names llm_decompose.py reads -- both files load the identical on-disk
# model (MODEL_PATH above defaults to llm_decompose.py's own model file),
# so the right hardware-tuning VALUE is identical too (same model, same
# machine). One `export N_GPU_LAYERS=-1` (offload every layer) in the
# shell before running server.py turns on GPU offload for both 3di-men's
# CAD decomposition and Mitsugen's free-form decomposition at once, on a
# machine with a CUDA- or Metal-capable llama-cpp-python build -- never
# flipped on by default here either, same reasoning as that file's own
# comment: this is hardware-dependent, not safe to assume for every
# environment this project might run in.
N_GPU_LAYERS = int(os.environ.get("N_GPU_LAYERS", "0"))
_N_THREADS_ENV = os.environ.get("N_THREADS")
N_THREADS = int(_N_THREADS_ENV) if _N_THREADS_ENV else None


@dataclass
class PrimitiveSpec:
    id: str
    kind: str  # one of PRIMITIVE_KINDS
    position_mm: tuple[float, float, float] = (0.0, 0.0, 0.0)
    rotation_deg: tuple[float, float, float] = (0.0, 0.0, 0.0)
    size_mm: "tuple[float, float, float] | None" = None  # box
    radius_mm: "float | None" = None  # cylinder / cone / sphere
    height_mm: "float | None" = None  # cylinder / cone


@dataclass
class BooleanStep:
    op: str  # one of BOOLEAN_OPS
    target_id: str


@dataclass
class FreeformSpec:
    object_description: str
    primitives: list[PrimitiveSpec] = field(default_factory=list)
    operations: list[BooleanStep] = field(default_factory=list)
    dropped_primitives: list[str] = field(default_factory=list)  # malformed entries dropped, disclosed, never silently built wrong
    auto_unioned_ids: list[str] = field(default_factory=list)  # primitives the model never wired into an operation -- unioned in anyway rather than silently dropped
    fell_back: bool = False  # true if the LLM output was unusable and a single default-box approximation was substituted
    raw_llm_output: str = ""


_FREEFORM_SYSTEM_PROMPT = """You are decomposing a description of a physical object into a small set of simple geometric primitives (box, cylinder, sphere, cone) combined with boolean operations, for a 3D-printing CSG builder. This is a CONSTRUCTIVE APPROXIMATION: pick primitives and booleans that best resemble the object's overall form and major features. Organic, sculptural, or highly detailed surface geometry is honestly out of reach for this vocabulary -- approximate it as closely as box/cylinder/sphere/cone allow, never invent a fifth primitive type.

Use plausible real-world sizing for a named everyday object (a toy car, a mug, a bracket) even when no dimensions are given -- never an arbitrary or token-small size.

Each primitive is an object with:
- "id": a short stable string, unique within this object
- "type": one of "box", "cylinder", "sphere", "cone"
- "box" needs "size_mm": [length_x, width_y, height_z]
- "cylinder" and "cone" need "radius_mm" and "height_mm" (a cone tapers from radius_mm at its base to a point at the top)
- "sphere" needs "radius_mm"
- "position_mm": [x, y, z] -- the primitive's own CENTER in world space (every primitive spans +/- half its own size/height around this center along each axis, even the cone)
- "rotation_deg": [x, y, z] -- optional Euler angles in degrees, applied about the primitive's own center in X-then-Y-then-Z order; omit or [0,0,0] for no rotation

List primitives in a sensible build order: the FIRST primitive is the object's main body. Every OTHER primitive must be referenced by exactly one operation.

"operations" is an array of {"op": "union"|"difference"|"intersection", "target_id": "<id of a primitive other than the first>"}, applied in order starting from the first primitive as the running result. Use "union" to add a feature (a wheel, a handle, a boss, a leg), "difference" to cut a hole or cavity, "intersection" rarely (clipping one shape by another).

Respond with ONLY one JSON object: {"primitives": [...], "operations": [...]}. No prose, no explanation, no markdown fences, no extra top-level keys.

/no_think"""

_FREEFORM_JSON_SCHEMA = json.dumps({
    "type": "object",
    "properties": {
        "primitives": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "type": {"type": "string", "enum": sorted(PRIMITIVE_KINDS)},
                    "size_mm": {"type": "array", "items": {"type": "number"}, "minItems": 3, "maxItems": 3},
                    "radius_mm": {"type": "number"},
                    "height_mm": {"type": "number"},
                    "position_mm": {"type": "array", "items": {"type": "number"}, "minItems": 3, "maxItems": 3},
                    "rotation_deg": {"type": "array", "items": {"type": "number"}, "minItems": 3, "maxItems": 3},
                },
                "required": ["id", "type", "position_mm"],
            },
        },
        "operations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "op": {"type": "string", "enum": sorted(BOOLEAN_OPS)},
                    "target_id": {"type": "string"},
                },
                "required": ["op", "target_id"],
            },
        },
    },
    "required": ["primitives", "operations"],
})

_llm: Any = None
_freeform_grammar: Any = None


def _get_llm(model_path: Path = MODEL_PATH):
    global _llm
    if _llm is None:
        from llama_cpp import Llama

        _llm = Llama(
            model_path=str(model_path), n_ctx=MAX_CONTEXT, verbose=False,
            n_gpu_layers=N_GPU_LAYERS, n_threads=N_THREADS,
        )
    return _llm


def _get_freeform_grammar():
    global _freeform_grammar
    if _freeform_grammar is None:
        from llama_cpp import LlamaGrammar

        _freeform_grammar = LlamaGrammar.from_json_schema(_FREEFORM_JSON_SCHEMA, verbose=False)
    return _freeform_grammar


def _find_schema_shaped_objects(text: str) -> list[dict]:
    """Brace-balanced scan for every well-formed {...} substring that looks
    like a freeform decomposition (a dict with a "primitives" list), in
    order of appearance -- mirrors the bracket-balanced scanning technique
    llm_decompose.py uses for its own, differently-shaped JSON array, but
    is fresh code written against this module's own object schema."""
    candidates: list[dict] = []
    depth = 0
    start = None
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start is not None:
                    candidate_text = text[start : i + 1]
                    try:
                        parsed = json.loads(candidate_text)
                    except json.JSONDecodeError:
                        continue
                    if (
                        isinstance(parsed, dict) and isinstance(parsed.get("primitives"), list)
                        and len(parsed["primitives"]) > 0
                        and all(isinstance(p, dict) and "id" in p and ("type" in p or "kind" in p) for p in parsed["primitives"])
                    ):
                        candidates.append(parsed)
    return candidates


def _extract_json_object(raw: str) -> dict:
    """Only looks after the model's LAST </think> tag, same reasoning-vs-
    answer split convention llm_decompose.py established for this same
    Qwen3 model family."""
    answer_segment = raw.rsplit("</think>", 1)[-1]
    candidates = _find_schema_shaped_objects(answer_segment)
    if not candidates:
        raise ValueError("no schema-shaped freeform JSON object found in LLM output")
    return candidates[0]


def _coerce_vec3(raw: Any, default: tuple[float, float, float] = (0.0, 0.0, 0.0)) -> tuple[float, float, float]:
    if not isinstance(raw, (list, tuple)) or len(raw) != 3:
        return default
    try:
        return (float(raw[0]), float(raw[1]), float(raw[2]))
    except (TypeError, ValueError):
        return default


def _parse_primitive(raw: dict) -> "PrimitiveSpec | None":
    """None if this one entry is too malformed to build -- the caller
    drops it and discloses that, rather than failing the whole spec over
    one bad primitive."""
    if not isinstance(raw, dict):
        return None
    pid = raw.get("id")
    kind = raw.get("type") or raw.get("kind")
    if not isinstance(pid, str) or not pid or kind not in PRIMITIVE_KINDS:
        return None
    position_mm = _coerce_vec3(raw.get("position_mm"))
    rotation_deg = _coerce_vec3(raw.get("rotation_deg"))
    size_mm = None
    radius_mm = None
    height_mm = None
    if kind == "box":
        size_mm = _coerce_vec3(raw.get("size_mm"), default=None)
        if size_mm is None or any(s <= 0 for s in size_mm):
            return None
    elif kind == "sphere":
        try:
            radius_mm = float(raw.get("radius_mm"))
        except (TypeError, ValueError):
            return None
        if radius_mm <= 0:
            return None
    else:  # cylinder / cone
        try:
            radius_mm = float(raw.get("radius_mm"))
            height_mm = float(raw.get("height_mm"))
        except (TypeError, ValueError):
            return None
        if radius_mm <= 0 or height_mm <= 0:
            return None
    return PrimitiveSpec(
        id=pid, kind=kind, position_mm=position_mm, rotation_deg=rotation_deg,
        size_mm=size_mm, radius_mm=radius_mm, height_mm=height_mm,
    )


def _parse_freeform(raw: str, object_description: str) -> FreeformSpec:
    parsed = _extract_json_object(raw)
    raw_primitives = parsed.get("primitives") or []
    raw_operations = parsed.get("operations") or []

    primitives: list[PrimitiveSpec] = []
    dropped: list[str] = []
    seen_ids: set[str] = set()
    for item in raw_primitives:
        p = _parse_primitive(item)
        if p is None:
            dropped.append(str(item.get("id", "<unnamed>")) if isinstance(item, dict) else "<unnamed>")
            continue
        if p.id in seen_ids:
            dropped.append(p.id)  # duplicate id -- can't disambiguate which one was meant
            continue
        seen_ids.add(p.id)
        primitives.append(p)

    if not primitives:
        raise ValueError("every primitive in the LLM output was malformed or unparseable")

    base_id = primitives[0].id
    valid_ids = {p.id for p in primitives}
    operations: list[BooleanStep] = []
    referenced: set[str] = set()
    for op_raw in raw_operations:
        if not isinstance(op_raw, dict):
            continue
        op = op_raw.get("op")
        target_id = op_raw.get("target_id")
        if op not in BOOLEAN_OPS or target_id not in valid_ids or target_id == base_id:
            continue
        operations.append(BooleanStep(op=op, target_id=target_id))
        referenced.add(target_id)

    auto_unioned: list[str] = []
    for p in primitives[1:]:
        if p.id not in referenced:
            operations.append(BooleanStep(op="union", target_id=p.id))
            auto_unioned.append(p.id)

    return FreeformSpec(
        object_description=object_description, primitives=primitives, operations=operations,
        dropped_primitives=dropped, auto_unioned_ids=auto_unioned, fell_back=False, raw_llm_output=raw,
    )


_FALLBACK_SIZE_RE = re.compile(r"(\d+(?:\.\d+)?)\s*mm", re.IGNORECASE)


def _fallback_spec(object_description: str, raw_llm_output: str) -> FreeformSpec:
    """Never silently claims success: a single box, sized from whatever
    plain mm number the prompt happens to contain (else a generic 80mm
    default), explicitly marked fell_back=True so nothing downstream
    mistakes this box for the actually-requested shape."""
    sizes = [float(m) for m in _FALLBACK_SIZE_RE.findall(object_description)][:3]
    while len(sizes) < 3:
        sizes.append(sizes[-1] if sizes else 80.0)
    box = PrimitiveSpec(id="fallback_box", kind="box", position_mm=(0.0, 0.0, sizes[2] / 2.0), size_mm=tuple(sizes))
    return FreeformSpec(
        object_description=object_description, primitives=[box], operations=[], fell_back=True,
        raw_llm_output=raw_llm_output,
    )


def decompose_freeform_prompt(object_description: str, model_path: Path = MODEL_PATH) -> FreeformSpec:
    """Never raises: an LLM response that can't be parsed into a usable
    primitive+operation spec falls back to a single disclosed
    approximate box (fell_back=True), never a crash and never a silent
    claim that something more accurate was built."""
    llm = _get_llm(model_path)
    try:
        grammar = _get_freeform_grammar()
    except Exception:
        grammar = None

    response = llm.create_chat_completion(
        messages=[
            {"role": "system", "content": _FREEFORM_SYSTEM_PROMPT},
            {"role": "user", "content": object_description},
        ],
        temperature=0.1, max_tokens=2048, repeat_penalty=1.15, grammar=grammar,
    )
    raw = response["choices"][0]["message"]["content"] or ""

    try:
        return _parse_freeform(raw, object_description)
    except Exception:
        return _fallback_spec(object_description, raw)
