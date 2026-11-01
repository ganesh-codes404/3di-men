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

PRIMITIVE_KINDS = {"box", "cylinder", "sphere", "cone", "torus"}
BOOLEAN_OPS = {"union", "difference", "intersection"}
# Repeats a primitive N times and unions every copy into the running
# result in one operation -- added so the LLM can declare ONE instance
# of a repeated feature (a leg, a tooth, a spoke) plus ONE pattern
# operation, instead of enumerating N separate primitives by hand
# (confirmed real risk of that: the more copies a request implies, the
# more chances to drop one or get the count wrong -- exactly the class
# of bug a dedicated pattern op removes by construction, same reasoning
# 3di-men's own sandbox.py linear_pattern/circular_pattern tools were
# added for). Fresh, independent geometry below (freeform_builder.py)
# -- not a call into sandbox.py's own tools.
PATTERN_OPS = {"linear_pattern", "circular_pattern"}

# Confirmed real, severe bug this fixes (live-reported "Mitsugen
# generates cubes for nearly every prompt"): this used to read ONLY its
# own MITSUGEN_FREEFORM_MODEL_PATH, never the shared MODEL_PATH env var
# llm_decompose.py (and this whole project's .env/.env.rtx4060.example
# profiles) already use. On the RTX4060 laptop profile specifically,
# .env sets MODEL_PATH=models/Qwen3-8B-Q4_K_M.gguf (a deliberately
# BIGGER model than the Mac's default, see .env.rtx4060.example's own
# comment) -- but MITSUGEN_FREEFORM_MODEL_PATH is never set there, so
# this fell through to the Mac-specific hardcoded Qwen3-4B path, which
# does not exist on that machine. Every single freeform decomposition
# call on that machine failed to even load the model, was caught by
# decompose_freeform_prompt's own try/except, and silently (from the
# user's perspective -- the fallback IS disclosed in the result, but
# easy to miss) substituted the single default box -- for EVERY
# non-vase prompt, since the vase/bottle/bowl/cup radial-sweep path is
# pure regex and never calls an LLM at all, so it alone kept working.
# Falling back through the SHARED MODEL_PATH before the Mac-specific
# default (not instead of MITSUGEN_FREEFORM_MODEL_PATH, which still
# wins if someone deliberately wants Mitsugen on a DIFFERENT model than
# 3di-men's own CAD decomposition) fixes this without coupling the two
# products' CODE -- only the env var NAME is shared, the same choice
# already made for N_GPU_LAYERS/N_THREADS below.
def _resolve_model_path() -> Path:
    """Pulled out of the plain MODEL_PATH = Path(os.environ.get(...))
    assignment so tests can exercise this resolution order directly
    (with monkeypatched env vars) without reloading the whole module --
    reloading recreates every class in it (BooleanStep, PrimitiveSpec,
    ...) as a NEW object, which breaks == for anything that imported a
    name from this module before the reload still holding the OLD
    class. Confirmed real: an earlier version of this test file's own
    MODEL_PATH tests did exactly that and intermittently broke an
    unrelated pattern-operation test's BooleanStep equality check,
    purely from test ordering."""
    return Path(os.environ.get(
        "MITSUGEN_FREEFORM_MODEL_PATH",
        os.environ.get(
            "MODEL_PATH",
            str(Path(__file__).resolve().parent.parent / "models" / "qwen" / "quantized" / "Qwen3-4B-Instruct-2507-Q4_K_M.gguf"),
        ),
    ))


MODEL_PATH = _resolve_model_path()
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
    major_radius_mm: "float | None" = None  # torus: center of the ring to the center of its tube
    minor_radius_mm: "float | None" = None  # torus: the tube's own thickness radius


@dataclass
class BooleanStep:
    op: str  # one of BOOLEAN_OPS | PATTERN_OPS
    target_id: str
    count: "int | None" = None  # linear_pattern / circular_pattern
    spacing_mm: "float | None" = None  # linear_pattern: center-to-center distance between consecutive copies
    pattern_radius_mm: "float | None" = None  # circular_pattern: distance from the target's own position to each copy
    axis: str = "z"  # linear_pattern: direction copies are offset along; circular_pattern: the circle's own normal


@dataclass
class FreeformSpec:
    object_description: str
    primitives: list[PrimitiveSpec] = field(default_factory=list)
    operations: list[BooleanStep] = field(default_factory=list)
    dropped_primitives: list[str] = field(default_factory=list)  # malformed entries dropped, disclosed, never silently built wrong
    auto_unioned_ids: list[str] = field(default_factory=list)  # primitives the model never wired into an operation -- unioned in anyway rather than silently dropped
    fell_back: bool = False  # true if the LLM output was unusable and a single default-box approximation was substituted
    raw_llm_output: str = ""


_FREEFORM_SYSTEM_PROMPT = """You are decomposing a description of a physical object into a small set of simple geometric primitives (box, cylinder, sphere, cone, torus) combined with boolean operations, for a 3D-printing CSG builder. This is a CONSTRUCTIVE APPROXIMATION: pick primitives and booleans that best resemble the object's overall form and major features. Organic, sculptural, or highly detailed surface geometry is honestly out of reach for this vocabulary -- approximate it as closely as box/cylinder/sphere/cone/torus allow, never invent a sixth primitive type. Use "torus" for any curved ring/loop/handle/hook shape (a mug's handle, a key ring, a hook's curved part) -- a straight cylinder cannot represent a curve, a torus is the one primitive here that can.

Use plausible real-world sizing for a named everyday object (a toy car, a mug, a bracket) even when no dimensions are given -- never an arbitrary or token-small size.

Each primitive is an object with:
- "id": a short stable string, unique within this object
- "type": one of "box", "cylinder", "sphere", "cone", "torus"
- "box" needs "size_mm": [length_x, width_y, height_z]
- "cylinder" and "cone" need "radius_mm" and "height_mm" (a cone tapers from radius_mm at its base to a point at the top)
- "sphere" needs "radius_mm"
- "torus" needs "major_radius_mm" (center of the ring to the center of its tube) and "minor_radius_mm" (the tube's own thickness) -- its own local Z axis runs through the hole in the middle of the ring (like a donut lying flat), so rotate it if the ring should stand upright or face sideways
- "position_mm": [x, y, z] -- the primitive's own CENTER in world space (every primitive spans +/- half its own size/height around this center along each axis, even the cone)
- "rotation_deg": [x, y, z] -- optional Euler angles in degrees, applied about the primitive's own center in X-then-Y-then-Z order; omit or [0,0,0] for no rotation

List primitives in a sensible build order: the FIRST primitive is the object's main body. Every OTHER primitive must be referenced by exactly one operation.

"operations" is an array applied in order starting from the first primitive as the running result. Each entry is one of:
- {"op": "union"|"difference"|"intersection", "target_id": "<id of a primitive other than the first>"} -- "union" to add a feature (a wheel, a handle, a boss, a leg), "difference" to cut a hole or cavity, "intersection" rarely (clipping one shape by another).
- {"op": "linear_pattern", "target_id": "<id>", "count": N, "spacing_mm": S, "axis": "x"|"y"|"z"} -- repeats that ONE primitive N times in a straight row along "axis", each copy S mm apart, centered on the primitive's own stated position, and unions every copy in. Use this for a row of repeated features (teeth, ribs, slats) INSTEAD OF declaring N separate primitives by hand -- declare the feature ONCE, at the row's own center position, then pattern it.
- {"op": "circular_pattern", "target_id": "<id>", "count": N, "pattern_radius_mm": R, "axis": "x"|"y"|"z"} -- repeats that ONE primitive N times evenly spaced around a circle of radius R (in the plane perpendicular to "axis"), centered on the primitive's own stated position, and unions every copy in. Use this ONLY for a feature that is genuinely, rotationally arranged around a true circle -- every copy the same distance from the center, evenly spaced by angle (legs under a round stool/table, spokes on a wheel, bolts on a bolt circle). Do NOT use it for a feature arranged at the CORNERS of a rectangle instead (a car's four wheels, a box's four corner feet) -- evenly-spaced-around-a-circle math puts points at 0/90/180/270 degrees, i.e. centered on each SIDE of a rectangular body, not at its corners, which is the wrong layout entirely. For a rectangular/corner arrangement, declare each instance as its own separate primitive with its own explicit position instead (see the toy car example below).

EVERY FEATURE THE REQUEST NAMES MUST APPEAR AS ITS OWN PRIMITIVE -- this is the single most common mistake: if the request says "with a handle" / "with a hook" / "with four legs" / "with a spout", that feature is NOT optional and is NOT satisfied by the main body alone, no matter how plausible the main body looks on its own. A mug with no handle primitive at all is wrong even if the cylinder itself is perfect. For a COUNTED repeated feature arranged on a true circle (N legs, N spokes), prefer ONE primitive + a circular_pattern operation over N separate primitives -- it is far less likely to silently end up with the wrong count. For a feature at rectangular corners (wheels), declare each one explicitly instead, as shown below.

Every secondary feature's own "position_mm" should place it so it genuinely OVERLAPS the body it attaches to by a few mm, not just reaches its surface exactly -- a feature positioned to merely touch the body at a single mathematical point, with no real volume in common, can come out as a separate disconnected piece instead of one solid object.

Example -- request: "a coffee mug with a handle, 90mm tall, 75mm diameter". The body is a plain cylinder; the handle is a curved loop, so it must be a torus (never a straight cylinder, which cannot bend), rotated so its own hole faces sideways (through the ring, toward the mug) and unioned onto the body's own side wall:
{"primitives": [
  {"id": "body", "type": "cylinder", "radius_mm": 37.5, "height_mm": 90, "position_mm": [0, 0, 0]},
  {"id": "handle", "type": "torus", "major_radius_mm": 22, "minor_radius_mm": 6, "position_mm": [0, 43, 0], "rotation_deg": [0, 90, 0]}
], "operations": [{"op": "union", "target_id": "handle"}]}

Example -- request: "a round stool with a seat and four legs, 300mm diameter seat, 450mm tall". Declare ONE leg at the ring's own center position, then pattern it -- never four separate leg primitives:
{"primitives": [
  {"id": "seat", "type": "cylinder", "radius_mm": 150, "height_mm": 20, "position_mm": [0, 0, 440]},
  {"id": "leg", "type": "cylinder", "radius_mm": 12, "height_mm": 430, "position_mm": [0, 0, 215]}
], "operations": [{"op": "circular_pattern", "target_id": "leg", "count": 4, "pattern_radius_mm": 120, "axis": "z"}]}

Example -- request: "a simple toy car: a rectangular body on four cylindrical wheels". Four wheels at the four CORNERS of a rectangular body is NOT a circular arrangement -- declare each wheel as its own primitive, at its own corner position, rotated 90 degrees so its flat round face points sideways (axle along y) like a real wheel rather than standing up like a short can, and positioned so it overlaps the body's own side wall by a couple mm rather than merely reaching it:
{"primitives": [
  {"id": "body", "type": "box", "size_mm": [80, 40, 20], "position_mm": [0, 0, 15]},
  {"id": "wheel_fl", "type": "cylinder", "radius_mm": 8, "height_mm": 6, "position_mm": [25, 22, 8], "rotation_deg": [90, 0, 0]},
  {"id": "wheel_fr", "type": "cylinder", "radius_mm": 8, "height_mm": 6, "position_mm": [25, -22, 8], "rotation_deg": [90, 0, 0]},
  {"id": "wheel_bl", "type": "cylinder", "radius_mm": 8, "height_mm": 6, "position_mm": [-25, 22, 8], "rotation_deg": [90, 0, 0]},
  {"id": "wheel_br", "type": "cylinder", "radius_mm": 8, "height_mm": 6, "position_mm": [-25, -22, 8], "rotation_deg": [90, 0, 0]}
], "operations": [
  {"op": "union", "target_id": "wheel_fl"}, {"op": "union", "target_id": "wheel_fr"},
  {"op": "union", "target_id": "wheel_bl"}, {"op": "union", "target_id": "wheel_br"}
]}
(body half-width is 20mm along y; each wheel's own center sits at y=22mm, only 2mm past that edge, so its 6mm-thick body genuinely overlaps the box's side wall instead of merely grazing it.)

BEFORE ANSWERING, check your own primitives array against the request one more time: for every feature word in the request (handle, hook, leg, wheel, spout, wing, ear, tail, boss, foot, arm, strap, hinge, lid), confirm there is a primitive for it AND an operation referencing that primitive's id. A primitives array that only describes the main body when the request named an additional feature is incomplete -- add the missing primitive and operation before responding, never answer with the feature silently left out.

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
                    "major_radius_mm": {"type": "number"},
                    "minor_radius_mm": {"type": "number"},
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
                    "op": {"type": "string", "enum": sorted(BOOLEAN_OPS | PATTERN_OPS)},
                    "target_id": {"type": "string"},
                    "count": {"type": "integer"},
                    "spacing_mm": {"type": "number"},
                    "pattern_radius_mm": {"type": "number"},
                    "axis": {"type": "string", "enum": ["x", "y", "z"]},
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
    major_radius_mm = None
    minor_radius_mm = None
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
    elif kind == "torus":
        try:
            major_radius_mm = float(raw.get("major_radius_mm"))
            minor_radius_mm = float(raw.get("minor_radius_mm"))
        except (TypeError, ValueError):
            return None
        if major_radius_mm <= 0 or minor_radius_mm <= 0 or minor_radius_mm >= major_radius_mm:
            return None  # a tube thicker than the ring itself isn't a real torus -- rejected, not clamped
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
        major_radius_mm=major_radius_mm, minor_radius_mm=minor_radius_mm,
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
        if target_id not in valid_ids or target_id == base_id:
            continue
        axis = op_raw.get("axis")
        axis = axis if axis in ("x", "y", "z") else "z"  # a sensible default, same as rotation_deg/position_mm defaulting elsewhere -- never a reason to drop the whole operation
        if op in BOOLEAN_OPS:
            operations.append(BooleanStep(op=op, target_id=target_id))
        elif op == "linear_pattern":
            try:
                count = int(op_raw.get("count"))
                spacing_mm = float(op_raw.get("spacing_mm"))
            except (TypeError, ValueError):
                continue  # malformed pattern fields -- drop the operation, not the whole spec
            if count < 1 or spacing_mm <= 0:
                continue
            operations.append(BooleanStep(op=op, target_id=target_id, count=count, spacing_mm=spacing_mm, axis=axis))
        elif op == "circular_pattern":
            try:
                count = int(op_raw.get("count"))
                pattern_radius_mm = float(op_raw.get("pattern_radius_mm"))
            except (TypeError, ValueError):
                continue
            if count < 1 or pattern_radius_mm <= 0:
                continue
            operations.append(BooleanStep(op=op, target_id=target_id, count=count, pattern_radius_mm=pattern_radius_mm, axis=axis))
        else:
            continue
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
    """Never raises -- not even if the model itself fails to load or the
    inference call errors out (confirmed necessary, not defensive: an
    uncaught exception here propagates all the way through
    pipeline.py/server.py and surfaces to the browser as Flask's own
    HTML error page, which the frontend's res.json() then fails to parse
    as JSON -- a real, reported failure mode, not a hypothetical one).
    Any such failure -- model load, inference, or parsing -- falls back
    to the same single disclosed approximate box (fell_back=True) a bad
    LLM response already falls back to."""
    try:
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
    except Exception as exc:
        return _fallback_spec(object_description, f"<LLM call failed: {exc}>")

    try:
        return _parse_freeform(raw, object_description)
    except Exception:
        return _fallback_spec(object_description, raw)


# ---------------- Stage A(b): optional clarify step ----------------
# Mitsugen's own independent analogue of llm_decompose.py's
# ask_clarifying_questions/fold_answers_into_prompt -- same OPTIONAL,
# never-raises, never-blocks shape (a caller can always skip straight
# to decompose_freeform_prompt with the original text), but with its
# own system prompt scoped to THIS module's own primitive vocabulary
# and concerns (which primitive a feature should be, how many of a
# repeated feature, overall size for a custom object), never a mode of
# or an import from llm_decompose.py.

_CLARIFY_SYSTEM_PROMPT = """You are reviewing a request for a 3D-printable object before decomposing it into simple geometric primitives (box, cylinder, sphere, cone, torus) combined with boolean operations. Given the request, decide whether you need more information before decomposing it, and if so, ask up to 3 short, specific questions -- the kind that would actually change which primitives get used or how they're arranged, not generic follow-ups.

Only ask about things that actually change the decomposition:
- A missing overall size for a CUSTOM object with no sensible typical size (not a named everyday object like a mug, vase, bottle, or toy car).
- An ambiguous count of a repeated feature the request itself implies (how many legs, how many spokes, how many wheels) when it isn't stated.
- Whether a named feature's own shape is genuinely ambiguous given the request's own wording (rare -- most named features, like "a handle" or "a hook", have one obvious shape and need no question).

Do NOT ask about dimensions for a well-known everyday object (a mug, a vase, a bottle, a toy car, a bracket) -- typical real-world sizing is used automatically. Do NOT ask about anything this primitive vocabulary cannot represent anyway: material, color, surface finish, texture, or printing settings.

If the request already gives enough to decompose (even using typical sizing for a named everyday object), return an empty list. Most requests need zero questions.

Example 1 -- request: "a toy car". A well-known object with sensible typical sizing and no ambiguous repeated feature (a car has 4 wheels by default). Correct output: {"questions": []}

Example 2 -- request: "a stand with some legs". A custom object with no typical size and an unstated leg count. Correct output: {"questions": ["Roughly how big should this be overall (in mm)?", "How many legs?"]}

Respond with ONLY a JSON object: {"questions": [...]}. No other text, no explanation.

/no_think"""

_CLARIFY_JSON_SCHEMA = json.dumps({
    "type": "object",
    "properties": {"questions": {"type": "array", "items": {"type": "string"}, "maxItems": 3}},
    "required": ["questions"],
})

_clarify_grammar: Any = None


def _get_clarify_grammar():
    global _clarify_grammar
    if _clarify_grammar is None:
        from llama_cpp import LlamaGrammar

        _clarify_grammar = LlamaGrammar.from_json_schema(_CLARIFY_JSON_SCHEMA, verbose=False)
    return _clarify_grammar


@dataclass
class ClarifyResult:
    questions: list[str]
    raw_llm_output: str


def _find_clarify_shaped_objects(text: str) -> list[dict]:
    """Brace-balanced scan for a {"questions": [...]} object -- the SAME
    scanning technique _find_schema_shaped_objects above uses, but
    against this different schema (a "questions" list, not a
    "primitives" list) -- kept as its own small function rather than
    overloading the primitives-shaped one with an extra parameter."""
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
                    if isinstance(parsed, dict) and isinstance(parsed.get("questions"), list):
                        candidates.append(parsed)
    return candidates


def ask_clarifying_questions(object_description: str, model_path: Path = MODEL_PATH, max_questions: int = 3) -> ClarifyResult:
    """Never raises: any failure (model load, inference, or parsing) is
    treated the same as "no questions needed" -- an optional step that
    could itself crash or block generation would defeat the point of
    it being optional."""
    try:
        llm = _get_llm(model_path)
        try:
            grammar = _get_clarify_grammar()
        except Exception:
            grammar = None
        response = llm.create_chat_completion(
            messages=[
                {"role": "system", "content": _CLARIFY_SYSTEM_PROMPT},
                {"role": "user", "content": object_description},
            ],
            temperature=0.2, max_tokens=512, repeat_penalty=1.15, grammar=grammar,
        )
        raw = response["choices"][0]["message"]["content"] or ""
    except Exception:
        return ClarifyResult(questions=[], raw_llm_output="")

    answer_segment = raw.rsplit("</think>", 1)[-1]
    candidates = _find_clarify_shaped_objects(answer_segment)
    if not candidates:
        return ClarifyResult(questions=[], raw_llm_output=raw)
    questions = [str(q).strip() for q in candidates[0]["questions"] if str(q).strip()]
    return ClarifyResult(questions=questions[:max_questions], raw_llm_output=raw)


def fold_answers_into_prompt(object_description: str, qa_pairs: list[tuple[str, str]]) -> str:
    """Combines the original description with the user's answers into
    one enriched description -- a plain string append, so
    decompose_freeform_prompt never needs its own code path for "a
    clarify stage happened." Independent copy of the same simple
    technique llm_decompose.py's own fold_answers_into_prompt uses --
    there is no CAD-specific (or Mitsugen-specific) logic in a plain
    string join worth importing across the architectural boundary for."""
    answered = [(q, a) for q, a in qa_pairs if a and a.strip()]
    if not answered:
        return object_description
    details = "\n".join(f"- {q} {a.strip()}" for q, a in answered)
    return f"{object_description}\n\nAdditional details:\n{details}"
