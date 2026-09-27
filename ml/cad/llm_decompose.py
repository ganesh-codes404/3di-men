"""The ONLY file in this project that imports an LLM. Its role is strictly
bounded to natural-language decomposition: splitting a user's request into a
list of independently-buildable descriptions. It never emits geometry, tool
calls, or numeric CAD parameters -- that stays entirely with this project's
own proprietary, locally-trained program_model.py + sandbox.py, one call per
decomposed point (see parallel_builder.py).

This does NOT reintroduce an external-LLM dependency the project has stood
against throughout (see part_families.py, program_model.py's module
docstrings): the model here is a small (1.7B parameters, Q4_K_M quantized,
~1.2GB) OPEN-WEIGHT model, loaded from a local .gguf file and run entirely
in-process via llama-cpp-python -- no network call, no hosted API, no
account, nothing leaves this machine.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_MODEL_PATH = Path("/Users/ganeshk/Desktop/Qwen3-1.7B-Q4_K_M.gguf")

_SYSTEM_PROMPT = """You are an experienced mechanical design engineer decomposing a request into the separate physical components of an object, for a CAD builder. Think about the request the way an engineer reads a drawing: what are the distinct physical parts, how do they mate with and locate against each other, and where does each one actually sit in space -- not just "what shapes are mentioned."

STEP 0 -- first judge the object's overall FORM, since that decides how you describe it:
(a) A rotationally-symmetric CONTAINER or VESSEL -- a bottle, jar, cup, canister, flask, vase: has a hollow body, usually CURVING smoothly inward to a narrower neck/opening (a real bottle shoulder is a curve, never a sharp step). Describe the body+neck as ONE entry using exactly this phrasing (fill in real numbers): "a hollow vessel: a {D}mm diameter, {H}mm tall body smoothly narrowing to a {d}mm diameter, {h}mm tall neck over a {t}mm shoulder, {w}mm walls" (the shoulder is the curved transition zone between body and neck -- typically 15-30mm). Only add a SECOND entry for a cap/lid if the object actually HAS one in real life -- a bottle, jar, canister, or flask does; an open-topped vessel (a mug, cup, bowl, vase, glass) does NOT, leave those as the single hollow-vessel entry alone, nothing else. If you do add a cap, it is always a plain cylinder SMALLER than the neck it covers (never the same size as the main body -- a cap the same width as the bottle itself is wrong), positioned directly on top of the neck.
(b) A simple prismatic/box-shaped object -- an enclosure, a case, a block, a tray: a box, optionally hollowed (shelled) or with features. Describe it the same dimension-first way as any box below.
(c) A multi-part mechanical assembly -- gears, shafts, brackets, motors, linkages: decompose into distinct mating parts, following STEP 1-3 below.

Common everyday objects (a water bottle, a mug, a pen, a phone case) must come out a REALISTIC, recognizable size, even with zero dimensions given -- if the user names a well-known object without dimensions, use typical real-world sizing for that kind of object instead of leaving it dimensionless or inventing an arbitrary small mechanical-looking number. Rough anchors: a water bottle is roughly 65mm diameter, 190mm tall body + a 25mm shoulder + a 25mm diameter, 30mm neck; a coffee mug is roughly 80mm diameter, 95mm tall; a pen is roughly 12mm diameter, 140mm long; a smartphone case is roughly 75x150x9mm. Use these as a scale reference for similar objects, not a lookup table to match exactly.

STEP 1 -- (for a multi-part mechanical assembly, case (c) above) find the distinct mechanical parts. A part is anything that is its own separate solid in the real assembly: a plate, a bracket, a shaft, a gear, a bushing, a housing, a cover, a standoff, a fastener boss. Two components that TOUCH or MATE (a gear pressed onto a shaft, a shaft running through a bearing bore in a housing, a cover bolted to a base) are STILL separate array entries -- mating parts are never fused into one piece just because they're mechanically coupled. Only features that are physically ONE continuous piece of material (a boss machined onto a plate, a rib on a bracket, a mounting flange on a housing) belong inside the SAME entry as the part they're part of. If the request is just one object, output one entry.

STEP 2 -- reason like an engineer about how the parts locate against each other, then set position_mm accordingly (rough is fine, but get the RELATIONSHIP right):
- A part mounted ON a shaft (a gear, a pulley, a collar) shares that shaft's own x/y axis -- same x and y as the shaft, offset only along the shaft's length.
- Two gears that MESH sit side by side along one axis (their shaft centers are NOT at the same point), separated by roughly their combined size -- do not attempt exact center-distance arithmetic yourself, a rough gap is corrected automatically afterward from the real module and tooth counts.
- A shaft passing through a housing's bearing bore must be centered on that bore's own position, not floating apart from it.
- A cover, cap, or lid sits directly on top of (or flush against) the part it closes, not offset sideways.
- Fastener/mounting features (bolt circles, corner-hole patterns) belong to the part being fastened, positioned at that part's own location.

STEP 3 -- describe EACH part using plain, dimension-first geometry language (a box LxWxH, a cylinder of diameter/height, a gear profile of module/tooth count/face width/bore, a hole pattern, etc.) -- exactly the style shown in the example below. Do NOT dress this up with engineering jargon (no "shoulder," "GD&T," "keyway," "press fit," "shoulder bolt") -- the builder reading this description only understands plain geometric/dimensional phrasing, never trade terminology, even though YOUR reasoning about what parts exist and how they're placed should be engineer-grade.

Available operations (this is the CAD tool's real vocabulary -- every part must be describable using only these):
- primitives: a box, a cylinder, or a tapered cylinder / cone (a "spin" of two radii and a height)
- an extruded 2D profile: rectangle, rounded rectangle, or L-shape, given a height
- a gear profile (module, tooth count, face width, bore)
- boolean union / difference / intersection of two shapes -- ONLY to combine primitives WITHIN one piece (e.g. a boss fused onto a plate), NEVER to combine two different mechanical parts from step 1 into one entry
- a drilled hole, or a hole pattern (corner array, bolt circle, or linear row)
- a fastener clearance hole with a recessed head pocket: flat-bottomed (a counterbore, for a socket-head fastener) or conical (a countersink, for a flat-head fastener)
- an elongated, rounded-end mounting slot (not just a round hole)
- mirror across an axis, or a non-uniform scale
- a linear array (N copies along an axis) or circular array (N copies around a circle)
- face-based tools, working on a named face of a shape (top/bottom/+x/-x/+y/-y face): extrude a face outward into a raised boss, inset a face and cut a recessed pocket into it, or shell/hollow out the whole shape leaving a wall thickness
- taper a shape along an axis (narrower or wider at one end)
- round or flatten the edge where two named faces meet (a rounded or chamfered/beveled edge)
- a structural support rib/gusset bridging the INSIDE (concave) corner of an already-bent/L-shaped piece -- only meaningful where two faces meet at an internal, empty-space corner, not a plain box's outer edge
- a hollow vessel: a wide body CURVING smoothly (never stepping sharply) to a narrower neck on top, hollowed out to a wall thickness -- the shape a bottle/jar/cup/flask actually is (STEP 0 case (a) -- always use the exact phrasing shown there)
- a solid shape that smoothly curves from one diameter to another (no hollowing) -- a table leg, a lamp base, a bowling-pin-like shape

Example 1 -- request: "a water bottle" (a well-known object, no dimensions given -- STEP 0 case (a), use realistic typical sizing). Correct reasoning: one hollow-vessel entry for the body+shoulder+neck using the exact required phrasing, plus a separate cap sitting directly on top of the neck (cap z = body height + shoulder height + neck height):
[{"description": "a hollow vessel: a 65mm diameter, 190mm tall body smoothly narrowing to a 25mm diameter, 30mm tall neck over a 25mm shoulder, 2mm walls", "position_mm": {"x": 0, "y": 0, "z": 0}}, {"description": "a cylinder 27mm diameter, 15mm tall", "position_mm": {"x": 0, "y": 0, "z": 245}}]

Example 2 -- request: "a small gearbox: two spur gears meshing on parallel shafts, module 2.5mm, gear A has 14 teeth, gear B has 22 teeth, each gear is mounted on its own 10mm diameter, 60mm long shaft" (STEP 0 case (c)). Correct reasoning: 4 separate mechanical parts (2 shafts, 2 gears), each gear shares its own shaft's x/y position, and gear B's shaft is placed a rough, reasonable gap away from gear A's along one axis (the exact spacing is corrected afterward, not your job to get precisely right):
[{"description": "a cylinder 10mm diameter, 60mm long", "position_mm": {"x": 0, "y": 0, "z": 0}}, {"description": "a gear profile, module 2.5, 14 teeth", "position_mm": {"x": 0, "y": 0, "z": 0}}, {"description": "a cylinder 10mm diameter, 60mm long", "position_mm": {"x": 40, "y": 0, "z": 0}}, {"description": "a gear profile, module 2.5, 22 teeth", "position_mm": {"x": 40, "y": 0, "z": 0}}]

Do NOT design exact numeric geometry yourself beyond what the user already gave (dimensions the user didn't specify stay unspecified in the description, not invented) -- the two exceptions are position_mm placement math (center distances, shared axes) and realistic default sizing for a well-known common object with no dimensions given at all (STEP 0), both of which ARE your job.

Respond with ONLY a JSON array in the same format as the example, no other text, no explanation.

/no_think"""


_CLARIFY_SYSTEM_PROMPT = """You are an experienced mechanical design engineer reviewing a build request before committing to a design, for 3di-men, a CAD builder. Given a user's request, decide whether you need more information before it can be built, and if so, ask up to 3 short, specific questions -- the kind an engineer would actually need answered before laying out the assembly, not generic follow-ups.

Only ask about things that actually block building the geometry:
- a missing dimension needed for a piece the user clearly wants built, when NO reasonable default exists for it (a custom bracket, a shaft, a gear -- there is no "typical" size to guess)
- an ambiguous count or arrangement of a repeated feature the request itself implies exists (how many holes, how many teeth, arranged how) -- never invent a feature to ask about that the request never mentioned or implied
- an ambiguous mechanical relationship between named parts (e.g. where along a shaft a gear sits, which two faces of two housings mate, how far apart two meshing gears' shafts are meant to be if tooth counts are given but spacing isn't)

Do NOT ask about dimensions for a well-known EVERYDAY object (a water bottle, a coffee mug, a pen, a phone case, a box) even if none were given -- realistic, typical real-world sizing for that kind of object is used automatically, and asking wastes the user's time on something that already has a sensible default. Only ask about such an object if the request itself signals a genuinely unusual/custom size ("a giant water bottle", "a tiny mug") without saying how unusual.

NEVER ask about anything this builder cannot represent: material, color, finish, tolerances, manufacturing process, cost, or aesthetic style. Those questions are useless here -- do not ask them.

If the request already gives enough detail to build (even if some dimensions use round/typical numbers, or the object is a well-known everyday item with a sensible default size), return an empty list. Do not ask a question just to be thorough -- most complete, concrete requests, and most well-known everyday objects, need zero questions.

Example 1 -- request: "a bracket with some holes in it". This is a custom mechanical part with no sensible default size or hole count -- missing the bracket's own size and how many holes / where. Correct output:
{"questions": ["What size should the bracket be (length x width x thickness, in mm)?", "How many holes, and roughly where should they go?"]}

Example 1b -- request: "a water bottle". A well-known everyday object with realistic typical sizing available and nothing about holes, slots, or fasteners implied anywhere in the request. Correct output:
{"questions": []}

Example 2 -- request: "a box 50mm x 40mm x 10mm with a 16mm diameter, 20mm tall cylinder centered on top". Every dimension needed to build this is already given. Correct output:
{"questions": []}

Respond with ONLY a JSON object in the form {"questions": [...]}, no other text, no explanation.

/no_think"""


@dataclass
class ActionablePoint:
    description: str
    position_mm: tuple[float, float, float]


@dataclass
class DecomposeResult:
    points: list[ActionablePoint]
    raw_llm_output: str
    fell_back: bool  # True if the LLM's output couldn't be parsed and we fell back to one whole-prompt point


@dataclass
class ClarifyResult:
    questions: list[str]
    raw_llm_output: str


_llm: Any = None  # lazy-loaded singleton -- loading a ~1.2GB model per call would defeat the point


def _get_llm(model_path: Path = DEFAULT_MODEL_PATH):
    global _llm
    if _llm is None:
        from llama_cpp import Llama

        _llm = Llama(model_path=str(model_path), n_ctx=4096, verbose=False)
    return _llm


def _looks_like_actionable_points(parsed: Any) -> bool:
    return (
        isinstance(parsed, list) and len(parsed) > 0
        and all(isinstance(item, dict) and "description" in item for item in parsed)
    )


# Confirmed necessary, not hypothetical: on a real 5-part decomposition this
# model otherwise got completely right, it emitted a stray extra quote
# immediately after two of the numeric position_mm values -- '"z": 0"}}'
# instead of '"z": 0}}' -- which breaks strict JSON parsing outright even
# though the intent is completely unambiguous. Scoped narrowly to the
# "x"/"y"/"z" KEY specifically (not a blanket digit-followed-by-quote
# pattern anywhere in the text), so it can't misfire on a legitimate string
# field that happens to end in a digit followed by its own real closing
# quote (e.g. a description like "...123").
_STRAY_QUOTE_AFTER_POSITION_NUMBER_RE = re.compile(r'("[xyz]"\s*:\s*-?\d+(?:\.\d+)?)"')


def _repair_json_text(text: str) -> str:
    return _STRAY_QUOTE_AFTER_POSITION_NUMBER_RE.sub(r"\1", text)


def _find_schema_shaped_arrays(text: str) -> list[list[dict]]:
    """Finds every well-formed, bracket-balanced [...] substring in `text`
    and keeps only the ones that look like actionable points (a list of
    dicts each with a "description" key), in order of appearance."""
    candidates: list[list[dict]] = []
    depth = 0
    start = None
    for i, ch in enumerate(text):
        if ch == "[":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "]":
            if depth > 0:
                depth -= 1
                if depth == 0 and start is not None:
                    candidate_text = text[start : i + 1]
                    try:
                        parsed = json.loads(candidate_text)
                    except json.JSONDecodeError:
                        try:
                            parsed = json.loads(_repair_json_text(candidate_text))
                        except json.JSONDecodeError:
                            continue
                    if _looks_like_actionable_points(parsed):
                        candidates.append(parsed)
    return candidates


def _extract_json_array(raw: str) -> list[dict]:
    """Only looks after the model's LAST </think> tag (if any -- Qwen3-style
    reasoning-before-answer), so reasoning-time chatter (which can easily
    contain an incidental, unrelated, but still well-formed array, e.g. "the
    user wants [1, 2, 3] things") is never mistaken for the actual answer.

    Within that final-answer segment, ALL schema-shaped arrays found are
    concatenated together, not just the last one -- confirmed necessary,
    not hypothetical: this 1.7B model reliably enumerates every part
    correctly in its reasoning, but frequently formats its final answer as
    several separate single-item arrays back to back
    ("[{...}], [{...}], [{...}]]") instead of one combined array. Taking
    only the last such array silently dropped every part but the final one
    on every multi-part prompt tested. Concatenating is a safe superset for
    the well-formed case too: a model that DOES emit one single combined
    array just produces one candidate, unchanged behavior."""
    answer_segment = raw.rsplit("</think>", 1)[-1]
    candidates = _find_schema_shaped_arrays(answer_segment)
    if not candidates:
        raise ValueError("no schema-shaped JSON array found in LLM output")
    merged: list[dict] = []
    for c in candidates:
        merged.extend(c)
    return merged


def _parse_points(raw: str) -> list[ActionablePoint]:
    items = _extract_json_array(raw)
    points = []
    for item in items:
        pos = item.get("position_mm") or {}
        points.append(
            ActionablePoint(
                description=str(item["description"]),
                position_mm=(float(pos.get("x", 0.0)), float(pos.get("y", 0.0)), float(pos.get("z", 0.0))),
            )
        )
    if not points:
        raise ValueError("LLM returned an empty list")
    return points


# Matches this project's own established gear-description phrasing
# ("a gear profile, module 2, 16 teeth") -- see the _SYSTEM_PROMPT example
# and data/program_dataset.py's generic-composition patterns, which use the
# same wording. Deliberately tolerant of "2mm"/"2" and "module"/"mod".
_GEAR_DESC_RE = re.compile(r"mod(?:ule)?\.?\s*(\d+(?:\.\d+)?)\s*(?:mm)?,?\s*(\d+)\s*teeth", re.IGNORECASE)


def _extract_gear_params(description: str) -> tuple[float, int] | None:
    m = _GEAR_DESC_RE.search(description)
    if not m:
        return None
    return float(m.group(1)), int(m.group(2))


def _correct_gear_mesh_spacing(points: list[ActionablePoint]) -> list[ActionablePoint]:
    """The one piece of real arithmetic a mechanical engineer's decomposition
    needs -- a meshing gear pair's center distance, module * (teeth_a +
    teeth_b) / 2 -- confirmed EMPIRICALLY, not assumed, that this 1.7B local
    model cannot reliably compute: tested against a real prompt with known
    module/tooth counts, it consistently copied a distance straight out of
    the system prompt's own worked examples instead of recomputing for the
    request's actual numbers, even with an explicit "don't copy the
    example" instruction and multiple examples with different results.
    This is exactly the class of precise, deterministic computation this
    project keeps OUT of the LLM's hands on principle (see program_model.py
    and sandbox.py's whole design) -- so it's corrected here in plain
    Python instead of asked of the model at all; the LLM's job stays
    "recognize these are 2 separate meshing parts, roughly apart," which it
    handles fine.

    Scoped to exactly 2 gear-profile points: a single meshing pair is
    unambiguous. 3+ gears (a gear train) have ambiguous meshing topology --
    which gear meshes with which isn't recoverable from a flat points list
    -- so that case is left uncorrected rather than guessed at.

    A gear's own mounting shaft (any OTHER point sharing that gear's exact
    position -- the LLM reliably places a gear and its shaft together, see
    _SYSTEM_PROMPT's own instruction) is shifted by the same delta, so the
    "a part mounted on a shaft shares that shaft's axis" relationship the
    LLM does get right stays intact after the correction.
    """
    gear_indices = [i for i, p in enumerate(points) if _extract_gear_params(p.description) is not None]
    if len(gear_indices) != 2:
        return points

    a, b = gear_indices
    module_a, teeth_a = _extract_gear_params(points[a].description)
    module_b, teeth_b = _extract_gear_params(points[b].description)
    if abs(module_a - module_b) > 1e-6:
        return points  # different modules can't mesh -- something else is going on here, don't guess at it
    expected_distance = module_a * (teeth_a + teeth_b) / 2.0

    pos_a, pos_b = points[a].position_mm, points[b].position_mm
    delta = tuple(pos_b[i] - pos_a[i] for i in range(3))
    current_distance = math.sqrt(sum(c * c for c in delta))
    axis_dir = (1.0, 0.0, 0.0) if current_distance < 1e-6 else tuple(c / current_distance for c in delta)
    corrected_pos_b = tuple(pos_a[i] + axis_dir[i] * expected_distance for i in range(3))
    shift = tuple(corrected_pos_b[i] - pos_b[i] for i in range(3))

    def _same_position(p: tuple[float, float, float], q: tuple[float, float, float]) -> bool:
        return all(abs(p[i] - q[i]) < 1e-6 for i in range(3))

    corrected = []
    for i, p in enumerate(points):
        if i == b or _same_position(p.position_mm, pos_b):
            shifted = tuple(p.position_mm[i] + shift[i] for i in range(3))
            corrected.append(ActionablePoint(description=p.description, position_mm=shifted))
        else:
            corrected.append(p)
    return corrected


def decompose_prompt(prompt: str, model_path: Path = DEFAULT_MODEL_PATH) -> DecomposeResult:
    """Never raises and never returns zero points: a malformed/unparseable
    LLM response falls back to treating the WHOLE prompt as one actionable
    point at the origin -- a typed, disclosed degradation (fell_back=True),
    not a crash and not a silently-wrong decomposition."""
    llm = _get_llm(model_path)
    response = llm.create_chat_completion(
        messages=[{"role": "system", "content": _SYSTEM_PROMPT}, {"role": "user", "content": prompt}],
        temperature=0.1,
        max_tokens=2048,
        # Confirmed necessary, not defensive: on a genuinely complex 5-part
        # prompt, this model got stuck in a repetitive <think> loop
        # ("So the position is... but the user might have intended... So
        # the position is...") and never reached its JSON answer within the
        # old 1536-token budget -- a real truncation, not a parsing bug.
        # /no_think in the system prompt (a documented Qwen3 convention)
        # suppresses the reasoning block that was looping in the first
        # place; repeat_penalty is defense in depth in case some reasoning
        # still slips through.
        repeat_penalty=1.15,
    )
    raw = response["choices"][0]["message"]["content"] or ""

    try:
        points = _parse_points(raw)
        points = _correct_gear_mesh_spacing(points)
        return DecomposeResult(points=points, raw_llm_output=raw, fell_back=False)
    except Exception:
        return DecomposeResult(
            points=[ActionablePoint(description=prompt, position_mm=(0.0, 0.0, 0.0))],
            raw_llm_output=raw, fell_back=True,
        )


def _find_schema_shaped_objects(text: str) -> list[dict]:
    """Brace-balanced counterpart to _find_schema_shaped_arrays: finds every
    well-formed {...} substring that looks like a clarify-stage answer (a
    dict with a "questions" key holding a list), in order of appearance."""
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
                        try:
                            parsed = json.loads(_repair_json_text(candidate_text))
                        except json.JSONDecodeError:
                            continue
                    if isinstance(parsed, dict) and isinstance(parsed.get("questions"), list):
                        candidates.append(parsed)
    return candidates


def ask_clarifying_questions(prompt: str, model_path: Path = DEFAULT_MODEL_PATH, max_questions: int = 3) -> ClarifyResult:
    """Never raises: any parse failure is treated the same as "no questions
    needed" (an empty list) so a clarification-stage hiccup degrades to
    building straight from the original prompt, exactly like decompose_prompt's
    own fallback -- a clarification step that could itself crash or block the
    user would defeat the point of it being a helpful, optional step."""
    llm = _get_llm(model_path)
    try:
        response = llm.create_chat_completion(
            messages=[{"role": "system", "content": _CLARIFY_SYSTEM_PROMPT}, {"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=512,
            repeat_penalty=1.15,
        )
        raw = response["choices"][0]["message"]["content"] or ""
    except Exception:
        return ClarifyResult(questions=[], raw_llm_output="")

    answer_segment = raw.rsplit("</think>", 1)[-1]
    candidates = _find_schema_shaped_objects(answer_segment)
    if not candidates:
        return ClarifyResult(questions=[], raw_llm_output=raw)

    questions = [str(q).strip() for q in candidates[0]["questions"] if str(q).strip()]
    return ClarifyResult(questions=questions[:max_questions], raw_llm_output=raw)


def fold_answers_into_prompt(prompt: str, qa_pairs: list[tuple[str, str]]) -> str:
    """Combines the original prompt with the user's answers to the
    clarify-stage questions into one enriched prompt text -- kept as a
    plain string append rather than a new decompose_prompt code path, so
    decompose_prompt's own extraction/fallback logic never needs to know
    a clarify stage happened at all."""
    answered = [(q, a) for q, a in qa_pairs if a and a.strip()]
    if not answered:
        return prompt
    details = "\n".join(f"- {q} {a.strip()}" for q, a in answered)
    return f"{prompt}\n\nAdditional details:\n{details}"
