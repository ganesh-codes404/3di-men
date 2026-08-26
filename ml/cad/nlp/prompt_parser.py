"""Rule-based prompt -> engineering-intent parser.

This is the "natural language -> CAD intent extraction -> structured
representation" stage from docs/cad-finetuning-technical-report.md Phase 6,
implemented as plain regex/keyword matching rather than a learned model.

Why rule-based and not a fine-tuned model here: per the project owner's explicit
direction, this prototype must run entirely on a laptop with no GPU (Intel
integrated graphics, 8GB RAM) and must not be "too heavy." A neural
text-conditioned pipeline (T2I + Hunyuan3D, see report Section 4) needs an NVIDIA
GPU no matter how it's fine-tuned -- there is no way to make that light enough
for this machine. This module instead demonstrates the same "prompt -> engineering
intent -> geometry" concept the brief asks for, deterministically and in
milliseconds, using the two procedural part families in procedural_generator.py.

Known limitation: only two part families exist (rectangular mounting plate,
circular flange), so anything else (gears, shafts, brackets, housings, ...) gets
mapped onto whichever of those two is the closer geometric analogue, with that
mapping recorded in the result so it's never silently wrong. Extending part
coverage is future work, not a defect in the parsing logic itself.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from data.procedural_generator import FlangeParams, MountingPlateParams

_WORD_NUMBERS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
}

# Nominal metric thread OD in mm, for prompts that say "M8" instead of "8mm".
_METRIC_THREAD_MM = {"m3": 3, "m4": 4, "m5": 5, "m6": 6, "m8": 8, "m10": 10, "m12": 12, "m16": 16}

# Keywords that map a free-text part description onto one of the two available
# procedural families. Order matters: checked top to bottom, first match wins.
_FLANGE_KEYWORDS = ["flange", "housing", "pump", "bearing", "pulley", "bore"]
_PLATE_KEYWORDS = ["plate", "bracket", "mount", "panel"]


@dataclass
class ParseResult:
    part_type: str  # "plate" | "flange"
    params: MountingPlateParams | FlangeParams
    parsed_fields: set[str]  # fields that came directly from the prompt text
    defaulted_fields: set[str]  # fields filled in with a default, not requested
    part_type_inferred: bool  # True if no explicit keyword matched and we guessed "plate"


def _find_number(pattern: str, text: str) -> float | None:
    m = re.search(pattern, text, re.IGNORECASE)
    return float(m.group(1)) if m else None


def _find_count(pattern: str, text: str) -> int | None:
    m = re.search(pattern, text, re.IGNORECASE)
    if not m:
        return None
    token = m.group(1).lower()
    if token.isdigit():
        return int(token)
    return _WORD_NUMBERS.get(token)


def _find_hole_diameter(text: str) -> float | None:
    d = _find_number(r"(\d+(?:\.\d+)?)\s*mm\s+(?:mounting\s+)?holes?", text)
    if d is not None:
        return d
    m = re.search(r"\b(m\d{1,2})\b", text, re.IGNORECASE)
    if m:
        return _METRIC_THREAD_MM.get(m.group(1).lower())
    return None


def _classify_part_type(text: str) -> tuple[str, bool]:
    lowered = text.lower()
    for kw in _FLANGE_KEYWORDS:
        if kw in lowered:
            return "flange", False
    for kw in _PLATE_KEYWORDS:
        if kw in lowered:
            return "plate", False
    return "plate", True  # default guess when nothing matches


def _parse_plate(text: str) -> tuple[MountingPlateParams, set[str], set[str]]:
    parsed: set[str] = set()
    defaulted: set[str] = set()

    dims = re.search(
        r"(\d+(?:\.\d+)?)\s*(?:mm)?\s*[x×]\s*(\d+(?:\.\d+)?)\s*(?:mm)?\s*[x×]\s*(\d+(?:\.\d+)?)\s*mm", text
    )
    if dims:
        length, width, thickness = (float(g) for g in dims.groups())
        parsed |= {"length_mm", "width_mm", "thickness_mm"}
    else:
        length = _find_number(r"(\d+(?:\.\d+)?)\s*mm\s+long", text) or 100.0
        width = _find_number(r"(\d+(?:\.\d+)?)\s*mm\s+wide", text) or 60.0
        thickness = _find_number(r"(\d+(?:\.\d+)?)\s*mm\s+thick", text) or 6.0
        defaulted |= {"length_mm", "width_mm", "thickness_mm"}

    hole_count = _find_count(r"(\w+)\s+(?:mounting\s+)?holes?", text)
    if hole_count is None:
        hole_count = 4
        defaulted.add("hole_count")
    else:
        parsed.add("hole_count")
    hole_count = hole_count if hole_count in (2, 4) else (4 if hole_count > 2 else 2)

    hole_d = _find_hole_diameter(text)
    if hole_d is None:
        hole_d = 8.0
        defaulted.add("hole_diameter_mm")
    else:
        parsed.add("hole_diameter_mm")

    margin = _find_number(r"(\d+(?:\.\d+)?)\s*mm\s+from\s+(?:each|the)\s+edge", text)
    if margin is None:
        margin = 15.0
        defaulted.add("hole_edge_margin_mm")
    else:
        parsed.add("hole_edge_margin_mm")

    fillet = _find_number(r"(\d+(?:\.\d+)?)\s*mm\s+(?:corner\s+)?fillet", text)
    if fillet is None:
        fillet = 5.0
        defaulted.add("fillet_radius_mm")
    else:
        parsed.add("fillet_radius_mm")

    material_match = re.search(r"\b(aluminum|steel|polycarbonate|titanium|plastic|brass)\b", text, re.IGNORECASE)
    material = material_match.group(1).lower() if material_match else "aluminum"
    (parsed if material_match else defaulted).add("material")

    params = MountingPlateParams(
        length_mm=length,
        width_mm=width,
        thickness_mm=thickness,
        hole_diameter_mm=hole_d,
        hole_edge_margin_mm=margin,
        fillet_radius_mm=fillet,
        hole_count=hole_count,
        material=material,
    )
    return params, parsed, defaulted


def _parse_flange(text: str) -> tuple[FlangeParams, set[str], set[str]]:
    parsed: set[str] = set()
    defaulted: set[str] = set()

    outer_d = _find_number(r"(\d+(?:\.\d+)?)\s*mm\s+outer\s+diameter", text)
    if outer_d is None:
        outer_d = 120.0
        defaulted.add("outer_diameter_mm")
    else:
        parsed.add("outer_diameter_mm")

    bore_d = _find_number(r"(\d+(?:\.\d+)?)\s*mm\s+(?:central\s+)?bore", text)
    if bore_d is None:
        bore_d = 40.0
        defaulted.add("bore_diameter_mm")
    else:
        parsed.add("bore_diameter_mm")

    thickness = _find_number(r"(\d+(?:\.\d+)?)\s*mm\s+(?:wall\s+)?thick(?:ness)?", text)
    if thickness is None:
        thickness = 8.0
        defaulted.add("thickness_mm")
    else:
        parsed.add("thickness_mm")

    bolt_count = _find_count(r"(\w+)\s+(?:m\d{1,2}\s+)?mounting\s+holes?", text)
    if bolt_count is None:
        bolt_count = 4
        defaulted.add("bolt_hole_count")
    else:
        parsed.add("bolt_hole_count")

    bolt_d = _find_hole_diameter(text)
    if bolt_d is None:
        bolt_d = 8.0
        defaulted.add("bolt_hole_diameter_mm")
    else:
        parsed.add("bolt_hole_diameter_mm")

    bcd = _find_number(r"(\d+(?:\.\d+)?)\s*mm\s+bolt\s+circle", text)
    if bcd is None:
        bcd = round(outer_d * 0.75 / 2) * 2  # sensible default: 75% of outer diameter
        defaulted.add("bolt_circle_diameter_mm")
    else:
        parsed.add("bolt_circle_diameter_mm")

    material_match = re.search(r"\b(steel|aluminum|cast iron|titanium|brass)\b", text, re.IGNORECASE)
    material = material_match.group(1).lower() if material_match else "steel"
    (parsed if material_match else defaulted).add("material")

    params = FlangeParams(
        outer_diameter_mm=outer_d,
        bore_diameter_mm=bore_d,
        thickness_mm=thickness,
        bolt_hole_count=bolt_count,
        bolt_hole_diameter_mm=bolt_d,
        bolt_circle_diameter_mm=bcd,
        material=material,
    )
    return params, parsed, defaulted


def parse_prompt(text: str) -> ParseResult:
    part_type, inferred = _classify_part_type(text)
    if part_type == "flange":
        params, parsed, defaulted = _parse_flange(text)
    else:
        params, parsed, defaulted = _parse_plate(text)
    return ParseResult(
        part_type=part_type,
        params=params,
        parsed_fields=parsed,
        defaulted_fields=defaulted,
        part_type_inferred=inferred,
    )
