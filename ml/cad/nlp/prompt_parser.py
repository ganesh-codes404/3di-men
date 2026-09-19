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
milliseconds, using the procedural part families in procedural_generator.py.

Six part families exist today (plate, flange, bracket, shaft, enclosure, gear
-- see part_families.py for the registry). Anything not covered by one of
these six still gets mapped onto whichever is the closer geometric analogue
(default: plate), with that mapping recorded in the result (part_type_inferred)
so it's never silently wrong. Extending part coverage further is future work,
not a defect in the parsing logic itself. One specific, documented gap: a bare
"housing" always classifies as flange (round), not enclosure (rectangular) --
see _FLANGE_KEYWORDS/_ENCLOSURE_KEYWORDS below for why that's deliberate.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from data.procedural_generator import BracketParams, EnclosureParams, FlangeParams, GearParams, MountingPlateParams, ShaftParams
from part_families import resolve_default

_WORD_NUMBERS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
}

# Nominal metric thread OD in mm, for prompts that say "M8" instead of "8mm".
_METRIC_THREAD_MM = {"m3": 3, "m4": 4, "m5": 5, "m6": 6, "m8": 8, "m10": 10, "m12": 12, "m16": 16}

# Keywords that map a free-text part description onto one of the six available
# procedural families. Order matters: checked top to bottom, first match wins.
#
# "bracket" moved here from _PLATE_KEYWORDS once a real L-shaped bracket family
# existed -- a deliberate, more-correct behavior change, not an oversight.
# "housing" deliberately stays flange-only, not added to _ENCLOSURE_KEYWORDS:
# a bare "housing" is genuinely ambiguous between a round flange-housing and a
# rectangular box, unlike bracket where the new family is an unambiguous
# improvement. "electronics housing" -> classified as flange is an accepted,
# documented v1 gap, not a bug to chase.
#
# "bore" was REMOVED from here after real testing (not just reasoning) showed
# it was wrong: a "bore" is legitimately standard terminology for a shaft's
# or gear's center hole too, not flange-exclusive -- with "bore" as a flange
# keyword checked first, ~20% of synthetic shaft/gear prompts that happened
# to mention "bore" were misclassified as flange. Flange doesn't need it as
# a fallback signal: its own generated/realistic text always says "flange"
# (or "housing"/"pump"/"bearing"/"pulley"), which are all still unambiguous.
_FLANGE_KEYWORDS = ["flange", "housing", "pump", "bearing", "pulley"]
_PLATE_KEYWORDS = ["plate", "mount", "panel"]
_BRACKET_KEYWORDS = ["bracket"]  # deliberately NOT "angle" -- "20 degree pressure angle" in a
                                  # gear clause would spuriously anchor a bracket
_SHAFT_KEYWORDS = ["shaft", "axle", "rod", "spindle"]
_ENCLOSURE_KEYWORDS = ["enclosure", "box", "case", "cover"]
_GEAR_KEYWORDS = ["gear", "cog", "sprocket"]


@dataclass
class ParseResult:
    part_type: str  # "plate" | "flange" | "bracket" | "shaft" | "enclosure" | "gear"
    params: MountingPlateParams | FlangeParams | BracketParams | ShaftParams | EnclosureParams | GearParams
    parsed_fields: set[str]  # fields that came directly from the prompt text
    defaulted_fields: set[str]  # fields filled in with a default, not requested
    part_type_inferred: bool  # True if no explicit keyword matched and we guessed "plate"


def _find_number(pattern: str | list[str], text: str) -> float | None:
    """Accepts either one pattern (the original call shape, unchanged for
    every existing caller) or a list of patterns tried in order, first
    match wins -- used where a field needs to match more than one common
    phrasing (see _prepositional and _infix below)."""
    for p in [pattern] if isinstance(pattern, str) else pattern:
        m = re.search(p, text, re.IGNORECASE)
        if m:
            return float(m.group(1))
    return None


def _prepositional(descriptor: str) -> str:
    """"<descriptor> of N mm" -- e.g. "a diameter of 35 mm", "a length of
    200 mm". Common in verbose engineering-spec prose; the rest of this
    file's patterns are all the postpositive "N mm <descriptor>" style
    ("35mm diameter"), which this doesn't replace, only supplements."""
    return rf"{descriptor}\s+of\s+(\d+(?:\.\d+)?)\s*mm"


def _infix(descriptor: str) -> str:
    """"N mm in <descriptor>" -- e.g. "14 mm in diameter"."""
    return rf"(\d+(?:\.\d+)?)\s*mm\s+in\s+{descriptor}"


def _find_count(pattern: str, text: str) -> int | None:
    m = re.search(pattern, text, re.IGNORECASE)
    if not m:
        return None
    token = m.group(1).lower()
    if token.isdigit():
        return int(token)
    return _WORD_NUMBERS.get(token)


def _find_hole_diameter(text: str) -> float | None:
    d = _find_number([
        r"(\d+(?:\.\d+)?)\s*mm\s+(?:mounting\s+)?holes?",
        # a hole's own diameter given as a trailing/prepositional clause
        # rather than "<N>mm hole(s)" -- common in verbose engineering-spec
        # phrasing ("each 14 mm in diameter", "a diameter of 40 mm").
        _infix("diameter"), _prepositional("diameter"),
    ], text)
    if d is not None:
        return d
    m = re.search(r"\b(m\d{1,2})\b", text, re.IGNORECASE)
    if m:
        return _METRIC_THREAD_MM.get(m.group(1).lower())
    return None


def _classify_part_type(text: str) -> tuple[str, bool]:
    """Leftmost keyword match wins -- the family whose keyword appears
    EARLIEST in the text, not a fixed family-priority order. This matters
    once text gets longer than a short single-part description (e.g. a
    multi-sentence assembly-paragraph clause fed in via assembly.py): a
    paragraph about a shaft that mentions "brackets" later for spatial
    context ("the shaft passes through the circular holes in both
    brackets") must still classify as shaft, not bracket, just because
    bracket happens to be checked first. A fixed-order "first family that
    matches anywhere" rule got this wrong -- confirmed concretely on such a
    paragraph, which classified as bracket instead of shaft.

    One deliberate exception to plain leftmost-wins: a genuine "bracket"
    keyword match always beats a _PLATE_KEYWORDS "mount" match, regardless
    of position. "mount" is a loose, substring-based keyword (it has to
    match "mounting"/"mounted" too), so "mounting bracket" would otherwise
    misclassify as plate whenever "mount" happens to appear before
    "bracket" -- which it usually does, since "mounting" is often the
    adjective leading into "bracket". This is the same fix the previous
    fixed-order scheme existed to make (see git history), just narrowed to
    the one collision it actually needs to cover instead of reordering
    every family around it."""
    lowered = text.lower()
    matches: list[tuple[int, str]] = []
    for part_type, keywords in (
        ("flange", _FLANGE_KEYWORDS),
        ("bracket", _BRACKET_KEYWORDS),
        ("plate", _PLATE_KEYWORDS),
        ("shaft", _SHAFT_KEYWORDS),
        ("enclosure", _ENCLOSURE_KEYWORDS),
        ("gear", _GEAR_KEYWORDS),
    ):
        for kw in keywords:
            idx = lowered.find(kw)
            if idx != -1:
                matches.append((idx, part_type))
                break  # only the earliest occurrence per family matters
    if not matches:
        return "plate", True  # default guess when nothing matches
    matches.sort(key=lambda m: m[0])
    if matches[0][1] == "plate" and any(pt == "bracket" for _, pt in matches):
        return "bracket", False
    return matches[0][1], False


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
        # Each dimension is checked (and credited to parsed/defaulted)
        # independently -- a prompt giving only some of the three ("300 mm
        # long, 220 mm wide, and 15 mm thick" as three separate postpositive
        # clauses, not one combined "LxWxH mm") must not have all three
        # blanket-marked "defaulted" just because they didn't come from the
        # single combined-pattern branch above. This was a real bug: the
        # VALUES were always parsed correctly here, but every one of them
        # was mislabeled "defaulted" regardless, which is exactly the
        # metadata nlp/clarify.py's clarifying-question feature relies on
        # to decide what to ask about -- an already-answered field would
        # otherwise get asked about again.
        length = _find_number(r"(\d+(?:\.\d+)?)\s*mm\s+long", text)
        width = _find_number(r"(\d+(?:\.\d+)?)\s*mm\s+wide", text)
        thickness = _find_number(r"(\d+(?:\.\d+)?)\s*mm\s+thick", text)
        for name, value in (("length_mm", length), ("width_mm", width), ("thickness_mm", thickness)):
            (parsed if value is not None else defaulted).add(name)
        length = length if length is not None else resolve_default("plate", "length_mm", {})
        width = width if width is not None else resolve_default("plate", "width_mm", {})
        thickness = thickness if thickness is not None else resolve_default("plate", "thickness_mm", {})

    hole_count = _find_count(r"(\w+)\s+(?:mounting\s+)?holes?", text)
    if hole_count is None:
        hole_count = resolve_default("plate", "hole_count", {})
        defaulted.add("hole_count")
    else:
        parsed.add("hole_count")
    hole_count = hole_count if hole_count in (2, 4) else (4 if hole_count > 2 else 2)

    hole_d = _find_hole_diameter(text)
    if hole_d is None:
        hole_d = resolve_default("plate", "hole_diameter_mm", {})
        defaulted.add("hole_diameter_mm")
    else:
        parsed.add("hole_diameter_mm")

    margin = _find_number(r"(\d+(?:\.\d+)?)\s*mm\s+from\s+(?:each|the)\s+edge", text)
    if margin is None:
        margin = resolve_default("plate", "hole_edge_margin_mm", {})
        defaulted.add("hole_edge_margin_mm")
    else:
        parsed.add("hole_edge_margin_mm")

    fillet = _find_number(r"(\d+(?:\.\d+)?)\s*mm\s+(?:corner\s+)?fillet", text)
    if fillet is None:
        fillet = resolve_default("plate", "fillet_radius_mm", {})
        defaulted.add("fillet_radius_mm")
    else:
        parsed.add("fillet_radius_mm")

    material_match = re.search(r"\b(aluminum|steel|polycarbonate|titanium|plastic|brass)\b", text, re.IGNORECASE)
    material = material_match.group(1).lower() if material_match else resolve_default("plate", "material", {})
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
        outer_d = resolve_default("flange", "outer_diameter_mm", {})
        defaulted.add("outer_diameter_mm")
    else:
        parsed.add("outer_diameter_mm")

    bore_d = _find_number(r"(\d+(?:\.\d+)?)\s*mm\s+(?:central\s+)?bore", text)
    if bore_d is None:
        bore_d = resolve_default("flange", "bore_diameter_mm", {})
        defaulted.add("bore_diameter_mm")
    else:
        parsed.add("bore_diameter_mm")

    thickness = _find_number(r"(\d+(?:\.\d+)?)\s*mm\s+(?:wall\s+)?thick(?:ness)?", text)
    if thickness is None:
        thickness = resolve_default("flange", "thickness_mm", {})
        defaulted.add("thickness_mm")
    else:
        parsed.add("thickness_mm")

    bolt_count = _find_count(r"(\w+)\s+(?:m\d{1,2}\s+)?mounting\s+holes?", text)
    if bolt_count is None:
        bolt_count = resolve_default("flange", "bolt_hole_count", {})
        defaulted.add("bolt_hole_count")
    else:
        parsed.add("bolt_hole_count")

    bolt_d = _find_hole_diameter(text)
    if bolt_d is None:
        bolt_d = resolve_default("flange", "bolt_hole_diameter_mm", {})
        defaulted.add("bolt_hole_diameter_mm")
    else:
        parsed.add("bolt_hole_diameter_mm")

    bcd = _find_number(r"(\d+(?:\.\d+)?)\s*mm\s+bolt\s+circle", text)
    if bcd is None:
        bcd = resolve_default("flange", "bolt_circle_diameter_mm", {"outer_diameter_mm": outer_d})
        defaulted.add("bolt_circle_diameter_mm")
    else:
        parsed.add("bolt_circle_diameter_mm")

    material_match = re.search(r"\b(steel|aluminum|cast iron|titanium|brass)\b", text, re.IGNORECASE)
    material = material_match.group(1).lower() if material_match else resolve_default("flange", "material", {})
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


def _parse_bracket(text: str) -> tuple[BracketParams, set[str], set[str]]:
    parsed: set[str] = set()
    defaulted: set[str] = set()

    legs = re.search(r"(\d+(?:\.\d+)?)\s*(?:mm)?\s*[x×]\s*(\d+(?:\.\d+)?)\s*mm\s+legs?", text, re.IGNORECASE)
    if legs:
        leg1, leg2 = float(legs.group(1)), float(legs.group(2))
        parsed |= {"leg1_length_mm", "leg2_length_mm"}
    else:
        leg1 = resolve_default("bracket", "leg1_length_mm", {})
        leg2 = resolve_default("bracket", "leg2_length_mm", {})
        defaulted |= {"leg1_length_mm", "leg2_length_mm"}

    thickness = _find_number(r"(\d+(?:\.\d+)?)\s*mm\s+thick", text)
    if thickness is None:
        thickness = resolve_default("bracket", "thickness_mm", {})
        defaulted.add("thickness_mm")
    else:
        parsed.add("thickness_mm")

    length = _find_number(r"(\d+(?:\.\d+)?)\s*mm\s+long", text)
    if length is None:
        length = resolve_default("bracket", "length_mm", {})
        defaulted.add("length_mm")
    else:
        parsed.add("length_mm")

    hole_d = _find_hole_diameter(text)
    if hole_d is None:
        hole_d = resolve_default("bracket", "hole_diameter_mm", {})
        defaulted.add("hole_diameter_mm")
    else:
        parsed.add("hole_diameter_mm")

    margin = _find_number(r"(\d+(?:\.\d+)?)\s*mm\s+from\s+(?:each|the)\s+edge", text)
    if margin is None:
        margin = resolve_default("bracket", "hole_edge_margin_mm", {})
        defaulted.add("hole_edge_margin_mm")
    else:
        parsed.add("hole_edge_margin_mm")

    holes_per_arm = _find_count(r"(\w+)\s+holes?\s+per\s+arm", text)
    if holes_per_arm is None:
        holes_per_arm = resolve_default("bracket", "holes_per_arm", {})
        defaulted.add("holes_per_arm")
    else:
        holes_per_arm = 1 if holes_per_arm < 2 else 2
        parsed.add("holes_per_arm")

    material_match = re.search(r"\b(steel|aluminum|cast iron|titanium|brass)\b", text, re.IGNORECASE)
    material = material_match.group(1).lower() if material_match else resolve_default("bracket", "material", {})
    (parsed if material_match else defaulted).add("material")

    params = BracketParams(
        leg1_length_mm=leg1, leg2_length_mm=leg2, thickness_mm=thickness, length_mm=length,
        hole_diameter_mm=hole_d, hole_edge_margin_mm=margin, holes_per_arm=holes_per_arm, material=material,
    )
    return params, parsed, defaulted


def _parse_shaft(text: str) -> tuple[ShaftParams, set[str], set[str]]:
    parsed: set[str] = set()
    defaulted: set[str] = set()

    od = _find_number([
        r"(\d+(?:\.\d+)?)\s*mm\s+(?:outer\s+)?diameter", _prepositional(r"(?:outer\s+)?diameter"), _infix("diameter"),
    ], text)
    if od is None:
        od = resolve_default("shaft", "outer_diameter_mm", {})
        defaulted.add("outer_diameter_mm")
    else:
        parsed.add("outer_diameter_mm")

    length = _find_number([r"(\d+(?:\.\d+)?)\s*mm\s+long", _prepositional("length")], text)
    if length is None:
        length = resolve_default("shaft", "length_mm", {})
        defaulted.add("length_mm")
    else:
        parsed.add("length_mm")

    inner_d = _find_number(r"(\d+(?:\.\d+)?)\s*mm\s+bore", text)
    if inner_d is None:
        inner_d = resolve_default("shaft", "inner_diameter_mm", {})
        defaulted.add("inner_diameter_mm")
    else:
        parsed.add("inner_diameter_mm")

    keyway_w = _find_number(r"(\d+(?:\.\d+)?)\s*mm\s+keyway", text)
    if keyway_w is None:
        keyway_w = resolve_default("shaft", "keyway_width_mm", {})
        defaulted.add("keyway_width_mm")
    else:
        parsed.add("keyway_width_mm")

    material_match = re.search(r"\b(steel|aluminum|titanium|brass)\b", text, re.IGNORECASE)
    material = material_match.group(1).lower() if material_match else resolve_default("shaft", "material", {})
    (parsed if material_match else defaulted).add("material")

    # not directly parsed from text (rarely stated explicitly) -- always defaulted
    keyway_depth = resolve_default("shaft", "keyway_depth_mm", {})
    keyway_length = resolve_default("shaft", "keyway_length_mm", {})
    defaulted |= {"keyway_depth_mm", "keyway_length_mm"}

    params = ShaftParams(
        outer_diameter_mm=od, length_mm=length, inner_diameter_mm=inner_d,
        keyway_width_mm=keyway_w, keyway_depth_mm=keyway_depth, keyway_length_mm=keyway_length, material=material,
    )
    return params, parsed, defaulted


def _parse_enclosure(text: str) -> tuple[EnclosureParams, set[str], set[str]]:
    parsed: set[str] = set()
    defaulted: set[str] = set()

    dims = re.search(
        r"(\d+(?:\.\d+)?)\s*(?:mm)?\s*[x×]\s*(\d+(?:\.\d+)?)\s*(?:mm)?\s*[x×]\s*(\d+(?:\.\d+)?)\s*mm", text
    )
    if dims:
        length, width, height = (float(g) for g in dims.groups())
        parsed |= {"length_mm", "width_mm", "height_mm"}
    else:
        # Postpositive fallback ("240 mm long, 180 mm wide, and 100 mm
        # tall") -- mirrors _parse_plate/_parse_bracket's own fallback
        # exactly, just missing here until now. Each dimension is checked
        # independently so a prompt giving only some of the three still
        # gets credit for the ones it did specify.
        length = _find_number(r"(\d+(?:\.\d+)?)\s*mm\s+long", text)
        width = _find_number(r"(\d+(?:\.\d+)?)\s*mm\s+wide", text)
        height = _find_number(r"(\d+(?:\.\d+)?)\s*mm\s+tall", text)
        for name, value in (("length_mm", length), ("width_mm", width), ("height_mm", height)):
            (parsed if value is not None else defaulted).add(name)
        length = length if length is not None else resolve_default("enclosure", "length_mm", {})
        width = width if width is not None else resolve_default("enclosure", "width_mm", {})
        height = height if height is not None else resolve_default("enclosure", "height_mm", {})

    wall = _find_number(r"(\d+(?:\.\d+)?)\s*mm\s+wall", text)
    if wall is None:
        wall = resolve_default("enclosure", "wall_thickness_mm", {})
        defaulted.add("wall_thickness_mm")
    else:
        parsed.add("wall_thickness_mm")

    if re.search(r"\bopen\s+top\b", text, re.IGNORECASE):
        open_face = "top"
        parsed.add("open_face")
    else:
        open_face = resolve_default("enclosure", "open_face", {})
        defaulted.add("open_face")

    vent_count = _find_count(r"(\w+)\s+(?:\d+(?:\.\d+)?\s*mm\s+)?vent(?:ilation)?\s+holes?", text)
    vent_d = _find_number(r"(\d+(?:\.\d+)?)\s*mm\s+vent(?:ilation)?\s+holes?", text)
    if vent_count is None and vent_d is None:
        vent_count = resolve_default("enclosure", "vent_hole_count", {})
        vent_d = resolve_default("enclosure", "vent_hole_diameter_mm", {})
        defaulted |= {"vent_hole_count", "vent_hole_diameter_mm"}
    else:
        vent_count = vent_count if vent_count is not None else 4
        vent_d = vent_d if vent_d is not None else resolve_default("enclosure", "vent_hole_diameter_mm", {})
        parsed |= {"vent_hole_count", "vent_hole_diameter_mm"}

    material_match = re.search(r"\b(aluminum|steel|plastic|polycarbonate)\b", text, re.IGNORECASE)
    material = material_match.group(1).lower() if material_match else resolve_default("enclosure", "material", {})
    (parsed if material_match else defaulted).add("material")

    params = EnclosureParams(
        length_mm=length, width_mm=width, height_mm=height, wall_thickness_mm=wall, open_face=open_face,
        vent_hole_count=vent_count, vent_hole_diameter_mm=vent_d, material=material,
    )
    return params, parsed, defaulted


def _parse_gear(text: str) -> tuple[GearParams, set[str], set[str]]:
    parsed: set[str] = set()
    defaulted: set[str] = set()

    module = _find_number(r"module\s+(\d+(?:\.\d+)?)", text)
    if module is None:
        module = resolve_default("gear", "module_mm", {})
        defaulted.add("module_mm")
    else:
        parsed.add("module_mm")

    tooth_count = _find_count(r"(\w+)\s+teeth", text)
    if tooth_count is None:
        tooth_count = resolve_default("gear", "tooth_count", {})
        defaulted.add("tooth_count")
    else:
        parsed.add("tooth_count")

    face_width = _find_number(r"(\d+(?:\.\d+)?)\s*mm\s+face\s+width", text)
    if face_width is None:
        face_width = resolve_default("gear", "face_width_mm", {})
        defaulted.add("face_width_mm")
    else:
        parsed.add("face_width_mm")

    bore_d = _find_number(r"(\d+(?:\.\d+)?)\s*mm\s+bore", text)
    if bore_d is None:
        bore_d = resolve_default("gear", "bore_diameter_mm", {})
        defaulted.add("bore_diameter_mm")
    else:
        parsed.add("bore_diameter_mm")

    # pressure angle is essentially never stated in casual prompts -- always defaulted
    pressure_angle = resolve_default("gear", "pressure_angle_deg", {})
    defaulted.add("pressure_angle_deg")

    material_match = re.search(r"\b(steel|aluminum|brass|titanium)\b", text, re.IGNORECASE)
    material = material_match.group(1).lower() if material_match else resolve_default("gear", "material", {})
    (parsed if material_match else defaulted).add("material")

    params = GearParams(
        module_mm=module, tooth_count=int(tooth_count), face_width_mm=face_width,
        bore_diameter_mm=bore_d, pressure_angle_deg=pressure_angle, material=material,
    )
    return params, parsed, defaulted


_PARSERS = {
    "plate": _parse_plate,
    "flange": _parse_flange,
    "bracket": _parse_bracket,
    "shaft": _parse_shaft,
    "enclosure": _parse_enclosure,
    "gear": _parse_gear,
}


def parse_prompt(text: str) -> ParseResult:
    part_type, inferred = _classify_part_type(text)
    params, parsed, defaulted = _PARSERS[part_type](text)
    return ParseResult(
        part_type=part_type,
        params=params,
        parsed_fields=parsed,
        defaulted_fields=defaulted,
        part_type_inferred=inferred,
    )
