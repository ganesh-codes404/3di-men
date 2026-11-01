"""Typed engineering-dimension and feature-count requirements, extracted
from a build request's own text -- the shared representation
dimension_assignment.py uses to attach each requirement to a specific
(operation, parameter) pair, and the final reconciliation report checks
the BUILT geometry against.

Confirmed real root causes this exists to fix (live-reproduced against
the actual trained bundle, not hypothetical):

Failure A -- "A bracket with a 5 mm thick base and two 6 mm mounting
holes.": predict_program_checked predicted add_extrusion with
height=58.29mm (requested: 5mm) and profile_kind="l_shape" (never
requested). Root cause: program_model._override_primary_shape_dimensions
extracts every literal "Nmm" in the WHOLE text as one flat, unlabeled
list (here: [5, 6]) and only overrides an op's numeric fields when that
list has AT LEAST as many entries as the op has numeric fields -- here
add_extrusion needs 4 (param_a, param_b, param_c, height) but the text
only gives 2, so the override bails out entirely and even the one
UNAMBIGUOUS number (5mm, sitting right next to the word "thick") never
reaches anything. The fix has to label each number with what it
actually describes, not require the text to supply exactly as many
numbers as the op has slots.

Failure C -- "A cylindrical component with 40 mm outer diameter, 12 mm
through-bore, and 30 mm height.": predicted add_cylinder height=12mm
(requested: 30mm) -- the bore's OWN diameter (12) leaked into the base
shape's height field. Root cause: the same flat-list override zips
literals IN DOCUMENT ORDER against ARG_SCHEMA's field order
([radius, height] -> [40, 12]), with zero regard for the fact that the
"12" sits right next to "bore," not "height" -- the real height value
(30) is never even reached. This is the textbook "numeric parameter
contamination" this module's own extraction is built to prevent: every
extracted number carries its own `source_phrase` and the descriptive
keyword that justified its `parameter` tag, so assignment (see
dimension_assignment.py) never has to guess from position alone.

Deliberately reuses this project's own proven extraction techniques
(spec_validation.extract_claimed_hole_count's curated descriptor-word
list and count-word table, op_requirements.bore_type's through/blind
detection) rather than re-deriving them -- "one authoritative
representation," not a second competing parser for the same facts."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from itertools import count as _counter

from spec_validation import _HOLE_COUNT_WORDS, _HOLE_DESCRIPTOR_WORDS, extract_claimed_hole_count

_id_counter = _counter(1)


def _next_id(prefix: str) -> str:
    return f"{prefix}-{next(_id_counter):03d}"


class RequirementStatus(Enum):
    """The lifecycle a single requirement moves through. EXTRACTED is
    the only status extract_dimension_requirements()/
    extract_feature_count_requirements() ever produce -- every later
    status is set by dimension_assignment.py/the verification stage,
    never guessed here."""

    EXTRACTED = "EXTRACTED"
    ASSIGNED = "ASSIGNED"
    PLANNED = "PLANNED"
    EXECUTED = "EXECUTED"
    GEOMETRY_VERIFIED = "GEOMETRY_VERIFIED"
    UNRESOLVED = "UNRESOLVED"  # explicit number, but no parameter could be determined with confidence
    UNASSIGNED = "UNASSIGNED"  # a parameter WAS determined, but no op in the chosen program has it
    UNSUPPORTED = "UNSUPPORTED"  # the chosen program's op doesn't accept this requirement at all
    FAILED = "FAILED"  # assigned and executed, but the built geometry doesn't satisfy it, and repair couldn't fix it
    UNSUPPORTED_TO_VERIFY = "UNSUPPORTED_TO_VERIFY"  # the requirement WAS planned/executed, but this
        # project's own geometry-evaluation tooling has no independent, reliable way to re-measure
        # whether the feature actually exists (e.g. an exact fillet radius/chamfer distance) -- added
        # for feature_verification.py's own Required Feature Ledger (reusing this enum rather than a
        # second one, matching assembly_compatibility.CompatibilityStatus's own identically-named,
        # identically-motivated status for the same "never claim precision we don't have" reason).
        # Distinct from UNSUPPORTED: that means the CHOSEN OP can't satisfy this requirement at all;
        # this means the feature may well be present and correct, but is not independently verified.


@dataclass
class DimensionRequirement:
    id: str
    source_phrase: str
    value: float
    original_unit: str
    normalized_value: float
    normalized_unit: str = "mm"
    component: "str | None" = None
    feature: "str | None" = None  # e.g. "base", "mounting_hole", "bore", "cylinder", "step"
    parameter: "str | None" = None  # canonical requirement_type -- see dimension_assignment.OPERATION_PARAMETER_SEMANTICS
    explicit: bool = True
    mandatory: bool = True
    extraction_confidence: float = 1.0
    status: RequirementStatus = RequirementStatus.EXTRACTED
    assigned_op_index: "int | None" = None
    assigned_arg_name: "str | None" = None
    actual_value: "float | None" = None
    notes: str = ""


@dataclass
class FeatureCountRequirement:
    id: str
    source_phrase: str
    count: int
    component: "str | None" = None
    feature: str = "hole"
    diameter_mm: "float | None" = None
    pattern_hint: "str | None" = None  # "corner_array" | "linear_array" | "bolt_circle" | None -- sandbox.PatternKind's own literal values
    explicit: bool = True
    mandatory: bool = True
    status: RequirementStatus = RequirementStatus.EXTRACTED
    assigned_op_index: "int | None" = None
    instance_ids: list[str] = field(default_factory=list)
    actual_count: "int | None" = None
    notes: str = ""


# ---------------- unit normalization ----------------

_UNIT_TO_MM = {"mm": 1.0, "millimeter": 1.0, "millimeters": 1.0, "cm": 10.0, "centimeter": 10.0, "centimeters": 10.0, "m": 1000.0, "meter": 1000.0, "meters": 1000.0}
_UNIT_PATTERN = r"(?:mm|millimeters?|cm|centimeters?|m|meters?)"
# A bare number token, with its OWN unit -- never assumes "mm" for an
# unmarked number (this project's own standing "never invent a number"
# discipline extends to units: a number with no stated unit anywhere
# near it is not extracted as a dimension at all).
_NUMBER_WITH_UNIT_RE = re.compile(rf"(\d+(?:\.\d+)?)\s*({_UNIT_PATTERN})\b", re.IGNORECASE)


def _normalize_unit(value: float, unit: str) -> tuple[float, str]:
    unit_lower = unit.lower().rstrip("s")
    if unit_lower not in _UNIT_TO_MM and unit.lower() not in _UNIT_TO_MM:
        unit_lower = unit.lower()
    factor = _UNIT_TO_MM.get(unit.lower(), _UNIT_TO_MM.get(unit_lower, 1.0))
    return value * factor, "mm"


# ---------------- descriptive-keyword -> canonical parameter tag ----------------

# SPECIFIC: a phrase that names its own feature/role unambiguously
# (a stated hole/bore, an explicitly "outer"/"main"/"reduced" qualifier,
# "overall length", an edge/corner offset). GENERIC: a bare dimension
# word (diameter/thick/tall/long/wide/deep) that could describe EITHER
# the base shape or a feature depending on context alone. A SPECIFIC
# match always wins over a GENERIC one for the same number (see
# _nearest_keyword below) -- confirmed necessary, not just tidy: for
# "10 mm diameter blind hole," the bare word "diameter" sits literally
# closer to "10mm" (by character count) than "blind hole" does, but
# "blind hole" is what actually tells us this is a HOLE's diameter, not
# the primary shape's own -- pure proximity alone gets this wrong.
_SPECIFIC_KEYWORD_TO_PARAMETER: list[tuple[str, str]] = [
    (r"through-?\s*bore|through-?\s*hole", "hole_diameter"),
    (r"blind\s*(?:bore|hole)", "hole_diameter"),
    (rf"(?:{_HOLE_DESCRIPTOR_WORDS}-?\s*){{0,2}}(?:bore|hole)s?", "hole_diameter"),
    (r"main\s+(?:section|diameter)", "main_diameter"),
    (r"reduced\s+(?:section|diameter)|stepped\s+(?:section|diameter)|\bshoulder\b", "reduced_diameter"),
    # Confirmed real gap (Phase 6 benchmark Part 3's own Test 7): "Four
    # holes on a 60mm pitch circle diameter" used to tag 60mm as
    # hole_diameter -- "hole(s)" alone already matches this tier's own
    # bare-hole pattern, and _nearest_keyword's closest-by-gap tiebreak
    # only ever compares candidates that actually EXIST; with no PCD
    # keyword in this list at all, "holes" (textually farther away) won
    # by default, not because it was actually the better match. Checked
    # here so PCD wins by its own, genuinely much closer proximity once
    # it exists as a candidate at all.
    (r"pitch[\s-]*circle(?:\s+diameter)?|\bpcd\b", "pitch_circle_diameter"),
    (r"outer\s+diameter|outside\s+diameter|\bod\b", "outer_diameter"),
    # "ID"/"inner diameter" tags as a HOLE diameter requirement, never a
    # new base-shape parameter -- confirmed by this project's own tool
    # roster: a straight hollow tube (OD/ID/length) is built here as
    # add_cylinder(OD) + drill_hole(ID) (add_tube is a CURVED swept
    # tube, a different shape entirely -- see sandbox.py's own
    # docstring), so "ID" is semantically a through-bore's own
    # diameter, which the EXISTING hole_diameter assignment pathway in
    # dimension_assignment.py already fans out to drill_hole-family ops
    # correctly, with zero new wiring needed.
    (r"inner\s+diameter|inside\s+diameter|\bid\b", "hole_diameter"),
    (r"overall\s+length", "overall_length"),
    (r"edge\s+offsets?|corner\s+offsets?", "edge_offset"),
]
_GENERIC_KEYWORD_TO_PARAMETER: list[tuple[str, str]] = [
    (r"thick(?:ness)?", "thickness"),
    (r"deep|depth", "hole_depth"),
    (r"tall|height", "height"),
    (r"long|length", "length"),
    (r"wide|width", "width"),
    (r"diameter", "outer_diameter"),  # unqualified -- describes the primary/base shape by default
]
_SPECIFIC_KEYWORD_RE = [(re.compile(p, re.IGNORECASE), tag) for p, tag in _SPECIFIC_KEYWORD_TO_PARAMETER]
_GENERIC_KEYWORD_RE = [(re.compile(p, re.IGNORECASE), tag) for p, tag in _GENERIC_KEYWORD_TO_PARAMETER]

_FEATURE_BY_PARAMETER = {
    "hole_diameter": "hole", "hole_depth": "hole", "main_diameter": "base", "reduced_diameter": "step",
    "outer_diameter": "base", "overall_length": "base", "thickness": "base", "height": "base",
    "length": "base", "width": "base", "edge_offset": "hole_pattern", "pitch_circle_diameter": "hole_pattern",
}

# ---------------- shared/trailing-unit dimensional chains ----------------
#
# Confirmed real root cause (Phase 6 benchmark, live-reproduced): "A 100
# x 60 x 10 mm rectangular plate" extracted NOTHING for "100" and "60" --
# _NUMBER_WITH_UNIT_RE (the ENTIRE extraction pipeline's own number
# finder) requires every number to carry its OWN unit, so a number with
# no directly-attached unit is invisible to extraction from the start,
# never even reaching the UNRESOLVED bucket. _ordered_box_dimensions
# above already handles the fully-explicit case ("120mm x 80mm x 10mm")
# where every number DOES carry its own unit; this is the general,
# PART of a chain can omit its own unit case ("100 x 60 x 10 mm" --
# shared trailing -- or any mix of the two).
#
# Scoped deliberately narrow, matching this project's own "never
# silently guess" discipline: a chain is only reinterpreted this way
# when (a) every member resolves to SOME unit (its own, or the chain's
# shared trailing one -- never defaulting to "mm" for a genuinely
# unitless number), (b) the chain is immediately followed by a BOX-
# shaped noun or nothing identifiable at all, never a ROUND-shaped noun
# (cylinder/shaft/rod/tube/sphere) -- a bare "40 x 80" before "cylinder"
# is genuinely ambiguous (diameter+height, not length+width) and is left
# UNRESOLVED rather than guessed, and (c) the chain is NOT immediately
# followed by a recognized FEATURE noun (hole/slot/bore/pocket/...) --
# that is a feature's own paired dimension (e.g. "30mm x 8mm slot"),
# which op_requirements.py's own separate, existing extraction already
# owns; duplicating it here would create the exact "two competing
# representations for the same fact" this project stands against.
_CONNECTOR = r"\s*(?:x|×|by|,)\s*(?:and\s+)?\s*"
_CHAIN_UNIT_OPT = rf"(?:\s*({_UNIT_PATTERN}))?"
_DIMENSION_CHAIN_RE = re.compile(
    rf"(\d+(?:\.\d+)?){_CHAIN_UNIT_OPT}"
    rf"{_CONNECTOR}(\d+(?:\.\d+)?){_CHAIN_UNIT_OPT}"
    rf"(?:{_CONNECTOR}(\d+(?:\.\d+)?){_CHAIN_UNIT_OPT})?"
    rf"{_CHAIN_UNIT_OPT}",
    re.IGNORECASE,
)
_FEATURE_NOUN_RE = re.compile(
    r"\b(?:holes?|slots?|bores?|pockets?|gussets?|ribs?|bosses?|counterbores?|countersinks?|keyways?)\b", re.IGNORECASE,
)
_ROUND_SHAPE_NOUN_RE = re.compile(r"\b(?:cylinders?|cylindrical|shafts?|rods?|tubes?|spheres?|balls?|torus|toroidal)\b", re.IGNORECASE)
_CHAIN_TAGS_BY_LENGTH = {2: ("length", "width"), 3: ("length", "width", "height")}


def _find_dimension_chains(text: str) -> list[dict]:
    """Returns every chain match as {start, end, values: [(value,
    resolved_unit_or_None), ...]}, earliest first. `resolved_unit` is
    None when that specific chain could not resolve a unit for every
    member -- callers must treat that as unresolved, never default it."""
    chains = []
    for m in _DIMENSION_CHAIN_RE.finditer(text):
        v1, u1, v2, u2, v3, u3, trailing = m.groups()
        members = [(v1, u1), (v2, u2)] + ([(v3, u3)] if v3 is not None else [])
        # The shared/trailing unit is whichever explicit unit landed LAST
        # in the chain, regardless of which specific group captured it
        # (a 3-member chain's own trailing slot, or its last member's own
        # unit; a 2-member chain's trailing slot, or ITS last member's
        # own unit) -- used as the fallback for any EARLIER member
        # lacking its own unit.
        last_explicit = next((u for u in ([u3, trailing] if v3 is not None else [u2, trailing]) if u), None)
        resolved = []
        ok = True
        for v, u in members:
            unit = u or last_explicit
            if unit is None:
                ok = False
                break
            resolved.append((float(v), unit))
        if not ok:
            continue
        chains.append({"start": m.start(), "end": m.end(), "values": resolved})
    return chains

# A clause boundary -- comma, semicolon, period, or the word "and" -- is
# where one dimension's own descriptive context ends and another's
# begins. Confirmed necessary (not just tidy): without it, "a 5 mm thick
# base and two 6 mm mounting holes" lets "mounting holes" (25 chars away,
# but still within a loose single-number gap threshold) outrank "thick"
# for the 5mm number, because nothing stops a keyword from a LATER
# clause being considered for an EARLIER number's assignment.
_CLAUSE_BOUNDARY_RE = re.compile(r"[,;.]|\band\b", re.IGNORECASE)
_MAX_KEYWORD_GAP_CHARS = 60  # a backstop only -- clause + no-intervening-number already does the real disambiguation


def _clause_spans(text: str) -> list[tuple[int, int]]:
    spans = []
    start = 0
    for m in _CLAUSE_BOUNDARY_RE.finditer(text):
        spans.append((start, m.start()))
        start = m.end()
    spans.append((start, len(text)))
    return spans


def _clause_containing(spans: list[tuple[int, int]], pos: int) -> tuple[int, int]:
    for s, e in spans:
        if s <= pos <= e:
            return s, e
    return spans[-1]


def _gap(match_start: int, match_end: int, number_start: int, number_end: int) -> int:
    if match_end <= number_start:
        return number_start - match_end
    if match_start >= number_end:
        return match_start - number_end
    return 0


def _nearest_keyword(
    text: str, number_start: int, number_end: int, all_number_spans: list[tuple[int, int]], clause_spans: list[tuple[int, int]],
) -> "tuple[str, str, int, int] | None":
    """Returns (parameter, matched_phrase, match_start, match_end) for the
    best descriptive keyword for this number: restricted to the SAME
    clause, never crossing another number's own span, SPECIFIC-tier
    preferred over GENERIC-tier regardless of relative distance, and
    closest-by-character-gap as the final tiebreaker. None when nothing
    qualifies -- callers must treat that as genuinely UNRESOLVED, never
    guess a default."""
    clause_start, clause_end = _clause_containing(clause_spans, number_start)

    def _valid_candidates(keyword_res: list[tuple[re.Pattern, str]]) -> list[tuple[str, str, int, int, int]]:
        out = []
        for regex, tag in keyword_res:
            for m in regex.finditer(text, clause_start, clause_end):
                match_start, match_end = m.start(), m.end()
                gap = _gap(match_start, match_end, number_start, number_end)
                if gap > _MAX_KEYWORD_GAP_CHARS:
                    continue
                blocked = any(
                    (number_end <= ns and ns < match_start) or (match_end <= ns and ns < number_start)
                    for ns, ne in all_number_spans if (ns, ne) != (number_start, number_end)
                )
                if blocked:
                    continue
                out.append((tag, m.group(0), match_start, match_end, gap))
        return out

    for tier in (_SPECIFIC_KEYWORD_RE, _GENERIC_KEYWORD_RE):
        candidates = _valid_candidates(tier)
        if candidates:
            tag, phrase, match_start, match_end, _gap_val = min(candidates, key=lambda c: c[4])
            return tag, phrase, match_start, match_end
    return None


def _bore_type_near(text: str, start: int, end: int) -> "str | None":
    window = text[max(0, start - 40):min(len(text), end + 40)]
    m = re.search(r"\b(through|blind)\b", window, re.IGNORECASE)
    return m.group(1).lower() if m else None


# A bare number with no descriptive keyword at all, immediately before
# one of these LINEAR shape nouns -- "a 100mm shaft," "a 150mm rod" --
# conventionally states that part's own LENGTH by this project's own
# positional convention (the same convention "a 50mm cube"/"a 40mm
# diameter, 80mm tall cylinder" already rely on for an object's primary
# dimension). Deliberately scoped to shapes whose single free dimension
# (beyond a cross-section already covered by an explicit "diameter"
# elsewhere in the same clause) IS length -- never box-like nouns
# (plate/block/bracket), which have no single unambiguous "the" length
# without width/height also being stated, and never round-cross-section
# nouns handled by the base-chain path above.
_LINEAR_SHAPE_NOUN_RE = re.compile(r"\b(?:shafts?|rods?|bars?|tubes?)\b", re.IGNORECASE)


def _implicit_length_noun_near(text: str, number_end: int, clause_spans: list[tuple[int, int]], number_start: int) -> bool:
    clause_start, clause_end = _clause_containing(clause_spans, number_start)
    window = text[number_end:min(clause_end, number_end + 15)]
    return bool(_LINEAR_SHAPE_NOUN_RE.search(window))


def _ordered_box_dimensions(text: str) -> "tuple[list[DimensionRequirement], tuple[int, int] | None]":
    """"120 mm by 80 mm [and N mm]" / "120mm x 80mm[ x Nmm]" / "100 x 60
    x 10 mm" (shared trailing unit) / "100, 60, 10 mm" -- a box-like
    compound dimension statement, assigned by STATED ORDER (length,
    width[, height]) rather than keyword, since there is no reliable
    keyword distinguishing "length" from "width" for a plain plate/box
    beyond the order the user themselves chose to list them in -- the
    same engineering convention "120x80" plans already use. Deliberately
    only fires when NO individual keyword (thick/tall/wide/long) already
    covers these same numbers elsewhere in the text (checked by the
    caller, which only runs its own per-number keyword pass on whatever
    this leaves unconsumed), to avoid double-counting the same literal
    twice under two different requirement tags.

    Built on _find_dimension_chains (see its own docstring for the
    confirmed root cause this generalizes beyond the fully-explicit-
    units case _ordered_box_dimensions originally only handled). Takes
    the FIRST chain that is neither (a) immediately followed by a
    recognized FEATURE noun -- a feature's own paired dimension (e.g.
    "30mm x 8mm slot") is op_requirements.py's own separate, existing
    extraction's job, never duplicated here -- nor (b) immediately
    followed by a ROUND-shape noun (cylinder/shaft/rod/tube/sphere) --
    a bare "40 x 80" before "cylinder" is genuinely ambiguous between
    (diameter, height) and (length, width), and this project's own
    standing "never silently guess an ambiguous expression" rule means
    that case is left for the per-number keyword pass to (correctly)
    leave UNRESOLVED, not force-tagged length/width here.

    Returns (requirements, consumed_span) -- consumed_span is None when
    no qualifying chain was found, so the caller knows not to subtract
    anything from the per-number pass that follows."""
    for chain in _find_dimension_chains(text):
        start, end = chain["start"], chain["end"]
        after = text[end:end + 25]
        if _FEATURE_NOUN_RE.search(after):
            continue
        if len(chain["values"]) not in _CHAIN_TAGS_BY_LENGTH:
            continue
        if _ROUND_SHAPE_NOUN_RE.search(after):
            continue
        tags = _CHAIN_TAGS_BY_LENGTH[len(chain["values"])]
        out = []
        for (value, raw_unit), tag in zip(chain["values"], tags):
            normalized_value, normalized_unit = _normalize_unit(value, raw_unit)
            out.append(DimensionRequirement(
                id=_next_id("REQ"), source_phrase=text[start:end], value=value, original_unit=raw_unit,
                normalized_value=normalized_value, normalized_unit=normalized_unit,
                feature="base", parameter=tag, explicit=True, mandatory=True,
            ))
        return out, (start, end)
    return [], None


def extract_dimension_requirements(text: str) -> list[DimensionRequirement]:
    """Never raises, never invents a parameter for a number with no
    nearby descriptive keyword -- that number becomes a DimensionRequirement
    with parameter=None and status=UNRESOLVED instead (see this
    function's own handling below), so an ambiguous prompt like "a 20mm
    bracket with a 6mm hole" surfaces the ambiguity rather than silently
    guessing "20mm" is the thickness."""
    requirements: list[DimensionRequirement] = []
    consumed_spans: list[tuple[int, int]] = []

    box_requirements, box_span = _ordered_box_dimensions(text)
    requirements.extend(box_requirements)
    if box_span is not None:
        consumed_spans.append(box_span)

    all_number_spans = [m.span() for m in _NUMBER_WITH_UNIT_RE.finditer(text)]
    clause_spans = _clause_spans(text)

    for m in _NUMBER_WITH_UNIT_RE.finditer(text):
        start, end = m.span()
        if any(s <= start < e for s, e in consumed_spans):
            continue  # already captured by the ordered box-dimension pass above
        value = float(m.group(1))
        raw_unit = m.group(2)
        normalized_value, normalized_unit = _normalize_unit(value, raw_unit)
        nearest = _nearest_keyword(text, start, end, all_number_spans, clause_spans)
        if nearest is None:
            implicit_length = (
                _implicit_length_noun_near(text, end, clause_spans, start)
                and not any(r.parameter == "length" for r in requirements)
            )
            if implicit_length:
                requirements.append(DimensionRequirement(
                    id=_next_id("REQ"), source_phrase=text[max(0, start - 5):min(len(text), end + 15)].strip(),
                    value=value, original_unit=raw_unit, normalized_value=normalized_value, normalized_unit=normalized_unit,
                    feature="base", parameter="length", explicit=True, mandatory=True,
                    notes="no explicit keyword (long/length) -- resolved positionally: an unqualified number immediately before a linear base-shape noun (shaft/rod/bar/tube) conventionally states its own length",
                ))
                continue
            requirements.append(DimensionRequirement(
                id=_next_id("REQ"), source_phrase=text[max(0, start - 15):min(len(text), end + 15)].strip(),
                value=value, original_unit=raw_unit, normalized_value=normalized_value, normalized_unit=normalized_unit,
                parameter=None, explicit=True, mandatory=True, status=RequirementStatus.UNRESOLVED,
                notes="no descriptive keyword (thick/tall/wide/long/diameter/deep/...) found close enough to this number to determine what it describes",
            ))
            continue
        parameter, phrase, kw_start, kw_end = nearest
        span_start, span_end = min(start, kw_start), max(end, kw_end)
        feature = _FEATURE_BY_PARAMETER.get(parameter, "base")
        notes = ""
        if parameter in ("hole_diameter", "hole_depth"):
            # Both the hole's own diameter AND its depth need to know
            # blind-vs-through -- confirmed real gap: "a 10mm diameter
            # blind hole, 8mm deep" only ever attached this note to the
            # diameter requirement, so dimension_assignment.py's blind-
            # hole center-offset logic (keyed off the DEPTH requirement
            # specifically, since offset is only meaningful alongside a
            # depth value) never saw it and built a through hole
            # regardless of the stated depth.
            bore_type = _bore_type_near(text, start, end)
            if bore_type:
                notes = f"bore_type={bore_type}"
        requirements.append(DimensionRequirement(
            id=_next_id("REQ"), source_phrase=text[span_start:span_end],
            value=value, original_unit=raw_unit, normalized_value=normalized_value, normalized_unit=normalized_unit,
            feature=feature, parameter=parameter, explicit=True, mandatory=True, notes=notes,
        ))
    return requirements


def extract_feature_count_requirements(text: str) -> list[FeatureCountRequirement]:
    """Reuses spec_validation's own proven hole-count extraction wholesale
    (same curated descriptor-word list, same count-word table) rather
    than re-deriving a second, competing pattern -- the one addition
    here is attaching the matching hole DIAMETER (if one is stated
    nearby) and a pattern-type hint ("one near each corner" ->
    corner_array; "equally spaced along its centreline" -> linear),
    neither of which spec_validation.extract_claimed_hole_count needs
    for its own narrower claimed-vs-detected comparison."""
    total = extract_claimed_hole_count(text)
    if total is None:
        return []

    diameter_mm = None
    for req in extract_dimension_requirements(text):
        if req.parameter == "hole_diameter":
            diameter_mm = req.normalized_value
            break

    pattern_hint = None
    if re.search(r"each\s+corner|corners?\b", text, re.IGNORECASE):
        pattern_hint = "corner_array"
    elif re.search(r"equally\s+spaced|evenly\s+spaced|centreline|centerline|in\s+a\s+(?:straight\s+)?(?:row|line)", text, re.IGNORECASE):
        pattern_hint = "linear_array"
    elif re.search(r"bolt\s+circle|circular(?:ly)?\s+(?:spaced|arranged)|around\s+(?:a\s+)?circle", text, re.IGNORECASE):
        pattern_hint = "bolt_circle"

    return [FeatureCountRequirement(
        id=_next_id("REQ-COUNT"), source_phrase=text, count=total, feature="hole",
        diameter_mm=diameter_mm, pattern_hint=pattern_hint, explicit=True, mandatory=True,
    )]
