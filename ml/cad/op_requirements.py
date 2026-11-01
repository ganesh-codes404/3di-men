"""Structured engineering-requirement extraction from a build request's
own text -- the shared input both the classifier's own op-sequence
proposal and RAG's retrieved one get checked against during arbitration
(see op_arbitration.py).

Confirmed real root cause this exists to fix (mission-brief audit, live-
reproduced against the actual trained bundle, not hypothetical): for "a
cylinder, 40mm diameter, 30mm tall, with a 12mm diameter bore through
it," ALL top-5 retrieved training neighbors but one are the SAME wrong
(add_box, add_cylinder, boolean_intersection) pattern -- a genuine
training-data cluster sharing surface vocabulary ("cylinder",
"diameter") with the query, not a one-off bad match. No amount of
threshold tuning fixes this (confirmed separately: raising the override
threshold to block this exact case cost 16.6 points on hand_written and
1-1.9 on generic_composition, breaking two OTHER legitimate RAG
rescues -- a plate's hole pattern and a bracket's base geometry, see
program_model.ProgramModelConfig.rag_override_similarity_threshold's
own docstring for that investigation). The fix has to be requirement-
aware, not similarity-aware: a candidate whose base geometry or missing
features CONTRADICT what the text actually asked for must be rejected
regardless of how textually similar its retrieved source was.

Deliberately deterministic (plain regex/keyword matching, reusing the
same proven techniques program_model.py's own _DIMENSION_RE/
_BORE_HOLE_DIAMETER_RE/_GEAR_MODULE_RE and spec_validation.py's own
extract_claimed_hole_count already use elsewhere in this project) --
not a second LLM call. This runs on every single-part prediction, so it
must stay fast, and it must never introduce a new non-deterministic
dependency into something whose whole point is to make selection MORE
reliable."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from dimension_requirements import RequirementStatus, _find_dimension_chains
from spec_validation import extract_claimed_hole_count

# ---------------- base geometry ----------------

# keyword -> geometry_kind. Deliberately only UNAMBIGUOUS shape words --
# a word like "housing" or "enclosure" is genuinely used for both
# cylindrical and box-shaped parts in real engineering language, so it
# contributes no opinion here; the numeric-pattern fallback below
# (extract_base_geometry_requirement's own second half) covers those
# cases instead, from the shape of the STATED DIMENSIONS rather than a
# guessable word.
_BASE_GEOMETRY_KEYWORDS: dict[str, str] = {
    "cylinder": "cylindrical", "cylindrical": "cylindrical", "rod": "cylindrical",
    "shaft": "cylindrical", "tube": "cylindrical", "pipe": "cylindrical",
    "box": "rectangular", "plate": "rectangular", "bracket": "rectangular",
    "panel": "rectangular", "block": "rectangular", "rectangular": "rectangular",
    "gear": "gear",
    "sphere": "spherical", "ball": "spherical",
}

# "40mm x 30mm x 10mm" (three independent dimensions) implies a box-like
# shape; "40mm diameter, 30mm tall" (a diameter paired with a height/
# length) implies a cylindrical one. Only used when no keyword above
# matched -- an explicit shape word always wins, same "explicit beats
# inferred" precedence this project applies everywhere else.
_THREE_DIMENSION_RE = re.compile(r"\d+(?:\.\d+)?\s*mm\s*x\s*\d+(?:\.\d+)?\s*mm\s*x\s*\d+(?:\.\d+)?\s*mm", re.IGNORECASE)
_DIAMETER_PLUS_HEIGHT_RE = re.compile(
    r"\d+(?:\.\d+)?\s*mm\s*(?:outer\s+|outside\s+)?diameter.{0,40}?\d+(?:\.\d+)?\s*mm\s*(?:tall|long|height|length)", re.IGNORECASE,
)

# add_loft/add_sweep are deliberately absent -- their own predicted
# profile can be either shape (a lofted/swept cross-section is not
# fixed by the op name alone), so they carry no opinion and are never
# rejected OR credited by the base-geometry check below.
CREATES_GEOMETRY_KIND: dict[str, str] = {
    "add_box": "rectangular",
    "add_cylinder": "cylindrical",
    "add_extrusion": "rectangular",  # this project's own training data overwhelmingly uses rectangle/rounded_rect/l_shape profiles for this op -- a coarse but evidence-backed default, not a guess
    "add_gear_profile": "gear",
    "add_revolve": "cylindrical",
    "add_shouldered_cylinder": "cylindrical",
    "add_sphere": "spherical",
    "add_torus": "toroidal",
    "add_capsule": "cylindrical",
    "add_ellipsoid": "spherical",
    "add_polygon_frustum": "cylindrical",
    "add_dome": "spherical",
    "add_ring": "toroidal",
    "add_tube": "cylindrical",
}

# ---------------- features ----------------

# keyword -> (feature_type, ops that can satisfy it). Generalizes to
# plates/brackets/shafts/gears/holes/slots/pockets/ribs via the SAME
# mechanism (a keyword maps to a set of compatible ops in the real
# operation registry), not a per-object special case.
_FEATURE_KEYWORDS: list[tuple[str, str, frozenset[str]]] = [
    (r"through-?\s*hole", "subtractive_hole", frozenset({"drill_hole", "drill_hole_pattern", "counterbore_hole", "counterbore_hole_pattern"})),
    (r"blind\s*hole", "subtractive_hole", frozenset({"drill_hole", "drill_hole_pattern", "counterbore_hole", "counterbore_hole_pattern"})),
    (r"bore", "subtractive_hole", frozenset({"drill_hole", "drill_hole_pattern", "counterbore_hole", "counterbore_hole_pattern", "add_boss_with_bore"})),
    (r"holes?", "subtractive_hole", frozenset({"drill_hole", "drill_hole_pattern", "counterbore_hole", "counterbore_hole_pattern"})),
    (r"slots?", "subtractive_slot", frozenset({"drill_slot", "add_slot_pattern"})),
    (r"pockets?", "subtractive_pocket", frozenset({"inset_pocket"})),
    (r"stepped|shoulder(?:ed)?", "stepped_section", frozenset({"add_shouldered_cylinder"})),
    # "shell" is the dedicated hollowing tool, but this project's own
    # training data (data/program_dataset.py's enclosure family) builds a
    # hollow box just as validly via a smaller inner box + boolean_
    # difference -- confirmed live (program_model's own classifier/RAG
    # BOTH independently choose that pattern for "a rectangular enclosure
    # with a hollow interior"). Excluding it would wrongly flag a
    # perfectly good hollowed shape as violating this requirement.
    (r"hollow|shell(?:ed)?", "hollow_shell", frozenset({"shell", "boolean_difference"})),
    (r"knurl(?:ed)?", "knurled_surface", frozenset({"add_knurl_pattern"})),
    # "gusset" aliased onto the SAME feature_type as "rib", not a
    # separate one -- confirmed real gap this closes: a gusset request
    # (sandbox.py's own add_rib is the one tool that builds this kind of
    # triangular structural web; there is no separate "gusset" tool) was
    # completely invisible to this extraction before, not even as
    # UNSUPPORTED_TO_VERIFY -- op_requirements.extract_requirements
    # returned ONLY a base_geometry requirement for "an L-bracket with a
    # 30mm triangular gusset, 4mm thick," with nothing at all tracking
    # the gusset itself.
    (r"rib(?:s|bed)?|gussets?", "rib", frozenset({"add_rib"})),
    (r"fillet(?:ed)?", "edge_treatment", frozenset({"fillet_edge", "round_box_edges"})),
    (r"chamfer(?:ed)?", "edge_treatment", frozenset({"chamfer_edge", "chamfer_box_edges"})),
]

_BORE_TYPE_RE = re.compile(r"(through|blind)[\s-]*(?:hole|bore)?", re.IGNORECASE)


@dataclass
class Requirement:
    """One structured engineering requirement extracted from the
    request's own text. `compatible_ops` is empty for a base_geometry
    requirement (checked by CREATES_GEOMETRY_KIND against the
    sequence's own first op instead, see op_arbitration.py).

    This is also the canonical Required Feature Ledger entry (see
    feature_verification.py) -- extended here, rather than introduced as
    a second competing representation, because this dataclass is already
    the one op_arbitration.py checks a chosen op sequence against in
    production. The fields below carry the requirement PAST that
    op-sequence-structural check (Requirement.required/compatible_ops
    only ever answer "did the PLANNED sequence include a plausible op,"
    never "does the BUILT mesh actually have this feature") into the
    full EXTRACTED -> ... -> GEOMETRY_VERIFIED/FAILED/UNSUPPORTED_TO_
    VERIFY lifecycle (dimension_requirements.RequirementStatus, reused
    rather than a second enum) that feature_verification.py's
    independent, built-geometry-only checks populate."""

    id: str
    feature_type: str  # "base_geometry" | "subtractive_hole" | "subtractive_slot" | "stepped_section" | "hollow_shell" | ...
    geometry_type: "str | None"  # "cylindrical" | "rectangular" | "gear" | "spherical" -- only set for base_geometry
    explicit: bool
    source_text: str
    required: bool = True
    compatible_ops: frozenset[str] = field(default_factory=frozenset)
    count: "int | None" = None  # requested count, e.g. 4 for "four holes" -- None when no count was stated
    component_id: "str | None" = None  # which assembly component this feature belongs to -- None for a
        # single-point build (the only component there is), set by the assembly-level caller for a
        # multi-point build where a feature's own component assignment matters (Part 4's own rule)
    dimensions_mm: dict = field(default_factory=dict)  # named numeric expectations stated in the text for
        # THIS feature specifically (e.g. {"diameter": 6.0} for a hole, {"length": 30.0, "width": 8.0} for
        # a slot) -- deliberately a plain dict, not a second DimensionRequirement-style dataclass per
        # field, since feature_verification.py only ever needs to read named values back out, not
        # re-run dimension_requirements.py's own clause/keyword disambiguation a second time here
    status: RequirementStatus = RequirementStatus.EXTRACTED
    actual_count: "int | None" = None  # populated by feature_verification.py from the BUILT mesh
    verification_method: str = ""  # discloses HOW status was determined, e.g. "hole_count_and_diameter" |
        # "wall_opening_elongated_loop" | "edge_dihedral_angle_presence" | "axial_cross_section_radius_change"
        # | "op_sequence_only_not_geometry_verified" -- never left implicit, per this project's own standing
        # "disclosed, never silent" discipline (see assembly_compatibility.py's own RelationshipEvaluation)
    failure_reason: str = ""  # populated only when status is FAILED/UNSUPPORTED/UNSUPPORTED_TO_VERIFY/UNRESOLVED


def _extract_base_geometry_requirement(text: str) -> "Requirement | None":
    lowered = text.lower()
    for keyword, kind in _BASE_GEOMETRY_KEYWORDS.items():
        if re.search(rf"\b{keyword}\b", lowered):
            return Requirement(id="base_geometry", feature_type="base_geometry", geometry_type=kind, explicit=True, source_text=keyword)

    if _DIAMETER_PLUS_HEIGHT_RE.search(text):
        return Requirement(id="base_geometry", feature_type="base_geometry", geometry_type="cylindrical", explicit=False, source_text="diameter+height/length phrasing")
    if _THREE_DIMENSION_RE.search(text):
        return Requirement(id="base_geometry", feature_type="base_geometry", geometry_type="rectangular", explicit=False, source_text="NxNxN dimension phrasing")
    return None


_NUM = r"(\d+(?:\.\d+)?)"
# Slot/edge-treatment dimension phrasings -- deliberately narrow (this
# module's own standing discipline, see its own module docstring): only
# the specific wordings this project's own worked examples actually use,
# never a forced guess at which number belongs to which feature when the
# text doesn't make it unambiguous.
_SLOT_WORD_RE = re.compile(r"\bslots?\b", re.IGNORECASE)
_MAX_FEATURE_DIMENSION_GAP_CHARS = 40


def _extract_feature_dimensions_mm(feature_type: str, text: str) -> dict:
    """Named numeric expectations for the one feature TYPE this returns
    non-empty for -- deliberately scoped to exactly the types feature_
    verification.py can independently re-measure from built geometry
    (subtractive_slot, edge_treatment); every other feature_type gets an
    empty dict here, not a guessed one, same "never invent a number"
    discipline as dimension_requirements.py.

    Closest-match, bounded-gap, not a single greedy regex -- confirmed
    real bug this fixes: "a 100mm x 50mm x 8mm plate with a 30mm x 8mm
    central through-slot" has TWO dimension-chain matches in the
    sentence; a plain `.*?slot` regex matches the FIRST one (the
    plate's own base dimensions), not the one actually describing the
    slot. Picking whichever CHAIN is textually CLOSEST to "slot"
    (within a bounded gap) is the same disambiguation discipline
    dimension_requirements._nearest_keyword already applies elsewhere
    in this project.

    Reuses dimension_requirements._find_dimension_chains -- the SAME
    centralized chain parser the base-shape path relies on -- rather
    than a second, narrower "NNmm x NNmm" regex of its own. Confirmed
    real gap this closes: a plain "{NUM}mm x {NUM}mm" pattern requires
    BOTH numbers to carry their own unit, so "a 30 x 8 mm slot" (a
    shared TRAILING unit, exactly the Phase 6/7 root cause) was
    invisible here too, even though the base-shape chain a few words
    earlier in the SAME sentence was already fixed to handle it."""
    if feature_type == "subtractive_slot":
        slot_spans = [m.span() for m in _SLOT_WORD_RE.finditer(text)]
        if not slot_spans:
            return {}
        best = None
        for chain in _find_dimension_chains(text):
            if len(chain["values"]) != 2:
                continue
            gap = min(
                (s[0] - chain["end"] if s[0] >= chain["end"] else chain["start"] - s[1] if chain["start"] >= s[1] else 0)
                for s in slot_spans
            )
            if gap <= _MAX_FEATURE_DIMENSION_GAP_CHARS and (best is None or gap < best[0]):
                best = (gap, chain)
        if best is not None:
            a, b = best[1]["values"][0][0], best[1]["values"][1][0]
            return {"length": max(a, b), "width": min(a, b)}
    elif feature_type == "edge_treatment":
        word_spans = [m.span() for m in re.finditer(r"\b(?:fillet|chamfer)(?:ed)?\b", text, re.IGNORECASE)]
        if not word_spans:
            return {}
        best = None
        for m in re.finditer(rf"{_NUM}\s*mm", text, re.IGNORECASE):
            gap = min(
                (s[0] - m.end() if s[0] >= m.end() else m.start() - s[1] if m.start() >= s[1] else 0)
                for s in word_spans
            )
            if gap <= _MAX_FEATURE_DIMENSION_GAP_CHARS and (best is None or gap < best[0]):
                best = (gap, m)
        if best is not None:
            return {"distance": float(best[1].group(1))}
    return {}


def _extract_feature_requirements(text: str) -> list[Requirement]:
    requirements: list[Requirement] = []
    seen_types: set[str] = set()
    for pattern, feature_type, compatible_ops in _FEATURE_KEYWORDS:
        if feature_type in seen_types:
            continue  # the first, most specific match for a feature type wins -- "through-hole" before the plainer "hole" already covers the same requirement
        m = re.search(rf"\b{pattern}\b", text, re.IGNORECASE)
        if not m:
            continue
        seen_types.add(feature_type)
        count = extract_claimed_hole_count(text) if feature_type == "subtractive_hole" else None
        requirements.append(Requirement(
            id=feature_type, feature_type=feature_type, geometry_type=None, explicit=True,
            source_text=m.group(0), compatible_ops=compatible_ops, count=count,
            dimensions_mm=_extract_feature_dimensions_mm(feature_type, text),
        ))
    return requirements


def extract_requirements(text: str) -> list[Requirement]:
    """Never raises, never returns a requirement it isn't confident
    about -- an object with no recognizable base-geometry word and no
    recognizable dimension pattern simply gets no base_geometry
    requirement at all (nothing to check candidates against), same
    "never invent a number" discipline this project applies everywhere
    else. Order is base geometry first, then features, matching how a
    human would read the request."""
    requirements: list[Requirement] = []
    base = _extract_base_geometry_requirement(text)
    if base:
        requirements.append(base)
    requirements.extend(_extract_feature_requirements(text))
    return requirements


def bore_type(text: str) -> "str | None":
    """"through" vs "blind" -- a SOFT signal (affects whether depth
    should plausibly reach the full height or not), not a hard
    op-selection rule like the requirements above, since depth is a
    regressed dimension, not a choice between ops. Exposed separately
    so a caller that wants it (e.g. a future depth-plausibility check)
    doesn't have to re-derive it."""
    m = _BORE_TYPE_RE.search(text)
    return m.group(1).lower() if m else None
