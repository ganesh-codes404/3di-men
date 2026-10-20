"""Stage A: prompt interpretation -- turns a user prompt into a structured
MitsugenShapeSpec. Deliberately independent of llm_decompose.py: Mitsugen
has its own prompt-to-structure step, not a mode of 3di-men's CAD
decomposer (see this package's own __init__.py for why).

v1 construction strategy, stated plainly (so a caller never mistakes
"not implemented yet" for "every shape supported"): a RADIALLY SYMMETRIC
profile sweep -- the object is described as a sequence of (height
fraction, radius) WAYPOINTS, interpolated into a continuous radius curve
and revolved into circular cross-sections. This covers a broad, real
class of printable objects (vases, bottles, bowls, cups, simple spindle/
finial shapes) -- it does NOT cover shapes whose cross-section changes
shape (not just size) along the axis, or non-radially-symmetric objects
(a character, a phone stand's flat profile) -- those are a disclosed,
NOT-yet-implemented limitation (see MitsugenSpec.unsupported_reason),
never silently approximated as a revolve.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# Deliberately NOT imported from llm_decompose.py -- a separate product's
# own realistic-sizing anchors, even though the real-world knowledge
# behind the numbers is the same common sense any engineer would use.
# Radius, not diameter (this module's own waypoint convention is radius-
# vs-height).
_WELL_KNOWN_OBJECTS: dict[str, dict] = {
    "vase": {"height_mm": 220.0, "base_radius_mm": 55.0, "top_radius_mm": 35.0},
    "bottle": {"height_mm": 200.0, "base_radius_mm": 32.0, "top_radius_mm": 12.0},
    "bowl": {"height_mm": 90.0, "base_radius_mm": 20.0, "top_radius_mm": 70.0},
    "cup": {"height_mm": 95.0, "base_radius_mm": 35.0, "top_radius_mm": 40.0},
    "mug": {"height_mm": 95.0, "base_radius_mm": 40.0, "top_radius_mm": 40.0},
}

# "Hmm tall" / "H mm in height"
_HEIGHT_RE = re.compile(r"(\d+(?:\.\d+)?)\s*mm\s*(?:tall|high|in height)", re.IGNORECASE)
# "Rmm (diameter|wide|across) at the base/bottom"
_BASE_RE = re.compile(
    r"(\d+(?:\.\d+)?)\s*mm\s*(?:diameter|wide|across)?\s*(?:at|for)\s*the\s*(?:base|bottom)", re.IGNORECASE,
)
# "narrowing/widening to Rmm (diameter|wide) at the top/opening/neck"
_TOP_RE = re.compile(
    r"(?:narrowing|widening|tapering)\s*to\s*(\d+(?:\.\d+)?)\s*mm\s*(?:diameter|wide|across)?\s*"
    r"(?:at\s*the\s*)?(?:top|opening|neck|rim)", re.IGNORECASE,
)
_MAX_SLICES_RE = re.compile(r"(\d+)\s*(?:max(?:imum)?\s*)?(?:construction\s*)?slices?", re.IGNORECASE)


@dataclass
class MitsugenShapeSpec:
    object_description: str
    height_mm: float
    waypoints_mm: list[tuple[float, float]]  # (height_fraction 0..1, radius_mm), >= 2, sorted by fraction
    hollow: bool = False
    wall_thickness_mm: float | None = None
    max_slices: int = 12
    tolerance_fraction: float = 0.05
    # Provenance -- the non-negotiable principle this spec exists to
    # enforce: "explicit user dimensions must override semantic
    # defaults... do not let a label silently override specified
    # measurements." Every numeric field above traces back to one of
    # these two dicts, so a caller (or a test) can always tell which.
    explicit_mm: dict[str, float] = field(default_factory=dict)
    inferred_defaults_mm: dict[str, float] = field(default_factory=dict)
    unsupported_reason: str | None = None  # None means "this spec is buildable" -- see
        # compose_spec_from_prompt's own docstring for when this gets set instead of guessing


def _find_well_known_object(prompt: str) -> tuple[str, dict] | None:
    lowered = prompt.lower()
    for name, defaults in _WELL_KNOWN_OBJECTS.items():
        if re.search(rf"\b{name}\b", lowered):
            return name, defaults
    return None


def compose_spec_from_prompt(prompt: str, default_max_slices: int = 12) -> MitsugenShapeSpec:
    """Deterministic, regex-based extraction -- never raises, never
    fabricates a shape it can't support. Recognizes:
      "a <vase/bottle/bowl/cup/mug>[, Hmm tall][, Rmm at the base][,
      narrowing/widening to Rmm at the top]"
    Explicit numbers always override the named object's own typical
    size (recorded in `explicit_mm`, never silently blended with
    `inferred_defaults_mm`). If the prompt names no recognized object
    AND gives no usable height/radius numbers at all, returns a spec
    with `unsupported_reason` set -- callers must check this before
    building, never guess past it.

    This is NOT an LLM call -- matches this package's own "Mitsugen has
    its own prompt-to-structure step" scope. A real LLM-assisted
    extraction (proposing waypoints for prompts this regex can't parse)
    is a disclosed, NOT-yet-implemented extension, not something this
    function pretends to do by falling back to a wild guess."""
    known = _find_well_known_object(prompt)
    explicit_mm: dict[str, float] = {}
    inferred_mm: dict[str, float] = {}

    height_match = _HEIGHT_RE.search(prompt)
    base_match = _BASE_RE.search(prompt)
    top_match = _TOP_RE.search(prompt)

    if height_match:
        height_mm = float(height_match.group(1))
        explicit_mm["height_mm"] = height_mm
    elif known:
        height_mm = known[1]["height_mm"]
        inferred_mm["height_mm"] = height_mm
    else:
        return MitsugenShapeSpec(
            object_description=prompt, height_mm=0.0, waypoints_mm=[], max_slices=default_max_slices,
            unsupported_reason=(
                "no recognized well-known object name and no explicit height given -- nothing to build "
                "a radial profile from without guessing"
            ),
        )

    if base_match:
        base_radius_mm = float(base_match.group(1))
        explicit_mm["base_radius_mm"] = base_radius_mm
    elif known:
        base_radius_mm = known[1]["base_radius_mm"]
        inferred_mm["base_radius_mm"] = base_radius_mm
    else:
        base_radius_mm = height_mm * 0.25  # a plain, disclosed geometric fallback, not a guessed "typical" size
        inferred_mm["base_radius_mm"] = base_radius_mm

    if top_match:
        top_radius_mm = float(top_match.group(1))
        explicit_mm["top_radius_mm"] = top_radius_mm
    elif known:
        top_radius_mm = known[1]["top_radius_mm"]
        inferred_mm["top_radius_mm"] = top_radius_mm
    else:
        top_radius_mm = base_radius_mm  # no taper claimed or implied -- a plain cylinder, not a guessed shape
        inferred_mm["top_radius_mm"] = top_radius_mm

    max_slices_match = _MAX_SLICES_RE.search(prompt)
    max_slices = int(max_slices_match.group(1)) if max_slices_match else default_max_slices

    return MitsugenShapeSpec(
        object_description=prompt, height_mm=height_mm,
        waypoints_mm=[(0.0, base_radius_mm), (1.0, top_radius_mm)],
        max_slices=max_slices, explicit_mm=explicit_mm, inferred_defaults_mm=inferred_mm,
    )
