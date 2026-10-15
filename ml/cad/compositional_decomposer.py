"""A general, compositional alternative to program_model.py's whole-sentence
prediction, built specifically to remove the architectural ceiling
documented in ACCURACY_AUDIT.md root cause #1 and demonstrated concretely
by the gearbox-mounting-assembly audit (MISSION_STATUS.md Addendum 7):
program_model.py has to have already seen a WHOLE op-sequence shape (or
5 close TF-IDF neighbors of one) before it can predict it reliably. A
genuinely novel COMBINATION of individually-ordinary pieces -- "a box with
a circular opening on a face" was never a trained whole-sequence shape,
even though "a box" and "a hole through it" are each completely ordinary
on their own -- gets mapped to whichever training example happens to be
closest by TF-IDF, which can be a totally unrelated primitive. Confirmed,
real, from that audit: a box-with-a-face-opening request predicted a solid
sphere; a cylinder-with-a-stepped-section request also predicted a solid
sphere; a triangular-rib request predicted a 6-cylinder circular pattern.

Patching program_dataset.py with a new training example every time a new
combination trips the classifier is NOT a general fix -- it only ever
covers combinations you've already watched fail, one at a time, forever.
The actual general lever is to stop asking one classifier to guess an
ENTIRE tool sequence from a fuzzy whole-sentence embedding, and instead
decompose a description into one base-shape clause plus feature clauses,
resolve EACH clause independently against a small, literal, deterministic
pattern library, and compose the tool sequence mechanically. Coverage
becomes (number of base patterns) x (number of feature patterns) -- every
COMBINATION of already-known pieces generalizes for free, including
combinations no training example has ever shown, which is the property
whole-sequence classification structurally cannot have.

Deliberately narrow and honest about it, same discipline as repair.py and
spec_validation.py: if the base clause isn't recognized by any pattern
here, or a stated feature clause matches none of the feature patterns,
this returns None -- NEVER a forced/guessed match -- so the caller falls
back to program_model.py's existing statistical path, unchanged. This is
strictly additive: every description this module resolves is one
program_model.py would otherwise have had to guess at; every description
it doesn't resolve behaves exactly as it did before this module existed.
This also means an unbuildable request (e.g. "a triangular prism" -- there
is genuinely no triangular profile anywhere in sandbox.py's tool roster)
correctly falls through to the existing path rather than being silently
forced into some unrelated shape, which is itself a real improvement: the
existing path at least discloses in_distribution/nearest_neighbor_similarity
for that guess, whereas a forced-wrong compositional match would hide the
mismatch behind a confident-looking deterministic program.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from sandbox import ToolCall

_NUM = r"(-?\d+(?:\.\d+)?)"
_WORD_TO_COUNT = {
    "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
    "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8,
}


def _parse_count(word: str) -> int | None:
    return _WORD_TO_COUNT.get(word.strip().lower())


@dataclass
class _BaseShape:
    op: str
    args: dict
    # the shape's own extent (length/width/height, or diameter/diameter/length
    # for a cylinder) in its own local frame, in mm -- feature resolvers that
    # need to reason about the shape's own size (e.g. picking a through-hole
    # axis, or a through-depth that's guaranteed to fully cut it) use this
    # instead of re-deriving it, since sandbox.py's own _cutter_height floors
    # at the TARGET's actual bounds anyway -- this only needs to be a safe,
    # generous estimate, never an exact value.
    extent: tuple[float, float, float]
    # True for a base pattern whose own regex already spans the entire
    # relevant text (e.g. a stepped cylinder's "... with a stepped section
    # of ..." is ONE primitive, not base+feature) -- compose_program_from_
    # description must not also try to resolve a separate feature clause
    # from the same span in that case.
    consumes_whole_text: bool = False


def _resolve_stepped_cylinder(text: str) -> _BaseShape | None:
    """"a cylinder Dmm diameter, Lmm long with a concentric stepped section
    of dmm diameter and lmm length at one end" -- a SINGLE sandbox
    primitive (add_shouldered_cylinder), not base+feature, since there is
    no separate "step this existing cylinder down" tool; the step has to
    be built into the primitive itself. Must be tried BEFORE
    _resolve_cylinder, whose plain regex would otherwise also match this
    text's own leading clause as a (wrong, incomplete) prefix match."""
    m = re.search(
        rf"\bcylinder\b.{{0,30}}?{_NUM}\s*mm\s*diameter,?\s*{_NUM}\s*mm\s*long"
        rf".{{0,60}}?stepped section of\s*{_NUM}\s*mm\s*diameter\s*(?:and)?\s*{_NUM}\s*mm\s*length",
        text, re.IGNORECASE,
    )
    if not m:
        return None
    main_d, main_l, step_d, step_l = (float(m.group(i)) for i in (1, 2, 3, 4))
    main_r, step_r = main_d / 2.0, step_d / 2.0
    bottom_l = max(main_l - step_l, 1.0)
    return _BaseShape(
        op="add_shouldered_cylinder",
        args={"bottom_radius": main_r, "bottom_height": bottom_l, "top_radius": step_r, "top_height": step_l,
              "transition_height": 0.5, "center": [0.0, 0.0, 0.0]},
        extent=(main_d, main_d, main_l),
        consumes_whole_text=True,
    )


def _resolve_cylinder(text: str) -> _BaseShape | None:
    m = re.search(rf"\bcylinder\b.{{0,30}}?{_NUM}\s*mm\s*diameter,?\s*{_NUM}\s*mm\s*(?:tall|long)", text, re.IGNORECASE)
    if not m:
        return None
    diameter, length = float(m.group(1)), float(m.group(2))
    return _BaseShape(op="add_cylinder", args={"radius": diameter / 2.0, "height": length, "center": [0.0, 0.0, 0.0], "axis": "z"},
                       extent=(diameter, diameter, length))


def _resolve_box(text: str) -> _BaseShape | None:
    m = re.search(rf"\bbox\b.{{0,30}}?{_NUM}\s*mm\s*x\s*{_NUM}\s*mm\s*x\s*{_NUM}\s*mm", text, re.IGNORECASE)
    if not m:
        return None
    length, width, height = (float(m.group(i)) for i in (1, 2, 3))
    return _BaseShape(op="add_box", args={"length": length, "width": width, "height": height, "center": [0.0, 0.0, 0.0]},
                       extent=(length, width, height))


def _resolve_sphere(text: str) -> _BaseShape | None:
    m = re.search(rf"\bsphere\b.{{0,30}}?{_NUM}\s*mm\s*diameter", text, re.IGNORECASE)
    if not m:
        return None
    diameter = float(m.group(1))
    return _BaseShape(op="add_sphere", args={"radius": diameter / 2.0, "center": [0.0, 0.0, 0.0]},
                       extent=(diameter, diameter, diameter))


# Order matters: compound/specific patterns before the generic ones they'd
# otherwise also match as an incomplete prefix.
_BASE_RESOLVERS = [_resolve_stepped_cylinder, _resolve_cylinder, _resolve_box, _resolve_sphere]


def _resolve_base_shape(text: str) -> _BaseShape | None:
    for resolver in _BASE_RESOLVERS:
        shape = resolver(text)
        if shape is not None:
            return shape
    return None


# -- Feature clauses: each takes the text AFTER the connector "with", plus
# the already-resolved base shape, and returns the list of {op, args} steps
# to append (each consuming the running target handle -- "target" is filled
# in by compose_program_from_description, not here), or None if this
# resolver doesn't recognize the text at all. Tried in a fixed priority
# order, not found-order: edge-finishing ops (chamfer/round) are tried
# before hole-drilling ops on purpose, because sandbox.py's
# round_box_edges/chamfer_box_edges require an unmodified, still-axis-
# aligned box target (Sandbox._is_axis_aligned_box) -- drilling holes
# first would make a later chamfer/round step correctly, but unhelpfully,
# rejected.

def _feature_chamfer_box_edges(text: str, base: _BaseShape) -> list[dict] | None:
    if base.op != "add_box":
        return None
    m = re.search(rf"(?:all edges chamfered|chamfer(?:ed)?(?: the)?(?: exposed)?(?: outer)? edges(?: of[^,.]*)?)\D*?{_NUM}\s*mm", text, re.IGNORECASE)
    if not m:
        m = re.search(rf"{_NUM}\s*mm\s*chamfer\b", text, re.IGNORECASE)
    if not m:
        return None
    return [{"op": "chamfer_box_edges", "args": {"distance": float(m.group(1))}}]


def _feature_round_box_edges(text: str, base: _BaseShape) -> list[dict] | None:
    if base.op != "add_box":
        return None
    m = re.search(rf"(?:all edges rounded|rounded edges)\D*?{_NUM}\s*mm", text, re.IGNORECASE)
    if not m:
        return None
    return [{"op": "round_box_edges", "args": {"radius": float(m.group(1))}}]


def _feature_corner_array_holes(text: str, base: _BaseShape) -> list[dict] | None:
    """Matches the exact phrasing that failed in the audited build:
    "a corner array of four holes, each 8mm diameter and centered at 12mm
    from adjacent edges" -- count, hole diameter, AND edge margin, all
    read literally, not predicted. `margin` is sandbox.py's own inset-
    from-each-edge parameter (drill_hole_pattern: hx = length/2 - margin),
    so "Nmm from adjacent edges" maps onto it directly, with no unit
    translation needed."""
    if base.op != "add_box":
        return None
    m = re.search(
        rf"corner array of\s*(\w+)\s*holes,?\s*each\s*{_NUM}\s*mm\s*diameter.{{0,50}}?{_NUM}\s*mm\s*from",
        text, re.IGNORECASE,
    )
    if not m:
        return None
    count = _parse_count(m.group(1))
    if count is None:
        return None
    diameter, margin = float(m.group(2)), float(m.group(3))
    _length, _width, height = base.extent
    return [{"op": "drill_hole_pattern", "args": {
        "pattern": "corner_array", "count": count, "radius": diameter / 2.0, "depth": height + 10.0,
        "axis": "z", "margin": margin, "circle_diameter": 0.0, "z": 0.0,
    }}]


def _feature_bolt_circle_holes(text: str, base: _BaseShape) -> list[dict] | None:
    """"a circular array of four holes, each 5mm diameter on a 36mm
    bolt-circle" -- the flange-style pattern. `circle_diameter` > 0 tells
    sandbox.py's drill_hole_pattern to use this exact bolt-circle diameter
    instead of deriving one from the target's own bounds."""
    m = re.search(
        rf"circular array of\s*(\w+)\s*holes,?\s*each\s*{_NUM}\s*mm\s*diameter\s*on\s*a\s*{_NUM}\s*mm\s*bolt.?circle",
        text, re.IGNORECASE,
    )
    if not m:
        return None
    count = _parse_count(m.group(1))
    if count is None:
        return None
    hole_diameter, circle_diameter = float(m.group(2)), float(m.group(3))
    _length, _width, height = base.extent
    return [{"op": "drill_hole_pattern", "args": {
        "pattern": "bolt_circle", "count": count, "radius": hole_diameter / 2.0, "depth": height + 10.0,
        "axis": "z", "margin": 0.0, "circle_diameter": circle_diameter, "z": 0.0,
    }}]


def _feature_through_hole(text: str, base: _BaseShape) -> list[dict] | None:
    """The general "a shape with a hole through it" case -- deliberately
    scoped to a hole CENTERED on the base shape, which is both the common
    phrasing (matches this project's own worked system-prompt examples,
    e.g. "a cylinder ... with a 6mm diameter hole through it") and the one
    case with no placement ambiguity. An explicitly off-center face
    position (e.g. "centered on its face at z=35mm") is NOT handled here
    and falls back to the statistical path -- resolving it correctly needs
    assembly-level context (which other part this hole has to line up
    with) that a single point's own description text doesn't carry, so
    guessing would trade one wrong answer for another instead of actually
    fixing anything.

    Axis choice: a cylinder's "hole through it" is, in every real worked
    example in this project, along the cylinder's OWN axis (lengthwise --
    a shaft's own bore, a boss's screw hole), never across its diameter,
    so that case is hardcoded to "z" (sandbox.py's add_cylinder default
    axis). A box has no such convention, so the shortest of its 3 extents
    is used -- correct for the common "thin wall/plate with a hole through
    it" case; a box that needs the hole along its LONGEST axis instead
    (e.g. a tall post, hole along its height) is not covered and falls
    back to the statistical path too."""
    m = re.search(rf"{_NUM}\s*mm\s*diameter\s*hole\s*through\s*(?:it|the\s*middle|the\s*post)", text, re.IGNORECASE)
    if not m:
        m = re.search(rf"circular through-opening of\s*{_NUM}\s*mm\s*diameter", text, re.IGNORECASE)
    if not m:
        return None
    radius = float(m.group(1)) / 2.0
    length, width, height = base.extent
    if base.op in ("add_cylinder", "add_shouldered_cylinder"):
        axis, depth = "z", height + 10.0
    else:
        extents = {"x": length, "y": width, "z": height}
        axis = min(extents, key=extents.get)
        depth = extents[axis] + 10.0
    return [{"op": "drill_hole", "args": {"radius": radius, "depth": depth, "center": [0.0, 0.0, 0.0], "axis": axis}}]


_FEATURE_RESOLVERS = [
    _feature_chamfer_box_edges,
    _feature_round_box_edges,
    _feature_corner_array_holes,
    _feature_bolt_circle_holes,
    _feature_through_hole,
]

_WITH_RE = re.compile(r"\bwith\b", re.IGNORECASE)


def compose_program_from_description(description: str) -> list[ToolCall] | None:
    """The entry point: resolves `description` into a deterministic
    ToolCall program by independent clause resolution, or returns None if
    any part of it (the base shape, or a stated feature clause) isn't
    recognized by any pattern above -- NEVER a partial/forced guess. A
    caller should try this FIRST and fall back to
    program_model.predict_program_checked() only on a None result, which
    leaves every description this module doesn't yet cover exactly as
    reliable (or unreliable) as it was before this module existed."""
    base = _resolve_base_shape(description)
    if base is None:
        return None

    steps: list[dict] = [{"op": base.op, "args": base.args}]

    if not base.consumes_whole_text:
        m = _WITH_RE.search(description)
        feature_text = description[m.end():] if m else ""
        if feature_text.strip():
            matched = None
            for resolver in _FEATURE_RESOLVERS:
                result = resolver(feature_text, base)
                if result is not None:
                    matched = result
                    break
            if matched is None:
                # A feature clause is present but nothing here recognizes
                # it -- never silently drop a requested feature by
                # returning the base shape alone; fall back whole instead.
                return None
            steps.extend(matched)

    program = [ToolCall(op=steps[0]["op"], args=dict(steps[0]["args"]))]
    for step in steps[1:]:
        args = dict(step["args"])
        args["target"] = len(program) - 1
        program.append(ToolCall(op=step["op"], args=args))
    program.append(ToolCall(op="finalize", args={"target": len(program) - 1}))
    return program
