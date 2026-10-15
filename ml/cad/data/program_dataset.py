"""(text, program) training data for program_model.py -- two sources, kept
permanently separate (never merged for training OR eval, same principle as
parser_eval.py's "never merge synthetic and hand-written scores"):

1. "family" samples: every (text, params) pair data/prompt_dataset.py already
   generates for the 6 known part families, additionally traced through
   data/program_builders.py to get a (text, program) label for free. Because
   every family's program has a FIXED op-sequence shape (only arguments
   vary), this teaches the model to reproduce the 6 known shapes through the
   tool vocabulary -- it is NOT, by itself, evidence of compositional
   generalization. See module-level HONEST_LIMITATIONS below.

2. "generic_composition" samples: a modest (few hundred), separately-tagged
   corpus of hand-designed composable PATTERNS (not free-form random tool
   graphs, which mostly produce geometric garbage) -- the only source of any
   real exposure to primitive composition beyond the 6 fixed families. Text
   for these is mechanically rendered from the program itself, one clause
   per tool call -- there is no natural vocabulary for a random composition
   the way "gear" is a real word, so this reads as telegraphic, not fluent
   English. That is a known, deliberate limitation, not an oversight.

HONEST_LIMITATIONS: true open-ended compositional generalization (correctly
building an arbitrary novel object) is not realistically achievable under
this project's constraints (CPU-only, scikit-learn, no LLM API, a few-
hundred-example generic corpus) -- that needs large pretrained sequence
models trained on huge diverse CAD corpora, a different project. The honest
target here is: reliable reproduction of the 6 known shapes through a
uniform tool-call representation, modest and honestly-measured competence on
simple 2-3-primitive novel combinations resembling the generic-composition
grammar, and visible, typed failure beyond that (see sandbox.py's
ProgramExecutionResult) -- never silent degradation.
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from typing import Literal

from data.program_builders import traced_build
from data.prompt_dataset import sample_prompt_dataset
from part_families import list_part_families
from sandbox import ToolCall

SourceKind = Literal["family", "generic_composition"]


@dataclass(frozen=True)
class ProgramSample:
    id: str
    text: str
    program: tuple[ToolCall, ...]
    source_kind: SourceKind
    family: str | None  # only set for source_kind == "family"
    canonical_key: str

    def to_dict(self) -> dict:
        return {
            "id": self.id, "text": self.text,
            "program": [{"op": c.op, "args": c.args} for c in self.program],
            "source_kind": self.source_kind, "family": self.family, "canonical_key": self.canonical_key,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ProgramSample":
        return cls(
            id=d["id"], text=d["text"],
            program=tuple(ToolCall(op=c["op"], args=c["args"]) for c in d["program"]),
            source_kind=d["source_kind"], family=d["family"], canonical_key=d["canonical_key"],
        )


def _program_canonical_key(text: str, program: list[ToolCall]) -> str:
    payload = json.dumps({"text": text, "program": [{"op": c.op, "args": c.args} for c in program]}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()


def sample_family_program_dataset(counts: dict[str, int], seed: int) -> list[ProgramSample]:
    """Free (text, program) pairs: reuses sample_prompt_dataset() unmodified,
    then traces each sample's already-resolved params through the matching
    program builder."""
    base_samples = sample_prompt_dataset(counts, seed=seed)
    out = []
    for s in base_samples:
        spec_params_cls = _params_cls_for(s.part_family)
        params = spec_params_cls(**s.params)
        _, program = traced_build(s.part_family, params)
        out.append(
            ProgramSample(
                id=f"family_{s.id}", text=s.text, program=tuple(program), source_kind="family",
                family=s.part_family, canonical_key=_program_canonical_key(s.text, program),
            )
        )
    return out


def _params_cls_for(family: str):
    from part_families import get_part_family

    return get_part_family(family).params_cls


# ---------------------------------------------------------------------------
# Generic composition corpus: hand-designed composable patterns, not
# free-form random tool graphs (most random sequences are geometric
# garbage -- same reasoning as random_gear flooring tooth_count at 12).
# ---------------------------------------------------------------------------


def _pattern_union(rng: random.Random) -> tuple[str, list[ToolCall]]:
    l, w, h = rng.choice([30, 40, 50, 60]), rng.choice([20, 30, 40]), rng.choice([8, 10, 15])
    r, ch = rng.choice([5, 8, 10]), rng.choice([15, 20, 25])
    program = [
        ToolCall("add_box", {"length": l, "width": w, "height": h, "center": [0, 0, 0]}),
        ToolCall("add_cylinder", {"radius": r, "height": ch, "center": [0, 0, h / 2 + ch / 2], "axis": "z"}),
        ToolCall("boolean_union", {"a": 0, "b": 1}),
        ToolCall("finalize", {"target": 2}),
    ]
    text = f"a box {l:g}mm x {w:g}mm x {h:g}mm with a cylinder {2*r:g}mm diameter x {ch:g}mm tall on top, unioned together"
    return text, program


def _pattern_cavity(rng: random.Random) -> tuple[str, list[ToolCall]]:
    L, W, H = rng.choice([60, 80, 100]), rng.choice([40, 60, 80]), rng.choice([20, 30, 40])
    wall = rng.choice([2, 3, 4])
    program = [
        ToolCall("add_box", {"length": L, "width": W, "height": H, "center": [0, 0, 0]}),
        ToolCall("add_box", {"length": L - 2 * wall, "width": W - 2 * wall, "height": H - wall, "center": [0, 0, wall / 2]}),
        ToolCall("boolean_difference", {"a": 0, "b": 1}),
        ToolCall("finalize", {"target": 2}),
    ]
    text = f"a box {L:g}mm x {W:g}mm x {H:g}mm with an inset cavity leaving {wall:g}mm walls"
    return text, program


def _pattern_ring(rng: random.Random) -> tuple[str, list[ToolCall]]:
    R, H = rng.choice([30, 40, 50]), rng.choice([5, 8, 10])
    r = R * rng.choice([0.4, 0.5, 0.6])
    program = [
        ToolCall("add_cylinder", {"radius": R, "height": H, "center": [0, 0, 0], "axis": "z"}),
        ToolCall("add_cylinder", {"radius": r, "height": H + 10, "center": [0, 0, 0], "axis": "z"}),
        ToolCall("boolean_difference", {"a": 0, "b": 1}),
        ToolCall("finalize", {"target": 2}),
    ]
    text = f"a ring, {2*R:g}mm outer diameter, {2*r:g}mm inner diameter, {H:g}mm thick"
    return text, program


_NUMBER_WORDS = {2: "two", 3: "three", 4: "four"}


def _pattern_holes(rng: random.Random) -> tuple[str, list[ToolCall]]:
    shape = rng.choice(["box", "cylinder"])
    count = rng.choice([2, 3, 4])
    hole_r = rng.choice([2, 3, 4])
    margin = rng.choice([8.0, 15.0, 20.0, 25.0])
    # two real phrasing styles, both trained: the plain "N holes" count-only
    # style, and the "corner array... Xmm from a corner" positional style --
    # confirmed live that _SYSTEM_PROMPT's own worked phrasing ("a corner
    # array of four holes, each 20mm from a corner") shares NO vocabulary
    # with the count-only style, so a classifier trained on count-only text
    # alone mispredicted this as a box-minus-box cavity instead of a hole
    # pattern. Both styles must be present so either phrasing is recognized.
    # weighted toward the verbose style -- confirmed directly (same density
    # problem as add_sphere's earlier fix) that a 50/50 split left this
    # phrasing too thin relative to other patterns' box+"with a ..." text
    # for the per-slot classifier to reliably distinguish it from
    # _pattern_cavity's box-minus-box shape.
    verbose_style = rng.random() < 0.85
    if shape == "box":
        l, w, h = rng.choice([40, 60, 80]), rng.choice([30, 40, 60]), rng.choice([6, 8, 10])
        program = [ToolCall("add_box", {"length": l, "width": w, "height": h, "center": [0, 0, 0]})]
        shape_desc = f"a box {l:g}mm x {w:g}mm x {h:g}mm"
        depth = h
        pattern_kind = "corner_array"
    else:
        r, h = rng.choice([20, 30, 40]), rng.choice([6, 8, 10])
        program = [ToolCall("add_cylinder", {"radius": r, "height": h, "center": [0, 0, 0], "axis": "z"})]
        shape_desc = f"a cylinder {2*r:g}mm diameter x {h:g}mm tall"
        depth = h
        pattern_kind = "bolt_circle"
    program.append(
        ToolCall("drill_hole_pattern", {"target": 0, "pattern": pattern_kind,
                                         "count": count, "radius": hole_r, "depth": depth, "axis": "z",
                                         "margin": margin, "circle_diameter": 0.0, "z": 0.0})
    )
    program.append(ToolCall("finalize", {"target": 1}))
    if verbose_style and shape == "box":
        count_word = _NUMBER_WORDS[count]
        text = f"{shape_desc} with a corner array of {count_word} holes, each {margin:g}mm from a corner"
    elif verbose_style:
        text = f"{shape_desc} with {count} holes arranged in a bolt circle, {hole_r*2:g}mm diameter, {margin:g}mm margin from the edge"
    else:
        text = f"{shape_desc} with {count} {2*hole_r:g}mm holes"
    return text, program


def _pattern_intersection(rng: random.Random) -> tuple[str, list[ToolCall]]:
    l, w, h = rng.choice([20, 30, 40]), rng.choice([15, 20, 30]), rng.choice([20, 30, 40])
    r, ch = rng.choice([10, 15, 20]), rng.choice([30, 40, 50])
    program = [
        ToolCall("add_box", {"length": l, "width": w, "height": h, "center": [0, 0, 0]}),
        ToolCall("add_cylinder", {"radius": r, "height": ch, "center": [0, 0, 0], "axis": "z"}),
        ToolCall("boolean_intersection", {"a": 0, "b": 1}),
        ToolCall("finalize", {"target": 2}),
    ]
    text = f"a box {l:g}mm x {w:g}mm x {h:g}mm intersected with a cylinder {2*r:g}mm diameter x {ch:g}mm tall"
    return text, program


def _pattern_mirror(rng: random.Random) -> tuple[str, list[ToolCall]]:
    l, w, h = rng.choice([20, 30, 40]), rng.choice([15, 20, 30]), rng.choice([6, 8, 10])
    offset = rng.choice([15, 20, 25])
    box_args = {"length": l, "width": w, "height": h, "center": [offset, 0, 0]}
    program = [
        ToolCall("add_box", box_args),  # 0 -- gets mirrored (consumed)
        ToolCall("add_box", dict(box_args)),  # 1 -- survives to be unioned with the mirror
        ToolCall("mirror", {"target": 0, "axis": "x"}),  # 2
        ToolCall("boolean_union", {"a": 1, "b": 2}),  # 3
        ToolCall("finalize", {"target": 3}),
    ]
    text = f"a box {l:g}mm x {w:g}mm x {h:g}mm positioned at x={offset:g}mm, mirrored across the x axis, and unioned with the original"
    return text, program


def _pattern_scaled(rng: random.Random) -> tuple[str, list[ToolCall]]:
    l, w, h = rng.choice([20, 30, 40]), rng.choice([15, 20, 30]), rng.choice([6, 8, 10])
    fx, fy, fz = rng.choice([0.5, 1.5, 2.0]), rng.choice([0.5, 1.5, 2.0]), rng.choice([0.5, 1.5, 2.0])
    program = [
        ToolCall("add_box", {"length": l, "width": w, "height": h, "center": [0, 0, 0]}),
        ToolCall("scale", {"target": 0, "factor_x": fx, "factor_y": fy, "factor_z": fz}),
        ToolCall("finalize", {"target": 1}),
    ]
    text = f"a box {l:g}mm x {w:g}mm x {h:g}mm scaled by {fx:g}x in x, {fy:g}x in y, and {fz:g}x in z"
    return text, program


def _pattern_linear_array(rng: random.Random) -> tuple[str, list[ToolCall]]:
    r, ch = rng.choice([3, 4, 5]), rng.choice([10, 15, 20])
    count, spacing = rng.choice([3, 4, 5]), rng.choice([15, 20, 25])
    program = [
        ToolCall("add_cylinder", {"radius": r, "height": ch, "center": [0, 0, 0], "axis": "z"}),
        ToolCall("linear_pattern", {"target": 0, "axis": "x", "count": count, "spacing": spacing}),
        ToolCall("finalize", {"target": 1}),
    ]
    text = f"{count} cylinders {2*r:g}mm diameter x {ch:g}mm tall spaced {spacing:g}mm apart along the x axis"
    return text, program


def _pattern_circular_array(rng: random.Random) -> tuple[str, list[ToolCall]]:
    r, ch = rng.choice([3, 4, 5]), rng.choice([8, 10, 15])
    count, ring_radius = rng.choice([4, 5, 6, 8]), rng.choice([25, 30, 40])
    program = [
        ToolCall("add_cylinder", {"radius": r, "height": ch, "center": [0, 0, 0], "axis": "z"}),
        ToolCall("circular_pattern", {"target": 0, "count": count, "radius": ring_radius}),
        ToolCall("finalize", {"target": 1}),
    ]
    text = f"{count} cylinders {2*r:g}mm diameter x {ch:g}mm tall arranged in a circle of {ring_radius:g}mm radius"
    return text, program


_FACE_DIRECTIONS = ["+x", "-x", "+y", "-y", "+z", "-z"]
_FACE_DIRECTION_WORDS = {"+x": "the +x face", "-x": "the -x face", "+y": "the +y face", "-y": "the -y face", "+z": "the top face", "-z": "the bottom face"}


def _pattern_boss(rng: random.Random) -> tuple[str, list[ToolCall]]:
    l, w, h = rng.choice([30, 40, 50]), rng.choice([20, 30, 40]), rng.choice([8, 10, 15])
    direction = rng.choice(_FACE_DIRECTIONS)
    distance = rng.choice([8, 10, 15, 20])
    program = [
        ToolCall("add_box", {"length": l, "width": w, "height": h, "center": [0, 0, 0]}),
        ToolCall("extrude_face", {"target": 0, "direction": direction, "distance": distance}),
        ToolCall("finalize", {"target": 1}),
    ]
    text = f"a box {l:g}mm x {w:g}mm x {h:g}mm with {_FACE_DIRECTION_WORDS[direction]} extruded out {distance:g}mm"
    return text, program


def _pattern_pocket(rng: random.Random) -> tuple[str, list[ToolCall]]:
    l, w, h = rng.choice([40, 50, 60]), rng.choice([30, 40, 50]), rng.choice([10, 15, 20])
    direction = rng.choice(_FACE_DIRECTIONS)
    depth = rng.choice([2, 3, 4])
    # margin must leave SOMETHING after insetting BOTH in-plane dimensions of
    # whichever face got picked -- a +x/-x face's in-plane dims are (w, h),
    # a +y/-y face's are (l, h), only +z/-z uses (l, w). Deriving margin from
    # min(l, w, h) rather than a fixed list guarantees 2*margin stays well
    # under the smallest possible face dimension regardless of which
    # direction was chosen. Confirmed necessary, not defensive: the fixed
    # list [5, 8, 10] combined with h as small as 10 let margin=10 on a
    # +x/-x face (in-plane height=10) inset the face away entirely,
    # producing an unbuildable "ground truth" training example.
    margin = max(2, min(l, w, h) // 5)
    program = [
        ToolCall("add_box", {"length": l, "width": w, "height": h, "center": [0, 0, 0]}),
        ToolCall("inset_pocket", {"target": 0, "direction": direction, "depth": depth, "margin": margin}),
        ToolCall("finalize", {"target": 1}),
    ]
    text = f"a box {l:g}mm x {w:g}mm x {h:g}mm with a recessed pocket {depth:g}mm deep, {margin:g}mm margin, in {_FACE_DIRECTION_WORDS[direction]}"
    return text, program


def _pattern_shell(rng: random.Random) -> tuple[str, list[ToolCall]]:
    l, w, h = rng.choice([60, 80, 100]), rng.choice([40, 60, 80]), rng.choice([30, 40, 50])
    wall = rng.choice([2, 3, 4, 5])
    program = [
        ToolCall("add_box", {"length": l, "width": w, "height": h, "center": [0, 0, 0]}),
        ToolCall("shell", {"target": 0, "wall_thickness": wall}),
        ToolCall("finalize", {"target": 1}),
    ]
    text = f"a box {l:g}mm x {w:g}mm x {h:g}mm hollowed out leaving {wall:g}mm walls"
    return text, program


def _pattern_shell_cylinder(rng: random.Random) -> tuple[str, list[ToolCall]]:
    """shell() had only ever been paired with add_box in this corpus --
    program_model.py's classifier had no exposure at all to hollowing a
    CYLINDER (a canister, a tube, a cup) and would likely mispredict the
    primitive itself, not just the wall thickness, for that phrasing.
    Confirmed as a real gap, not a hypothetical one, while investigating
    why a "water bottle" prompt decomposed into nonsense."""
    radius, height = rng.choice([20, 30, 40]), rng.choice([40, 60, 80])
    wall = max(2, min(radius, height) // 15)
    program = [
        ToolCall("add_cylinder", {"radius": radius, "height": height, "center": [0, 0, 0], "axis": "z"}),
        ToolCall("shell", {"target": 0, "wall_thickness": wall}),
        ToolCall("finalize", {"target": 1}),
    ]
    text = f"a cylinder {2*radius:g}mm diameter, {height:g}mm tall, hollowed out leaving {wall:g}mm walls"
    return text, program


def _pattern_shouldered_cylinder(rng: random.Random) -> tuple[str, list[ToolCall]]:
    """The solid (non-hollowed) case of add_shouldered_cylinder -- a table
    leg, a lamp base, a bowling pin, a doorknob spindle: anything that
    smoothly changes diameter partway up without being a container.
    Kept as its own pattern (not just a step inside _pattern_hollow_vessel)
    so program_model.py sees add_shouldered_cylinder used ALONE too, not
    only ever immediately followed by shell."""
    bottom_radius = rng.choice([15, 20, 25])
    bottom_height = rng.choice([60, 80, 100])
    top_radius = rng.choice([5, 8, 10])
    top_height = rng.choice([20, 30, 40])
    shoulder_height = rng.choice([10, 15, 20])
    program = [
        ToolCall("add_shouldered_cylinder", {
            "bottom_radius": bottom_radius, "bottom_height": bottom_height, "top_radius": top_radius,
            "top_height": top_height, "transition_height": shoulder_height, "center": [0, 0, 0],
        }),
        ToolCall("finalize", {"target": 0}),
    ]
    text = (
        f"a {2*bottom_radius:g}mm diameter, {bottom_height:g}mm tall cylinder smoothly narrowing to a "
        f"{2*top_radius:g}mm diameter, {top_height:g}mm tall section over a {shoulder_height:g}mm shoulder"
    )
    return text, program


def _pattern_hollow_vessel(rng: random.Random) -> tuple[str, list[ToolCall]]:
    """A genuine container/vessel composition -- a wide body CURVING
    smoothly (not stepping) into a narrower neck, then hollowed -- the
    shape family a bottle, jar, flask, or vase actually needs, which this
    corpus had NO example of at all before (see _pattern_shell_cylinder's
    note). Uses add_shouldered_cylinder rather than unioning two plain
    cylinders: confirmed directly (not assumed) that the union approach
    leaves an actual sharp 90-degree step at the shoulder, which looks
    nothing like a real bottle -- add_shouldered_cylinder's smoothstep
    profile is tangent-continuous with both the body and the neck, a real
    curve. Also confirmed directly that shell() still produces a real,
    watertight cavity on this single curved-profile shape: a point at the
    body's own center tests outside the solid material, a point just
    inside the outer wall tests inside it."""
    body_radius = rng.choice([25, 30, 35])
    body_height = rng.choice([120, 150, 180])
    neck_radius = rng.choice([8, 10, 12])
    neck_height = rng.choice([20, 25, 30])
    shoulder_height = rng.choice([15, 20, 25])
    wall = 2
    program = [
        ToolCall("add_shouldered_cylinder", {
            "bottom_radius": body_radius, "bottom_height": body_height, "top_radius": neck_radius,
            "top_height": neck_height, "transition_height": shoulder_height, "center": [0, 0, 0],
        }),
        ToolCall("shell", {"target": 0, "wall_thickness": wall}),
        ToolCall("finalize", {"target": 1}),
    ]
    text = (
        f"a hollow vessel: a {2*body_radius:g}mm diameter, {body_height:g}mm tall body smoothly narrowing to a "
        f"{2*neck_radius:g}mm diameter, {neck_height:g}mm tall neck over a {shoulder_height:g}mm shoulder, {wall:g}mm walls"
    )
    return text, program


def _pattern_widening_vessel(rng: random.Random) -> tuple[str, list[ToolCall]]:
    """The OTHER real vessel direction -- a vase, funnel, or flowerpot that
    WIDENS toward its opening, not a bottle-style narrowing neck. Confirmed
    as a genuine, previously-uncovered gap live: asked the actual LLM
    decomposer for "a flower vase, wider at the top than the bottom" and it
    defaulted to the narrowing-neck template anyway (body wider than the
    requested opening), because this corpus had zero examples of the
    opposite direction to weigh against that default. Structurally this is
    just add_shouldered_cylinder with bottom_radius < top_radius -- the
    tool already supports it fine, only the training/prompt guidance was
    missing."""
    base_radius = rng.choice([15, 20, 25])
    base_height = rng.choice([20, 30, 40])
    top_radius = rng.choice([35, 40, 50])
    top_height = rng.choice([100, 130, 160])
    shoulder_height = rng.choice([15, 20, 25])
    wall = 2
    program = [
        ToolCall("add_shouldered_cylinder", {
            "bottom_radius": base_radius, "bottom_height": base_height, "top_radius": top_radius,
            "top_height": top_height, "transition_height": shoulder_height, "center": [0, 0, 0],
        }),
        ToolCall("shell", {"target": 0, "wall_thickness": wall}),
        ToolCall("finalize", {"target": 1}),
    ]
    text = (
        f"a hollow vessel: a {2*base_radius:g}mm diameter, {base_height:g}mm tall base smoothly widening to a "
        f"{2*top_radius:g}mm diameter, {top_height:g}mm tall top opening over a {shoulder_height:g}mm shoulder, {wall:g}mm walls"
    )
    return text, program


def _pattern_frustum(rng: random.Random) -> tuple[str, list[ToolCall]]:
    bottom_r, top_r = rng.choice([15, 20, 25]), rng.choice([5, 8, 10])
    height = rng.choice([20, 30, 40])
    program = [
        ToolCall("add_revolve", {"bottom_radius": bottom_r, "top_radius": top_r, "height": height, "center": [0, 0, 0]}),
        ToolCall("finalize", {"target": 0}),
    ]
    text = f"a tapered cylinder, {2*bottom_r:g}mm diameter at the bottom, {2*top_r:g}mm diameter at the top, {height:g}mm tall"
    return text, program


def _pattern_cone(rng: random.Random) -> tuple[str, list[ToolCall]]:
    radius, height = rng.choice([15, 20, 25, 30]), rng.choice([20, 30, 40])
    program = [
        ToolCall("add_revolve", {"bottom_radius": radius, "top_radius": 0, "height": height, "center": [0, 0, 0]}),
        ToolCall("finalize", {"target": 0}),
    ]
    text = f"a cone, {2*radius:g}mm base diameter, {height:g}mm tall"
    return text, program


def _pattern_tapered_box(rng: random.Random) -> tuple[str, list[ToolCall]]:
    l, w, h = rng.choice([30, 40, 50]), rng.choice([30, 40, 50]), rng.choice([20, 30, 40])
    top_factor = rng.choice([0.3, 0.5, 0.7])
    program = [
        ToolCall("add_box", {"length": l, "width": w, "height": h, "center": [0, 0, 0]}),
        ToolCall("taper", {"target": 0, "axis": "z", "bottom_factor": 1.0, "top_factor": top_factor}),
        ToolCall("finalize", {"target": 1}),
    ]
    text = f"a box {l:g}mm x {w:g}mm x {h:g}mm tapering to {top_factor*100:g}% width at the top"
    return text, program


_PERPENDICULAR_DIRECTION_PAIRS = [
    ("+z", "+x"), ("+z", "-x"), ("+z", "+y"), ("+z", "-y"),
    ("-z", "+x"), ("-z", "-x"), ("-z", "+y"), ("-z", "-y"),
    ("+x", "+y"), ("+x", "-y"), ("-x", "+y"), ("-x", "-y"),
]


def _pattern_fillet(rng: random.Random) -> tuple[str, list[ToolCall]]:
    l, w, h = rng.choice([40, 50, 60]), rng.choice([30, 40, 50]), rng.choice([20, 30, 40])
    d1, d2 = rng.choice(_PERPENDICULAR_DIRECTION_PAIRS)
    # kept safely small relative to the box's own smallest dimension -- same
    # reasoning as _pattern_pocket's margin fix: an edge tool's radius/distance
    # too close to half the smallest box dimension risks an unbuildable
    # ground-truth example, not something to discover via a failing sweep test.
    radius = max(2, min(l, w, h) // 6)
    program = [
        ToolCall("add_box", {"length": l, "width": w, "height": h, "center": [0, 0, 0]}),
        ToolCall("fillet_edge", {"target": 0, "direction1": d1, "direction2": d2, "radius": radius}),
        ToolCall("finalize", {"target": 1}),
    ]
    text = f"a box {l:g}mm x {w:g}mm x {h:g}mm with a {radius:g}mm rounded edge where the {d1} and {d2} faces meet"
    return text, program


def _pattern_chamfer(rng: random.Random) -> tuple[str, list[ToolCall]]:
    l, w, h = rng.choice([40, 50, 60]), rng.choice([30, 40, 50]), rng.choice([20, 30, 40])
    d1, d2 = rng.choice(_PERPENDICULAR_DIRECTION_PAIRS)
    distance = max(2, min(l, w, h) // 6)
    program = [
        ToolCall("add_box", {"length": l, "width": w, "height": h, "center": [0, 0, 0]}),
        ToolCall("chamfer_edge", {"target": 0, "direction1": d1, "direction2": d2, "distance": distance}),
        ToolCall("finalize", {"target": 1}),
    ]
    text = f"a box {l:g}mm x {w:g}mm x {h:g}mm with a {distance:g}mm chamfered edge where the {d1} and {d2} faces meet"
    return text, program


def _pattern_counterbore(rng: random.Random) -> tuple[str, list[ToolCall]]:
    l, w, h = rng.choice([50, 60, 70]), rng.choice([40, 50, 60]), rng.choice([15, 20, 25])
    direction = rng.choice(_FACE_DIRECTIONS)
    # kept safely small relative to the box's own smallest dimension --
    # same reasoning as _pattern_fillet/_pattern_pocket's margin fix: a
    # hole/head too close to half the smallest relevant face dimension
    # risks an unbuildable ground-truth example.
    hole_radius = max(2, min(l, w, h) // 12)
    head_radius = hole_radius * 2
    head_depth = max(2, min(l, w, h) // 8)
    head_style = rng.choice(["flat", "conical"])
    program = [
        ToolCall("add_box", {"length": l, "width": w, "height": h, "center": [0, 0, 0]}),
        ToolCall("counterbore_hole", {
            "target": 0, "direction": direction, "offset_u": 0, "offset_v": 0,
            "hole_radius": hole_radius, "head_radius": head_radius, "head_depth": head_depth, "head_style": head_style,
        }),
        ToolCall("finalize", {"target": 1}),
    ]
    style_word = "flat-bottomed counterbore" if head_style == "flat" else "conical countersink"
    text = (
        f"a box {l:g}mm x {w:g}mm x {h:g}mm with a {hole_radius*2:g}mm fastener clearance hole and a "
        f"{head_radius*2:g}mm {style_word} recess on the {direction} face"
    )
    return text, program


def _pattern_drill_slot(rng: random.Random) -> tuple[str, list[ToolCall]]:
    l, w, h = rng.choice([60, 70, 80]), rng.choice([40, 50, 60]), rng.choice([10, 15, 20])
    direction = rng.choice(["+z", "-z"])  # the two large flat faces -- keeps the slot generously within bounds
    radius = max(2, min(l, w) // 20)
    slot_length = min(l, w) // 3
    slot_axis = rng.choice(["u", "v"])
    program = [
        ToolCall("add_box", {"length": l, "width": w, "height": h, "center": [0, 0, 0]}),
        ToolCall("drill_slot", {
            "target": 0, "direction": direction, "offset_u": 0, "offset_v": 0,
            "radius": radius, "slot_length": slot_length, "slot_axis": slot_axis, "depth": h,
        }),
        ToolCall("finalize", {"target": 1}),
    ]
    text = f"a box {l:g}mm x {w:g}mm x {h:g}mm with a {radius*2:g}mm wide, {slot_length:g}mm long mounting slot through the {direction} face"
    return text, program


def _pattern_rib(rng: random.Random) -> tuple[str, list[ToolCall]]:
    thickness = rng.choice([8, 10, 12])
    leg1_len = rng.choice([40, 50, 60])
    leg2_len = rng.choice([40, 50, 60])
    depth = rng.choice([30, 40, 50])
    size = max(2, thickness // 2)
    # An L-bracket (two boxes unioned at a right angle) -- the simplest
    # shape with a genuine CONCAVE corner. add_rib only does anything
    # useful at a concave corner (see sandbox.py's own note: the same
    # wedge at a plain convex box corner would already lie inside existing
    # material and add nothing) -- so this pattern is the union itself,
    # not a bare box, to keep the ground truth meaningful.
    program = [
        ToolCall("add_box", {"length": thickness, "width": depth, "height": leg1_len, "center": [thickness / 2, depth / 2, leg1_len / 2]}),
        ToolCall("add_box", {"length": leg2_len, "width": depth, "height": thickness, "center": [leg2_len / 2, depth / 2, thickness / 2]}),
        ToolCall("boolean_union", {"a": 0, "b": 1}),
        ToolCall("add_rib", {"target": 2, "direction1": "+x", "direction2": "+z", "size": size}),
        ToolCall("finalize", {"target": 3}),
    ]
    text = (
        f"an L-bracket {thickness:g}mm thick, legs {leg1_len:g}mm and {leg2_len:g}mm long, {depth:g}mm deep, "
        f"with a {size:g}mm support rib in the inside corner"
    )
    return text, program


def _pattern_sphere(rng: random.Random) -> tuple[str, list[ToolCall]]:
    # deliberately 10 distinct values, not 3-4 like most patterns here --
    # confirmed directly that with only 4 choices, canonical-key-hash-based
    # train/val/test splitting can route ALL duplicate occurrences of most
    # values away from train (every occurrence of a given text hashes to
    # the same split), leaving the classifier having only ever seen ONE
    # exact diameter string during training and misclassifying every other
    # (lexically near-identical) diameter as add_cylinder+drill_hole. A
    # single-parameter pattern like this is uniquely exposed to that
    # failure mode -- most other new patterns vary 2-3 dimensions, so the
    # combinatorial spread already avoids this.
    radius = rng.choice([8, 10, 12, 15, 18, 20, 22, 25, 28, 30])
    program = [
        ToolCall("add_sphere", {"radius": radius, "center": [0, 0, 0]}),
        ToolCall("finalize", {"target": 0}),
    ]
    text = f"a sphere {2*radius:g}mm diameter"
    return text, program


def _pattern_torus(rng: random.Random) -> tuple[str, list[ToolCall]]:
    major_radius = rng.choice([20, 30, 40])
    minor_radius = rng.choice([4, 6, 8])
    axis = rng.choice(["x", "y", "z"])
    program = [
        ToolCall("add_torus", {"major_radius": major_radius, "minor_radius": minor_radius, "center": [0, 0, 0], "axis": axis}),
        ToolCall("finalize", {"target": 0}),
    ]
    text = f"a torus (ring donut shape) {2*major_radius:g}mm across, {2*minor_radius:g}mm tube diameter, around the {axis} axis"
    return text, program


def _pattern_capsule(rng: random.Random) -> tuple[str, list[ToolCall]]:
    radius = rng.choice([6, 8, 10])
    height = rng.choice([40, 60, 80])
    axis = rng.choice(["x", "y", "z"])
    program = [
        ToolCall("add_capsule", {"radius": radius, "height": height, "center": [0, 0, 0], "axis": axis}),
        ToolCall("finalize", {"target": 0}),
    ]
    text = f"a capsule (pill shape) {2*radius:g}mm diameter, {height:g}mm long, along the {axis} axis"
    return text, program


def _pattern_ellipsoid(rng: random.Random) -> tuple[str, list[ToolCall]]:
    rx, ry, rz = rng.choice([15, 20, 25]), rng.choice([10, 15, 20]), rng.choice([8, 12, 16])
    program = [
        ToolCall("add_ellipsoid", {"radius_x": rx, "radius_y": ry, "radius_z": rz, "center": [0, 0, 0]}),
        ToolCall("finalize", {"target": 0}),
    ]
    text = f"an egg-like ellipsoid {2*rx:g}mm x {2*ry:g}mm x {2*rz:g}mm"
    return text, program


def _pattern_polygon_frustum(rng: random.Random) -> tuple[str, list[ToolCall]]:
    sides = rng.choice([5, 6, 8])
    bottom_r, top_r = rng.choice([15, 20, 25]), rng.choice([8, 10, 12])
    height = rng.choice([20, 30, 40])
    program = [
        ToolCall("add_polygon_frustum", {
            "sides": sides, "bottom_radius": bottom_r, "top_radius": top_r, "height": height,
            "center": [0, 0, 0], "axis": "z",
        }),
        ToolCall("finalize", {"target": 0}),
    ]
    text = f"a {sides}-sided prism, {2*bottom_r:g}mm across at the bottom narrowing to {2*top_r:g}mm at the top, {height:g}mm tall"
    return text, program


def _pattern_dome(rng: random.Random) -> tuple[str, list[ToolCall]]:
    radius = rng.choice([20, 25, 30])
    height = rng.choice([10, 15, 20])
    program = [
        ToolCall("add_dome", {"radius": radius, "height": height, "center": [0, 0, 0], "axis": "z"}),
        ToolCall("finalize", {"target": 0}),
    ]
    text = f"a dome {2*radius:g}mm diameter base, {height:g}mm tall"
    return text, program


def _pattern_annulus_primitive(rng: random.Random) -> tuple[str, list[ToolCall]]:
    """add_ring, the direct annulus primitive -- distinct from the
    existing _pattern_ring, which builds its ring via two unioned
    cylinders + boolean_difference. Both belong in the corpus: the
    classifier needs exposure to the direct one-call primitive too, not
    only the boolean-composition route to the same shape family."""
    outer_radius, height = rng.choice([25, 35, 45]), rng.choice([5, 8, 10])
    inner_radius = outer_radius * rng.choice([0.4, 0.5, 0.6])
    program = [
        ToolCall("add_ring", {
            "outer_radius": outer_radius, "inner_radius": inner_radius, "height": height,
            "center": [0, 0, 0], "axis": "z",
        }),
        ToolCall("finalize", {"target": 0}),
    ]
    text = f"a flat ring/washer, {2*outer_radius:g}mm outer diameter, {2*inner_radius:g}mm inner diameter, {height:g}mm thick"
    return text, program


def _pattern_flare(rng: random.Random) -> tuple[str, list[ToolCall]]:
    radius, height = rng.choice([15, 20, 25]), rng.choice([60, 80, 100])
    top_factor = rng.choice([1.4, 1.6, 1.8])
    program = [
        ToolCall("add_cylinder", {"radius": radius, "height": height, "center": [0, 0, 0], "axis": "z"}),
        ToolCall("flare", {"target": 0, "axis": "z", "bottom_factor": 1.0, "top_factor": top_factor}),
        ToolCall("finalize", {"target": 1}),
    ]
    text = f"a cylinder {2*radius:g}mm diameter, {height:g}mm tall, smoothly flaring out to {top_factor*100:g}% width at the top"
    return text, program


def _pattern_twist(rng: random.Random) -> tuple[str, list[ToolCall]]:
    l, w, h = rng.choice([20, 25, 30]), rng.choice([20, 25, 30]), rng.choice([80, 100, 120])
    angle = rng.choice([45, 90, 180])
    program = [
        ToolCall("add_box", {"length": l, "width": w, "height": h, "center": [0, 0, 0]}),
        ToolCall("twist", {"target": 0, "axis": "z", "angle_deg": angle}),
        ToolCall("finalize", {"target": 1}),
    ]
    text = f"a box {l:g}mm x {w:g}mm x {h:g}mm twisted {angle:g} degrees along its height"
    return text, program


def _pattern_bend(rng: random.Random) -> tuple[str, list[ToolCall]]:
    l, w, h = rng.choice([100, 120, 150]), rng.choice([15, 20, 25]), rng.choice([10, 15, 20])
    angle = rng.choice([30, 45, 60, 90])
    program = [
        ToolCall("add_box", {"length": l, "width": w, "height": h, "center": [0, 0, 0]}),
        ToolCall("bend", {"target": 0, "axis": "x", "bend_direction": "z", "angle_deg": angle}),
        ToolCall("finalize", {"target": 1}),
    ]
    text = f"a bar {l:g}mm x {w:g}mm x {h:g}mm bent {angle:g} degrees along its length"
    return text, program


def _pattern_round_box_edges(rng: random.Random) -> tuple[str, list[ToolCall]]:
    l, w, h = rng.choice([40, 50, 60]), rng.choice([30, 40, 50]), rng.choice([20, 25, 30])
    radius = max(2, min(l, w, h) // 8)
    program = [
        ToolCall("add_box", {"length": l, "width": w, "height": h, "center": [0, 0, 0]}),
        ToolCall("round_box_edges", {"target": 0, "radius": radius}),
        ToolCall("finalize", {"target": 1}),
    ]
    text = f"a box {l:g}mm x {w:g}mm x {h:g}mm with all edges rounded to a {radius:g}mm radius"
    return text, program


def _pattern_chamfer_box_edges(rng: random.Random) -> tuple[str, list[ToolCall]]:
    l, w, h = rng.choice([40, 50, 60]), rng.choice([30, 40, 50]), rng.choice([20, 25, 30])
    distance = max(2, min(l, w, h) // 8)
    program = [
        ToolCall("add_box", {"length": l, "width": w, "height": h, "center": [0, 0, 0]}),
        ToolCall("chamfer_box_edges", {"target": 0, "distance": distance}),
        ToolCall("finalize", {"target": 1}),
    ]
    text = f"a box {l:g}mm x {w:g}mm x {h:g}mm with all edges chamfered {distance:g}mm"
    return text, program


def _pattern_mirror_merge(rng: random.Random) -> tuple[str, list[ToolCall]]:
    l, w, h = rng.choice([20, 30, 40]), rng.choice([15, 20, 30]), rng.choice([6, 8, 10])
    offset = rng.choice([15, 20, 25])
    axis = rng.choice(["x", "y"])
    center = [offset, 0, 0] if axis == "x" else [0, offset, 0]
    program = [
        ToolCall("add_box", {"length": l, "width": w, "height": h, "center": center}),
        ToolCall("mirror_merge", {"target": 0, "axis": axis}),
        ToolCall("finalize", {"target": 1}),
    ]
    text = f"a box {l:g}mm x {w:g}mm x {h:g}mm mirrored across the {axis} axis and merged with itself into one symmetric part"
    return text, program


def _pattern_boss_with_bore(rng: random.Random) -> tuple[str, list[ToolCall]]:
    l, w, h = rng.choice([50, 60, 70]), rng.choice([40, 50, 60]), rng.choice([6, 8, 10])
    direction = rng.choice(_FACE_DIRECTIONS)
    boss_radius = max(4, min(l, w) // 8)
    boss_height = rng.choice([6, 8, 10])
    bore_radius = max(1, boss_radius // 2)
    program = [
        ToolCall("add_box", {"length": l, "width": w, "height": h, "center": [0, 0, 0]}),
        ToolCall("add_boss_with_bore", {
            "target": 0, "direction": direction, "offset_u": 0, "offset_v": 0,
            "boss_radius": boss_radius, "boss_height": boss_height, "bore_radius": bore_radius,
        }),
        ToolCall("finalize", {"target": 1}),
    ]
    text = (
        f"an enclosure wall {l:g}mm x {w:g}mm x {h:g}mm with a screw boss on {_FACE_DIRECTION_WORDS[direction]}: "
        f"{2*boss_radius:g}mm diameter, {boss_height:g}mm tall, with a {2*bore_radius:g}mm bore through the middle"
    )
    return text, program


def _pattern_counterbore_hole_pattern(rng: random.Random) -> tuple[str, list[ToolCall]]:
    l, w, h = rng.choice([80, 100, 120]), rng.choice([60, 80, 100]), rng.choice([15, 20, 25])
    hole_radius = max(2, min(l, w, h) // 15)
    head_radius = hole_radius * 2
    head_depth = max(2, min(l, w, h) // 8)
    head_style = rng.choice(["flat", "conical"])
    pattern = rng.choice(["corner_array", "bolt_circle"])
    margin = max(10, min(l, w) // 6)
    if pattern == "corner_array":
        count, circle_diameter = 4, 0.0
    else:
        count, circle_diameter = rng.choice([4, 6, 8]), min(l, w) - 2 * margin
    program = [
        ToolCall("add_box", {"length": l, "width": w, "height": h, "center": [0, 0, 0]}),
        ToolCall("counterbore_hole_pattern", {
            "target": 0, "direction": "+z", "pattern": pattern, "count": count,
            "hole_radius": hole_radius, "head_radius": head_radius, "head_depth": head_depth, "head_style": head_style,
            "margin": margin, "circle_diameter": circle_diameter,
        }),
        ToolCall("finalize", {"target": 1}),
    ]
    layout_word = "at the four corners" if pattern == "corner_array" else f"in a bolt circle of {count}"
    text = (
        f"a plate {l:g}mm x {w:g}mm x {h:g}mm with counterbored fastener holes {layout_word}, "
        f"{hole_radius*2:g}mm holes with a {head_radius*2:g}mm recess"
    )
    return text, program


def _pattern_add_slot_pattern(rng: random.Random) -> tuple[str, list[ToolCall]]:
    l, w, h = rng.choice([100, 120, 150]), rng.choice([60, 80, 100]), rng.choice([10, 15, 20])
    radius = max(2, min(l, w) // 25)
    slot_length = min(l, w) // 4
    slot_axis = rng.choice(["u", "v"])
    count = rng.choice([2, 3, 4])
    margin = max(15, min(l, w) // 5)
    program = [
        ToolCall("add_box", {"length": l, "width": w, "height": h, "center": [0, 0, 0]}),
        ToolCall("add_slot_pattern", {
            "target": 0, "direction": "+z", "pattern": "linear_array", "count": count,
            "radius": radius, "slot_length": slot_length, "slot_axis": slot_axis, "depth": h,
            "margin": margin, "circle_diameter": 0.0,
        }),
        ToolCall("finalize", {"target": 1}),
    ]
    text = f"a plate {l:g}mm x {w:g}mm x {h:g}mm with a row of {count} adjustable mounting slots, {radius*2:g}mm wide, {slot_length:g}mm long"
    return text, program


def _pattern_knurl(rng: random.Random) -> tuple[str, list[ToolCall]]:
    radius, height = rng.choice([12, 15, 20]), rng.choice([15, 20, 25])
    ridge_count = rng.choice([12, 16, 20])
    knurl_radius = radius - 3
    knurl_height = rng.choice([1, 2, 3])
    program = [
        ToolCall("add_cylinder", {"radius": radius, "height": height, "center": [0, 0, 0], "axis": "z"}),
        ToolCall("add_knurl_pattern", {
            "target": 0, "direction": "+z", "offset_u": 0, "offset_v": 0,
            "radius": knurl_radius, "height": knurl_height, "ridge_count": ridge_count,
        }),
        ToolCall("finalize", {"target": 1}),
    ]
    text = f"a knob {2*radius:g}mm diameter, {height:g}mm tall, with a knurled grip texture of {ridge_count} ridges on top"
    return text, program


def _pattern_radial_array_oriented(rng: random.Random) -> tuple[str, list[ToolCall]]:
    l, w, h = rng.choice([2, 3, 4]), rng.choice([8, 10, 12]), rng.choice([4, 6, 8])
    count, ring_radius = rng.choice([6, 8, 10]), rng.choice([20, 30, 40])
    program = [
        ToolCall("add_box", {"length": l, "width": w, "height": h, "center": [0, 0, 0]}),
        ToolCall("radial_array_oriented", {"target": 0, "count": count, "radius": ring_radius}),
        ToolCall("finalize", {"target": 1}),
    ]
    text = f"{count} spokes {l:g}mm x {w:g}mm x {h:g}mm arranged in a circle of {ring_radius:g}mm radius, each pointing outward from the center"
    return text, program


def _pattern_linear_array_scaled(rng: random.Random) -> tuple[str, list[ToolCall]]:
    l, w, h = rng.choice([6, 8, 10]), rng.choice([6, 8, 10]), rng.choice([6, 8, 10])
    count, spacing = rng.choice([4, 5, 6]), rng.choice([20, 25, 30])
    scale_step = rng.choice([0.2, 0.3, 0.4])
    program = [
        ToolCall("add_box", {"length": l, "width": w, "height": h, "center": [0, 0, 0]}),
        ToolCall("linear_array_scaled", {"target": 0, "axis": "x", "count": count, "spacing": spacing, "scale_step": scale_step}),
        ToolCall("finalize", {"target": 1}),
    ]
    text = (
        f"a row of {count} boxes {l:g}mm x {w:g}mm x {h:g}mm spaced {spacing:g}mm apart along the x axis, "
        f"each one {scale_step*100:g}% larger than the last"
    )
    return text, program


def _pattern_hex_socket(rng: random.Random) -> tuple[str, list[ToolCall]]:
    l, w, h = rng.choice([40, 50, 60]), rng.choice([40, 50, 60]), rng.choice([15, 20, 25])
    across_flats = rng.choice([6, 8, 10])
    depth = max(3, h // 3)
    program = [
        ToolCall("add_box", {"length": l, "width": w, "height": h, "center": [0, 0, 0]}),
        ToolCall("add_hex_socket", {
            "target": 0, "direction": "+z", "offset_u": 0, "offset_v": 0,
            "across_flats": across_flats, "depth": depth,
        }),
        ToolCall("finalize", {"target": 1}),
    ]
    text = f"a block {l:g}mm x {w:g}mm x {h:g}mm with a {across_flats:g}mm hex socket, {depth:g}mm deep, in the top face"
    return text, program


def _pattern_loft(rng: random.Random) -> tuple[str, list[ToolCall]]:
    """add_loft had ZERO generic-composition training examples despite
    being registered in sandbox.py/ARG_SCHEMA since earlier this session --
    confirmed directly (grepped this whole file) rather than assumed, while
    auditing which tools are actually reachable from natural-language text.
    With no examples, program_model.py has nothing to learn the text->tool
    association from -- the tool exists but is functionally dead from the
    classifier's point of view. Phrasing matches llm_decompose.py's own
    "Available operations" vocabulary line for loft, added alongside this."""
    length, width = rng.choice([40, 60, 80]), rng.choice([30, 40, 50])
    radius = rng.choice([15, 20, 25])
    height = rng.choice([60, 80, 100])
    program = [
        ToolCall("add_loft", {
            "bottom_profile": "rounded_rect", "bottom_param_a": length, "bottom_param_b": width, "bottom_param_c": 5,
            "top_profile": "ellipse", "top_param_a": radius, "top_param_b": radius, "top_param_c": 0,
            "height": height, "center": [0, 0, 0], "sections": 24,
        }),
        ToolCall("finalize", {"target": 0}),
    ]
    text = (
        f"a lofted shape, {length:g}mm x {width:g}mm rounded rectangle at the bottom blending to a "
        f"{2*radius:g}mm diameter ellipse at the top, {height:g}mm tall"
    )
    return text, program


def _pattern_sweep_arc(rng: random.Random) -> tuple[str, list[ToolCall]]:
    """Same gap as add_loft above: add_sweep had zero training examples."""
    radius = rng.choice([5, 8, 10])
    path_radius = rng.choice([40, 60, 80])
    angle = rng.choice([45, 90, 120])
    program = [
        ToolCall("add_sweep", {
            "profile_kind": "ellipse", "param_a": radius, "param_b": radius, "param_c": 0,
            "path_kind": "arc", "path_param_a": path_radius, "path_param_b": angle, "path_param_c": 0,
            "center": [0, 0, 0], "sections": 24, "path_segments": 48,
        }),
        ToolCall("finalize", {"target": 0}),
    ]
    text = f"a {2*radius:g}mm diameter round rail swept along a {angle:g} degree arc of {path_radius:g}mm radius"
    return text, program


def _pattern_sweep_helix(rng: random.Random) -> tuple[str, list[ToolCall]]:
    """A coil/spring shape -- the helix path_kind case, kept as its own
    pattern (not folded into _pattern_sweep_arc) so program_model.py sees
    BOTH path kinds, not just arc."""
    radius = rng.choice([3, 5, 8])
    helix_radius = rng.choice([20, 30, 40])
    pitch = rng.choice([10, 15, 20])
    turns = rng.choice([3, 5, 8])
    program = [
        ToolCall("add_sweep", {
            "profile_kind": "ellipse", "param_a": radius, "param_b": radius, "param_c": 0,
            "path_kind": "helix", "path_param_a": helix_radius, "path_param_b": pitch, "path_param_c": turns,
            "center": [0, 0, 0], "sections": 20, "path_segments": 64,
        }),
        ToolCall("finalize", {"target": 0}),
    ]
    text = f"a {2*radius:g}mm diameter coil, {helix_radius:g}mm radius, {pitch:g}mm pitch, {turns:g} turns"
    return text, program


def _pattern_tube_straight(rng: random.Random) -> tuple[str, list[ToolCall]]:
    """Same gap as add_loft/add_sweep above: add_tube had zero training
    examples. Straight-path case -- a simple hollow tube/pipe segment."""
    outer = rng.choice([10, 15, 20])
    inner = max(2, outer - rng.choice([2, 3, 4]))
    length = rng.choice([80, 100, 150])
    program = [
        ToolCall("add_tube", {
            "outer_radius": outer, "inner_radius": inner, "path_kind": "straight",
            "path_param_a": length, "path_param_b": 0, "path_param_c": 0,
            "center": [0, 0, 0], "sections": 24, "path_segments": 8,
        }),
        ToolCall("finalize", {"target": 0}),
    ]
    text = f"a hollow tube, {2*outer:g}mm outer diameter, {2*inner:g}mm inner diameter, {length:g}mm long"
    return text, program


def _pattern_tube_arc(rng: random.Random) -> tuple[str, list[ToolCall]]:
    """The arc-path case -- a curved pipe/handrail run, kept separate from
    _pattern_tube_straight so program_model.py sees both path kinds."""
    outer = rng.choice([8, 10, 15])
    inner = max(2, outer - rng.choice([2, 3]))
    path_radius = rng.choice([50, 80, 100])
    angle = rng.choice([90, 120, 180])
    program = [
        ToolCall("add_tube", {
            "outer_radius": outer, "inner_radius": inner, "path_kind": "arc",
            "path_param_a": path_radius, "path_param_b": angle, "path_param_c": 0,
            "center": [0, 0, 0], "sections": 20, "path_segments": 48,
        }),
        ToolCall("finalize", {"target": 0}),
    ]
    text = (
        f"a curved pipe, {2*outer:g}mm outer diameter, {2*inner:g}mm inner diameter, "
        f"bent through a {angle:g} degree arc of {path_radius:g}mm radius"
    )
    return text, program


def _pattern_smooth_subdivide(rng: random.Random) -> tuple[str, list[ToolCall]]:
    """smooth_subdivide/add_wireframe_from_edges/weld_vertices were added
    this session (closing a gap found by cross-referencing acquired CAD
    documentation against the tool roster -- see knowledge/sources.py and
    MISSION_STATUS.md's Addendum 2) but, same as add_loft/add_sweep/
    add_tube above, had zero training examples at the time they were
    registered -- confirmed, not assumed."""
    l, w, h = rng.choice([40, 50, 60]), rng.choice([40, 50, 60]), rng.choice([20, 30, 40])
    subdivisions = rng.choice([1, 2])
    program = [
        ToolCall("add_box", {"length": l, "width": w, "height": h, "center": [0, 0, 0]}),
        ToolCall("smooth_subdivide", {"target": 0, "subdivisions": subdivisions, "iterations": 10}),
        ToolCall("finalize", {"target": 1}),
    ]
    text = f"a box {l:g}mm x {w:g}mm x {h:g}mm, smoothed and rounded into a softer organic form"
    return text, program


def _pattern_wireframe(rng: random.Random) -> tuple[str, list[ToolCall]]:
    l, w, h = rng.choice([40, 60, 80]), rng.choice([40, 60, 80]), rng.choice([40, 60, 80])
    edge_radius = rng.choice([1, 2, 3])
    program = [
        ToolCall("add_box", {"length": l, "width": w, "height": h, "center": [0, 0, 0]}),
        ToolCall("add_wireframe_from_edges", {"target": 0, "edge_radius": edge_radius, "sections": 6}),
        ToolCall("finalize", {"target": 1}),
    ]
    text = f"a box {l:g}mm x {w:g}mm x {h:g}mm converted into a wireframe skeleton, {edge_radius:g}mm thick edges"
    return text, program


def _pattern_weld(rng: random.Random) -> tuple[str, list[ToolCall]]:
    """weld_vertices's natural use case is cleanup after another operation
    (a mirror seam here), not a shape on its own -- kept as a 3-step
    program so program_model.py sees it in realistic context."""
    l, w, h = rng.choice([40, 60, 80]), rng.choice([30, 40, 50]), rng.choice([20, 30])
    program = [
        ToolCall("add_box", {"length": l, "width": w, "height": h, "center": [0, 0, 0]}),
        ToolCall("mirror_merge", {"target": 0, "axis": "x"}),
        ToolCall("weld_vertices", {"target": 1, "tolerance": 0.01}),
        ToolCall("finalize", {"target": 2}),
    ]
    text = (
        f"a box {l:g}mm x {w:g}mm x {h:g}mm mirrored and fused into one symmetric part, "
        "with the seam vertices welded clean"
    )
    return text, program


_GENERIC_PATTERNS = [
    _pattern_union, _pattern_cavity, _pattern_ring, _pattern_holes, _pattern_holes, _pattern_holes,
    _pattern_intersection, _pattern_mirror, _pattern_scaled, _pattern_linear_array, _pattern_circular_array,
    _pattern_boss, _pattern_pocket, _pattern_shell, _pattern_fillet, _pattern_chamfer,
    _pattern_frustum, _pattern_cone, _pattern_tapered_box,
    _pattern_counterbore, _pattern_drill_slot, _pattern_rib,
    _pattern_shell_cylinder, _pattern_hollow_vessel, _pattern_widening_vessel, _pattern_shouldered_cylinder,
    # 20 more Blender-like tools
    _pattern_sphere, _pattern_sphere, _pattern_sphere,  # 3x weight: confirmed directly that this
    # pattern's single, 10-value radius parameter otherwise gets too few raw
    # samples-per-value out of an even per-pattern share of n_generic for
    # canonical-key-hash train/val/test splitting to reliably put ANY given
    # exact diameter string in the train split -- every other new pattern
    # varies 2+ dimensions, so this collision risk is unique to add_sphere.
    _pattern_torus, _pattern_capsule, _pattern_ellipsoid, _pattern_ellipsoid,
    _pattern_polygon_frustum, _pattern_dome, _pattern_annulus_primitive,
    _pattern_flare, _pattern_twist, _pattern_bend,
    _pattern_round_box_edges, _pattern_chamfer_box_edges, _pattern_mirror_merge, _pattern_boss_with_bore,
    _pattern_counterbore_hole_pattern, _pattern_add_slot_pattern, _pattern_knurl,
    _pattern_radial_array_oriented, _pattern_linear_array_scaled, _pattern_hex_socket,
    # 6 tools added across two missions this session (add_loft/add_sweep/add_tube, then
    # smooth_subdivide/add_wireframe_from_edges/weld_vertices) that were registered in
    # sandbox.py/ARG_SCHEMA with zero generic-composition training examples -- confirmed
    # by direct grep, not assumed. Without these, program_model.py has no text->tool
    # association to learn for any of the 6, regardless of how well sandbox.py itself
    # implements them.
    _pattern_loft, _pattern_sweep_arc, _pattern_sweep_helix,
    _pattern_tube_straight, _pattern_tube_arc,
    _pattern_smooth_subdivide, _pattern_wireframe, _pattern_weld,
]


_POSITIONAL_SUFFIXES = [
    ", centered on the top face of the box base",
    ", mounted on top",
    ", centered on the front face",
    ", positioned on top of the base",
    ", centered on top of the box",
]


def _with_positional_noise(text: str, rng: random.Random) -> str:
    """Appends a realistic positional/relational clause to `text` with some
    probability, WITHOUT changing the program -- teaching the classifier
    that this kind of clause is irrelevant to which tool gets chosen.
    Confirmed as a genuinely necessary fix, not a hypothetical one: a live
    multi-part decomposition (a treasure chest) had the LLM describe EVERY
    single part with a trailing clause like "...centered on the top face of
    the box base" (position_mm is a separate field, but STEP 3's guidance
    didn't forbid restating it in the description too), and every single
    one of those descriptions -- a dome, a sphere, a capsule, an ellipsoid,
    a cone, a ring, a hollowed cylinder -- got dragged into the SAME wrong
    add_box+add_box+boolean_difference prediction regardless of the actual
    requested shape. A robustness probe confirmed this is systemic: 7 of 8
    generic patterns tested broke under this exact suffix pattern, with no
    relation to how well-represented the base pattern itself was."""
    if rng.random() < 0.35:
        return text + rng.choice(_POSITIONAL_SUFFIXES)
    return text


def sample_generic_composition_dataset(n: int, seed: int) -> list[ProgramSample]:
    rng = random.Random(seed)
    out = []
    for i in range(n):
        pattern_fn = rng.choice(_GENERIC_PATTERNS)
        text, program = pattern_fn(rng)
        text = _with_positional_noise(text, rng)
        out.append(
            ProgramSample(
                id=f"generic_{i:06d}", text=text, program=tuple(program), source_kind="generic_composition",
                family=None, canonical_key=_program_canonical_key(text, program),
            )
        )
    return out


def sample_program_dataset(family_counts: dict[str, int], n_generic: int, seed: int) -> list[ProgramSample]:
    samples = sample_family_program_dataset(family_counts, seed=seed)
    samples += sample_generic_composition_dataset(n_generic, seed=seed + 1)
    return samples


# ---------------------------------------------------------------------------
# Storage: same manifest+jsonl+atomic-write spirit as data/prompt_store.py,
# split via the SAME leakage-safe canonical-key hashing as dataset_split.py
# (imported directly, not reimplemented) -- one manifest.json + train/val/test
# .jsonl, no per-sample directories (no binary payload here either).
# ---------------------------------------------------------------------------

from pathlib import Path

from data.dataset_split import Split, assign_split

_SPLITS: tuple[Split, ...] = ("train", "val", "test")


def _atomic_write(path: Path, content: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(content)
    tmp.replace(path)


def write_program_dataset(root: Path, samples: list[ProgramSample], seed: int) -> dict:
    root.mkdir(parents=True, exist_ok=True)
    by_split: dict[Split, list[ProgramSample]] = {s: [] for s in _SPLITS}
    for sample in samples:
        by_split[assign_split(sample.canonical_key, seed=seed)].append(sample)

    counts: dict[str, dict[str, int]] = {}
    for split in _SPLITS:
        rows = by_split[split]
        content = "\n".join(json.dumps(s.to_dict()) for s in rows)
        _atomic_write(root / f"{split}.jsonl", content + ("\n" if content else ""))
        split_counts: dict[str, int] = {}
        for s in rows:
            key = s.family if s.source_kind == "family" else "generic_composition"
            split_counts[key] = split_counts.get(key, 0) + 1
        counts[split] = split_counts

    manifest = {"version": "1.0.0", "seed": seed, "counts": counts}
    _atomic_write(root / "manifest.json", json.dumps(manifest, indent=2))
    return manifest


def read_program_split(root: Path, split: Split) -> list[ProgramSample]:
    path = root / f"{split}.jsonl"
    if not path.exists():
        return []
    return [ProgramSample.from_dict(json.loads(line)) for line in path.read_text().splitlines() if line.strip()]
