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


def _pattern_holes(rng: random.Random) -> tuple[str, list[ToolCall]]:
    shape = rng.choice(["box", "cylinder"])
    count = rng.choice([2, 3, 4])
    hole_r = rng.choice([2, 3, 4])
    if shape == "box":
        l, w, h = rng.choice([40, 60, 80]), rng.choice([30, 40, 60]), rng.choice([6, 8, 10])
        program = [ToolCall("add_box", {"length": l, "width": w, "height": h, "center": [0, 0, 0]})]
        shape_desc = f"a box {l:g}mm x {w:g}mm x {h:g}mm"
        depth = h
    else:
        r, h = rng.choice([20, 30, 40]), rng.choice([6, 8, 10])
        program = [ToolCall("add_cylinder", {"radius": r, "height": h, "center": [0, 0, 0], "axis": "z"})]
        shape_desc = f"a cylinder {2*r:g}mm diameter x {h:g}mm tall"
        depth = h
    program.append(
        ToolCall("drill_hole_pattern", {"target": 0, "pattern": "corner_array" if shape == "box" else "bolt_circle",
                                         "count": count, "radius": hole_r, "depth": depth, "axis": "z",
                                         "margin": 8.0, "circle_diameter": 0.0, "z": 0.0})
    )
    program.append(ToolCall("finalize", {"target": 1}))
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


_GENERIC_PATTERNS = [
    _pattern_union, _pattern_cavity, _pattern_ring, _pattern_holes,
    _pattern_intersection, _pattern_mirror, _pattern_scaled, _pattern_linear_array, _pattern_circular_array,
    _pattern_boss, _pattern_pocket, _pattern_shell, _pattern_fillet, _pattern_chamfer,
    _pattern_frustum, _pattern_cone, _pattern_tapered_box,
    _pattern_counterbore, _pattern_drill_slot, _pattern_rib,
    _pattern_shell_cylinder, _pattern_hollow_vessel, _pattern_shouldered_cylinder,
]


def sample_generic_composition_dataset(n: int, seed: int) -> list[ProgramSample]:
    rng = random.Random(seed)
    out = []
    for i in range(n):
        pattern_fn = rng.choice(_GENERIC_PATTERNS)
        text, program = pattern_fn(rng)
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
