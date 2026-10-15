"""Tests for compositional_decomposer.py -- the general, clause-by-clause
alternative to program_model.py's whole-sentence prediction. Several cases
here are taken directly from the gearbox-mounting-assembly audit
(MISSION_STATUS.md Addendum 7) that motivated this module: real
descriptions where program_model.py predicted a totally unrelated
primitive (a solid sphere) because the WHOLE combined sentence had never
been seen as a trained example, even though each individual piece (a box,
a cylinder, a hole through it, a stepped section) is ordinary on its own."""

from compositional_decomposer import compose_program_from_description


def _ops(program):
    return [c.op for c in program]


def _args(program, op):
    return next(c.args for c in program if c.op == op)


def test_plain_box_with_no_feature_clause():
    program = compose_program_from_description("a box 140mm x 100mm x 10mm")
    assert program is not None
    assert _ops(program) == ["add_box", "finalize"]
    args = _args(program, "add_box")
    assert (args["length"], args["width"], args["height"]) == (140.0, 100.0, 10.0)


def test_plain_cylinder_with_no_feature_clause():
    program = compose_program_from_description("a cylinder 30mm diameter, 80mm tall")
    assert program is not None
    assert _ops(program) == ["add_cylinder", "finalize"]
    args = _args(program, "add_cylinder")
    assert args["radius"] == 15.0
    assert args["height"] == 80.0


def test_plain_sphere():
    program = compose_program_from_description("a sphere 40mm diameter")
    assert program is not None
    assert _args(program, "add_sphere")["radius"] == 20.0


def test_base_plate_corner_array_holes_matches_audited_real_prompt():
    """Real failure from the audit: program_model.py predicted
    drill_hole_pattern(count=3, radius=3.42mm, margin=16.49mm) for this
    exact text, against a requested 4 holes / 8mm diameter / 12mm margin
    -- the compositional path must read the literal numbers instead."""
    text = "a box 140mm x 100mm x 10mm with a corner array of four holes, each 8mm diameter and centered at 12mm from adjacent edges"
    program = compose_program_from_description(text)
    assert program is not None
    assert _ops(program) == ["add_box", "drill_hole_pattern", "finalize"]
    holes = _args(program, "drill_hole_pattern")
    assert holes["pattern"] == "corner_array"
    assert holes["count"] == 4
    assert holes["radius"] == 4.0  # 8mm diameter / 2
    assert holes["margin"] == 12.0
    # a through-hole: depth must exceed the plate's own thickness
    assert holes["depth"] > 10.0


def test_chamfer_box_edges_is_applied_before_any_hole_pattern():
    """round_box_edges/chamfer_box_edges require an unmodified, still-
    axis-aligned box target (Sandbox._is_axis_aligned_box) -- if a hole
    were drilled first, a later chamfer step would be correctly, but
    unhelpfully, rejected. Composition order must put edge-finishing
    first regardless of which order the text itself mentions clauses in."""
    text = "a box 80mm x 80mm x 6mm with a 1mm chamfer on the exposed outer edges"
    program = compose_program_from_description(text)
    assert program is not None
    assert _ops(program) == ["add_box", "chamfer_box_edges", "finalize"]
    assert _args(program, "chamfer_box_edges")["distance"] == 1.0


def test_round_box_edges():
    text = "a box 50mm x 50mm x 20mm with all edges rounded 3mm"
    program = compose_program_from_description(text)
    assert program is not None
    assert _ops(program) == ["add_box", "round_box_edges", "finalize"]
    assert _args(program, "round_box_edges")["radius"] == 3.0


def test_stepped_cylinder_matches_audited_shaft_prompt():
    """Real failure from the audit: program_model.py predicted a solid
    12mm-radius sphere for this exact text -- no length, no step at all.
    This must resolve to add_shouldered_cylinder with the literal
    diameters/lengths, as ONE primitive (there is no separate "step this
    cylinder down" tool in sandbox.py)."""
    text = "a cylinder 24mm diameter, 130mm long with a concentric stepped section of 18mm diameter and 20mm length at one end"
    program = compose_program_from_description(text)
    assert program is not None
    assert _ops(program) == ["add_shouldered_cylinder", "finalize"]
    args = _args(program, "add_shouldered_cylinder")
    assert args["bottom_radius"] == 12.0
    assert args["top_radius"] == 9.0
    assert args["top_height"] == 20.0
    assert args["bottom_height"] == 110.0  # 130 - 20


def test_bearing_support_gets_a_real_through_hole_not_a_sphere():
    """Real failure from the audit: program_model.py predicted a solid
    50mm sphere for this exact text -- not a box, and no opening of any
    kind. The compositional path must at least produce a real box with a
    real through-hole of the stated diameter (the exact off-face z=35mm
    placement is explicitly out of scope -- see _feature_through_hole's
    docstring -- but a real hole through the box's thinnest wall,
    centered on the box, is a measurable improvement over a solid sphere
    with zero holes)."""
    text = "a box 50mm x 8mm x 60mm with a circular through-opening of 24mm diameter centered on its face at z=35mm and aligned to the shaft axis"
    program = compose_program_from_description(text)
    assert program is not None
    assert _ops(program) == ["add_box", "drill_hole", "finalize"]
    hole = _args(program, "drill_hole")
    assert hole["radius"] == 12.0
    assert hole["axis"] == "y"  # the box's own shortest extent (8mm, its width)


def test_cylinder_hole_through_it_uses_the_cylinders_own_axis():
    """Confirmed in program_model.py's own _override_primary_shape_dimensions
    docstring as a real worked case this project already relies on --
    the hole must go along the cylinder's length (axis "z"), never across
    its diameter."""
    text = "a cylinder 20mm diameter, 15mm tall with a 6mm diameter hole through it"
    program = compose_program_from_description(text)
    assert program is not None
    assert _ops(program) == ["add_cylinder", "drill_hole", "finalize"]
    hole = _args(program, "drill_hole")
    assert hole["radius"] == 3.0
    assert hole["axis"] == "z"


def test_flange_bolt_circle_holes_on_a_cylinder_base():
    text = "a cylinder 50mm diameter, 8mm tall with a circular array of four holes, each 5mm diameter on a 36mm bolt-circle"
    program = compose_program_from_description(text)
    assert program is not None
    holes = _args(program, "drill_hole_pattern")
    assert holes["pattern"] == "bolt_circle"
    assert holes["count"] == 4
    assert holes["radius"] == 2.5
    assert holes["circle_diameter"] == 36.0


def test_unrecognized_base_shape_returns_none_not_a_guess():
    """"a triangular prism" has no matching base pattern -- and genuinely
    no triangular profile exists anywhere in sandbox.py's tool roster (see
    sandbox.py's ProfileKind: rect/rounded_rect/l_shape/ellipse only).
    This must return None (honest "I don't know"), never force a wrong
    primitive -- the caller falls back to program_model.py's existing
    statistical path, unchanged from before this module existed."""
    text = "a triangular prism with a horizontal footprint of 25mm and vertical height of 30mm"
    assert compose_program_from_description(text) is None


def test_unrecognized_feature_clause_returns_none_rather_than_dropping_it():
    """A recognized base shape with a feature clause that matches NO
    feature pattern must return None for the whole thing, not silently
    build just the base and drop the requested feature."""
    text = "a box 40mm x 40mm x 10mm with a spiral groove machined into the top face"
    assert compose_program_from_description(text) is None


def test_ring_without_inner_diameter_is_not_guessed():
    """The real flange text from the audit never states an inner
    diameter at all -- inventing one would be exactly the kind of
    unprincipled guess this module exists to avoid. No add_ring resolver
    is implemented, by design, so this (and any ring text) falls back to
    the statistical path rather than being silently approximated."""
    text = "a flat ring of outer diameter 50mm and thickness 8mm with a circular array of four holes, each 5mm diameter on a 36mm bolt-circle"
    assert compose_program_from_description(text) is None


def test_every_composed_program_is_a_valid_sandbox_program():
    """End-to-end sanity: a composed program must actually execute
    cleanly through the real Sandbox, not just look right structurally."""
    from sandbox import Sandbox

    texts = [
        "a box 140mm x 100mm x 10mm with a corner array of four holes, each 8mm diameter and centered at 12mm from adjacent edges",
        "a box 80mm x 80mm x 6mm with a 1mm chamfer on the exposed outer edges",
        "a cylinder 24mm diameter, 130mm long with a concentric stepped section of 18mm diameter and 20mm length at one end",
        "a box 50mm x 8mm x 60mm with a circular through-opening of 24mm diameter centered on its face at z=35mm and aligned to the shaft axis",
        "a cylinder 20mm diameter, 15mm tall with a 6mm diameter hole through it",
    ]
    for text in texts:
        program = compose_program_from_description(text)
        assert program is not None, text
        result = Sandbox().execute(program)
        assert result.status == "ok", f"{text!r} -> {result.status} ({result.error})"
        assert result.mesh is not None and result.mesh.is_watertight
