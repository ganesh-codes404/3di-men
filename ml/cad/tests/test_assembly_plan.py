"""assembly_plan.py -- deterministic relationship resolution for multi-part
assemblies. Pure-geometry tests (no LLM, no program_model.py classifier):
hand-built meshes via sandbox.py's own tools, checking resolve_position()'s
arithmetic directly against values computable by hand, plus the
validation/ordering logic's error paths."""
import pytest

from assembly_plan import Relationship, RelationshipError, order_by_dependency, resolve_position, resolve_rotation_deg, validate_relationships
from sandbox import Sandbox


def test_attached_to_face_top_lands_at_exact_half_height_sum():
    """The exact bug this module was built to fix: a part on top of
    another must sit at (target's own half-height) + (own half-height),
    never at the target's FULL height."""
    sb = Sandbox()
    plate = sb._get(sb.add_box(100, 60, 8, [0, 0, 0]))
    sb2 = Sandbox()
    boss = sb2._get(sb2.add_cylinder(10, 15, [0, 0, 0], "z"))

    rel = Relationship(kind="attached_to_face", target_id="plate", face="+z")
    pos = resolve_position(rel, plate, boss)
    assert pos == pytest.approx((0.0, 0.0, 4.0 + 7.5))


def test_attached_to_face_different_faces_give_different_positions():
    """The other confirmed bug: two features on different faces of the
    same target must NOT collapse to the same position."""
    sb = Sandbox()
    box = sb._get(sb.add_box(60, 40, 35, [0, 0, 0]))

    sb2 = Sandbox()
    dome = sb2._get(sb2.add_dome(10, 15, [0, 0, 0], "z"))
    sb3 = Sandbox()
    plate = sb3._get(sb3.add_polygon_frustum(6, 6, 6, 2, [0, 0, 0], "z"))

    top_pos = resolve_position(Relationship(kind="attached_to_face", target_id="box", face="+z"), box, dome)
    front_pos = resolve_position(Relationship(kind="attached_to_face", target_id="box", face="+y"), box, plate)
    assert top_pos != front_pos
    # box half-height + 0: add_dome's own local origin IS its flat base (see
    # resolve_position's own docstring) -- no extra offset beyond the box's
    # own half-height is needed for the base to land exactly on the face.
    # (Previously asserted 17.5 + 7.5 = 25.0, a confirmed real bug: that
    # value baked in a half-dome-height GAP between the box top and the
    # dome's actual base instead of flush contact -- caught by reproducing
    # this exact case through the real pipeline, position AND rotation
    # together, which this isolated assertion never did before.)
    assert top_pos[2] == pytest.approx(17.5)
    assert front_pos[1] > 0  # pushed out along +y, not left at the box's own center


def test_attached_to_face_offset_u_v_slides_within_the_face():
    """u/v are SOME valid orthonormal in-plane basis for the face (shared
    with sandbox.py's own face-based tools' _plane_basis convention) --
    not necessarily world X/Y -- so this checks the offset moves the
    EXPECTED in-plane DISTANCE and never touches the out-of-plane
    coordinate, rather than assuming a specific axis mapping."""
    sb = Sandbox()
    plate = sb._get(sb.add_box(100, 60, 8, [0, 0, 0]))
    sb2 = Sandbox()
    boss = sb2._get(sb2.add_cylinder(5, 10, [0, 0, 0], "z"))

    centered = resolve_position(Relationship(kind="attached_to_face", target_id="plate", face="+z"), plate, boss)
    offset = resolve_position(Relationship(kind="attached_to_face", target_id="plate", face="+z", offset_u=20, offset_v=-10), plate, boss)
    in_plane_distance = ((offset[0] - centered[0]) ** 2 + (offset[1] - centered[1]) ** 2) ** 0.5
    assert in_plane_distance == pytest.approx((20.0**2 + 10.0**2) ** 0.5)
    assert offset[2] == pytest.approx(centered[2])  # in-plane offsets never change the out-of-plane coordinate


def test_concentric_with_shares_the_other_two_axes_exactly():
    sb = Sandbox()
    boss = sb._get(sb.add_cylinder(10, 15, [3.0, -2.0, 11.5], "z"))
    sb2 = Sandbox()
    shaft = sb2._get(sb2.add_cylinder(3, 25, [0, 0, 0], "z"))

    pos = resolve_position(Relationship(kind="concentric_with", target_id="boss", axis="z"), boss, shaft)
    assert pos[0] == pytest.approx(3.0)
    assert pos[1] == pytest.approx(-2.0)
    assert pos[2] == pytest.approx(11.5)  # axis_offset defaults to 0 -- level with the target, not independent


def test_concentric_with_axis_offset_slides_along_the_shared_axis():
    sb = Sandbox()
    boss = sb._get(sb.add_cylinder(10, 15, [0, 0, 11.5], "z"))
    sb2 = Sandbox()
    shaft = sb2._get(sb2.add_cylinder(3, 25, [0, 0, 0], "z"))

    pos = resolve_position(Relationship(kind="concentric_with", target_id="boss", axis="z", axis_offset=5.0), boss, shaft)
    assert pos == pytest.approx((0.0, 0.0, 16.5))


def test_symmetric_about_mirrors_the_targets_position():
    sb = Sandbox()
    tab_a = sb._get(sb.add_box(10, 10, 10, [25.0, 4.0, 0.0]))
    sb2 = Sandbox()
    tab_b_proto = sb2._get(sb2.add_box(10, 10, 10, [0, 0, 0]))

    pos = resolve_position(Relationship(kind="symmetric_about", target_id="tab_a", axis="x"), tab_a, tab_b_proto)
    assert pos == pytest.approx((-25.0, 4.0, 0.0))


def test_resolve_rotation_deg_concentric_with_reorients_a_z_built_shaft():
    """Regression test for a confirmed real bug: resolve_position alone only
    computes WHERE a part's center lands -- every rotationally-symmetric
    sandbox.py primitive (cylinder, capsule, torus, dome, ring, tube) and
    every gear builds along its own local Z axis by default/always, so
    without a matching rotation, a shaft concentric_with a boss on the y
    axis ends up correctly CENTERED but still standing up along Z instead
    of running along y through the boss -- exactly "floats beside its
    housing," confirmed live with real meshes elsewhere. Applying this
    rotation (about the part's own local origin) before resolve_position's
    translation must make the part's own long axis actually point along
    the relationship's declared axis."""
    sb = Sandbox()
    boss = sb._get(sb.add_cylinder(10, 15, [0, 0, 0], "z"))
    sb2 = Sandbox()
    shaft = sb2._get(sb2.add_cylinder(5, 70, [0, 0, 0], "z"))
    assert shaft.extents == pytest.approx([10.0, 10.0, 70.0])

    from parallel_builder import PlacementTransform

    rel = Relationship(kind="concentric_with", target_id="boss", axis="y")
    pos = resolve_position(rel, boss, shaft)
    rot = resolve_rotation_deg(rel)
    placed = PlacementTransform.translation(pos).compose(PlacementTransform.rotation(rot)).apply_to_mesh(shaft)
    # the shaft's own 70mm length is now along y, not z
    assert placed.extents == pytest.approx([10.0, 70.0, 10.0])


def test_resolve_rotation_deg_concentric_with_z_is_a_no_op():
    """axis="z" (sandbox.py's own default build axis) needs no reorientation
    at all -- confirms the fix doesn't needlessly rotate the common case."""
    assert resolve_rotation_deg(Relationship(kind="concentric_with", target_id="t", axis="z")) == (0.0, 0.0, 0.0)


def test_resolve_rotation_deg_attached_to_face_points_outward_from_each_face():
    """A boss/standoff attached to a face must rise OUTWARD along that
    face's own normal, not stay pointing along its own default Z -- sign
    matters here (unlike concentric_with), since a boss pointing into its
    own target instead of away from it is a real, visible defect."""
    sb = Sandbox()
    housing = sb._get(sb.add_box(120, 80, 60, [0, 0, 0]))
    sb2 = Sandbox()
    boss = sb2._get(sb2.add_cylinder(10, 10, [0, 0, 0], "z"))
    assert boss.extents == pytest.approx([20.0, 20.0, 10.0])

    from parallel_builder import PlacementTransform

    for face, long_axis_idx in [("+y", 1), ("-y", 1), ("+x", 0), ("-x", 0), ("+z", 2), ("-z", 2)]:
        rel = Relationship(kind="attached_to_face", target_id="housing", face=face)
        pos = resolve_position(rel, housing, boss)
        rot = resolve_rotation_deg(rel)
        placed = PlacementTransform.translation(pos).compose(PlacementTransform.rotation(rot)).apply_to_mesh(boss)
        # the boss's own 10mm height (its rise direction) now lands along
        # whichever world axis that face's normal points along
        extents = list(placed.extents)
        assert extents[long_axis_idx] == pytest.approx(10.0), f"face {face}: {extents}"


def test_resolve_rotation_deg_symmetric_about_is_unchanged_for_now():
    """Scoped deliberately narrow, same as resolve_position's own three
    relationship kinds: symmetric_about's rotation is a separate, disclosed
    limitation, not silently guessed at here."""
    assert resolve_rotation_deg(Relationship(kind="symmetric_about", target_id="t", axis="x")) == (0.0, 0.0, 0.0)


def test_symmetric_about_raises_when_target_sits_on_the_mirror_plane():
    """Regression test for a confirmed real bug: 'a wheel symmetric_about
    frame' with the frame centered at the origin (this project's own
    default convention for a single main part) used to silently collapse
    the wheel onto the frame's own center -- "flush against the frame"
    instead of beside it. Mirroring something already on the mirror plane
    is meaningless, so this must fail loudly instead."""
    sb = Sandbox()
    frame = sb._get(sb.add_box(400, 30, 60, [0, 0, 0]))  # centered at the origin -- the degenerate case
    sb2 = Sandbox()
    wheel = sb2._get(sb2.add_cylinder(300, 20, [0, 0, 0], "x"))

    with pytest.raises(RelationshipError):
        resolve_position(Relationship(kind="symmetric_about", target_id="frame", axis="x"), frame, wheel)


def test_attached_to_face_places_a_triangular_gusset_flush_against_its_target():
    """Confirmed real bug, fixed at its actual source: a triangular gusset
    (sandbox.py's add_extrusion "triangle" profile) used to end up
    floating well off its target face instead of flush against it,
    because its own polygon's vertices started at a corner rather than
    being centered on their own bounding box -- resolve_position's
    attached_to_face math assumes a shape's local origin IS its own
    placement-relevant center (true for every other true primitive here).
    Fixed by centering the "triangle" profile itself
    (_right_triangle_polygon), not by special-casing this shared resolver
    -- confirming that fix end to end here, through the real,
    UNMODIFIED resolve_position.

    Applies resolve_rotation_deg too, not just resolve_position in
    isolation -- parallel_builder.py always applies both together for any
    attached_to_face relationship (resolve_rotation_deg rotates the
    gusset's own local Z, its 4mm thickness, to align with the face
    normal), and a SECOND confirmed real bug only shows up once rotation
    is actually applied: the old position formula used the gusset's
    in-plane "height" dimension (30mm) as the offset instead of its
    thickness (4mm), leaving a 13mm gap off the plate's face even though
    an isolated, no-rotation check of resolve_position alone couldn't see
    it (that was this test's own prior, misleading setup)."""
    sb = Sandbox()
    plate = sb._get(sb.add_box(60, 50, 6, [0, 0, 0]))
    sb2 = Sandbox()
    gusset = sb2._get(sb2.add_extrusion("triangle", param_a=30, param_b=30, param_c=0, height=4))
    # the triangle profile's own bbox center must be at its local origin,
    # matching every other true primitive (see _right_triangle_polygon)
    assert gusset.bounds.mean(axis=0) == pytest.approx([0.0, 0.0, 0.0])

    from parallel_builder import PlacementTransform

    rel = Relationship(kind="attached_to_face", target_id="plate", face="+y")
    pos = resolve_position(rel, plate, gusset)
    rot = resolve_rotation_deg(rel)
    placed = PlacementTransform.translation(pos).compose(PlacementTransform.rotation(rot)).apply_to_mesh(gusset)

    # flush against the plate's +y face (at y=25, since the plate is 50mm
    # wide, centered at the origin) -- not floating off it, and not
    # embedded past it either
    assert placed.bounds[0][1] == pytest.approx(25.0)
    assert placed.bounds[1][1] == pytest.approx(29.0)  # the gusset's own 4mm thickness, extending outward
    # centered on the plate's width (x=0), not shifted sideways
    assert placed.bounds.mean(axis=0)[0] == pytest.approx(0.0)


def test_attached_to_face_side_face_uses_the_parts_own_height_not_its_diameter():
    """Confirmed, reproduced real bug (not hypothetical): for any SIDE face
    (+-x/+-y), resolve_rotation_deg rotates a part so its own local Z axis
    (its height) ends up aligned with the face normal -- but the old
    resolve_position formula computed the normal-direction offset from the
    part's UNROTATED x/y/z extents dotted against the normal, which is
    only correct for +z (no rotation needed there). For an ANISOTROPIC
    part -- diameter very different from height, e.g. llm_decompose.py's
    own Example 5 boss (5mm radius, 30mm height) attached to a housing's
    +y face -- the old formula used the boss's DIAMETER (10mm) as the
    offset instead of its HEIGHT (30mm), embedding it 10mm into the
    housing instead of standing flush on its surface. This reproduces
    that exact case through BOTH resolve_position and resolve_rotation_deg
    together (the way parallel_builder.py actually uses them)."""
    sb = Sandbox()
    housing = sb._get(sb.add_box(120, 80, 60, [0, 0, 0]))  # +y face at y=40
    sb2 = Sandbox()
    boss = sb2._get(sb2.add_cylinder(5, 30, [0, 0, 0], "z"))  # 10mm diameter, 30mm height

    from parallel_builder import PlacementTransform

    rel = Relationship(kind="attached_to_face", target_id="housing", face="+y")
    pos = resolve_position(rel, housing, boss)
    rot = resolve_rotation_deg(rel)
    placed = PlacementTransform.translation(pos).compose(PlacementTransform.rotation(rot)).apply_to_mesh(boss)

    # the boss's own 30mm height now stands flush on the housing's +y face
    # (y=40) and rises outward to y=70 -- never embedded into the housing
    # (the old formula landed it at y=[30, 60], 10mm INSIDE the housing)
    assert placed.bounds[0][1] == pytest.approx(40.0)
    assert placed.bounds[1][1] == pytest.approx(70.0)


def test_attached_to_face_raises_when_offset_runs_off_the_targets_own_face():
    """Confirmed real failure mode this guards against: "a gusset at
    X=-45mm on a 60mm-wide plate" -- an offset large enough to push the
    attachment point past the target's own face boundary used to be
    accepted silently, placing the part off the edge of its target instead
    of onto it. Checked against the face's OWN actual extent, not a
    hardcoded number."""
    sb = Sandbox()
    plate = sb._get(sb.add_box(60, 50, 6, [0, 0, 0]))  # +y face spans roughly u in [-30, 30]
    sb2 = Sandbox()
    gusset = sb2._get(sb2.add_extrusion("triangle", param_a=30, param_b=30, param_c=0, height=4))

    with pytest.raises(RelationshipError):
        resolve_position(
            Relationship(kind="attached_to_face", target_id="plate", face="+y", offset_u=-45), plate, gusset
        )


def test_attached_to_face_accepts_an_offset_within_the_targets_own_face():
    """The other half of the same check: a legitimate offset (e.g.
    llm_decompose.py's own Example 6 left/right gusset pair, slid apart
    along a 60mm-wide plate) must still be accepted -- this isn't a
    blanket ban on any nonzero offset, only on one that actually runs off
    the target's own face.

    Uses offset_v, not offset_u, for this specific face -- a 60x50x6 box's
    +y face is only 6mm tall (its own Z extent) by 60mm wide (its own X
    extent), and _plane_basis's cross-product-derived (u, v) for a +y
    normal maps u to -Z (the 6mm dimension) and v to -X (the 60mm
    dimension) -- confirmed directly, not assumed. Which of offset_u/
    offset_v corresponds to which world direction depends on the face's
    own normal and isn't a fixed per-axis rule; see llm_decompose.py's
    Example 6 for the one worked case this project corrects for."""
    sb = Sandbox()
    plate = sb._get(sb.add_box(60, 50, 6, [0, 0, 0]))
    sb2 = Sandbox()
    gusset = sb2._get(sb2.add_extrusion("triangle", param_a=30, param_b=30, param_c=0, height=4))

    pos_left = resolve_position(
        Relationship(kind="attached_to_face", target_id="plate", face="+y", offset_v=-20), plate, gusset
    )
    pos_right = resolve_position(
        Relationship(kind="attached_to_face", target_id="plate", face="+y", offset_v=20), plate, gusset
    )
    assert pos_left[0] == pytest.approx(20.0)
    assert pos_right[0] == pytest.approx(-20.0)
    assert pos_left != pos_right  # Test J's own requirement: distinct instances, distinct transforms


def test_attached_to_face_raises_when_target_has_no_matching_face():
    # a sphere has no flat facet aligned with any direction at all -- unlike
    # a faceted cylinder, which (confirmed directly) DOES have a near-+x
    # triangulated side-wall quad close enough to pass find_facet_by_direction's
    # alignment threshold, so that isn't a genuine "no face" case.
    sb = Sandbox()
    sphere = sb._get(sb.add_sphere(10, [0, 0, 0]))
    sb2 = Sandbox()
    tiny = sb2._get(sb2.add_box(2, 2, 2, [0, 0, 0]))
    with pytest.raises(RelationshipError):
        resolve_position(Relationship(kind="attached_to_face", target_id="sphere", face="+x"), sphere, tiny)


# ---- validation / ordering (no geometry needed) ----

def test_validate_rejects_self_reference():
    with pytest.raises(RelationshipError):
        validate_relationships(["a"], {"a": Relationship(kind="concentric_with", target_id="a", axis="z")})


def test_validate_rejects_missing_target():
    with pytest.raises(RelationshipError):
        validate_relationships(["a", "b"], {"a": Relationship(kind="concentric_with", target_id="nope", axis="z")})


def test_validate_rejects_a_cycle():
    rels = {
        "a": Relationship(kind="concentric_with", target_id="b", axis="z"),
        "b": Relationship(kind="concentric_with", target_id="a", axis="z"),
    }
    with pytest.raises(RelationshipError):
        validate_relationships(["a", "b"], rels)


def test_validate_accepts_a_well_formed_chain():
    rels = {
        "plate": None,
        "boss": Relationship(kind="attached_to_face", target_id="plate", face="+z"),
        "shaft": Relationship(kind="concentric_with", target_id="boss", axis="z"),
    }
    validate_relationships(["plate", "boss", "shaft"], rels)  # must not raise


def test_order_by_dependency_produces_correct_waves():
    rels = {
        "plate": None,
        "boss": Relationship(kind="attached_to_face", target_id="plate", face="+z"),
        "shaft": Relationship(kind="concentric_with", target_id="boss", axis="z"),
        "tab": None,  # independent, same wave as plate
    }
    waves = order_by_dependency(["plate", "boss", "shaft", "tab"], rels)
    assert waves == [["plate", "tab"], ["boss"], ["shaft"]]


def test_order_by_dependency_handles_no_relationships_as_one_wave():
    waves = order_by_dependency(["a", "b", "c"], {"a": None, "b": None, "c": None})
    assert waves == [["a", "b", "c"]]
