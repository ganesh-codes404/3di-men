"""assembly_plan.py -- deterministic relationship resolution for multi-part
assemblies. Pure-geometry tests (no LLM, no program_model.py classifier):
hand-built meshes via sandbox.py's own tools, checking resolve_position()'s
arithmetic directly against values computable by hand, plus the
validation/ordering logic's error paths."""
import pytest

from assembly_plan import Relationship, RelationshipError, order_by_dependency, resolve_position, validate_relationships
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
    assert top_pos[2] == pytest.approx(17.5 + 7.5)  # box half-height + dome's own half-extent (flat-base convention)
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
