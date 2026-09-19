import random

from data.procedural_generator import EnclosureParams, build_enclosure


def test_enclosure_is_watertight_across_seeds():
    from part_families import get_part_family

    spec = get_part_family("enclosure")
    for seed in range(20):
        params = spec.sampler(random.Random(seed))
        mesh = spec.builder(params)
        assert mesh.is_watertight, f"seed {seed} produced a non-watertight enclosure"


def test_hollow_shell_has_much_less_volume_than_solid_box():
    import trimesh

    params = EnclosureParams(length_mm=100, width_mm=80, height_mm=40, wall_thickness_mm=3, material="aluminum")
    shell = build_enclosure(params)
    solid_box = trimesh.creation.box(extents=[100, 80, 40])
    assert shell.volume < solid_box.volume * 0.5  # a hollow shell should be dramatically less material
    assert shell.is_watertight


def test_open_top_removes_more_material_than_closed():
    closed = build_enclosure(
        EnclosureParams(length_mm=100, width_mm=80, height_mm=40, wall_thickness_mm=3, open_face="none", material="aluminum")
    )
    open_top = build_enclosure(
        EnclosureParams(length_mm=100, width_mm=80, height_mm=40, wall_thickness_mm=3, open_face="top", material="aluminum")
    )
    assert open_top.volume < closed.volume
    assert open_top.is_watertight  # a real container with wall thickness is still a closed 2-manifold


def test_vent_holes_remove_material():
    no_vents = build_enclosure(
        EnclosureParams(length_mm=100, width_mm=80, height_mm=40, wall_thickness_mm=3, material="aluminum")
    )
    with_vents = build_enclosure(
        EnclosureParams(length_mm=100, width_mm=80, height_mm=40, wall_thickness_mm=3,
                        vent_hole_count=4, vent_hole_diameter_mm=4, material="aluminum")
    )
    assert with_vents.volume < no_vents.volume
    assert with_vents.is_watertight
