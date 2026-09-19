import random

from data.procedural_generator import ShaftParams, build_shaft


def test_shaft_is_watertight_across_seeds():
    from part_families import get_part_family

    spec = get_part_family("shaft")
    for seed in range(20):
        params = spec.sampler(random.Random(seed))
        mesh = spec.builder(params)
        assert mesh.is_watertight, f"seed {seed} produced a non-watertight shaft"


def test_hollow_shaft_has_less_volume_than_solid():
    solid = build_shaft(ShaftParams(outer_diameter_mm=20, length_mm=100, material="steel"))
    hollow = build_shaft(ShaftParams(outer_diameter_mm=20, length_mm=100, inner_diameter_mm=8, material="steel"))
    assert hollow.volume < solid.volume
    assert hollow.is_watertight


def test_keyway_removes_material_and_stays_watertight():
    plain = build_shaft(ShaftParams(outer_diameter_mm=20, length_mm=100, material="steel"))
    keyed = build_shaft(
        ShaftParams(outer_diameter_mm=20, length_mm=100, keyway_width_mm=4, keyway_depth_mm=2, material="steel")
    )
    assert keyed.volume < plain.volume
    assert keyed.is_watertight
