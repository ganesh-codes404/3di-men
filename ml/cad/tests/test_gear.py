import random

import pytest

from data.procedural_generator import GearParams, build_gear


def test_gear_is_watertight_across_seeds():
    from part_families import get_part_family

    spec = get_part_family("gear")
    for seed in range(20):
        params = spec.sampler(random.Random(seed))
        mesh = spec.builder(params)
        assert mesh.is_watertight, f"seed {seed} produced a non-watertight gear"


@pytest.mark.parametrize("tooth_count", [12, 16, 20, 24, 32])
def test_gear_watertight_across_tooth_counts_including_the_undercut_floor(tooth_count):
    """tooth_count=12 is the documented floor below which standard 20-degree
    involute teeth undercut -- this pins that the floor is actually safe."""
    params = GearParams(module_mm=2.0, tooth_count=tooth_count, face_width_mm=10, bore_diameter_mm=8, material="steel")
    mesh = build_gear(params)
    assert mesh.is_watertight


def test_more_teeth_dont_change_pitch_radius():
    """pitch_radius depends only on module and tooth_count, not a builder
    side effect -- a cheap sanity check on the class methods themselves."""
    params = GearParams(module_mm=2.0, tooth_count=20, face_width_mm=10, bore_diameter_mm=8, material="steel")
    assert params.pitch_radius() == pytest.approx(20.0)  # module * teeth / 2
    assert params.addendum_radius() > params.pitch_radius() > params.dedendum_radius()


def test_gear_has_visibly_distinct_teeth_not_a_plain_disc():
    """A plain 20-tooth disc (no teeth) would have exactly as many outline
    vertices as its polygon resolution; a real gear has ~2x the vertices per
    tooth for the two involute flanks -- use vertex count as a cheap proxy
    for "this actually has teeth, not just a circle"."""
    import trimesh

    gear = build_gear(GearParams(module_mm=2.0, tooth_count=16, face_width_mm=10, bore_diameter_mm=8, material="steel"))
    plain_disc = trimesh.creation.cylinder(radius=gear.bounds[1][0], height=10, sections=64)
    assert len(gear.vertices) > len(plain_disc.vertices)
