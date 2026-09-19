import random

from data.procedural_generator import BracketParams, build_bracket


def test_bracket_is_watertight_across_seeds():
    from part_families import get_part_family

    spec = get_part_family("bracket")
    for seed in range(20):
        params = spec.sampler(random.Random(seed))
        mesh = spec.builder(params)
        assert mesh.is_watertight, f"seed {seed} produced a non-watertight bracket"


def test_holes_land_within_each_arms_material():
    """The real risk this guards: arm 2's holes must be rotated (Z-axis ->
    X-axis) and positioned correctly, or they'd be drilled in empty space
    (no-op) or through the wrong arm entirely."""
    params = BracketParams(
        leg1_length_mm=60, leg2_length_mm=60, thickness_mm=5, length_mm=100,
        hole_diameter_mm=6, hole_edge_margin_mm=15, holes_per_arm=2, material="steel",
    )
    without_holes = BracketParams(**{**params.__dict__, "hole_diameter_mm": 0.01})
    mesh_with = build_bracket(params)
    mesh_without = build_bracket(without_holes)
    # drilling real holes must remove material -- if a hole missed the solid
    # entirely (wrong position/rotation), volume would be unchanged. Expected
    # removed volume for 4x 6mm-diameter holes through 5mm material:
    # 4 * pi*(3)^2*5 ~= 565mm^3 -- assert it's in that ballpark, not just "any" decrease.
    removed = mesh_without.volume - mesh_with.volume
    assert 400 < removed < 750, f"removed {removed:.1f}mm^3, expected ~565mm^3"


def test_bracket_bounds_match_declared_dimensions():
    params = BracketParams(
        leg1_length_mm=60, leg2_length_mm=40, thickness_mm=5, length_mm=100,
        hole_diameter_mm=6, hole_edge_margin_mm=15, holes_per_arm=2, material="steel",
    )
    mesh = build_bracket(params)
    extents = mesh.extents
    assert abs(extents[0] - 60) < 1e-6  # X spans leg1
    assert abs(extents[1] - 40) < 1e-6  # Y spans leg2
    assert abs(extents[2] - 100) < 1e-6  # Z spans length (centered)
