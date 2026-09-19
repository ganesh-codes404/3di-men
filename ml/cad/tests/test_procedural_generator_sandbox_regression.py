"""Verifies the parallel Sandbox-based program builders (data/program_builders.py)
produce geometry EQUIVALENT to the primary, deeply-tested build_*() functions
in data/procedural_generator.py -- equivalent (watertight, comparable volume
and bounds), not byte-identical, since the Sandbox tools compute their own
CSG-robustness padding and mesh-quality heuristics internally rather than
replicating each builder's specific historical fudge factors. See
program_builders.py's module docstring for why these are two independent
implementations rather than one function with a sandbox branch."""

import random

import pytest

from data.procedural_generator import build_bracket, build_enclosure, build_flange, build_gear, build_mounting_plate, build_shaft
from data.program_builders import traced_build
from part_families import get_part_family

_ORIGINAL_BUILDERS = {
    "plate": build_mounting_plate,
    "flange": build_flange,
    "bracket": build_bracket,
    "shaft": build_shaft,
    "enclosure": build_enclosure,
    "gear": build_gear,
}

_FAMILIES = ["plate", "flange", "bracket", "shaft", "enclosure", "gear"]


@pytest.mark.parametrize("family", _FAMILIES)
def test_program_builder_matches_original_across_sampled_params(family):
    spec = get_part_family(family)
    original_builder = _ORIGINAL_BUILDERS[family]
    mismatches = []

    for seed in range(30):
        params = spec.sampler(random.Random(seed))
        original_mesh = original_builder(params)
        traced_mesh, program = traced_build(family, params)

        if not traced_mesh.is_watertight:
            mismatches.append(f"seed {seed}: traced mesh not watertight")
            continue
        if not original_mesh.is_watertight:
            continue  # a rare original-side edge case isn't this test's concern

        volume_ratio = traced_mesh.volume / original_mesh.volume if original_mesh.volume > 0 else None
        if volume_ratio is None or not (0.85 <= volume_ratio <= 1.15):
            mismatches.append(f"seed {seed}: volume ratio {volume_ratio} out of tolerance "
                               f"(original={original_mesh.volume:.1f}, traced={traced_mesh.volume:.1f})")

        orig_extents = original_mesh.extents
        traced_extents = traced_mesh.extents
        for axis_name, o, t in zip("xyz", orig_extents, traced_extents):
            if abs(o - t) > max(1.0, o * 0.1):
                mismatches.append(f"seed {seed}: {axis_name}-extent differs (original={o:.1f}, traced={t:.1f})")

    assert not mismatches, f"{family}: {len(mismatches)}/30 mismatches:\n" + "\n".join(mismatches[:10])


@pytest.mark.parametrize("family", _FAMILIES)
def test_traced_program_replays_to_the_same_mesh_via_sandbox_execute(family):
    """The trace itself must be faithful: replaying it through Sandbox.execute()
    (a fresh sandbox, no shortcuts) must reproduce the same mesh traced_build()
    returned directly."""
    from sandbox import Sandbox

    spec = get_part_family(family)
    params = spec.sampler(random.Random(0))
    direct_mesh, program = traced_build(family, params)

    result = Sandbox.execute(program)
    assert result.status == "ok", f"{family}: replay failed with status={result.status}, error={result.error}"
    assert abs(result.mesh.volume - direct_mesh.volume) < 1e-6
