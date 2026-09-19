from dataclasses import asdict

from assembly import (
    PlacementTransform,
    build_assembly,
    decompose_prompt,
    plan_assembly,
    run_assembly,
)
from nlp.prompt_parser import parse_prompt


def _regex_adapter(text: str) -> tuple[str, dict]:
    result = parse_prompt(text)
    return result.part_type, asdict(result.params)


def test_decompose_splits_base_and_satellite_from_one_with_clause():
    """The load-bearing case: "X with N Y" must become TWO groups (the base
    X, and N of satellite Y) -- not one group whose rightmost keyword
    swallows the base mention entirely."""
    groups = decompose_prompt("a base plate with four mounting brackets")
    assert len(groups) == 2
    assert groups[0]["count"] == 1
    assert groups[1]["count"] == 4
    assert "plate" in groups[0]["clauses"][0]
    assert "bracket" in groups[1]["clauses"][0]


def test_decompose_does_not_split_a_holes_clause_into_a_fake_second_component():
    """Regression test for a real bug: "mount" (a deliberately loose plate
    keyword, has to match "mounting"/"mounted" too) matches inside "four 10
    mm mounting holes" just as readily as inside "four mounting brackets"
    above -- but a hole is a FEATURE, never a standalone part family, so
    this must stay ONE component, not split into a plate plus a mystery
    second "component" for its own holes. This is the exact canonical
    single-part example from the frontend's own example prompts."""
    groups = decompose_prompt(
        "rectangular aluminum mounting plate, 120 mm x 80 mm x 8 mm, with four 10 mm mounting holes, "
        "20 mm from each edge, 5 mm corner fillets"
    )
    assert len(groups) == 1


def test_decompose_strips_leading_and_after_comma():
    groups = decompose_prompt("a shaft, and a top cover")
    assert len(groups) == 2
    assert not groups[1]["clauses"][0].lower().startswith("and ")


def test_decompose_unanchored_segment_attaches_to_previous_component():
    """Mirrors today's single-part behavior: per-part detail with no family
    keyword of its own must stay attached to the component it describes,
    not become a stray extra component."""
    groups = decompose_prompt("circular flange, 100mm outer diameter, with a 25mm bore")
    assert len(groups) == 1
    assert len(groups[0]["clauses"]) == 3


def test_plan_assembly_produces_correct_component_specs():
    spec = plan_assembly("a base plate with four mounting brackets, a shaft in the center", _regex_adapter)
    assert [c.family for c in spec.components] == ["plate", "bracket", "shaft"]
    assert [c.count for c in spec.components] == [1, 4, 1]


def test_count_expansion_builds_mesh_once_not_once_per_instance(monkeypatch):
    """CSG boolean ops are the expensive part of these builders -- N
    identical brackets must reuse one built local_mesh, not rebuild it N
    times."""
    import dataclasses

    import part_families

    spec = plan_assembly("four mounting brackets", _regex_adapter)
    original_spec = part_families.get_part_family("bracket")
    call_count = 0

    def counting_builder(params):
        nonlocal call_count
        call_count += 1
        return original_spec.builder(params)

    # PartFamilySpec is frozen -- patch the registry entry itself, not an attribute on it
    patched_spec = dataclasses.replace(original_spec, builder=counting_builder)
    monkeypatch.setitem(part_families._REGISTRY, "bracket", patched_spec)

    result = build_assembly(spec)
    assert call_count == 1
    assert len(result.placed) == 4  # 4 placed instances from that one built mesh


def test_ground_correction_places_every_component_touching_its_target_z():
    spec = plan_assembly("a base plate with a shaft in the center", _regex_adapter)
    result = build_assembly(spec)
    plate = next(p for p in result.placed if p.spec.family == "plate")
    shaft = next(p for p in result.placed if p.spec.family == "shaft")
    # both should sit with their bottom face at world Z=0 (target_z=0 for both
    # the base and ground-level satellites in this v1 placement scheme)
    assert abs(plate.world_mesh.bounds[0][2] - 0.0) < 1e-6
    assert abs(shaft.world_mesh.bounds[0][2] - 0.0) < 1e-6


def test_placement_transform_translation_matches_matrix_convention():
    t = PlacementTransform.translation([1.0, 2.0, 3.0])
    assert list(t.matrix[:3, 3]) == [1.0, 2.0, 3.0]
    identity = PlacementTransform.identity()
    composed = t.compose(identity)
    assert list(composed.matrix[:3, 3]) == [1.0, 2.0, 3.0]


def test_run_assembly_end_to_end_all_components_watertight():
    prompt = "a base plate with four mounting brackets, a shaft in the center, and a top cover with ventilation holes"
    result = run_assembly(prompt, _regex_adapter)

    assert len(result.placed) == 7  # 1 plate + 4 brackets + 1 shaft + 1 enclosure
    assert result.mesh is not None
    for key, report in result.geometry_reports.items():
        assert report.watertight, f"{key} is not watertight"
    # this specific prompt's default part sizes are expected to overlap (a
    # small default plate hosting 4 corner-positioned brackets) -- this is
    # the diagnostic doing its job, not asserted to be empty
    assert isinstance(result.overlapping_pairs, tuple)


def test_single_part_prompt_still_produces_exactly_one_component():
    """Assembly must not change behavior for the (still far more common)
    single-part case."""
    result = run_assembly("a steel flange, 100mm outer diameter", _regex_adapter)
    assert len(result.placed) == 1
    assert result.placed[0].spec.family == "flange"
