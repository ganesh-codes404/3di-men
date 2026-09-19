import random
from dataclasses import fields

import pytest

from part_families import get_part_family, list_part_families, resolve_default, schema_dict


_ALL_FAMILIES = ["bracket", "enclosure", "flange", "gear", "plate", "shaft"]


def test_known_families():
    assert list_part_families() == _ALL_FAMILIES


def test_unknown_family_lists_known_ones():
    with pytest.raises(ValueError) as exc_info:
        get_part_family("widget")  # not a real family -- "gear" would be now that it's registered
    msg = str(exc_info.value)
    assert "flange" in msg
    assert "plate" in msg


@pytest.mark.parametrize("family", _ALL_FAMILIES)
def test_sampler_and_builder_round_trip(family):
    spec = get_part_family(family)
    rng = random.Random(0)
    params = spec.sampler(rng)
    mesh = spec.builder(params)
    assert len(mesh.vertices) > 0
    assert len(mesh.faces) > 0
    assert mesh.is_watertight


@pytest.mark.parametrize("family", _ALL_FAMILIES)
def test_field_specs_match_dataclass_fields(family):
    """Drift guard: if MountingPlateParams/FlangeParams gains a field and the
    registry isn't updated, this must fail loudly instead of silently
    producing an incomplete schema."""
    spec = get_part_family(family)
    dataclass_field_names = {f.name for f in fields(spec.params_cls)}
    registered_field_names = {f.name for f in spec.fields}
    assert dataclass_field_names == registered_field_names


def test_resolve_default_flat_value():
    assert resolve_default("plate", "length_mm", {}) == 100.0
    assert resolve_default("plate", "material", {}) == "aluminum"


def test_resolve_default_derived_value():
    # matches nlp/prompt_parser.py's inline formula: round(outer_d * 0.75/2) * 2
    assert resolve_default("flange", "bolt_circle_diameter_mm", {"outer_diameter_mm": 120.0}) == 90


def test_schema_dict_is_json_safe_and_marks_derived_defaults():
    schema = schema_dict()
    assert set(schema.keys()) == set(_ALL_FAMILIES)
    flange_fields = {f["name"]: f for f in schema["flange"]}
    assert flange_fields["bolt_circle_diameter_mm"]["default"] == "<derived from other fields>"
    assert flange_fields["outer_diameter_mm"]["default"] == 120.0


def test_material_choices_match_widened_samplers():
    # both original samplers were widened to cover every material
    # nlp/prompt_parser.py's regex recognizes -- this pins that invariant
    plate = get_part_family("plate").field("material")
    assert set(plate.choices) == {"aluminum", "steel", "polycarbonate", "titanium", "plastic", "brass"}
    flange = get_part_family("flange").field("material")
    assert set(flange.choices) == {"steel", "aluminum", "cast iron", "titanium", "brass"}


@pytest.mark.parametrize("family", _ALL_FAMILIES)
def test_every_family_has_nonempty_material_choices(family):
    material = get_part_family(family).field("material")
    assert material.choices and len(material.choices) >= 1
