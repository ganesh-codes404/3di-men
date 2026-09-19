from data.prompt_dataset import (
    WordingConfig,
    _resolve_with_omissions,
    canonical_key,
    render_text,
    sample_prompt_dataset,
)
from part_families import resolve_default


def test_omitted_field_resolves_to_canonical_default_not_random_value():
    """Load-bearing test: this is the exact bug that would otherwise make
    training silently impossible (see module docstring)."""
    raw = {
        "length_mm": 137.0, "width_mm": 91.0, "thickness_mm": 7.0,
        "hole_diameter_mm": 9.0, "hole_edge_margin_mm": 17.0, "fillet_radius_mm": 2.0,
        "hole_count": 2, "material": "brass",
    }
    resolved = _resolve_with_omissions("plate", raw, omit_fields=frozenset({"length_mm", "material"}))
    assert resolved["length_mm"] == resolve_default("plate", "length_mm", {})
    assert resolved["material"] == resolve_default("plate", "material", {})
    # non-omitted fields keep the sampler's raw value untouched
    assert resolved["width_mm"] == 91.0
    assert resolved["hole_count"] == 2


def test_derived_default_uses_already_resolved_fields():
    raw = {
        "outer_diameter_mm": 200.0, "bore_diameter_mm": 40.0, "thickness_mm": 8.0,
        "bolt_hole_diameter_mm": 8.0, "bolt_hole_count": 4, "bolt_circle_diameter_mm": 999.0,
        "material": "steel",
    }
    resolved = _resolve_with_omissions("flange", raw, omit_fields=frozenset({"bolt_circle_diameter_mm"}))
    # 200 * 0.75 / 2 = 75, round(75)*2 = 150 -- derived from THIS sample's outer_diameter_mm, not a flat constant
    assert resolved["bolt_circle_diameter_mm"] == 150
    assert resolved["outer_diameter_mm"] == 200.0  # kept field unaffected


def test_two_wordings_of_same_params_share_canonical_key_but_differ_in_text():
    resolved = {
        "length_mm": 100.0, "width_mm": 60.0, "thickness_mm": 6.0,
        "hole_diameter_mm": 8.0, "hole_edge_margin_mm": 15.0, "fillet_radius_mm": 5.0,
        "hole_count": 4, "material": "aluminum",
    }
    cfg_a = WordingConfig(unit_style="mm_tight", clause_order_seed=1, omit_fields=frozenset(), material_synonym=None)
    cfg_b = WordingConfig(unit_style="word_millimeters", clause_order_seed=2, omit_fields=frozenset(), material_synonym=None)
    text_a = render_text("plate", resolved, frozenset(), cfg_a)
    text_b = render_text("plate", resolved, frozenset(), cfg_b)
    assert text_a != text_b
    assert canonical_key("plate", resolved) == canonical_key("plate", dict(resolved))  # pure function of params


def test_a_kept_fields_literal_value_appears_in_rendered_text():
    resolved = {
        "outer_diameter_mm": 137.0, "bore_diameter_mm": 40.0, "thickness_mm": 8.0,
        "bolt_hole_diameter_mm": 8.0, "bolt_hole_count": 4, "bolt_circle_diameter_mm": 100.0,
        "material": "steel",
    }
    cfg = WordingConfig(unit_style="mm_spaced", clause_order_seed=0, omit_fields=frozenset(), material_synonym=None)
    text = render_text("flange", resolved, frozenset(), cfg)
    assert "137" in text


def test_sample_prompt_dataset_produces_requested_counts_and_valid_fields():
    samples = sample_prompt_dataset({"plate": 10, "flange": 6}, seed=0)
    plate_samples = [s for s in samples if s.part_family == "plate"]
    flange_samples = [s for s in samples if s.part_family == "flange"]
    assert len(plate_samples) == 10
    assert len(flange_samples) == 6
    for s in samples:
        assert set(s.parsed_fields) & set(s.defaulted_fields) == set()
        assert set(s.parsed_fields) | set(s.defaulted_fields) == set(s.params)
        assert len(s.text) > 0
