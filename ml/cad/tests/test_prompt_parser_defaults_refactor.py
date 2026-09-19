"""Golden test for the DEFAULTS-extraction refactor of nlp/prompt_parser.py.

prompt_parser.py's inline default literals (100.0, 4, "aluminum", the
bolt_circle_diameter_mm formula, ...) were replaced with calls to
part_families.resolve_default() so dataset generation and the regex parser
share one source of truth (see part_families.py's module docstring). This
test pins the exact ParseResult each of benchmark.py's real prompts produced
BEFORE that refactor (verified via `git diff` to touch only the literal
values, never the regex/keyword matching logic itself), so any future change
that accidentally alters parsing behavior fails here immediately.

Reuses benchmark.py's actual prompt list rather than a separate one, so this
test exercises exactly what the shipped benchmark already exercises.
"""

from dataclasses import asdict

from benchmark import _PROMPTS
from nlp.prompt_parser import parse_prompt

_EXPECTED = [
    (
        "plate",
        {
            "length_mm": 120.0, "width_mm": 80.0, "thickness_mm": 8.0,
            "hole_diameter_mm": 10.0, "hole_edge_margin_mm": 20.0, "fillet_radius_mm": 5.0,
            "hole_count": 4, "material": "aluminum",
        },
        {"fillet_radius_mm", "hole_diameter_mm", "hole_edge_margin_mm", "length_mm", "material", "thickness_mm", "width_mm"},
        {"hole_count"},
    ),
    (
        # "bracket" moved from _PLATE_KEYWORDS to its own family once a real
        # L-shaped bracket family existed -- see nlp/prompt_parser.py's module
        # docstring. This golden value changed deliberately (not a regression).
        "bracket",
        {
            "leg1_length_mm": 50.0, "leg2_length_mm": 50.0, "thickness_mm": 5.0, "length_mm": 100.0,
            "hole_diameter_mm": 6.0, "hole_edge_margin_mm": 15.0, "holes_per_arm": 2, "material": "steel",
        },
        set(),
        {"hole_diameter_mm", "hole_edge_margin_mm", "holes_per_arm", "leg1_length_mm", "leg2_length_mm", "length_mm", "material", "thickness_mm"},
    ),
    (
        "plate",
        {
            "length_mm": 150.0, "width_mm": 100.0, "thickness_mm": 10.0,
            "hole_diameter_mm": 12.0, "hole_edge_margin_mm": 15.0, "fillet_radius_mm": 5.0,
            "hole_count": 4, "material": "steel",
        },
        {"hole_diameter_mm", "length_mm", "material", "thickness_mm", "width_mm"},
        {"fillet_radius_mm", "hole_count", "hole_edge_margin_mm"},
    ),
    (
        "flange",
        {
            "outer_diameter_mm": 120.0, "bore_diameter_mm": 40.0, "thickness_mm": 8.0,
            "bolt_hole_count": 4, "bolt_hole_diameter_mm": 8, "bolt_circle_diameter_mm": 90,
            "material": "steel",
        },
        {"bolt_hole_count", "bolt_hole_diameter_mm", "bore_diameter_mm", "outer_diameter_mm", "thickness_mm"},
        {"bolt_circle_diameter_mm", "material"},
    ),
    (
        # "a small aluminum bracket" -- same deliberate reclassification as above
        "bracket",
        {
            "leg1_length_mm": 50.0, "leg2_length_mm": 50.0, "thickness_mm": 5.0, "length_mm": 100.0,
            "hole_diameter_mm": 6.0, "hole_edge_margin_mm": 15.0, "holes_per_arm": 2, "material": "aluminum",
        },
        {"material"},
        {"hole_diameter_mm", "hole_edge_margin_mm", "holes_per_arm", "leg1_length_mm", "leg2_length_mm", "length_mm", "thickness_mm"},
    ),
    (
        "flange",
        {
            "outer_diameter_mm": 100.0, "bore_diameter_mm": 25.0, "thickness_mm": 8.0,
            "bolt_hole_count": 4, "bolt_hole_diameter_mm": 8.0, "bolt_circle_diameter_mm": 70.0,
            "material": "steel",
        },
        {"bolt_circle_diameter_mm", "bore_diameter_mm", "outer_diameter_mm"},
        {"bolt_hole_count", "bolt_hole_diameter_mm", "material", "thickness_mm"},
    ),
]


def test_prompt_parser_matches_pre_refactor_golden_output():
    assert len(_PROMPTS) == len(_EXPECTED), "benchmark.py's prompt list changed -- update _EXPECTED to match"
    for prompt, (exp_part_type, exp_params, exp_parsed, exp_defaulted) in zip(_PROMPTS, _EXPECTED):
        result = parse_prompt(prompt)
        assert result.part_type == exp_part_type, prompt
        assert asdict(result.params) == exp_params, prompt
        assert result.parsed_fields == exp_parsed, prompt
        assert result.defaulted_fields == exp_defaulted, prompt
