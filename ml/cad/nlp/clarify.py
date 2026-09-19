"""Turns a prompt into a plan of components PLUS a list of clarifying
questions for whatever the rule-based parser couldn't confidently fill in --
the frontend's "ask before building" step (see server.py's /api/plan and
/api/generate_assembly). No LLM anywhere: every question is generated from
data the existing rule-based system already computes (part_families.py's
FieldSpec registry, and parse_prompt()'s own parsed/defaulted field sets) --
this module adds no new extraction logic of its own beyond the one
deliberate family-ambiguity case below.

Reuses, rather than duplicates: assembly.decompose_prompt() for splitting
the prompt into per-component clause text (already fixed this session for
verbose multi-paragraph prompts), nlp.prompt_parser.parse_prompt()/_PARSERS
for per-family field extraction, and part_families.py's FieldSpec registry
for question metadata (kind/unit/choices). assembly.ComponentSpec/
build_assembly()/validate_assembly() are untouched -- this module only
produces the plan; server.py turns an approved plan into those.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from assembly import decompose_prompt
from nlp.prompt_parser import _FLANGE_KEYWORDS, _PARSERS, parse_prompt
from part_families import get_part_family

# A component whose ONLY flange-keyword match is "housing" (none of the
# other, unambiguous flange words) is genuinely ambiguous between a round
# flange-style housing and a rectangular enclosure -- nlp/prompt_parser.py's
# own module docstring documents this as a deliberate, accepted gap in the
# silent-default path. Here, instead of silently guessing, it becomes a
# question.
_UNAMBIGUOUS_FLANGE_WORDS = [kw for kw in _FLANGE_KEYWORDS if kw != "housing"]


@dataclass
class ClarifyingQuestion:
    component_id: str
    field: str  # a params field name, or "__family__" for the family-choice case
    kind: str  # "numeric" | "categorical" | "count" | "family_choice"
    label: str
    unit: str | None
    default: Any
    choices: list[str] | None = None


@dataclass
class PlannedComponent:
    id: str
    family: str
    family_options: list[str] | None  # only set for the ambiguous-housing case
    clause_text: str
    count: int
    params: dict[str, Any]  # fully resolved -- parsed values merged with defaults
    parsed_fields: list[str]
    defaulted_fields: list[str]


@dataclass
class ClarifyPlan:
    components: list[PlannedComponent] = field(default_factory=list)
    questions: list[ClarifyingQuestion] = field(default_factory=list)

    @property
    def needs_clarification(self) -> bool:
        return len(self.questions) > 0


def _field_label(name: str) -> str:
    """"leg1_length_mm" -> "Leg 1 length" -- the unit is shown separately
    (FieldSpec.unit), so strip the trailing unit-shaped suffix rather than
    showing it twice."""
    for suffix in ("_mm", "_deg"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break
    return " ".join(name.split("_")).capitalize()


def _is_ambiguous_housing(family: str, clause_text: str) -> bool:
    if family != "flange":
        return False
    lowered = clause_text.lower()
    if "housing" not in lowered:
        return False
    return not any(kw in lowered for kw in _UNAMBIGUOUS_FLANGE_WORDS)


def _questions_for_defaulted_fields(
    component_id: str, family: str, defaulted_fields: set[str], params: dict[str, Any],
) -> list[ClarifyingQuestion]:
    """`params` is this component's ALREADY fully-resolved param dict (from
    the real parser, parsed-or-defaulted for every field) -- used as the
    default value directly rather than calling resolve_default() again with
    a fresh, empty resolved_so_far dict. That matters for a field like
    flange's bolt_circle_diameter_mm, whose default is a FUNCTION of
    outer_diameter_mm: calling resolve_default(family, name, {}) here would
    KeyError whenever outer_diameter_mm isn't already in that empty dict --
    which is not a hypothetical, it's the common case whenever bolt-circle
    ends up in defaulted_fields at all. `params` already has every
    dependency resolved in the correct order (the parser did that), so
    reading the value straight from it is both simpler and correct."""
    spec = get_part_family(family)
    questions = []
    for name in sorted(defaulted_fields):
        f = spec.field(name)
        questions.append(
            ClarifyingQuestion(
                component_id=component_id, field=name, kind=f.kind, label=_field_label(name),
                unit=f.unit, default=params[name],
                choices=list(f.choices) if f.choices else None,
            )
        )
    return questions


def build_plan(prompt: str, family_overrides: dict[str, str] | None = None) -> ClarifyPlan:
    """Component ids are positional ("component_0", "component_1", ...),
    not family-prefixed -- decompose_prompt() is deterministic for a given
    prompt, so ids stay stable across a second build_plan() call with
    family_overrides filled in after the first call's family_choice
    question is answered, even though the override changes which family
    that component resolves to."""
    family_overrides = family_overrides or {}
    groups = decompose_prompt(prompt)
    plan = ClarifyPlan()

    for i, group in enumerate(groups):
        component_id = f"component_{i}"
        clause_text = ", ".join(group["clauses"])
        override = family_overrides.get(component_id)

        if override is not None:
            family = override
            params_obj, parsed_fields, defaulted_fields = _PARSERS[family](clause_text)
            part_type_inferred = False
        else:
            result = parse_prompt(clause_text)
            family = result.part_type
            params_obj, parsed_fields, defaulted_fields = result.params, result.parsed_fields, result.defaulted_fields
            part_type_inferred = result.part_type_inferred

        params_dict = {f.name: getattr(params_obj, f.name) for f in get_part_family(family).fields}

        component = PlannedComponent(
            id=component_id, family=family,
            family_options=["flange", "enclosure"] if _is_ambiguous_housing(family, clause_text) else None,
            clause_text=clause_text, count=group["count"], params=params_dict,
            parsed_fields=sorted(parsed_fields), defaulted_fields=sorted(defaulted_fields),
        )
        plan.components.append(component)

        if component.family_options is not None and override is None:
            # The family question takes priority over this component's
            # field questions -- answering it (a follow-up build_plan()
            # call with family_overrides set) is what reveals the CORRECT
            # family's own field questions, so asking both at once would
            # show questions for a family that might not even be the right
            # one.
            plan.questions.append(
                ClarifyingQuestion(
                    component_id=component_id, field="__family__", kind="family_choice",
                    label="Is this housing round or rectangular?", unit=None,
                    default="flange", choices=list(component.family_options),
                )
            )
            continue

        plan.questions.extend(_questions_for_defaulted_fields(component_id, family, defaulted_fields, params_dict))
        if part_type_inferred:
            # No family keyword matched at all (parse_prompt's own fallback
            # guess) -- worth a question in its own right, distinct from
            # any single field, since every field default is suspect if the
            # family itself was guessed.
            plan.questions.append(
                ClarifyingQuestion(
                    component_id=component_id, field="__family__", kind="family_choice",
                    label="Couldn't identify a part type for this -- which is it?", unit=None,
                    default=family, choices=sorted(_PARSERS.keys()),
                )
            )

    return plan
