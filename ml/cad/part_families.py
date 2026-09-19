"""Typed registry of "part families" -- the tool layer between a parsed prompt
and procedural_generator.py's deterministic mesh builders.

Why a registry of records, not a class hierarchy: a part family is just data
(a params dataclass + two free functions + field metadata), not a polymorphic
object with its own behavior beyond `to_description()` (which already lives on
the params dataclasses themselves). `SurfaceRepresentation`'s `__init_subclass__`
registry pattern (ml/threedimen) earns its complexity because its subclasses
really are distinct objects with instance methods; that complexity isn't needed
here, so this is a plain register/get/list API instead.

This module is the single source of truth for "what happens to a field when a
prompt doesn't mention it" (`resolve_default`), reused by both
nlp/prompt_parser.py (the existing regex parser) and data/prompt_dataset.py
(the synthetic training-data generator, added later). Before this module
existed, that default logic lived only inline in prompt_parser.py; duplicating
it in the dataset generator would have let the two silently drift apart --
most notably because FlangeParams.bolt_circle_diameter_mm's default is a
*function* of outer_diameter_mm, not a flat constant.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any, Callable, Literal

from data.procedural_generator import (
    BracketParams,
    EnclosureParams,
    FlangeParams,
    GearParams,
    MountingPlateParams,
    ShaftParams,
    build_bracket,
    build_enclosure,
    build_flange,
    build_gear,
    build_mounting_plate,
    build_shaft,
    random_bracket,
    random_enclosure,
    random_flange,
    random_gear,
    random_mounting_plate,
    random_shaft,
)

FieldKind = Literal["numeric", "categorical", "count"]


@dataclass(frozen=True)
class FieldSpec:
    name: str
    kind: FieldKind
    unit: str | None
    python_type: type
    # For "categorical"/"count": the discrete values this field is actually
    # trained/tested against (i.e. what the *sampler* produces), as strings for
    # JSON-safe display via schema_dict() -- not necessarily every value the
    # regex parser could theoretically extract from free text. See module
    # docstring in data/prompt_dataset.py for why that gap is a known,
    # explicitly-measured limitation rather than a bug.
    choices: tuple[str, ...] | None = None
    # A flat default, or a function of the OTHER fields already resolved for
    # this part (see FlangeParams.bolt_circle_diameter_mm below).
    default: float | int | str | Callable[[dict[str, Any]], float | int | str] | None = None


@dataclass(frozen=True)
class PartFamilySpec:
    name: str
    params_cls: type
    builder: Callable[[Any], Any]  # -> trimesh.Trimesh
    sampler: Callable[[Any], Any]  # (random.Random) -> params instance
    fields: tuple[FieldSpec, ...]

    def field(self, name: str) -> FieldSpec:
        for f in self.fields:
            if f.name == name:
                return f
        raise ValueError(f"{self.name!r} part family has no field {name!r}")


_REGISTRY: dict[str, PartFamilySpec] = {}


def register_part_family(spec: PartFamilySpec) -> None:
    _REGISTRY[spec.name] = spec


def get_part_family(name: str) -> PartFamilySpec:
    try:
        return _REGISTRY[name]
    except KeyError:
        raise ValueError(f"unknown part family {name!r} -- known families: {sorted(_REGISTRY)}") from None


def list_part_families() -> list[str]:
    return sorted(_REGISTRY)


def resolve_default(family: str, field: str, resolved_so_far: dict[str, Any]) -> float | int | str:
    """The value a field takes when a prompt doesn't mention it. Single
    source of truth -- see module docstring for why this matters."""
    spec = get_part_family(family).field(field)
    if spec.default is None:
        raise ValueError(f"{family}.{field} has no default registered")
    if callable(spec.default):
        return spec.default(resolved_so_far)
    return spec.default


def schema_dict() -> dict[str, list[dict]]:
    """JSON-safe schema for all registered families -- used by the CLI and as
    the model's output-space documentation. Defaults that are functions of
    other fields are shown as their source-derived form (not evaluated),
    since there's no fixed value without a concrete `resolved_so_far`."""
    out: dict[str, list[dict]] = {}
    for family_name in list_part_families():
        spec = _REGISTRY[family_name]
        out[family_name] = [
            {
                "name": f.name,
                "kind": f.kind,
                "unit": f.unit,
                "python_type": f.python_type.__name__,
                "choices": list(f.choices) if f.choices else None,
                "default": f.default if not callable(f.default) else "<derived from other fields>",
            }
            for f in spec.fields
        ]
    return out


# ---------------------------------------------------------------------------
# Registrations. Defaults below MUST match nlp/prompt_parser.py's existing
# inline fallback values exactly -- prompt_parser.py imports resolve_default
# from this module rather than hardcoding its own copies (see that file).
# ---------------------------------------------------------------------------

register_part_family(
    PartFamilySpec(
        name="plate",
        params_cls=MountingPlateParams,
        builder=build_mounting_plate,
        sampler=random_mounting_plate,
        fields=(
            FieldSpec("length_mm", "numeric", "mm", float, default=100.0),
            FieldSpec("width_mm", "numeric", "mm", float, default=60.0),
            FieldSpec("thickness_mm", "numeric", "mm", float, default=6.0),
            FieldSpec("hole_diameter_mm", "numeric", "mm", float, default=8.0),
            FieldSpec("hole_edge_margin_mm", "numeric", "mm", float, default=15.0),
            FieldSpec("fillet_radius_mm", "numeric", "mm", float, default=5.0),
            FieldSpec("hole_count", "count", None, int, choices=("2", "4"), default=4),
            FieldSpec(
                "material",
                "categorical",
                None,
                str,
                choices=("aluminum", "steel", "polycarbonate", "titanium", "plastic", "brass"),
                default="aluminum",
            ),
        ),
    )
)

register_part_family(
    PartFamilySpec(
        name="flange",
        params_cls=FlangeParams,
        builder=build_flange,
        sampler=random_flange,
        fields=(
            FieldSpec("outer_diameter_mm", "numeric", "mm", float, default=120.0),
            FieldSpec("bore_diameter_mm", "numeric", "mm", float, default=40.0),
            FieldSpec("thickness_mm", "numeric", "mm", float, default=8.0),
            FieldSpec("bolt_hole_diameter_mm", "numeric", "mm", float, default=8.0),
            FieldSpec("bolt_hole_count", "count", None, int, choices=("4", "6", "8"), default=4),
            FieldSpec(
                "bolt_circle_diameter_mm",
                "numeric",
                "mm",
                float,
                # matches prompt_parser._parse_flange's inline formula exactly
                default=lambda resolved: round(resolved["outer_diameter_mm"] * 0.75 / 2) * 2,
            ),
            FieldSpec(
                "material",
                "categorical",
                None,
                str,
                choices=("steel", "aluminum", "cast iron", "titanium", "brass"),
                default="steel",
            ),
        ),
    )
)

register_part_family(
    PartFamilySpec(
        name="bracket",
        params_cls=BracketParams,
        builder=build_bracket,
        sampler=random_bracket,
        fields=(
            FieldSpec("leg1_length_mm", "numeric", "mm", float, default=50.0),
            FieldSpec("leg2_length_mm", "numeric", "mm", float, default=50.0),
            FieldSpec("thickness_mm", "numeric", "mm", float, default=5.0),
            FieldSpec("length_mm", "numeric", "mm", float, default=100.0),
            FieldSpec("hole_diameter_mm", "numeric", "mm", float, default=6.0),
            FieldSpec("hole_edge_margin_mm", "numeric", "mm", float, default=15.0),
            FieldSpec("holes_per_arm", "count", None, int, choices=("1", "2"), default=2),
            FieldSpec(
                "material", "categorical", None, str,
                choices=("steel", "aluminum", "cast iron", "titanium", "brass"), default="steel",
            ),
        ),
    )
)

register_part_family(
    PartFamilySpec(
        name="shaft",
        params_cls=ShaftParams,
        builder=build_shaft,
        sampler=random_shaft,
        fields=(
            FieldSpec("outer_diameter_mm", "numeric", "mm", float, default=20.0),
            FieldSpec("length_mm", "numeric", "mm", float, default=100.0),
            FieldSpec("inner_diameter_mm", "numeric", "mm", float, default=0.0),  # 0 = solid
            FieldSpec("keyway_width_mm", "numeric", "mm", float, default=0.0),  # 0 = no keyway
            FieldSpec("keyway_depth_mm", "numeric", "mm", float, default=0.0),
            FieldSpec("keyway_length_mm", "numeric", "mm", float, default=0.0),  # 0 = full length
            FieldSpec(
                "material", "categorical", None, str,
                choices=("steel", "aluminum", "titanium", "brass"), default="steel",
            ),
        ),
    )
)

register_part_family(
    PartFamilySpec(
        name="enclosure",
        params_cls=EnclosureParams,
        builder=build_enclosure,
        sampler=random_enclosure,
        fields=(
            FieldSpec("length_mm", "numeric", "mm", float, default=100.0),
            FieldSpec("width_mm", "numeric", "mm", float, default=80.0),
            FieldSpec("height_mm", "numeric", "mm", float, default=40.0),
            FieldSpec("wall_thickness_mm", "numeric", "mm", float, default=3.0),
            FieldSpec("open_face", "categorical", None, str, choices=("none", "top"), default="none"),
            FieldSpec("vent_hole_count", "count", None, int, choices=("0", "2", "4", "6"), default=0),
            FieldSpec("vent_hole_diameter_mm", "numeric", "mm", float, default=0.0),
            FieldSpec(
                "material", "categorical", None, str,
                choices=("aluminum", "steel", "plastic", "polycarbonate"), default="aluminum",
            ),
        ),
    )
)

register_part_family(
    PartFamilySpec(
        name="gear",
        params_cls=GearParams,
        builder=build_gear,
        sampler=random_gear,
        fields=(
            FieldSpec("module_mm", "numeric", "mm", float, default=2.0),
            FieldSpec("tooth_count", "count", None, int, choices=("12", "16", "20", "24", "32"), default=16),
            FieldSpec("face_width_mm", "numeric", "mm", float, default=10.0),
            FieldSpec("bore_diameter_mm", "numeric", "mm", float, default=8.0),
            FieldSpec("pressure_angle_deg", "numeric", "deg", float, default=20.0),
            FieldSpec(
                "material", "categorical", None, str,
                choices=("steel", "aluminum", "brass", "titanium"), default="steel",
            ),
        ),
    )
)


def _check_field_specs_match_dataclass(family: str) -> None:
    """Drift guard, run at import time: if someone adds a field to
    MountingPlateParams/FlangeParams and forgets this registry, fail loudly
    immediately rather than silently producing an incomplete schema."""
    spec = get_part_family(family)
    dataclass_fields = {f.name for f in fields(spec.params_cls)}
    registered_fields = {f.name for f in spec.fields}
    missing = dataclass_fields - registered_fields
    if missing:
        raise RuntimeError(
            f"part_families.py registration for {family!r} is missing FieldSpec(s) for "
            f"{sorted(missing)} -- {spec.params_cls.__name__} has these fields but this "
            f"registry doesn't describe them"
        )


for _family in list_part_families():
    _check_field_specs_match_dataclass(_family)
