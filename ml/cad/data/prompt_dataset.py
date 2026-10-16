"""Synthetic (text, part_family, params) triples for training/evaluating a
prompt-understanding model, generated for free from the existing procedural
samplers -- no real-world data collection needed for this task, since the
target (structured params) is fully known at generation time.

Wording variation covers: unit notation (tight/spaced/word-form mm), clause
order, an alternate combined "LxWxT mm" dimension format (matching what
nlp/prompt_parser.py's regex already recognizes), material synonyms, and
field omission.

CRITICAL correctness rule, see part_families.py's module docstring: when a
field is chosen for omission from the text, its value in `params` is
overwritten with part_families.resolve_default() BEFORE rendering -- never
left as the sampler's raw random value. Otherwise the label for an omitted
field would be an unguessable random number uncorrelated with the (silent)
input, which would look like "the model failed to learn" when the labels
were actually structurally impossible to predict.
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

from part_families import get_part_family, list_part_families, resolve_default

UnitStyle = Literal["mm_tight", "mm_spaced", "word_millimeters"]

_MATERIAL_SYNONYMS: dict[str, tuple[str, ...]] = {
    "aluminum": ("aluminum", "aluminium"),
    "steel": ("steel", "mild steel"),
    "polycarbonate": ("polycarbonate", "clear polycarbonate"),
    "titanium": ("titanium",),
    "plastic": ("plastic", "ABS plastic"),
    "brass": ("brass",),
    "cast iron": ("cast iron",),
}


@dataclass(frozen=True)
class WordingConfig:
    unit_style: UnitStyle
    clause_order_seed: int
    omit_fields: frozenset[str]
    material_synonym: str | None  # a value from _MATERIAL_SYNONYMS[canonical_material], or None


@dataclass(frozen=True)
class PromptSample:
    id: str
    text: str
    part_family: str
    params: dict  # fully resolved -- see module docstring
    parsed_fields: tuple[str, ...]
    defaulted_fields: tuple[str, ...]
    wording_variant: str
    canonical_key: str

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "PromptSample":
        return cls(
            id=d["id"], text=d["text"], part_family=d["part_family"], params=d["params"],
            parsed_fields=tuple(d["parsed_fields"]), defaulted_fields=tuple(d["defaulted_fields"]),
            wording_variant=d["wording_variant"], canonical_key=d["canonical_key"],
        )


def _fmt_num(value: float, unit_style: UnitStyle) -> str:
    text = f"{value:g}"
    if unit_style == "mm_tight":
        return f"{text}mm"
    if unit_style == "mm_spaced":
        return f"{text} mm"
    if unit_style == "word_millimeters":
        return f"{text} millimeters"
    raise ValueError(f"unknown unit_style {unit_style!r}")


def _material_text(material: str, cfg: WordingConfig) -> str:
    if cfg.material_synonym and cfg.material_synonym in _MATERIAL_SYNONYMS.get(material, ()):
        return cfg.material_synonym
    return material


def _render_plate(resolved: dict, kept: frozenset[str], cfg: WordingConfig) -> str:
    material = _material_text(resolved["material"], cfg)
    clauses: list[str] = []

    dims_fields = {"length_mm", "width_mm", "thickness_mm"}
    # the combined "L x W x T mm" form only makes sense if all three dims are
    # kept -- otherwise fall back to per-field clauses, which also covers the
    # case where none/some of them are omitted
    use_combined_dims = dims_fields <= kept and cfg.clause_order_seed % 2 == 0

    lead = f"rectangular {material} mounting plate" if "material" in kept else "rectangular mounting plate"
    clauses.append(lead)

    if use_combined_dims:
        l, w, t = resolved["length_mm"], resolved["width_mm"], resolved["thickness_mm"]
        clauses.append(f"{l:g} mm x {w:g} mm x {t:g} mm")
    else:
        if "length_mm" in kept:
            clauses.append(f"{_fmt_num(resolved['length_mm'], cfg.unit_style)} long")
        if "width_mm" in kept:
            clauses.append(f"{_fmt_num(resolved['width_mm'], cfg.unit_style)} wide")
        if "thickness_mm" in kept:
            clauses.append(f"{_fmt_num(resolved['thickness_mm'], cfg.unit_style)} thick")

    if "hole_count" in kept or "hole_diameter_mm" in kept:
        count_str = f"{int(resolved['hole_count'])} " if "hole_count" in kept else ""
        d_str = f"{_fmt_num(resolved['hole_diameter_mm'], cfg.unit_style)} " if "hole_diameter_mm" in kept else ""
        clauses.append(f"with {count_str}{d_str}mounting holes".strip())
    if "hole_edge_margin_mm" in kept:
        clauses.append(f"{_fmt_num(resolved['hole_edge_margin_mm'], cfg.unit_style)} from each edge")
    if "fillet_radius_mm" in kept:
        clauses.append(f"{_fmt_num(resolved['fillet_radius_mm'], cfg.unit_style)} corner fillet")

    body = clauses[1:]
    rng = random.Random(cfg.clause_order_seed)
    rng.shuffle(body)
    return ", ".join([clauses[0], *body])


def _render_flange(resolved: dict, kept: frozenset[str], cfg: WordingConfig) -> str:
    material = _material_text(resolved["material"], cfg)
    lead = f"circular {material} flange" if "material" in kept else "circular flange"
    clauses = [lead]

    if "outer_diameter_mm" in kept:
        clauses.append(f"{_fmt_num(resolved['outer_diameter_mm'], cfg.unit_style)} outer diameter")
    if "thickness_mm" in kept:
        clauses.append(f"{_fmt_num(resolved['thickness_mm'], cfg.unit_style)} thick")
    if "bore_diameter_mm" in kept:
        clauses.append(f"a {_fmt_num(resolved['bore_diameter_mm'], cfg.unit_style)} central bore")
    if "bolt_hole_count" in kept or "bolt_hole_diameter_mm" in kept:
        count_str = f"{int(resolved['bolt_hole_count'])} " if "bolt_hole_count" in kept else ""
        d_str = f"{_fmt_num(resolved['bolt_hole_diameter_mm'], cfg.unit_style)} " if "bolt_hole_diameter_mm" in kept else ""
        clauses.append(f"{count_str}{d_str}mounting holes".strip())
    if "bolt_circle_diameter_mm" in kept:
        clauses.append(f"on a {_fmt_num(resolved['bolt_circle_diameter_mm'], cfg.unit_style)} bolt circle")

    body = clauses[1:]
    rng = random.Random(cfg.clause_order_seed)
    rng.shuffle(body)
    return ", ".join([clauses[0], *body])


def _render_bracket(resolved: dict, kept: frozenset[str], cfg: WordingConfig) -> str:
    material = _material_text(resolved["material"], cfg)
    # "L-shape", not "L-shaped" -- matches _SYSTEM_PROMPT's own wording
    # exactly (Example 1: "an L-shape bracket, ..."), for the same reason
    # as the "in each leg" fix below: this is the lead word a TF-IDF match
    # weighs most heavily, so it should match the LLM's real vocabulary.
    lead = f"L-shape {material} bracket" if "material" in kept else "L-shape bracket"
    clauses = [lead]

    if "leg1_length_mm" in kept and "leg2_length_mm" in kept:
        l1, l2 = resolved["leg1_length_mm"], resolved["leg2_length_mm"]
        clauses.append(f"{l1:g} mm x {l2:g} mm legs")
    else:
        if "leg1_length_mm" in kept:
            clauses.append(f"{_fmt_num(resolved['leg1_length_mm'], cfg.unit_style)} leg")
        if "leg2_length_mm" in kept:
            clauses.append(f"{_fmt_num(resolved['leg2_length_mm'], cfg.unit_style)} leg")
    if "thickness_mm" in kept:
        clauses.append(f"{_fmt_num(resolved['thickness_mm'], cfg.unit_style)} thick")
    if "length_mm" in kept:
        clauses.append(f"{_fmt_num(resolved['length_mm'], cfg.unit_style)} long")
    if "holes_per_arm" in kept or "hole_diameter_mm" in kept:
        count_str = f"{int(resolved['holes_per_arm'])} " if "holes_per_arm" in kept else ""
        d_str = f"{_fmt_num(resolved['hole_diameter_mm'], cfg.unit_style)} " if "hole_diameter_mm" in kept else ""
        # "in each leg", not "per arm" -- llm_decompose.py's own _SYSTEM_PROMPT
        # (Example 1) teaches the LLM to phrase this exact feature as "with
        # two holes in each leg". Training text using "arm" instead of "leg"
        # shares almost no vocabulary with that taught phrasing, which starves
        # program_model.py's TF-IDF matching of signal on real LLM output --
        # confirmed by a real decomposition producing a generic flat plate
        # instead of an L-bracket for exactly this phrasing.
        suffix = " in each leg" if "holes_per_arm" in kept else ""
        clauses.append(f"with {count_str}{d_str}holes{suffix}".strip())
    if "hole_edge_margin_mm" in kept:
        clauses.append(f"{_fmt_num(resolved['hole_edge_margin_mm'], cfg.unit_style)} from each edge")

    body = clauses[1:]
    rng = random.Random(cfg.clause_order_seed)
    rng.shuffle(body)
    return ", ".join([clauses[0], *body])


def _render_shaft(resolved: dict, kept: frozenset[str], cfg: WordingConfig) -> str:
    material = _material_text(resolved["material"], cfg)
    lead = f"{material} shaft" if "material" in kept else "shaft"
    clauses = [lead]

    if "outer_diameter_mm" in kept:
        clauses.append(f"{_fmt_num(resolved['outer_diameter_mm'], cfg.unit_style)} diameter")
    if "length_mm" in kept:
        clauses.append(f"{_fmt_num(resolved['length_mm'], cfg.unit_style)} long")
    if "inner_diameter_mm" in kept and resolved["inner_diameter_mm"] > 0:
        clauses.append(f"{_fmt_num(resolved['inner_diameter_mm'], cfg.unit_style)} bore")
    if "keyway_width_mm" in kept and resolved["keyway_width_mm"] > 0:
        clauses.append(f"{_fmt_num(resolved['keyway_width_mm'], cfg.unit_style)} keyway")

    body = clauses[1:]
    rng = random.Random(cfg.clause_order_seed)
    rng.shuffle(body)
    return ", ".join([clauses[0], *body])


def _render_enclosure(resolved: dict, kept: frozenset[str], cfg: WordingConfig) -> str:
    material = _material_text(resolved["material"], cfg)
    use_combined_dims = {"length_mm", "width_mm", "height_mm"} <= kept and cfg.clause_order_seed % 2 == 0
    lead = f"{material} enclosure" if "material" in kept else "enclosure"
    clauses = [lead]

    if use_combined_dims:
        l, w, h = resolved["length_mm"], resolved["width_mm"], resolved["height_mm"]
        clauses.append(f"{l:g} mm x {w:g} mm x {h:g} mm")
    else:
        if "length_mm" in kept:
            clauses.append(f"{_fmt_num(resolved['length_mm'], cfg.unit_style)} long")
        if "width_mm" in kept:
            clauses.append(f"{_fmt_num(resolved['width_mm'], cfg.unit_style)} wide")
        if "height_mm" in kept:
            clauses.append(f"{_fmt_num(resolved['height_mm'], cfg.unit_style)} tall")
    if "wall_thickness_mm" in kept:
        clauses.append(f"{_fmt_num(resolved['wall_thickness_mm'], cfg.unit_style)} wall")
    if "open_face" in kept and resolved["open_face"] == "top":
        clauses.append("open top")
    if "vent_hole_count" in kept and resolved["vent_hole_count"] > 0:
        d_str = f"{_fmt_num(resolved['vent_hole_diameter_mm'], cfg.unit_style)} " if "vent_hole_diameter_mm" in kept else ""
        clauses.append(f"{int(resolved['vent_hole_count'])} {d_str}vent holes".strip())

    body = clauses[1:]
    rng = random.Random(cfg.clause_order_seed)
    rng.shuffle(body)
    return ", ".join([clauses[0], *body])


def _render_gear(resolved: dict, kept: frozenset[str], cfg: WordingConfig) -> str:
    material = _material_text(resolved["material"], cfg)
    lead = f"{material} spur gear" if "material" in kept else "spur gear"
    clauses = [lead]

    if "module_mm" in kept:
        clauses.append(f"module {resolved['module_mm']:g}")
    if "tooth_count" in kept:
        clauses.append(f"{int(resolved['tooth_count'])} teeth")
    if "face_width_mm" in kept:
        clauses.append(f"{_fmt_num(resolved['face_width_mm'], cfg.unit_style)} face width")
    if "bore_diameter_mm" in kept:
        clauses.append(f"{_fmt_num(resolved['bore_diameter_mm'], cfg.unit_style)} bore")

    body = clauses[1:]
    rng = random.Random(cfg.clause_order_seed)
    rng.shuffle(body)
    return ", ".join([clauses[0], *body])


_RENDERERS = {
    "plate": _render_plate,
    "flange": _render_flange,
    "bracket": _render_bracket,
    "shaft": _render_shaft,
    "enclosure": _render_enclosure,
    "gear": _render_gear,
}


def render_text(part_family: str, resolved: dict, omit_fields: frozenset[str], cfg: WordingConfig) -> str:
    kept = frozenset(resolved.keys()) - omit_fields
    return _RENDERERS[part_family](resolved, kept, cfg)


def canonical_key(part_family: str, params: dict) -> str:
    payload = json.dumps({"family": part_family, "params": {k: params[k] for k in sorted(params)}}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()


def _resolve_with_omissions(family: str, raw: dict, omit_fields: frozenset[str]) -> dict:
    spec = get_part_family(family)
    resolved: dict = {}
    for f in spec.fields:  # declaration order matters: derived defaults (e.g.
        if f.name in omit_fields:  # bolt_circle_diameter_mm) depend on earlier fields
            resolved[f.name] = resolve_default(family, f.name, resolved)
        else:
            resolved[f.name] = raw[f.name]
    return resolved


def sample_prompt_dataset(counts: dict[str, int], seed: int, max_omit_fraction: float = 0.4) -> list[PromptSample]:
    """counts: e.g. {"plate": 4000, "flange": 4000}."""
    rng = random.Random(seed)
    samples: list[PromptSample] = []
    unit_styles: tuple[UnitStyle, ...] = ("mm_tight", "mm_spaced", "word_millimeters")

    for family in list_part_families():
        n = counts.get(family, 0)
        spec = get_part_family(family)
        optional_fields = [f.name for f in spec.fields]  # any field may be omitted, incl. material
        for i in range(n):
            raw = asdict(spec.sampler(rng))
            n_omit = rng.randint(0, max(1, int(len(optional_fields) * max_omit_fraction)))
            omit_fields = frozenset(rng.sample(optional_fields, k=min(n_omit, len(optional_fields) - 1)))

            resolved = _resolve_with_omissions(family, raw, omit_fields)
            material_canonical = resolved.get("material")
            synonym_choices = _MATERIAL_SYNONYMS.get(material_canonical, (material_canonical,)) if material_canonical else (None,)
            cfg = WordingConfig(
                unit_style=rng.choice(unit_styles),
                clause_order_seed=rng.randint(0, 2**31 - 1),
                omit_fields=omit_fields,
                material_synonym=rng.choice(synonym_choices),
            )
            text = render_text(family, resolved, omit_fields, cfg)
            kept_fields = tuple(sorted(set(resolved) - omit_fields))
            defaulted_fields = tuple(sorted(omit_fields))

            samples.append(
                PromptSample(
                    id=f"{family}_{i:06d}",
                    text=text,
                    part_family=family,
                    params=resolved,
                    parsed_fields=kept_fields,
                    defaulted_fields=defaulted_fields,
                    wording_variant=f"{cfg.unit_style}|{'combined' if cfg.clause_order_seed % 2 == 0 else 'separate'}",
                    canonical_key=canonical_key(family, resolved),
                )
            )
    return samples
