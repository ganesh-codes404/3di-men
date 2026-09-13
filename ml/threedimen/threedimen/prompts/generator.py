"""Part 5: synthetic prompt generation.

Each category is a small, independent spec (nouns/adjectives/materials/
feature phrases/dimension ranges) combined by random sampling + templating --
the same spirit as ml/cad/data/procedural_generator.py's random_mounting_plate,
but text-only: this module never touches geometry, that's the teacher
model's job (Part 3). Adding a category means adding one CategorySpec entry,
not touching the generation logic.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field


@dataclass
class CategorySpec:
    name: str
    nouns: list[str]
    adjectives: list[str]
    materials: list[str]
    features: list[str]  # may contain "{n}" for a randomly chosen count
    dimension_ranges: dict[str, tuple[float, float]] = field(default_factory=dict)


@dataclass
class GeneratedPrompt:
    text: str
    category: str
    attributes: dict


CATEGORIES: dict[str, CategorySpec] = {
    "pump": CategorySpec(
        "pump",
        nouns=["centrifugal water pump", "gear pump", "diaphragm pump", "submersible pump", "fuel pump"],
        adjectives=["compact", "industrial", "heavy-duty", "high-flow", "low-noise"],
        materials=["cast iron", "stainless steel", "aluminum", "bronze"],
        features=[
            "{n} mounting holes", "a cylindrical inlet", "a threaded outlet port",
            "an integrated motor housing", "a cooling fin array", "a flanged base",
        ],
        dimension_ranges={"diameter_mm": (60, 300), "length_mm": (100, 500)},
    ),
    "motor": CategorySpec(
        "motor",
        nouns=["brushless DC motor", "stepper motor", "servo motor", "induction motor"],
        adjectives=["compact", "high-torque", "industrial", "precision", "sealed"],
        materials=["aluminum", "steel", "cast iron"],
        features=[
            "{n} mounting holes on the face plate", "a splined output shaft", "cooling fins",
            "an integrated encoder housing", "a terminal box",
        ],
        dimension_ranges={"diameter_mm": (30, 200), "shaft_diameter_mm": (5, 30)},
    ),
    "bracket": CategorySpec(
        "bracket",
        nouns=["L-bracket", "mounting bracket", "shelf bracket", "wall mount bracket", "corner brace"],
        adjectives=["compact", "reinforced", "heavy-duty", "lightweight"],
        materials=["aluminum", "steel", "stainless steel", "polycarbonate"],
        features=["{n} mounting holes", "a reinforcing rib", "rounded corners", "a slotted adjustment hole"],
        dimension_ranges={"length_mm": (40, 300), "thickness_mm": (2, 12)},
    ),
    "pipe": CategorySpec(
        "pipe",
        nouns=["pipe fitting", "elbow fitting", "T-fitting", "reducer coupling", "flange fitting"],
        adjectives=["threaded", "compact", "heavy-duty", "corrosion-resistant"],
        materials=["PVC", "stainless steel", "brass", "cast iron"],
        features=["a threaded connection on each end", "a sealing gasket groove", "{n} bolt holes on the flange"],
        dimension_ranges={"outer_diameter_mm": (15, 150), "wall_thickness_mm": (2, 12)},
    ),
    "valve": CategorySpec(
        "valve",
        nouns=["ball valve", "gate valve", "check valve", "butterfly valve", "relief valve"],
        adjectives=["compact", "industrial", "high-pressure", "corrosion-resistant"],
        materials=["brass", "stainless steel", "cast iron", "PVC"],
        features=["a lever handle", "{n} bolt holes on each flange", "a threaded inlet and outlet"],
        dimension_ranges={"port_diameter_mm": (10, 100), "body_length_mm": (40, 250)},
    ),
    "enclosure": CategorySpec(
        "enclosure",
        nouns=["electronics enclosure", "junction box", "control panel housing", "weatherproof enclosure"],
        adjectives=["compact", "weatherproof", "ventilated", "industrial"],
        materials=["ABS plastic", "aluminum", "polycarbonate", "sheet steel"],
        features=[
            "a hinged lid", "{n} cable gland ports", "ventilation slots", "{n} corner mounting holes",
            "a gasket-sealed lid",
        ],
        dimension_ranges={"width_mm": (80, 400), "height_mm": (60, 300), "depth_mm": (40, 200)},
    ),
    "ev_charger": CategorySpec(
        "ev_charger",
        nouns=["EV wall charger", "EV charging connector housing", "EV charging station enclosure"],
        adjectives=["compact", "weatherproof", "wall-mounted", "sleek"],
        materials=["aluminum", "ABS plastic", "polycarbonate"],
        features=[
            "a cable management hook", "a status LED window", "{n} wall mounting holes",
            "a ventilated rear panel", "a connector holster",
        ],
        dimension_ranges={"width_mm": (150, 350), "height_mm": (200, 450)},
    ),
    "tool": CategorySpec(
        "tool",
        nouns=["hand drill housing", "wrench", "screwdriver handle", "clamp", "tool holder"],
        adjectives=["ergonomic", "compact", "heavy-duty", "lightweight"],
        materials=["polycarbonate", "aluminum", "steel", "rubber-coated steel"],
        features=["a textured grip", "a hanging hole", "a locking mechanism", "{n} finger grooves"],
        dimension_ranges={"length_mm": (100, 350)},
    ),
    "furniture": CategorySpec(
        "furniture",
        nouns=["stool", "side table", "shelf unit", "chair frame", "coat rack"],
        adjectives=["minimalist", "compact", "modern", "sturdy", "foldable"],
        materials=["oak wood", "powder-coated steel", "walnut wood", "aluminum"],
        features=["tapered legs", "a curved backrest", "{n} shelves", "rounded edges", "a cross-brace"],
        dimension_ranges={"height_mm": (300, 1200), "width_mm": (300, 1500)},
    ),
    "vehicle": CategorySpec(
        "vehicle",
        nouns=["go-kart chassis", "drone frame", "RC car body", "bicycle frame component"],
        adjectives=["lightweight", "aerodynamic", "compact", "rugged"],
        materials=["carbon fiber", "aluminum", "steel tubing", "ABS plastic"],
        features=["mounting points for four wheels", "a battery compartment", "{n} reinforcement ribs"],
        dimension_ranges={"length_mm": (200, 1500), "width_mm": (150, 800)},
    ),
    "consumer_product": CategorySpec(
        "consumer_product",
        nouns=["desk lamp base", "phone stand", "speaker enclosure", "kitchen utensil holder", "planter pot"],
        adjectives=["minimalist", "compact", "modern", "ergonomic", "sleek"],
        materials=["ABS plastic", "matte aluminum", "ceramic-look resin", "bamboo"],
        features=["a non-slip base", "a cable routing channel", "rounded edges", "a weighted bottom"],
        dimension_ranges={"height_mm": (50, 300), "base_diameter_mm": (40, 200)},
    ),
}


def generate_prompt(category: str | None = None, rng: random.Random | None = None) -> GeneratedPrompt:
    rng = rng or random.Random()
    category = category or rng.choice(list(CATEGORIES))
    if category not in CATEGORIES:
        raise ValueError(f"unknown category {category!r} -- known: {sorted(CATEGORIES)}")
    spec = CATEGORIES[category]

    noun = rng.choice(spec.nouns)
    adjective = rng.choice(spec.adjectives)
    material = rng.choice(spec.materials)

    n_features = rng.randint(1, min(3, len(spec.features)))
    feature_phrases = []
    dimensions: dict[str, float] = {}
    for f in rng.sample(spec.features, n_features):
        if "{n}" in f:
            count = rng.choice([2, 3, 4, 6, 8])
            feature_phrases.append(f.format(n=count))
        else:
            feature_phrases.append(f)

    dim_phrases = []
    for dim_name, (lo, hi) in spec.dimension_ranges.items():
        value = round(rng.uniform(lo, hi))
        dimensions[dim_name] = value
        pretty = dim_name.replace("_mm", "").replace("_", " ")
        dim_phrases.append(f"{value}mm {pretty}")

    text = f"{adjective} {material} {noun} with {', '.join(feature_phrases)}"
    if dim_phrases:
        text += f", {', '.join(dim_phrases)}"

    return GeneratedPrompt(
        text=text,
        category=category,
        attributes={
            "noun": noun, "adjective": adjective, "material": material,
            "features": feature_phrases, "dimensions": dimensions,
        },
    )


def generate_prompts(
    n: int, categories: list[str] | None = None, seed: int = 0
) -> list[GeneratedPrompt]:
    rng = random.Random(seed)
    pool = categories or list(CATEGORIES)
    for c in pool:
        if c not in CATEGORIES:
            raise ValueError(f"unknown category {c!r} -- known: {sorted(CATEGORIES)}")
    return [generate_prompt(rng.choice(pool), rng) for _ in range(n)]
