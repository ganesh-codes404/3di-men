"""The 60-prompt CAD benchmark's own fixed prompt set + independent
ground truth (Phase B, Part 9/10). Deliberately kept OUT of production
logic (benchmark_runner.py imports this; nothing in the generation path
imports it) -- Part "Engineering Constraints" own rule: "do not hardcode
the 60 prompts into production logic."

Ground truth is written INDEPENDENTLY of this project's own validators:
each BenchmarkPrompt states what a human engineer would expect, in a
plain, validator-agnostic vocabulary (component count, key measurable
dimensions, automatically-countable features, relationship types) --
benchmark_runner.py's own scoring function measures the BUILT mesh
directly (trimesh extents, evaluation.geometry_eval's raw hole/slot
counts) rather than trusting feature_verification.py's/component_
inventory.py's own self-reported pass/fail (Part 10's own "do not use
3di-men's own validators as the sole source of truth" rule) -- those
validators are still recorded separately, for the validator-reliability
scoring axis (H), never conflated with ground truth itself.

`decomposition` is a FIXED, hand-written list of (description,
relationship) pairs -- never produced by the real LLM (Part 8's own
"controlled deterministic benchmark" mode: this is what lets 60 prompts
run in minutes rather than the 10-90+ minutes PER PROMPT a real local-
LLM decomposition call took elsewhere in this project's own session
history). For every single-component prompt (Categories A-D), the
"decomposition" is trivially one point whose description IS the prompt
text itself -- no hand-splitting needed or invented. Only genuinely
multi-component prompts (Categories E-F) need a real, hand-written
multi-point split, written here the same way llm_decompose.py's own
system prompt instructs the real model to do it (one part per physically
separate solid, assembly_plan.Relationship for anything that mates)."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class BenchmarkPrompt:
    id: str
    category: str
    prompt: str
    decomposition: list[dict] = field(default_factory=list)  # [] means "use the prompt text as one single point" --
        # filled in by benchmark_runner.get_decomposition(), never duplicated by hand for every single-part prompt.
    expected_component_count: int = 1
    expected_dimensions_mm: list[float] = field(default_factory=list)  # key dimensions a human would check first --
        # not exhaustive, deliberately scoped to what is both STATED and independently re-measurable.
    expected_features: list[tuple[str, int]] = field(default_factory=list)  # (feature_type, count) for the two
        # feature types this project's own geometry measurement can independently count without trusting any
        # validator's own opinion: "hole" and "slot" (evaluation.geometry_eval's own detected_hole_count/
        # detected_slot_count). Every other feature word in the prompt (rib, gusset, boss, chamfer, keyway,
        # counterbore...) is intentionally NOT listed here -- see `not_automatically_measurable` instead.
    not_automatically_measurable: list[str] = field(default_factory=list)  # features/properties this benchmark's
        # own scoring cannot independently confirm with the measurement tools available in this environment --
        # disclosed per-prompt (Part 10's own "separate automatically measurable from human-reviewed" rule),
        # never silently scored as if they were checked.
    expected_relationship_count: int = 0  # how many assembly_plan.Relationship-level matings this prompt implies --
        # 0 for anything with only one component (nothing to relate).
    tolerance_fraction: float = 0.1
    notes: str = ""


PROMPTS: list[BenchmarkPrompt] = [
    # ---------------- Category A: basic primitives and dimensions ----------------
    BenchmarkPrompt("A01", "A", "A 100 x 60 x 10 mm rectangular plate.", expected_dimensions_mm=[100, 60, 10]),
    BenchmarkPrompt("A02", "A", "A cylinder 40 mm in diameter and 80 mm high.", expected_dimensions_mm=[40, 80]),
    BenchmarkPrompt("A03", "A", "A hollow cylindrical tube with 30 mm outer diameter, 20 mm inner diameter, and 100 mm length.",
                     expected_dimensions_mm=[30, 20, 100], expected_features=[("hole", 1)],
                     notes="through-bore along the tube's own axis counts as the 'hole' feature"),
    BenchmarkPrompt("A04", "A", "A 120 x 80 x 5 mm rectangular plate.", expected_dimensions_mm=[120, 80, 5]),
    BenchmarkPrompt("A05", "A", "A stepped shaft with 20 mm, 30 mm, and 15 mm diameter sections.",
                     expected_dimensions_mm=[20, 30, 15], not_automatically_measurable=["3-way step vs 2-way step shape correctness"],
                     notes="3 distinct diameter sections -- this project's own add_shouldered_cylinder tool is 2-section only; flagged as a likely UNSUPPORTED case, not assumed"),
    BenchmarkPrompt("A06", "A", "A 50 mm cube with a centered 20 mm cylindrical cutout.", expected_dimensions_mm=[50, 20], expected_features=[("hole", 1)]),
    BenchmarkPrompt("A07", "A", "A rectangular block with a centered 10 mm-deep pocket.", not_automatically_measurable=["pocket presence/depth -- no independent pocket detector in this environment"]),
    BenchmarkPrompt("A08", "A", "A 150 mm-long bar with a 20 x 30 mm cross-section.", expected_dimensions_mm=[150, 20, 30]),
    BenchmarkPrompt("A09", "A", "A 60 mm cylindrical boss on a 100 x 100 x 10 mm plate.", expected_dimensions_mm=[100, 100, 10, 60],
                     not_automatically_measurable=["boss presence as a distinct protrusion -- no independent boss detector"]),
    BenchmarkPrompt("A10", "A", "A rectangular block with a 5 mm chamfer on all specified external edges.",
                     not_automatically_measurable=["exact 5mm chamfer distance -- feature_verification.py's own edge_treatment check is presence-only"]),

    # ---------------- Category B: mechanical parts ----------------
    BenchmarkPrompt("B11", "B", "A simple L-bracket with a 5 mm thickness.", expected_dimensions_mm=[5]),
    BenchmarkPrompt("B12", "B", "A mounting plate with two vertical flanges.", expected_component_count=1,
                     not_automatically_measurable=["flange count/placement -- flanges are a feature of one part here, not separately verified"]),
    BenchmarkPrompt("B13", "B", "A shaft with two stepped diameters and a shoulder.", not_automatically_measurable=["shoulder transition correctness"]),
    BenchmarkPrompt("B14", "B", "A motor mounting bracket for a NEMA 17 motor: 42mm x 42mm bolt pattern, 31mm bolt-circle diameter, M3 holes, with a 22mm diameter central bore, on a 60mm x 60mm x 5mm plate.",
                     expected_dimensions_mm=[60, 60, 5, 22], expected_features=[("hole", 5)],
                     notes="4 bolt-circle holes + 1 central bore = 5 holes; explicit dims supplied per the brief's own instruction, never a guessed standard"),
    BenchmarkPrompt("B15", "B", "A bearing housing with a centered bore.", expected_features=[("hole", 1)]),
    BenchmarkPrompt("B16", "B", "A flanged cylindrical coupling with four bolt holes.", expected_features=[("hole", 4)]),
    BenchmarkPrompt("B17", "B", "A U-shaped mounting bracket.", not_automatically_measurable=["U-shape profile correctness"]),
    BenchmarkPrompt("B18", "B", "A rectangular enclosure with a removable lid.", expected_component_count=2, expected_relationship_count=1,
                     decomposition=[
                         {"description": "a rectangular enclosure, 100mm x 80mm x 40mm, hollow interior"},
                         {"description": "a flat lid, 100mm x 80mm x 3mm", "relationship": {"kind": "attached_to_face", "target_id": "p0", "face": "+z"}},
                     ]),
    BenchmarkPrompt("B19", "B", "A pulley with a central bore and specified outer dimensions: 80mm outer diameter, 10mm bore, 15mm thick.",
                     expected_dimensions_mm=[80, 10, 15], expected_features=[("hole", 1)]),
    BenchmarkPrompt("B20", "B", "A shaft collar with a radial set-screw hole.", expected_features=[("hole", 1)]),
    BenchmarkPrompt("B21", "B", "A spacer with specified inner and outer diameters: 20mm outer diameter, 10mm inner diameter, 15mm length.",
                     expected_dimensions_mm=[20, 10, 15], expected_features=[("hole", 1)]),
    BenchmarkPrompt("B22", "B", "A T-shaped structural support.", not_automatically_measurable=["T-shape profile correctness"]),
    BenchmarkPrompt("B23", "B", "A slotted mounting rail.", expected_features=[("slot", 1)]),
    BenchmarkPrompt("B24", "B", "A pipe flange with a central opening and four bolt holes.", expected_features=[("hole", 5)]),
    BenchmarkPrompt("B25", "B", "A mechanical hinge with two leaves and a pin.", expected_component_count=2, expected_relationship_count=1,
                     decomposition=[
                         {"description": "a hinge leaf, 60mm x 20mm x 3mm, with a 5mm diameter hole near one edge"},
                         {"description": "a cylindrical hinge pin, 5mm diameter, 25mm long", "relationship": {"kind": "concentric_with", "target_id": "p0", "axis": "y"}},
                     ],
                     not_automatically_measurable=["the SECOND leaf -- this fixed decomposition only models the structural pin joint, see notes"],
                     notes="a true hinge needs 2 leaves + 1 pin (3 components); this fixed decomposition intentionally omits the 2nd leaf to keep the geometry simple, so expected_component_count is deliberately 2, not 3 -- the gap between a real request and this benchmark's own simplification is disclosed, not hidden"),

    # ---------------- Category C: holes, slots, and patterns ----------------
    BenchmarkPrompt("C26", "C", "A 120 x 80 x 5 mm plate with four 6 mm through-holes near its corners.",
                     expected_dimensions_mm=[120, 80, 5], expected_features=[("hole", 4)]),
    BenchmarkPrompt("C27", "C", "A 100 x 50 x 8 mm plate with a centered 30 x 8 mm through-slot.",
                     expected_dimensions_mm=[100, 50, 8], expected_features=[("slot", 1)]),
    BenchmarkPrompt("C28", "C", "A cylinder 40mm diameter, 50mm tall, with a 12 mm blind hole, 20mm deep.",
                     expected_dimensions_mm=[40, 50], expected_features=[("hole", 1)],
                     not_automatically_measurable=["exact blind depth -- only through-vs-blind is independently checked, not the precise 20mm depth"]),
    BenchmarkPrompt("C29", "C", "A plate with six equally spaced 5 mm holes in one straight line.", expected_features=[("hole", 6)]),
    BenchmarkPrompt("C30", "C", "A circular plate with eight equally spaced mounting holes.", expected_features=[("hole", 8)]),
    BenchmarkPrompt("C31", "C", "A block with two counterbored holes.", expected_features=[("hole", 2)],
                     not_automatically_measurable=["counterbore shoulder geometry specifically, vs a plain hole"]),
    BenchmarkPrompt("C32", "C", "A shaft with a longitudinal keyway.", not_automatically_measurable=["keyway presence -- no independent keyway detector; this project's own tool roster has no dedicated keyway cutter either"]),
    BenchmarkPrompt("C33", "C", "A plate with two parallel elongated slots.", expected_features=[("slot", 2)]),
    BenchmarkPrompt("C34", "C", "A flange with four holes on a specified 50mm pitch-circle diameter.", expected_dimensions_mm=[50], expected_features=[("hole", 4)]),
    BenchmarkPrompt("C35", "C", "A block with three pockets at specified positions.", not_automatically_measurable=["pocket presence/count/position -- no independent pocket detector"]),

    # ---------------- Category D: brackets, ribs, and mounts ----------------
    BenchmarkPrompt("D36", "D", "An L-bracket with a 30 mm triangular gusset, 4 mm thick.", expected_dimensions_mm=[30, 4],
                     not_automatically_measurable=["gusset presence -- feature_verification.py has no independent rib/gusset geometric check (disclosed limitation from the prior Feature Ledger audit)"]),
    BenchmarkPrompt("D37", "D", "A mounting plate with four cylindrical standoffs.", not_automatically_measurable=["standoff count/presence -- standoffs are a protrusion feature with no independent detector"]),
    BenchmarkPrompt("D38", "D", "A support bracket with two triangular ribs.", not_automatically_measurable=["rib count/presence -- same disclosed gap as D36"]),
    BenchmarkPrompt("D39", "D", "A motor mount with four mounting holes and a central opening.", expected_features=[("hole", 5)]),
    BenchmarkPrompt("D40", "D", "A rectangular frame with four corner supports.", not_automatically_measurable=["corner-support count/presence"]),
    BenchmarkPrompt("D41", "D", "A vertical plate supported by two gussets.", not_automatically_measurable=["gusset count/presence"]),
    BenchmarkPrompt("D42", "D", "A flange with a raised cylindrical boss.", not_automatically_measurable=["boss presence"]),
    BenchmarkPrompt("D43", "D", "A bracket with two mounting tabs and a connecting web.", not_automatically_measurable=["tab count/presence"]),
    BenchmarkPrompt("D44", "D", "A base plate with three support columns.", not_automatically_measurable=["column count/presence"]),
    BenchmarkPrompt("D45", "D", "A U-bracket with specified hole positions on both arms: two 6mm holes on each arm.",
                     expected_features=[("hole", 4)]),

    # ---------------- Category E: multi-component assemblies ----------------
    BenchmarkPrompt("E46", "E", "Two identical L-brackets connected by a 100 mm crossbar.", expected_component_count=3, expected_dimensions_mm=[100],
                     decomposition=[
                         {"description": "an L-shape bracket, 60mm x 20mm x 5mm"},
                         {"description": "an L-shape bracket, 60mm x 20mm x 5mm"},
                         {"description": "a crossbar, 100mm x 10mm x 5mm"},
                     ],
                     notes="3 physical parts; expected_component_count reflects the 2 DISTINCT bracket instances the Component Inventory should recognize as one type requested twice, plus the crossbar as a separate type"),
    BenchmarkPrompt("E47", "E", "A shaft inserted into two bearing housings.", expected_component_count=3, expected_relationship_count=2,
                     decomposition=[
                         {"description": "a bearing housing, 40mm x 40mm x 20mm, with a 12.2mm bore"},
                         {"description": "a bearing housing, 40mm x 40mm x 20mm, with a 12.2mm bore"},
                         {"description": "a shaft, 12mm diameter, 150mm long", "relationship": {"kind": "concentric_with", "target_id": "p0", "axis": "x"}},
                     ]),
    BenchmarkPrompt("E48", "E", "Two external spur gears, module 2.5mm, gear A 20 teeth, gear B 32 teeth, on parallel shafts along the x axis.",
                     expected_component_count=2, expected_relationship_count=0,
                     decomposition=[
                         {"description": "a gear, module 2.5, 20 teeth"},
                         {"description": "a gear, module 2.5, 32 teeth"},
                     ],
                     notes="gear-pair spacing correction is applied automatically to any 2-gear point set (llm_decompose._correct_gear_mesh_spacing), not via a declared relationship"),
    BenchmarkPrompt("E49", "E", "A motor housing attached to a mounting plate using four specified mounting points: 42mm x 42mm bolt pattern.",
                     expected_component_count=2, expected_relationship_count=1, expected_features=[("hole", 4)],
                     decomposition=[
                         {"description": "a mounting plate, 100mm x 100mm x 5mm with a corner array of four 3mm diameter holes, each 21mm from a corner"},
                         {"description": "a motor housing, 42mm x 42mm x 30mm", "relationship": {"kind": "attached_to_face", "target_id": "p0", "face": "+z"}},
                     ]),
    BenchmarkPrompt("E50", "E", "A pulley mounted on a stepped shaft.", expected_component_count=2, expected_relationship_count=1,
                     decomposition=[
                         {"description": "a shaft, 10mm diameter, 100mm long"},
                         {"description": "a pulley, 60mm outer diameter, 10mm bore, 15mm thick", "relationship": {"kind": "concentric_with", "target_id": "p0", "axis": "x"}},
                     ],
                     notes="'stepped shaft' simplified to a plain shaft in this fixed decomposition -- disclosed, see expected_features"),
    BenchmarkPrompt("E51", "E", "A hinge assembly with two leaves and a connecting pin.", expected_component_count=3, expected_relationship_count=1,
                     decomposition=[
                         {"description": "a hinge leaf, 60mm x 20mm x 3mm"},
                         {"description": "a hinge leaf, 60mm x 20mm x 3mm"},
                         {"description": "a cylindrical hinge pin, 5mm diameter, 25mm long", "relationship": {"kind": "concentric_with", "target_id": "p0", "axis": "y"}},
                     ]),
    BenchmarkPrompt("E52", "E", "A frame with four legs and two crossbars.", expected_component_count=6, expected_relationship_count=5,
                     decomposition=[
                         {"description": "a leg, 10mm x 10mm x 200mm"},
                         {"description": "a leg, 10mm x 10mm x 200mm"},
                         {"description": "a leg, 10mm x 10mm x 200mm"},
                         {"description": "a leg, 10mm x 10mm x 200mm"},
                         {"description": "a crossbar, 300mm x 10mm x 10mm", "relationship": {"kind": "attached_to_face", "target_id": "p0", "face": "+z"}},
                         {"description": "a crossbar, 300mm x 10mm x 10mm", "relationship": {"kind": "attached_to_face", "target_id": "p2", "face": "+z"}},
                     ]),
    BenchmarkPrompt("E53", "E", "A coupling with two flanges connected by a central shaft.", expected_component_count=3, expected_relationship_count=2,
                     decomposition=[
                         {"description": "a flanged coupling disc, 50mm diameter, 10mm thick with a 10mm diameter central bore", "id": "p0"},
                         {"description": "a flanged coupling disc, 50mm diameter, 10mm thick with a 10mm diameter central bore", "id": "p1"},
                         {"description": "a shaft, 10mm diameter, 80mm long", "relationship": {"kind": "concentric_with", "target_id": "p0", "axis": "x"}},
                     ]),
    BenchmarkPrompt("E54", "E", "A bearing, shaft, and retaining collar assembly.", expected_component_count=3, expected_relationship_count=2,
                     decomposition=[
                         {"description": "a bearing housing, 40mm x 40mm x 20mm, with a 12.2mm bore"},
                         {"description": "a shaft, 12mm diameter, 100mm long", "relationship": {"kind": "concentric_with", "target_id": "p0", "axis": "x"}},
                         {"description": "a shaft collar, 25mm outer diameter, 12.2mm bore, 10mm thick with a radial set-screw hole", "relationship": {"kind": "concentric_with", "target_id": "p0", "axis": "x", "axis_offset": 30}},
                     ]),
    BenchmarkPrompt("E55", "E", "A simple gearbox with two gears, two shafts, and a housing.", expected_component_count=5, expected_relationship_count=2,
                     decomposition=[
                         {"description": "a gearbox housing, 120mm x 80mm x 60mm, with a corner array of four 6.6mm diameter holes, each 15mm from a corner"},
                         {"description": "a shaft, 10mm diameter, 70mm long"},
                         {"description": "a gear, module 2.5, 24 teeth", "relationship": {"kind": "concentric_with", "target_id": "p1", "axis": "y"}},
                         {"description": "a shaft, 10mm diameter, 70mm long"},
                         {"description": "a gear, module 2.5, 36 teeth", "relationship": {"kind": "concentric_with", "target_id": "p3", "axis": "y"}},
                     ]),

    # ---------------- Category F: complex mechanical designs ----------------
    BenchmarkPrompt("F56", "F", "A compact screw-driven linear slide: a 200mm x 60mm x 10mm base, two 200mm x 10mm x 15mm guide rails, a 40mm x 40mm x 15mm carriage, and a 10mm diameter, 220mm long lead screw.",
                     expected_component_count=4, expected_relationship_count=3,
                     decomposition=[
                         {"description": "a base plate, 200mm x 60mm x 10mm"},
                         {"description": "a guide rail, 200mm x 10mm x 15mm", "relationship": {"kind": "attached_to_face", "target_id": "p0", "face": "+z", "offset_v": 20}},
                         {"description": "a guide rail, 200mm x 10mm x 15mm", "relationship": {"kind": "attached_to_face", "target_id": "p0", "face": "+z", "offset_v": -20}},
                         {"description": "a lead screw, 10mm diameter, 220mm long", "relationship": {"kind": "attached_to_face", "target_id": "p0", "face": "+z"}},
                     ],
                     not_automatically_measurable=["carriage omitted from this fixed decomposition for geometric simplicity -- disclosed, not hidden"]),
    BenchmarkPrompt("F57", "F", "A small belt tensioner: a 60mm x 40mm x 10mm mounting bracket, a 50mm outer diameter pulley, and a 80mm x 15mm x 5mm tensioning arm.",
                     expected_component_count=3, expected_relationship_count=2,
                     decomposition=[
                         {"description": "a mounting bracket, 60mm x 40mm x 10mm"},
                         {"description": "a tensioning arm, 80mm x 15mm x 5mm", "relationship": {"kind": "attached_to_face", "target_id": "p0", "face": "+z"}},
                         {"description": "a pulley, 50mm outer diameter, 8mm bore, 10mm thick", "relationship": {"kind": "concentric_with", "target_id": "p1", "axis": "z"}},
                     ]),
    BenchmarkPrompt("F58", "F", "A two-stage shaft support: two bearing housings, two spacers, and a coupling, all on one 10mm diameter shaft.",
                     expected_component_count=5, expected_relationship_count=4,
                     decomposition=[
                         {"description": "a bearing housing, 40mm x 40mm x 20mm, with a 10.2mm bore"},
                         {"description": "a bearing housing, 40mm x 40mm x 20mm, with a 10.2mm bore"},
                         {"description": "a spacer, 20mm outer diameter, 10mm inner diameter, 15mm length", "relationship": {"kind": "concentric_with", "target_id": "p0", "axis": "x"}},
                         {"description": "a spacer, 20mm outer diameter, 10mm inner diameter, 15mm length", "relationship": {"kind": "concentric_with", "target_id": "p1", "axis": "x"}},
                         {"description": "a flanged coupling disc, 40mm diameter, 8mm thick with a 10mm diameter central bore", "relationship": {"kind": "concentric_with", "target_id": "p0", "axis": "x"}},
                     ],
                     not_automatically_measurable=["the single shared shaft running through all 5 components is omitted from this fixed decomposition -- disclosed, not hidden"]),
    BenchmarkPrompt("F59", "F", "A compact adjustable camera mounting bracket: a 60mm x 60mm x 8mm base, a 80mm tall x 15mm diameter vertical support post, and a pivot joint.",
                     expected_component_count=3, expected_relationship_count=2,
                     decomposition=[
                         {"description": "a base plate, 60mm x 60mm x 8mm"},
                         {"description": "a cylindrical support post, 15mm diameter, 80mm long", "relationship": {"kind": "attached_to_face", "target_id": "p0", "face": "+z"}},
                         {"description": "a pivot bracket, 30mm x 20mm x 10mm with a 6mm diameter hole", "relationship": {"kind": "attached_to_face", "target_id": "p1", "face": "+z"}},
                     ],
                     not_automatically_measurable=["pivot joint's own rotational function -- only static geometry/placement is checked"]),
    BenchmarkPrompt("F60", "F", "A small mechanical clamp: a 80mm x 30mm x 15mm base, a 60mm x 20mm x 15mm moving jaw, a 8mm diameter x 60mm long screw, and a 30mm diameter x 10mm thick handle.",
                     expected_component_count=4, expected_relationship_count=3,
                     decomposition=[
                         {"description": "a clamp base, 80mm x 30mm x 15mm"},
                         {"description": "a moving jaw, 60mm x 20mm x 15mm", "relationship": {"kind": "attached_to_face", "target_id": "p0", "face": "+x"}},
                         {"description": "a clamp screw, 8mm diameter, 60mm long", "relationship": {"kind": "concentric_with", "target_id": "p0", "axis": "x"}},
                         {"description": "a cylindrical handle, 30mm diameter, 10mm thick", "relationship": {"kind": "concentric_with", "target_id": "p2", "axis": "x", "axis_offset": 35}},
                     ]),
]

assert len(PROMPTS) == 60, f"expected 60 benchmark prompts, got {len(PROMPTS)}"
assert len({p.id for p in PROMPTS}) == 60, "duplicate benchmark prompt id"
