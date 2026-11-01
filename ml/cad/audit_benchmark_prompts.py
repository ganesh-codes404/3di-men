"""Part 7 of the multi-part assembly engineering audit -- 24 realistic
assembly prompts across the 5 required categories (A-E), each with
explicit, independently-authored ground truth. These ground-truth values
are NEVER fed into the generation system (see audit_benchmark_runner.py)
-- they exist only for this audit's own post-hoc scoring.

New audit-only artifact, not production code: imported only by
audit_benchmark_runner.py, never by server.py/parallel_builder.py/
llm_decompose.py or anything else the real build pipeline touches.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class AuditPrompt:
    id: str
    category: str  # "A".."E"
    prompt: str
    expected_component_count: int
    expected_identities: list[str]  # one entry per expected component, e.g. ["plate", "plate"]
    expected_dimensions_mm: dict[str, list[float]] = field(default_factory=dict)  # identity -> claimed numeric dims
    expected_features: list[tuple[str, int]] = field(default_factory=list)  # (feature_type, count)
    expected_relationships: list[str] = field(default_factory=list)  # human-readable relationship claims
    expected_placement_constraints: list[str] = field(default_factory=list)
    safely_inferable: list[str] = field(default_factory=list)  # details with a reasonable default
    needs_clarification: list[str] = field(default_factory=list)  # details genuinely ambiguous/missing
    notes: str = ""


PROMPTS: list[AuditPrompt] = [
    # ---------------- A. Simple assemblies ----------------
    AuditPrompt(
        id="A01", category="A",
        prompt="An L-shaped bracket made from two flat plates joined at 90 degrees, each plate 60mm x 40mm x 5mm.",
        expected_component_count=2, expected_identities=["plate", "plate"],
        expected_dimensions_mm={"plate": [60.0, 40.0, 5.0]},
        expected_relationships=["plate_b attached_to_face of plate_a, perpendicular (90 degree) join"],
        expected_placement_constraints=["the two plates must share a common edge, not float apart or overlap"],
        safely_inferable=["which face the second plate attaches to (an edge face, since '90 degrees' implies perpendicular)"],
        needs_clarification=["whether the join is itself reinforced (a gusset) -- not stated, should NOT be assumed"],
    ),
    AuditPrompt(
        id="A02", category="A",
        prompt="A 100mm x 100mm x 6mm mounting plate with four cylindrical standoffs, each 8mm diameter and 20mm tall, one at each corner, 10mm in from each edge.",
        expected_component_count=5, expected_identities=["plate", "standoff", "standoff", "standoff", "standoff"],
        expected_dimensions_mm={"plate": [100.0, 100.0, 6.0], "standoff": [8.0, 20.0]},
        expected_features=[], expected_relationships=["each standoff attached_to_face (+z) of the plate, at a corner offset"],
        expected_placement_constraints=["standoffs at the 4 corners, each 10mm in from both edges"],
        safely_inferable=["standoffs are solid cylinders (no stated hole)"],
        needs_clarification=[],
    ),
    AuditPrompt(
        id="A03", category="A",
        prompt="Two 80mm x 50mm x 4mm plates joined by four M4 bolts through aligned corner holes.",
        expected_component_count=2, expected_identities=["plate", "plate"],
        expected_dimensions_mm={"plate": [80.0, 50.0, 4.0]},
        expected_features=[("hole", 4)],  # per plate -- 4 aligned clearance holes
        expected_relationships=["the two plates' hole patterns must align (face contact, coaxial holes)"],
        expected_placement_constraints=["plates stacked face-to-face with holes coaxial"],
        safely_inferable=["M4 clearance diameter is 4.5mm (standard clearance conversion)", "hole positions are a standard corner pattern"],
        needs_clarification=["exact corner offset distance for the holes -- not stated"],
        notes="Bolts themselves are fasteners, not modeled as separate solid parts by this pipeline (no fastener library) -- component count counts the 2 plates only; this is a known, disclosed scope limit, not a defect to penalize.",
    ),
    AuditPrompt(
        id="A04", category="A",
        prompt="A 10mm diameter, 150mm long shaft passing through a single ball bearing with a 10mm bore and 26mm outer diameter.",
        expected_component_count=2, expected_identities=["shaft", "bearing"],
        expected_dimensions_mm={"shaft": [10.0, 150.0], "bearing": [10.0, 26.0]},
        expected_relationships=["shaft concentric_with bearing's bore"],
        expected_placement_constraints=["shaft axis coincides with bearing bore axis"],
        safely_inferable=["bearing width/thickness (not stated -- a reasonable standard-ish value, e.g. 8mm, is a safe default)"],
        needs_clarification=["where along the 150mm shaft the bearing sits -- not stated"],
    ),

    # ---------------- B. Mechanical fits ----------------
    AuditPrompt(
        id="B05", category="B",
        prompt="A 12mm diameter, 200mm long shaft running through two bearing housings, each a 50mm x 50mm x 20mm block with a 12.2mm bore, positioned at opposite ends of the shaft.",
        expected_component_count=3, expected_identities=["shaft", "housing", "housing"],
        expected_dimensions_mm={"shaft": [12.0, 200.0], "housing": [50.0, 50.0, 20.0]},
        expected_features=[("hole", 1)],  # per housing, the 12.2mm bore
        expected_relationships=["shaft concentric_with housing_a's bore", "shaft concentric_with housing_b's bore", "diametral clearance 12.2mm bore vs 12mm shaft = 0.2mm, a real clearance fit"],
        expected_placement_constraints=["both housings coaxial with the shaft, at opposite ends (near the shaft's own two ends)"],
        safely_inferable=[],
        needs_clarification=["the exact spacing between the two housings along the 200mm shaft -- 'opposite ends' is directional, not a precise number"],
    ),
    AuditPrompt(
        id="B06", category="B",
        prompt="A pulley, 60mm outer diameter and 15mm thick, with a 10mm bore, mounted on a 10mm diameter, 80mm long shaft.",
        expected_component_count=2, expected_identities=["pulley", "shaft"],
        expected_dimensions_mm={"pulley": [60.0, 15.0, 10.0], "shaft": [10.0, 80.0]},
        expected_relationships=["pulley concentric_with shaft, 10mm bore vs 10mm shaft = line-to-line/press fit"],
        expected_placement_constraints=["pulley coaxial with shaft"],
        safely_inferable=["pulley position along the 80mm shaft (not stated -- centered is a reasonable default)"],
        needs_clarification=["exact axial position of the pulley on the shaft"],
    ),
    AuditPrompt(
        id="B07", category="B",
        prompt="A NEMA 17 stepper motor, 42mm x 42mm x 40mm body, mounted on a 60mm x 60mm x 5mm plate via its standard 4-hole bolt pattern, with a central 22mm diameter clearance bore in the plate for the motor shaft.",
        expected_component_count=2, expected_identities=["motor", "plate"],
        expected_dimensions_mm={"motor": [42.0, 42.0, 40.0], "plate": [60.0, 60.0, 5.0]},
        expected_features=[("hole", 4), ("hole", 1)],  # 4 mounting holes + 1 central bore, on the plate
        expected_relationships=["motor attached_to_face of plate, mounting-hole pattern aligned, motor shaft coaxial with the plate's central bore"],
        expected_placement_constraints=["motor's 4-hole bolt pattern must align with 4 matching holes in the plate"],
        safely_inferable=["NEMA 17's own standard 31mm bolt-circle spacing / M3 hole size, since no conflicting explicit number is given"],
        needs_clarification=[],
        notes="Tests the exact 'NEMA 17 numbers != component count' confusion case the audit brief explicitly calls out.",
    ),
    AuditPrompt(
        id="B08", category="B",
        prompt="A deep-groove ball bearing, 15mm bore and 35mm outer diameter, pressed into a housing bore of 35.0mm diameter in a 60mm x 60mm x 25mm block.",
        expected_component_count=2, expected_identities=["bearing", "housing"],
        expected_dimensions_mm={"bearing": [15.0, 35.0], "housing": [60.0, 60.0, 25.0]},
        expected_features=[("hole", 1)],
        expected_relationships=["bearing concentric_with housing's bore, 35.0mm bore vs 35mm bearing OD -- a line-to-line/interference condition, not a clearance fit"],
        expected_placement_constraints=["bearing coaxial with the housing bore"],
        safely_inferable=["bearing width (not stated)"],
        needs_clarification=[],
        notes="The 35.0mm-vs-35mm pairing is deliberately a near-zero-clearance press fit -- a genuine mechanical-compatibility question, not an ambiguity.",
    ),

    # ---------------- C. Mechanical relationships ----------------
    AuditPrompt(
        id="C09", category="C",
        prompt="Two external spur gears meshing: gear A has 18 teeth, gear B has 30 teeth, both module 2mm, each mounted on its own 8mm diameter, 50mm long shaft.",
        expected_component_count=4, expected_identities=["gear", "shaft", "gear", "shaft"],
        expected_dimensions_mm={"shaft": [8.0, 50.0]},
        expected_features=[],
        expected_relationships=[
            "gear_a concentric_with shaft_a; gear_b concentric_with shaft_b",
            "gear_a and gear_b mesh -- expected center distance = module * (teeth_a + teeth_b) / 2 = 2 * (18 + 30) / 2 = 48mm",
        ],
        expected_placement_constraints=["shaft_a and shaft_b parallel, separated by exactly 48mm center-to-center"],
        safely_inferable=["gear face width/bore size (not stated, reasonable defaults)"],
        needs_clarification=[],
        notes="This is the EXACT center-distance formula the audit brief requires verifying: module*(teeth_A+teeth_B)/2.",
    ),
    AuditPrompt(
        id="C10", category="C",
        prompt="A stepped shaft: 20mm diameter for the first 40mm, then stepping down to 14mm diameter for the remaining 60mm, with a ball bearing (14mm bore, 32mm OD) mounted on the smaller-diameter section.",
        expected_component_count=2, expected_identities=["shaft", "bearing"],
        expected_dimensions_mm={"shaft": [20.0, 40.0, 14.0, 60.0], "bearing": [14.0, 32.0]},
        expected_relationships=["bearing concentric_with the shaft's 14mm-diameter section specifically (not the 20mm section)"],
        expected_placement_constraints=["bearing's own bore axis matches the shaft's TRUE built orientation on the stepped (smaller) section, not just nominally 'somewhere on the shaft'"],
        safely_inferable=[],
        needs_clarification=["which end of the shaft the stepped-down section is at -- not stated"],
        notes="Directly probes the confirmed historical coordinate-frame class of bug: does the bearing's own placement/measurement correctly track the SMALLER section's true diameter and position, not the shaft's overall/first-section diameter.",
    ),
    AuditPrompt(
        id="C11", category="C",
        prompt="A pulley with a 12mm bore mounted on a 12mm diameter, 100mm long shaft, with the pulley positioned 30mm from one end of the shaft.",
        expected_component_count=2, expected_identities=["pulley", "shaft"],
        expected_dimensions_mm={"shaft": [12.0, 100.0], "pulley": [None, None]},  # OD/thickness not stated
        expected_relationships=["pulley concentric_with shaft, 12mm bore vs 12mm shaft = line-to-line fit"],
        expected_placement_constraints=["pulley centered 30mm from one shaft end along the shaft axis"],
        safely_inferable=["pulley OD/thickness (not stated at all)"],
        needs_clarification=["pulley outer diameter and thickness are never given -- this should be flagged, not silently invented with an arbitrary number"],
    ),
    AuditPrompt(
        id="C12", category="C",
        prompt="Two coaxial 15mm diameter shafts, one 80mm long and one 60mm long, connected end-to-end by a cylindrical coupling 30mm diameter and 40mm long with a 15mm bore through each end.",
        expected_component_count=3, expected_identities=["shaft", "shaft", "coupling"],
        expected_dimensions_mm={"shaft": [15.0], "coupling": [30.0, 40.0, 15.0]},
        expected_relationships=["shaft_a concentric_with coupling (one end); shaft_b concentric_with coupling (other end); all three share one common axis"],
        expected_placement_constraints=["both shafts and the coupling must be truly coaxial -- a 3-way chain, not just 2 pairwise relationships that happen to look aligned"],
        safely_inferable=[],
        needs_clarification=[],
        notes="A 3-body coaxial chain is a genuine stress test for whether axis alignment is actually transitive in this pipeline, not just pairwise-correct.",
    ),

    # ---------------- D. Complex assemblies ----------------
    AuditPrompt(
        id="D13", category="D",
        prompt="A small gearbox: a housing 120mm x 80mm x 60mm, two spur gears (module 2.5mm, 16 and 28 teeth) each mounted on its own 10mm diameter, 70mm long shaft, with each shaft running through a bearing boss on the housing.",
        expected_component_count=6,
        expected_identities=["housing", "shaft", "gear", "shaft", "gear", "boss (x2, may be modeled separately or as housing features)"],
        expected_dimensions_mm={"housing": [120.0, 80.0, 60.0], "shaft": [10.0, 70.0]},
        expected_relationships=[
            "each shaft concentric_with its own bearing boss on the housing",
            "gear_a concentric_with shaft_a; gear_b concentric_with shaft_b",
            "gear_a meshes gear_b: expected center distance = 2.5*(16+28)/2 = 55mm",
        ],
        expected_placement_constraints=["both shafts parallel, 55mm apart, each passing through its own housing boss"],
        safely_inferable=["bearing boss dimensions (not stated numerically)"],
        needs_clarification=[],
        notes="Deliberately the most complex single prompt in this benchmark -- combines component decomposition, 2 relationship types, and the gear-mesh formula all at once, closely mirroring the audit brief's own 'gearbox' example.",
    ),
    AuditPrompt(
        id="D14", category="D",
        prompt="An assembly of a NEMA 17 motor, a cylindrical coupling (20mm diameter, 25mm long, 5mm bore each end), a 5mm diameter 60mm long shaft, and a single ball bearing (5mm bore, 16mm OD) supporting the far end of the shaft.",
        expected_component_count=4, expected_identities=["motor", "coupling", "shaft", "bearing"],
        expected_dimensions_mm={"coupling": [20.0, 25.0, 5.0], "shaft": [5.0, 60.0], "bearing": [5.0, 16.0]},
        expected_relationships=[
            "coupling concentric_with motor's own output shaft",
            "shaft concentric_with coupling (other end)",
            "bearing concentric_with shaft, at the far end from the motor",
        ],
        expected_placement_constraints=["motor, coupling, shaft, bearing all share one common axis, in that linear order"],
        safely_inferable=["motor body dimensions default to NEMA 17 standard (42.3mm square) since nothing conflicting is stated"],
        needs_clarification=["motor's own output-shaft diameter is not stated -- needed to confirm it matches the coupling's own bore"],
    ),
    AuditPrompt(
        id="D15", category="D",
        prompt="A belt-driven pulley assembly: a 40mm diameter drive pulley and an 80mm diameter driven pulley, each with a 10mm bore, mounted on separate parallel 10mm diameter shafts spaced 150mm apart center-to-center.",
        expected_component_count=4, expected_identities=["pulley", "shaft", "pulley", "shaft"],
        expected_dimensions_mm={"pulley": [40.0, 80.0], "shaft": [10.0]},
        expected_relationships=["each pulley concentric_with its own shaft", "the two shafts parallel, 150mm apart"],
        expected_placement_constraints=["both shafts parallel at exactly 150mm center distance -- an EXPLICIT, stated spacing, unlike the gear-mesh cases where spacing is derived"],
        safely_inferable=["pulley thickness/shaft length (not stated)"],
        needs_clarification=[],
        notes="Pulleys do not mesh like gears (a belt connects them, with no rigid center-distance formula) -- the 150mm figure is a literal, explicit requirement, and this prompt tests whether the system can tell that apart from a gear-mesh case and NOT silently apply gear-center-distance correction to an unrelated pulley pair.",
    ),
    AuditPrompt(
        id="D16", category="D",
        prompt="An adjustable mounting bracket: a 100mm x 40mm x 6mm base plate with a 60mm x 40mm x 6mm vertical plate attached to one end at 90 degrees, reinforced by a triangular gusset on each side of the joint, each gusset 30mm x 30mm x 4mm.",
        expected_component_count=4, expected_identities=["plate", "plate", "gusset", "gusset"],
        expected_dimensions_mm={"plate": [100.0, 40.0, 6.0], "vertical_plate": [60.0, 40.0, 6.0], "gusset": [30.0, 30.0, 4.0]},
        expected_relationships=["vertical plate attached_to_face of base plate", "each gusset attached_to_face at the same joint, on opposite sides (not stacked on top of each other)"],
        expected_placement_constraints=["the two gussets must be at OPPOSITE offsets along the shared edge, not identical/overlapping positions"],
        safely_inferable=[],
        needs_clarification=[],
        notes="Directly matches llm_decompose.py's own worked Example 6 pattern -- a strong prior the system SHOULD handle this correctly; a good test of whether that worked example generalizes beyond its exact wording.",
    ),

    # ---------------- E. Ambiguous or incomplete specifications ----------------
    AuditPrompt(
        id="E17", category="E",
        prompt="A motor mount plate for a small stepper motor.",
        expected_component_count=1, expected_identities=["plate"],
        expected_dimensions_mm={},
        expected_features=[("hole", None)],  # count genuinely unknown
        expected_relationships=[],
        expected_placement_constraints=[],
        safely_inferable=[],
        needs_clarification=["plate size", "mounting hole pattern/count/size", "which motor standard (if any)", "central shaft clearance bore size"],
        notes="Maximally underspecified on purpose -- the correct behavior is either a clarifying question or a clearly-disclosed assumption, never a silently invented bolt pattern presented as fact.",
    ),
    AuditPrompt(
        id="E18", category="E",
        prompt="A shaft assembly with two bearings supporting it.",
        expected_component_count=3, expected_identities=["shaft", "bearing", "bearing"],
        expected_dimensions_mm={},
        expected_relationships=["shaft concentric_with each bearing"],
        expected_placement_constraints=["bearing spacing along the shaft is NOT given"],
        safely_inferable=["shaft/bearing diameters do NOT have a safe default -- these are custom-sized parts with no standard to fall back on"],
        needs_clarification=["shaft diameter and length", "bearing bore/OD", "spacing between the two bearings"],
        notes="No numbers at all -- tests whether the system invents plausible-looking but fabricated numbers instead of disclosing the gap.",
    ),
    AuditPrompt(
        id="E19", category="E",
        prompt="Two meshing gears, module 2mm, one with 20 teeth.",
        expected_component_count=2, expected_identities=["gear", "gear"],
        expected_dimensions_mm={},
        expected_relationships=["gear_a meshes gear_b -- center distance formula needs BOTH tooth counts, and only one is given"],
        expected_placement_constraints=[],
        safely_inferable=[],
        needs_clarification=["the second gear's own tooth count is never stated -- the center-distance formula is literally unevaluable without it"],
        notes="A genuinely UNSUPPORTED/blocked case by construction, not just vague -- there is no reasonable default for a specific missing tooth count.",
    ),
    AuditPrompt(
        id="E20", category="E",
        prompt="A 60mm diameter, 100mm tall cylinder with an 80mm diameter counterbore hole through the center.",
        expected_component_count=1, expected_identities=["cylinder"],
        expected_dimensions_mm={"cylinder": [60.0, 100.0]},
        expected_features=[("hole", 1)],
        expected_relationships=[],
        expected_placement_constraints=[],
        safely_inferable=[],
        needs_clarification=["the stated hole (80mm) is LARGER than the cylinder's own diameter (60mm) -- an internally conflicting, physically impossible spec that should be flagged, never silently built as some unspecified resolution of the conflict"],
        notes="A genuine internal-conflict case (not just missing information) -- tests whether a physically impossible claim is caught at all, versus silently 'resolved' by clamping/guessing without disclosure.",
    ),
]

assert len(PROMPTS) == 20, f"expected exactly 20 prompts, got {len(PROMPTS)}"
assert len({p.id for p in PROMPTS}) == len(PROMPTS), "duplicate prompt ids"
