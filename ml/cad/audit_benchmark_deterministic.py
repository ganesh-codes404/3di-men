"""Part 8 of the audit, deterministic half: hand-authored ActionablePoint
decompositions (one per prompt in audit_benchmark_prompts.py), matching
THIS audit's own ground truth, fed directly to parallel_builder.
build_assembly_from_points -- bypassing llm_decompose.decompose_prompt
entirely. This isolates "is per-component geometry generation + placement
+ relationship resolution + validation correct" from "does the LLM
decompose the prompt correctly" (confirmed, separately, to be impractical
to benchmark at scale on this machine -- see the audit report's own Part 8
notes on measured LLM throughput).

This is NOT the production decomposition -- it is this audit's own
independently-authored best-effort translation of each prompt into the
same ActionablePoint/Relationship structures the real LLM is supposed to
produce, used only to exercise the REST of the real, unmodified pipeline.
New audit-only artifact, not production code.
"""

from __future__ import annotations

from llm_decompose import ActionablePoint
from assembly_plan import Relationship

# Each entry: prompt_id -> list[ActionablePoint], written independently from
# the ground truth in audit_benchmark_prompts.py (not derived from it
# programmatically, so an error in one isn't mechanically copied into the
# other).

DECOMPOSITIONS: dict[str, list[ActionablePoint]] = {
    "A01": [
        ActionablePoint(id="plate_a", description="a box 60mm x 40mm x 5mm", position_mm=(0, 0, 0)),
        ActionablePoint(id="plate_b", description="a box 60mm x 40mm x 5mm", position_mm=(0, 0, 0),
                         relationship=Relationship(kind="attached_to_face", target_id="plate_a", face="+y")),
    ],
    "A02": [
        ActionablePoint(id="plate", description="a box 100mm x 100mm x 6mm", position_mm=(0, 0, 0)),
        *[
            ActionablePoint(id=f"standoff_{i}", description="a cylinder 8mm diameter, 20mm tall", position_mm=(0, 0, 0),
                             relationship=Relationship(kind="attached_to_face", target_id="plate", face="+z", offset_u=ou, offset_v=ov))
            for i, (ou, ov) in enumerate([(40, 40), (-40, 40), (40, -40), (-40, -40)])
        ],
    ],
    "A03": [
        ActionablePoint(id="plate_a", description="a box 80mm x 50mm x 4mm with a corner array of four 4.5mm diameter holes, each 10mm from a corner", position_mm=(0, 0, 0)),
        ActionablePoint(id="plate_b", description="a box 80mm x 50mm x 4mm with a corner array of four 4.5mm diameter holes, each 10mm from a corner", position_mm=(0, 0, 0),
                         relationship=Relationship(kind="attached_to_face", target_id="plate_a", face="+z")),
    ],
    "A04": [
        ActionablePoint(id="shaft", description="a cylinder 10mm diameter, 150mm long", position_mm=(0, 0, 0)),
        ActionablePoint(id="bearing", description="a flat ring 26mm outer diameter, 10mm inner diameter, 8mm tall", position_mm=(0, 0, 0),
                         relationship=Relationship(kind="concentric_with", target_id="shaft", axis="z")),
    ],
    "B05": [
        ActionablePoint(id="shaft", description="a cylinder 12mm diameter, 200mm long", position_mm=(0, 0, 0)),
        ActionablePoint(id="housing_a", description="a box 50mm x 50mm x 20mm with a 12.2mm diameter hole through the center", position_mm=(0, 0, 0),
                         relationship=Relationship(kind="concentric_with", target_id="shaft", axis="z", axis_offset=-90)),
        ActionablePoint(id="housing_b", description="a box 50mm x 50mm x 20mm with a 12.2mm diameter hole through the center", position_mm=(0, 0, 0),
                         relationship=Relationship(kind="concentric_with", target_id="shaft", axis="z", axis_offset=90)),
    ],
    "B06": [
        ActionablePoint(id="shaft", description="a cylinder 10mm diameter, 80mm long", position_mm=(0, 0, 0)),
        ActionablePoint(id="pulley", description="a flat ring 60mm outer diameter, 10mm inner diameter, 15mm tall", position_mm=(0, 0, 0),
                         relationship=Relationship(kind="concentric_with", target_id="shaft", axis="z")),
    ],
    "B07": [
        ActionablePoint(id="plate", description="a box 60mm x 60mm x 5mm with a corner array of four 3.4mm diameter holes, each 15.5mm from a corner, and a 22mm diameter hole through the center", position_mm=(0, 0, 0)),
        ActionablePoint(id="motor", description="a box 42mm x 42mm x 40mm", position_mm=(0, 0, 0),
                         relationship=Relationship(kind="attached_to_face", target_id="plate", face="+z")),
    ],
    "B08": [
        ActionablePoint(id="housing", description="a box 60mm x 60mm x 25mm with a 35mm diameter hole through the center", position_mm=(0, 0, 0)),
        ActionablePoint(id="bearing", description="a flat ring 35mm outer diameter, 15mm inner diameter, 10mm tall", position_mm=(0, 0, 0),
                         relationship=Relationship(kind="concentric_with", target_id="housing", axis="z")),
    ],
    "C09": [
        ActionablePoint(id="shaft_a", description="a cylinder 8mm diameter, 50mm long", position_mm=(0, 0, 0)),
        ActionablePoint(id="gear_a", description="a gear profile, module 2, 18 teeth", position_mm=(0, 0, 0),
                         relationship=Relationship(kind="concentric_with", target_id="shaft_a", axis="z")),
        ActionablePoint(id="shaft_b", description="a cylinder 8mm diameter, 50mm long", position_mm=(48, 0, 0)),
        ActionablePoint(id="gear_b", description="a gear profile, module 2, 30 teeth", position_mm=(0, 0, 0),
                         relationship=Relationship(kind="concentric_with", target_id="shaft_b", axis="z")),
    ],
    "C10": [
        ActionablePoint(id="shaft", description="a cylinder 20mm diameter, 40mm tall, stepping down to a 14mm diameter, 60mm tall section", position_mm=(0, 0, 0)),
        ActionablePoint(id="bearing", description="a flat ring 32mm outer diameter, 14mm inner diameter, 10mm tall", position_mm=(0, 0, 70),
                         relationship=Relationship(kind="concentric_with", target_id="shaft", axis="z")),
    ],
    "C11": [
        ActionablePoint(id="shaft", description="a cylinder 12mm diameter, 100mm long", position_mm=(0, 0, 0)),
        ActionablePoint(id="pulley", description="a flat ring 40mm outer diameter, 12mm inner diameter, 15mm tall", position_mm=(0, 0, 0),
                         relationship=Relationship(kind="concentric_with", target_id="shaft", axis="z", axis_offset=-20)),
    ],
    "C12": [
        ActionablePoint(id="shaft_a", description="a cylinder 15mm diameter, 80mm long", position_mm=(0, 0, 0)),
        ActionablePoint(id="coupling", description="a cylinder 30mm diameter, 40mm tall", position_mm=(0, 0, 0),
                         relationship=Relationship(kind="concentric_with", target_id="shaft_a", axis="z", axis_offset=60)),
        ActionablePoint(id="shaft_b", description="a cylinder 15mm diameter, 60mm long", position_mm=(0, 0, 0),
                         relationship=Relationship(kind="concentric_with", target_id="coupling", axis="z", axis_offset=50)),
    ],
    "D13": [
        ActionablePoint(id="housing", description="a box 120mm x 80mm x 60mm", position_mm=(0, 0, 0)),
        ActionablePoint(id="shaft_a", description="a cylinder 10mm diameter, 70mm long", position_mm=(0, 0, 0),
                         relationship=Relationship(kind="attached_to_face", target_id="housing", face="+y")),
        ActionablePoint(id="gear_a", description="a gear profile, module 2.5, 16 teeth", position_mm=(0, 0, 0),
                         relationship=Relationship(kind="concentric_with", target_id="shaft_a", axis="y")),
        ActionablePoint(id="shaft_b", description="a cylinder 10mm diameter, 70mm long", position_mm=(0, 0, 0),
                         relationship=Relationship(kind="parallel_to", target_id="shaft_a", axis="y", lateral_axis="x", lateral_offset=55)),
        ActionablePoint(id="gear_b", description="a gear profile, module 2.5, 28 teeth", position_mm=(0, 0, 0),
                         relationship=Relationship(kind="concentric_with", target_id="shaft_b", axis="y")),
    ],
    "D14": [
        ActionablePoint(id="motor", description="a box 42mm x 42mm x 40mm", position_mm=(0, 0, 0)),
        ActionablePoint(id="coupling", description="a cylinder 20mm diameter, 25mm long", position_mm=(0, 0, 0),
                         relationship=Relationship(kind="attached_to_face", target_id="motor", face="+z")),
        ActionablePoint(id="shaft", description="a cylinder 5mm diameter, 60mm long", position_mm=(0, 0, 0),
                         relationship=Relationship(kind="concentric_with", target_id="coupling", axis="z", axis_offset=20)),
        ActionablePoint(id="bearing", description="a flat ring 16mm outer diameter, 5mm inner diameter, 8mm tall", position_mm=(0, 0, 0),
                         relationship=Relationship(kind="concentric_with", target_id="shaft", axis="z", axis_offset=55)),
    ],
    "D15": [
        ActionablePoint(id="shaft_a", description="a cylinder 10mm diameter, 60mm long", position_mm=(0, 0, 0)),
        ActionablePoint(id="pulley_a", description="a flat ring 40mm outer diameter, 10mm inner diameter, 15mm tall", position_mm=(0, 0, 0),
                         relationship=Relationship(kind="concentric_with", target_id="shaft_a", axis="z")),
        ActionablePoint(id="shaft_b", description="a cylinder 10mm diameter, 60mm long", position_mm=(0, 0, 0),
                         relationship=Relationship(kind="parallel_to", target_id="shaft_a", axis="z", lateral_axis="x", lateral_offset=150)),
        ActionablePoint(id="pulley_b", description="a flat ring 80mm outer diameter, 10mm inner diameter, 15mm tall", position_mm=(0, 0, 0),
                         relationship=Relationship(kind="concentric_with", target_id="shaft_b", axis="z")),
    ],
    "D16": [
        ActionablePoint(id="base", description="a box 100mm x 40mm x 6mm", position_mm=(0, 0, 0)),
        ActionablePoint(id="side", description="a box 60mm x 40mm x 6mm", position_mm=(0, 0, 0),
                         relationship=Relationship(kind="attached_to_face", target_id="base", face="+y")),
        # Confirmed live (this audit's own first attempt used offset_u here and was REJECTED --
        # see the audit report's own Part 4/5 findings): a box's +y face's own u axis spans only
        # the box's 6mm HEIGHT, not its 100mm length -- offset_v is the axis that actually spans
        # the length for this specific face. The u/v-to-world-axis mapping is face-dependent and
        # non-obvious, exactly as assembly_plan.py's own code comments already warn.
        ActionablePoint(id="gusset_left", description="a triangular gusset, 30mm x 30mm, 4mm thick", position_mm=(0, 0, 0),
                         relationship=Relationship(kind="attached_to_face", target_id="base", face="+y", offset_v=-25)),
        ActionablePoint(id="gusset_right", description="a triangular gusset, 30mm x 30mm, 4mm thick", position_mm=(0, 0, 0),
                         relationship=Relationship(kind="attached_to_face", target_id="base", face="+y", offset_v=25)),
    ],
    # E17-E20 are deliberately ambiguous/underspecified in the PROMPT TEXT (that's the point of
    # category E); this audit's own deterministic ground-truth decomposition necessarily has to
    # pick SOME concrete numbers to even run the rest of the pipeline, documented as this audit's
    # own best-effort filling-in, never claimed to be "the one correct interpretation."
    "E17": [
        ActionablePoint(id="plate", description="a box 50mm x 50mm x 5mm with a corner array of four 3.4mm diameter holes, each 10mm from a corner", position_mm=(0, 0, 0)),
    ],
    "E18": [
        ActionablePoint(id="shaft", description="a cylinder 10mm diameter, 120mm long", position_mm=(0, 0, 0)),
        ActionablePoint(id="bearing_a", description="a flat ring 26mm outer diameter, 10mm inner diameter, 8mm tall", position_mm=(0, 0, 0),
                         relationship=Relationship(kind="concentric_with", target_id="shaft", axis="z", axis_offset=-40)),
        ActionablePoint(id="bearing_b", description="a flat ring 26mm outer diameter, 10mm inner diameter, 8mm tall", position_mm=(0, 0, 0),
                         relationship=Relationship(kind="concentric_with", target_id="shaft", axis="z", axis_offset=40)),
    ],
    "E19": [
        # Genuinely unsupported -- second tooth count missing. This audit deliberately does NOT
        # invent one; it feeds exactly what IS stated, to see what the pipeline does with an
        # unevaluable gear-mesh pair (one gear only, no legitimate second point to construct).
        ActionablePoint(id="gear_a", description="a gear profile, module 2, 20 teeth", position_mm=(0, 0, 0)),
    ],
    "E20": [
        # Genuinely conflicting -- 80mm hole in a 60mm cylinder. Fed through AS STATED, deliberately
        # not "fixed," to see whether the pipeline catches the physical impossibility.
        ActionablePoint(id="cylinder", description="a cylinder 60mm diameter, 100mm tall with an 80mm diameter hole through the center", position_mm=(0, 0, 0)),
    ],
}

assert set(DECOMPOSITIONS) == {f"{c}{n:02d}" for c, rng in [("A", range(1, 5)), ("B", range(5, 9)), ("C", range(9, 13)), ("D", range(13, 17)), ("E", range(17, 21))] for n in rng}, \
    "every one of the 20 prompt ids must have a deterministic decomposition"
