"""The adversarial regression matrix (Part 12) for the Component
Inventory -- every test runs the REAL production path
(parallel_builder.build_assembly_from_points with `original_prompt` set,
the same wiring build_assembly_from_prompt uses for every real LLM-
decomposed build) end to end, with REAL geometry inspected afterward,
never a mocked mesh or a hand-asserted string.

`program_override` (via Sandbox(trace=True)) supplies deterministic,
hand-built geometry instead of letting the statistical model guess one
-- these tests are about whether the RECONCILIATION layer correctly
distinguishes a complete component inventory from an incomplete one, not
about re-testing llm_decompose.py's own decomposition accuracy (which
cannot be driven deterministically without the real model -- see
test_llm_decompose.py for that, separately, gated on the model file
actually being present)."""

from pathlib import Path

import pytest

pytest.importorskip("sklearn")

from assembly_plan import Relationship
from component_inventory import ReconciliationStatus
from data.program_dataset import sample_program_dataset, write_program_dataset
from llm_decompose import ActionablePoint
from parallel_builder import build_assembly_from_points
from program_model import ProgramModelConfig, train_program_model
from sandbox import Sandbox


@pytest.fixture(scope="module")
def tiny_bundle_path(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("tiny_program_dataset")
    samples = sample_program_dataset({"plate": 40, "flange": 40, "bracket": 40, "shaft": 40, "enclosure": 40, "gear": 40}, n_generic=40, seed=0)
    write_program_dataset(root, samples, seed=0)
    bundle, _ = train_program_model(root, ProgramModelConfig(seed=0))
    bundle_path = tmp_path_factory.mktemp("bundle") / "bundle.joblib"
    bundle.save(bundle_path)
    return bundle_path


def _program(sb: Sandbox, final_handle: int) -> list[dict]:
    program = [{"op": c.op, "args": dict(c.args)} for c in sb.get_trace()]
    program.append({"op": "finalize", "args": {"target": final_handle}})
    return program


def _box_point(description: str, pid: str, extents=(60, 20, 5), position_mm=(0.0, 0.0, 0.0), relationship=None) -> ActionablePoint:
    sb = Sandbox(trace=True)
    h = sb.add_box(*extents, center=(0, 0, 0))
    return ActionablePoint(description=description, position_mm=position_mm, id=pid, relationship=relationship, program_override=_program(sb, h))


def _inventory_by_id(result):
    return {c.component_id: c for c in result.component_inventory}


# ---------------- Test A: missing repeated component ----------------

def test_a_missing_repeated_component_is_count_mismatch(tiny_bundle_path):
    prompt = "Create two identical L-brackets connected by one 100mm crossbar."
    points = [
        _box_point("an L-shape bracket, 60mm long", "bracket_01"),
        _box_point("a crossbar, 100mm long", "crossbar_01", extents=(100, 10, 5), position_mm=(200, 0, 0)),
    ]
    result = build_assembly_from_points(points, tiny_bundle_path, max_workers=1, original_prompt=prompt)
    inv = _inventory_by_id(result)
    assert inv["bracket_type_A"].reconciliation is ReconciliationStatus.COUNT_MISMATCH
    assert inv["bracket_type_A"].generated_count == 1 and inv["bracket_type_A"].requested_count == 2
    assert inv["crossbar_type_B"].reconciliation is ReconciliationStatus.MATCHED
    assert result.components_complete is False
    assert result.overall_status == "INCOMPLETE"


# ---------------- Test B: correct repeated components ----------------

def test_b_correct_repeated_components_are_matched(tiny_bundle_path):
    prompt = "Create two identical L-brackets connected by one 100mm crossbar."
    points = [
        _box_point("an L-shape bracket, 60mm long", "bracket_01"),
        _box_point("an L-shape bracket, 60mm long", "bracket_02", position_mm=(0, 100, 0)),
        _box_point("a crossbar, 100mm long", "crossbar_01", extents=(100, 10, 5), position_mm=(200, 0, 0)),
    ]
    result = build_assembly_from_points(points, tiny_bundle_path, max_workers=1, original_prompt=prompt)
    inv = _inventory_by_id(result)
    assert inv["bracket_type_A"].reconciliation is ReconciliationStatus.MATCHED
    assert set(inv["bracket_type_A"].instance_ids) == {"bracket_01", "bracket_02"}
    assert inv["crossbar_type_B"].reconciliation is ReconciliationStatus.MATCHED
    assert result.components_complete is True
    assert result.overall_status == "PASS"


# ---------------- Test C: missing third component ----------------

def test_c_missing_third_component_type_is_an_explicit_failure(tiny_bundle_path):
    prompt = "Create one motor housing, one mounting plate, and two mounting brackets."
    points = [
        _box_point("a motor housing, 50mm x 50mm x 30mm", "housing_01"),
        _box_point("a mounting plate, 100mm x 80mm x 5mm", "plate_01", position_mm=(0, 100, 0)),
    ]
    result = build_assembly_from_points(points, tiny_bundle_path, max_workers=1, original_prompt=prompt)
    inv = _inventory_by_id(result)
    assert inv["housing_type_A"].reconciliation is ReconciliationStatus.MATCHED
    assert inv["plate_type_B"].reconciliation is ReconciliationStatus.MATCHED
    assert inv["bracket_type_C"].reconciliation is ReconciliationStatus.MISSING_FROM_PLAN
    assert inv["bracket_type_C"].generated_count == 0 and inv["bracket_type_C"].requested_count == 2
    assert result.components_complete is False
    assert result.overall_status == "INCOMPLETE"


# ---------------- Test D: duplicate component ----------------

def test_d_duplicate_component_is_detected_not_silently_accepted(tiny_bundle_path):
    prompt = "Create one mounting plate."
    points = [
        _box_point("a mounting plate, 100mm x 80mm x 5mm", "plate_01", extents=(100, 80, 5)),
        _box_point("a mounting plate, 100mm x 80mm x 5mm", "plate_02", extents=(100, 80, 5), position_mm=(0, 150, 0)),
    ]
    result = build_assembly_from_points(points, tiny_bundle_path, max_workers=1, original_prompt=prompt)
    inv = _inventory_by_id(result)
    assert inv["plate_type_A"].reconciliation is ReconciliationStatus.DUPLICATED_IN_PLAN
    assert inv["plate_type_A"].generated_count == 2 and inv["plate_type_A"].requested_count == 1
    assert result.components_complete is False
    assert result.overall_status == "INCOMPLETE"


# ---------------- Test E: identical component identity ----------------

def test_e_three_identical_support_legs_keep_distinct_instance_ids(tiny_bundle_path):
    prompt = "Create three identical support legs."
    points = [
        _box_point("a support leg, 10mm x 10mm x 200mm", "leg_01", extents=(10, 10, 200)),
        _box_point("a support leg, 10mm x 10mm x 200mm", "leg_02", extents=(10, 10, 200), position_mm=(50, 0, 0)),
        _box_point("a support leg, 10mm x 10mm x 200mm", "leg_03", extents=(10, 10, 200), position_mm=(100, 0, 0)),
    ]
    result = build_assembly_from_points(points, tiny_bundle_path, max_workers=1, original_prompt=prompt)
    inv = _inventory_by_id(result)
    comp = inv["support_type_A"]
    assert comp.reconciliation is ReconciliationStatus.MATCHED
    assert comp.generated_count == 3
    assert sorted(comp.instance_ids) == ["leg_01", "leg_02", "leg_03"]  # distinct identities, never collapsed
    assert len(set(comp.instance_ids)) == 3
    assert result.components_complete is True


# ---------------- Test F: features distributed across components ----------------

def test_f_holes_must_be_correctly_distributed_across_both_bracket_instances(tiny_bundle_path):
    """"Two L-brackets, each with two 6mm mounting holes" -- integrates
    with the Required Feature Ledger (feature_verification.py):
    PointBuildResult.feature_requirements is computed PER POINT, so each
    bracket instance's own hole count is checked independently -- the
    total (4) is never enough on its own if the DISTRIBUTION is wrong."""
    sb1 = Sandbox(trace=True)
    h1 = sb1.add_box(60, 20, 6, center=(0, 0, 0))
    h1 = sb1.drill_hole(h1, radius=3.0, depth=6.0, center=(20, 0, 0), axis="z")
    h1 = sb1.drill_hole(h1, radius=3.0, depth=6.0, center=(-20, 0, 0), axis="z")
    desc1 = "an L-bracket 60mm x 20mm x 6mm with two 6mm mounting holes"

    sb2 = Sandbox(trace=True)
    h2 = sb2.add_box(60, 20, 6, center=(0, 0, 0))  # correctly 0 of its OWN two holes, not "4 total exist somewhere"
    desc2 = "an L-bracket 60mm x 20mm x 6mm with two 6mm mounting holes"

    prompt = "Create two L-brackets, each with two 6mm mounting holes."
    points = [
        ActionablePoint(description=desc1, position_mm=(0, 0, 0), id="bracket_01", relationship=None, program_override=_program(sb1, h1)),
        ActionablePoint(description=desc2, position_mm=(0, 100, 0), id="bracket_02", relationship=None, program_override=_program(sb2, h2)),
    ]
    result = build_assembly_from_points(points, tiny_bundle_path, max_workers=1, original_prompt=prompt)
    inv = _inventory_by_id(result)
    assert inv["bracket_type_A"].reconciliation is ReconciliationStatus.MATCHED  # both bracket INSTANCES are present
    b1, b2 = result.points
    assert b1.status == "ok"
    assert b2.status == "suspect"  # present, but its OWN holes are missing -- the feature ledger catches this, not the inventory
    assert result.requirements_complete is False
    assert result.overall_status == "INCOMPLETE"


# ---------------- Test G: connected assembly preserves identity ----------------

def test_g_connected_assembly_does_not_collapse_component_identities(tiny_bundle_path):
    prompt = "Create two brackets connected by a crossbar."
    points = [
        _box_point("a bracket, 60mm x 20mm x 5mm", "bracket_01", position_mm=(0, 0, 0), relationship=Relationship(kind="attached_to_face", target_id="crossbar_01", face="+x")),
        _box_point("a bracket, 60mm x 20mm x 5mm", "bracket_02", position_mm=(0, 0, 0), relationship=Relationship(kind="attached_to_face", target_id="crossbar_01", face="-x")),
        _box_point("a crossbar, 100mm x 10mm x 5mm", "crossbar_01", extents=(100, 10, 5)),
    ]
    result = build_assembly_from_points(points, tiny_bundle_path, max_workers=1, original_prompt=prompt)
    # the final combined mesh is one connected solid, but the inventory/points still name 3 distinct instances
    assert len(result.points) == 3
    inv = _inventory_by_id(result)
    assert inv["bracket_type_A"].reconciliation is ReconciliationStatus.MATCHED
    assert inv["bracket_type_A"].generated_count == 2
    assert inv["crossbar_type_B"].reconciliation is ReconciliationStatus.MATCHED
    assert result.components_complete is True


# ---------------- Test H: single component with multiple features ----------------

def test_h_one_plate_with_four_bosses_is_one_component_not_five(tiny_bundle_path):
    prompt = "Create one mounting plate with four bosses."
    points = [_box_point("a mounting plate, 100mm x 80mm x 5mm with four bosses", "plate_01", extents=(100, 80, 5))]
    result = build_assembly_from_points(points, tiny_bundle_path, max_workers=1, original_prompt=prompt)
    assert len(result.component_inventory) == 1  # never a false "4 bosses == 4 more components"
    inv = _inventory_by_id(result)
    assert inv["plate_type_A"].reconciliation is ReconciliationStatus.MATCHED
    assert inv["plate_type_A"].requested_count == 1
    assert result.components_complete is True


# ---------------- Test I: relationship endpoint missing ----------------

def test_i_missing_shaft_leaves_the_bearing_relationship_unevaluated_not_compatible(tiny_bundle_path):
    """The shaft is entirely omitted from generation -- both the
    component inventory (a missing 'shaft' component) AND assembly
    compatibility (a relationship whose own endpoint never built, see
    assembly_compatibility.AssemblyCompatibilityReport.overall_status's
    own NOT_EVALUATED fix) must disclose this; neither may report a
    clean pass."""
    prompt = "Create a bearing housing with a shaft inserted into it."
    points = [_box_point("a bearing housing with a 12mm bore", "housing_01", extents=(40, 40, 20))]
    result = build_assembly_from_points(points, tiny_bundle_path, max_workers=1, original_prompt=prompt)
    inv = _inventory_by_id(result)
    assert inv["housing_type_A"].reconciliation is ReconciliationStatus.MATCHED
    assert inv["shaft_type_B"].reconciliation is ReconciliationStatus.MISSING_FROM_PLAN
    assert result.components_complete is False
    assert result.overall_status == "INCOMPLETE"


# ---------------- Test J: incorrect component assignment ----------------

def test_j_a_point_mislabeled_by_id_is_still_matched_by_its_own_description_not_its_id(tiny_bundle_path):
    """Reconciliation matches by the point's own DESCRIPTION text, never
    its id -- a point whose id suggests one type but whose description
    actually describes another must not be silently miscounted toward
    the id-suggested type."""
    prompt = "Create two brackets and one plate."
    points = [
        _box_point("a bracket, 60mm x 20mm x 5mm", "bracket_01"),
        _box_point("a bracket, 60mm x 20mm x 5mm", "bracket_02", position_mm=(0, 100, 0)),
        # mislabeled id ("bracket_03") but its OWN description is a plate
        _box_point("a rectangular mounting plate, 100mm x 80mm x 5mm", "bracket_03", extents=(100, 80, 5), position_mm=(0, 200, 0)),
    ]
    result = build_assembly_from_points(points, tiny_bundle_path, max_workers=1, original_prompt=prompt)
    inv = _inventory_by_id(result)
    assert inv["bracket_type_A"].reconciliation is ReconciliationStatus.MATCHED
    assert set(inv["bracket_type_A"].instance_ids) == {"bracket_01", "bracket_02"}  # the mislabeled plate is NOT counted as a bracket
    assert inv["plate_type_B"].reconciliation is ReconciliationStatus.MATCHED
    assert inv["plate_type_B"].instance_ids == ["bracket_03"]  # correctly matched by its own description despite its misleading id
    assert result.components_complete is True


# ---------------- Test K: ambiguous request ----------------

def test_k_a_vague_quantity_is_preserved_as_ambiguous_not_guessed(tiny_bundle_path):
    from component_inventory import extract_component_requirements
    from dimension_requirements import RequirementStatus

    expected = extract_component_requirements("Create a support structure with several standoffs.")
    unresolved = [c for c in expected if c.status is RequirementStatus.UNRESOLVED]
    assert len(unresolved) == 1
    assert unresolved[0].component_type is None  # never a guessed count or type
    assert "several standoffs" in unresolved[0].source_text


# ---------------- Test M: excluded component ----------------

def test_m_a_component_excluded_from_the_final_combined_mesh_is_detected(tiny_bundle_path):
    """A component that built a perfectly valid mesh on its own, then
    was EXCLUDED from the final combined assembly by collision
    resolution (parallel_builder._find_and_resolve_collisions), must
    still be caught as absent from the FINAL result -- not quietly
    waved through because it technically "generated" earlier."""
    prompt = "Create two identical brackets."
    # both hand-placed at the SAME position with no relationship connecting them --
    # an unrelated collision, which _find_and_resolve_collisions excludes one of.
    points = [
        _box_point("a bracket, 60mm x 20mm x 5mm", "bracket_01", position_mm=(0, 0, 0)),
        _box_point("a bracket, 60mm x 20mm x 5mm", "bracket_02", position_mm=(0, 0, 0)),
    ]
    result = build_assembly_from_points(points, tiny_bundle_path, max_workers=1, original_prompt=prompt)
    excluded = [p for p in result.points if p.excluded_from_assembly]
    assert len(excluded) == 1  # confirms the collision-exclusion path actually triggered for this scenario
    assert result.components_complete is False  # components_complete's own docstring: an excluded point's own mandatory requirements still count against it
    assert result.overall_status == "INCOMPLETE"


# ---------------- Test N: unexpected inferred component ----------------

def test_n_an_extra_component_type_the_prompt_never_asked_for_does_not_block_completeness(tiny_bundle_path):
    """The planner/decomposer adds an extra, unrequested component (e.g.
    an inferred cover) -- it must not inflate MANDATORY completion
    counts (an extra point matching no EXPECTED type simply has no
    ComponentTypeRequirement entry to affect at all), and the explicitly
    requested component must still independently reconcile as complete."""
    prompt = "Create one mounting plate."
    points = [
        _box_point("a mounting plate, 100mm x 80mm x 5mm", "plate_01", extents=(100, 80, 5)),
        _box_point("a protective cover", "cover_01", position_mm=(0, 100, 0)),  # unrequested, inferred extra
    ]
    result = build_assembly_from_points(points, tiny_bundle_path, max_workers=1, original_prompt=prompt)
    inv = _inventory_by_id(result)
    assert len(inv) == 1  # only the EXPECTED "plate" type has an inventory entry at all -- the extra cover never became a requirement
    assert inv["plate_type_A"].reconciliation is ReconciliationStatus.MATCHED
    assert result.components_complete is True
    assert result.overall_status == "PASS"
