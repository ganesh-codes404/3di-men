"""Mocks llm_decompose.decompose_prompt() entirely -- these tests exercise
the parallel-build-and-combine logic, not the LLM, so they run fast and
don't depend on the real .gguf model file being present."""

from pathlib import Path

import pytest

pytest.importorskip("sklearn")

from data.program_dataset import sample_program_dataset, write_program_dataset
from llm_decompose import ActionablePoint, DecomposeResult
from parallel_builder import build_assembly_from_points
from program_model import ProgramModelConfig, train_program_model


@pytest.fixture(scope="module")
def tiny_bundle_path(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("tiny_program_dataset")
    samples = sample_program_dataset(
        {"plate": 60, "flange": 60, "bracket": 60, "shaft": 60, "enclosure": 60, "gear": 60},
        n_generic=60, seed=0,
    )
    write_program_dataset(root, samples, seed=0)
    bundle, _ = train_program_model(root, ProgramModelConfig(seed=0))
    bundle_path = tmp_path_factory.mktemp("bundle") / "bundle.joblib"
    bundle.save(bundle_path)
    return bundle_path


def _mock_decompose(points):
    def fn(prompt, model_path=None):
        return DecomposeResult(points=points, raw_llm_output="", fell_back=False)
    return fn


def test_multi_point_prompt_builds_all_points_in_parallel_and_combines(tiny_bundle_path, monkeypatch):
    import parallel_builder

    points = [
        ActionablePoint(description="a steel shaft, 20mm diameter, 100mm long", position_mm=(0.0, 0.0, 0.0)),
        ActionablePoint(description="a steel spur gear, module 2, 16 teeth", position_mm=(0.0, 0.0, 100.0)),
    ]
    monkeypatch.setattr(parallel_builder, "decompose_prompt", _mock_decompose(points))

    result = parallel_builder.build_assembly_from_prompt("a shaft with a gear on top", tiny_bundle_path, max_workers=2)
    assert len(result.points) == 2
    assert result.mesh is not None
    assert not result.fell_back_to_single_point


def test_a_point_with_no_mesh_is_reported_not_silently_dropped(tiny_bundle_path, monkeypatch):
    import parallel_builder

    # a handcrafted description with no real geometry signal -- program_model
    # still predicts SOMETHING, so instead we directly verify the reporting
    # contract via a point whose build we force to fail by monkeypatching
    # _build_one for this one test, since reliably forcing a "rejected"
    # status through the real predictor isn't deterministic.
    points = [ActionablePoint(description="a steel shaft, 20mm diameter, 100mm long", position_mm=(0.0, 0.0, 0.0))]
    monkeypatch.setattr(parallel_builder, "decompose_prompt", _mock_decompose(points))

    def _fake_build_one(description):
        return {
            "status": "rejected", "executed_steps": 0, "error": "forced failure for test",
            "nearest_neighbor_similarity": 0.9, "program": [], "mesh": None, "topology": None,
        }

    # ProcessPoolExecutor workers can't easily be monkeypatched (separate
    # processes) -- run with max_workers=1 and patch the module-level
    # function the pool would otherwise call via its own import, verifying
    # the CALLER's handling of a None-mesh result instead of the pool
    # mechanics themselves (those are exercised by the test above).
    monkeypatch.setattr(parallel_builder, "_build_one", _fake_build_one)
    import concurrent.futures

    class _ImmediateExecutor:
        def __init__(self, *a, **k): pass
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def map(self, fn, iterable):
            return [_fake_build_one(x) for x in iterable]

    monkeypatch.setattr(concurrent.futures, "ProcessPoolExecutor", _ImmediateExecutor)
    monkeypatch.setattr(parallel_builder, "ProcessPoolExecutor", _ImmediateExecutor)

    result = parallel_builder.build_assembly_from_prompt("something ungeneratable", tiny_bundle_path, max_workers=1)
    assert len(result.points) == 1  # reported, not dropped
    assert result.points[0].status == "rejected"
    assert result.mesh is None  # no successful points to combine


def test_successful_point_carries_topology_stats(tiny_bundle_path, monkeypatch):
    import parallel_builder

    points = [ActionablePoint(description="a steel shaft, 20mm diameter, 100mm long", position_mm=(0.0, 0.0, 0.0))]
    monkeypatch.setattr(parallel_builder, "decompose_prompt", _mock_decompose(points))

    result = parallel_builder.build_assembly_from_prompt("a shaft", tiny_bundle_path, max_workers=1)
    assert len(result.points) == 1
    topo = result.points[0].topology
    assert topo is not None
    assert topo["vertex_count"] > 0
    assert topo["face_count"] > 0
    assert "is_watertight" in topo


def test_build_assembly_from_points_never_calls_the_llm(tiny_bundle_path, monkeypatch):
    """The playground's manual-edit path must go straight through
    program_model.py + sandbox.py, exactly like the LLM-decomposed path,
    but WITHOUT ever touching decompose_prompt -- monkeypatching it to
    raise proves that a hand-supplied points list bypasses the LLM
    entirely rather than silently still calling it."""
    import parallel_builder

    def _boom(prompt, model_path=None):
        raise AssertionError("build_assembly_from_points must never call decompose_prompt")

    monkeypatch.setattr(parallel_builder, "decompose_prompt", _boom)

    points = [
        ActionablePoint(description="a steel shaft, 20mm diameter, 100mm long", position_mm=(0.0, 0.0, 0.0)),
        ActionablePoint(description="a steel spur gear, module 2, 16 teeth", position_mm=(0.0, 0.0, 100.0)),
    ]
    result = build_assembly_from_points(points, tiny_bundle_path, max_workers=2)
    assert len(result.points) == 2
    assert result.mesh is not None
    assert result.fell_back_to_single_point is False


def test_placement_offsets_are_applied_before_combining(tiny_bundle_path, monkeypatch):
    import parallel_builder

    points = [
        ActionablePoint(description="aluminum shaft, 10mm diameter, 20mm long", position_mm=(0.0, 0.0, 0.0)),
        ActionablePoint(description="aluminum shaft, 10mm diameter, 20mm long", position_mm=(500.0, 0.0, 0.0)),
    ]
    monkeypatch.setattr(parallel_builder, "decompose_prompt", _mock_decompose(points))

    result = parallel_builder.build_assembly_from_prompt("two shafts far apart", tiny_bundle_path, max_workers=2)
    assert result.mesh is not None
    # two shafts 500mm apart must not overlap -- combined bounding box
    # should span roughly 500mm+ in x, not just one shaft's own extent
    span_x = result.mesh.bounds[1][0] - result.mesh.bounds[0][0]
    assert span_x > 400


# ---- relationship-based (dependency-wave) building ----

def test_relationship_based_points_build_in_dependency_order(tiny_bundle_path):
    """A boss attached_to_face of a plate, built through the REAL
    predict+build pipeline (not hand-constructed meshes) -- proves the
    wave ordering actually waits for the target's mesh before resolving
    the dependent part's position, end to end."""
    from assembly_plan import Relationship

    points = [
        ActionablePoint(description="a box 100mm x 60mm x 8mm", position_mm=(0.0, 0.0, 0.0), id="plate"),
        ActionablePoint(
            description="a cylinder 10mm diameter, 15mm tall", position_mm=(0.0, 0.0, 0.0), id="boss",
            relationship=Relationship(kind="attached_to_face", target_id="plate", face="+z"),
        ),
    ]
    result = build_assembly_from_points(points, tiny_bundle_path, max_workers=2)
    by_id = {p.point.id: p for p in result.points}
    assert by_id["plate"].status == "ok"
    assert by_id["boss"].status == "ok"
    # the boss's resolved z must be positive (pushed OUT from the plate's
    # top face, not left at the origin where it would be buried in the plate)
    assert by_id["boss"].point.position_mm[2] > 0
    assert result.mesh is not None
    assert result.mesh.is_watertight


def test_compatibility_report_is_populated_for_a_real_relationship_based_assembly(tiny_bundle_path):
    """assembly_compatibility.py's own engine, wired into build_assembly_
    from_points right where collision resolution already runs -- a boss
    attached_to_face of a plate has no mechanical-fit keyword on either
    side (no shaft/bore/gear/motor/bolt), so this classifies as a
    generic FACE_CONTACT relationship and must come back COMPATIBLE
    (the boss really is flush against the plate, same real build the
    test right above this one already confirms)."""
    from assembly_compatibility import CompatibilityStatus, MechanicalRelationshipType
    from assembly_plan import Relationship

    points = [
        ActionablePoint(description="a box 100mm x 60mm x 8mm", position_mm=(0.0, 0.0, 0.0), id="plate"),
        ActionablePoint(
            description="a cylinder 10mm diameter, 15mm tall", position_mm=(0.0, 0.0, 0.0), id="boss",
            relationship=Relationship(kind="attached_to_face", target_id="plate", face="+z"),
        ),
    ]
    result = build_assembly_from_points(points, tiny_bundle_path, max_workers=2)
    assert result.compatibility_report is not None
    assert len(result.compatibility_report.evaluations) == 1
    eva = result.compatibility_report.evaluations[0]
    assert eva.relationship_type is MechanicalRelationshipType.FACE_CONTACT
    assert eva.status is CompatibilityStatus.COMPATIBLE
    assert result.compatibility_report.overall_status is CompatibilityStatus.COMPATIBLE


def test_hand_constructed_points_get_the_exact_gear_mesh_correction_too(tiny_bundle_path):
    """Confirmed real gap this closes: llm_decompose._correct_gear_mesh_spacing
    previously only ran inside decompose_prompt() -- hand-constructed points
    (the frontend's manual playground, or any other direct caller of
    build_assembly_from_points) never got the exact center-distance
    correction at all, even though the correction itself has nothing
    LLM-specific about it. Two gears deliberately placed at the WRONG
    distance (40mm, no relationship -- Example 3's simplest pattern) via
    plain ActionablePoints must still land at the real, computed 75mm
    (module 2.5 * (24+36) teeth / 2) once build_assembly_from_points runs,
    with no LLM involved anywhere in this test."""
    from assembly_plan import Relationship

    points = [
        ActionablePoint(description="a cylinder 10mm diameter, 70mm long", position_mm=(0.0, 0.0, 0.0), id="shaft_a"),
        ActionablePoint(
            description="a gear profile, module 2.5, 24 teeth", position_mm=(0.0, 0.0, 0.0), id="gear_a",
            relationship=Relationship(kind="concentric_with", target_id="shaft_a", axis="z"),
        ),
        ActionablePoint(description="a cylinder 10mm diameter, 70mm long", position_mm=(40.0, 0.0, 0.0), id="shaft_b"),
        ActionablePoint(
            description="a gear profile, module 2.5, 36 teeth", position_mm=(40.0, 0.0, 0.0), id="gear_b",
            relationship=Relationship(kind="concentric_with", target_id="shaft_b", axis="z"),
        ),
    ]
    result = build_assembly_from_points(points, tiny_bundle_path, max_workers=2)
    by_id = {p.point.id: p for p in result.points}
    assert by_id["shaft_b"].point.position_mm[0] == pytest.approx(75.0)  # 2.5 * (24+36) / 2, not the original 40mm guess


def test_second_gear_on_a_parallel_shaft_stays_level_and_clear_of_the_base(tiny_bundle_path):
    """Confirmed, independently-reproduced regression (a real motor-mount
    run, twice): "gear_B was excluded after a 52% overlap with the
    base." Reproduces the full chain (base -> boss -> shaft_A -> gear_A,
    plus shaft_B parallel_to shaft_A -> gear_B) through the REAL build
    pipeline -- gear_B must land level with gear_A (both well above the
    base, not near its own level), survive without being excluded for a
    collision with the base, and sit the correct gear-mesh distance from
    gear_A.

    Uses the already-correct lateral_offset=65.0 directly (what a real
    decompose_prompt() call would hand off AFTER
    _correct_gear_mesh_spacing runs -- that correction step has its own
    separate, dedicated tests in test_llm_decompose.py) -- this test's
    own job is the placement/collision behavior once the offset is
    right, not re-testing the correction arithmetic itself.
    build_assembly_from_points (unlike decompose_prompt) never applies
    that correction on its own, by design: it's also the entry point for
    hand-constructed/manual points, which have no "declared module/teeth
    text" for a correction to even read."""
    from assembly_plan import Relationship

    points = [
        ActionablePoint(description="a box 150mm x 100mm x 10mm", position_mm=(0.0, 0.0, 0.0), id="base"),
        ActionablePoint(
            description="a cylinder 30mm diameter, 20mm tall", position_mm=(0.0, 0.0, 0.0), id="boss",
            relationship=Relationship(kind="attached_to_face", target_id="base", face="+z"),
        ),
        ActionablePoint(
            description="a cylinder 12mm diameter, 90mm long", position_mm=(0.0, 0.0, 0.0), id="shaft_a",
            relationship=Relationship(kind="concentric_with", target_id="boss", axis="z"),
        ),
        ActionablePoint(
            description="a gear profile, module 2.5, 20 teeth", position_mm=(0.0, 0.0, 0.0), id="gear_a",
            relationship=Relationship(kind="concentric_with", target_id="shaft_a", axis="z"),
        ),
        ActionablePoint(
            description="a cylinder 12mm diameter, 90mm long", position_mm=(0.0, 0.0, 0.0), id="shaft_b",
            relationship=Relationship(kind="parallel_to", target_id="shaft_a", axis="z", lateral_axis="x", lateral_offset=65.0),
        ),
        ActionablePoint(
            description="a gear profile, module 2.5, 32 teeth", position_mm=(0.0, 0.0, 0.0), id="gear_b",
            relationship=Relationship(kind="concentric_with", target_id="shaft_b", axis="z"),
        ),
    ]
    result = build_assembly_from_points(points, tiny_bundle_path, max_workers=3)
    by_id = {p.point.id: p for p in result.points}
    for pid in ("base", "boss", "shaft_a", "gear_a", "shaft_b", "gear_b"):
        assert by_id[pid].status == "ok", f"{pid}: {by_id[pid].error}"
        assert by_id[pid].excluded_from_assembly is False, f"{pid} was excluded: {by_id[pid].error}"
    assert result.collision_warnings == []

    gear_a_z, gear_b_z = by_id["gear_a"].point.position_mm[2], by_id["gear_b"].point.position_mm[2]
    assert gear_a_z == pytest.approx(gear_b_z, abs=0.5)  # level with each other, not one near the base
    base_top_z = 5.0  # 10mm-thick base centered at the origin
    assert gear_a_z > base_top_z and gear_b_z > base_top_z  # both clear of the base entirely

    gear_a_x, gear_b_x = by_id["gear_a"].point.position_mm[0], by_id["gear_b"].point.position_mm[0]
    expected_distance = 2.5 * (20 + 32) / 2.0  # 65mm
    assert abs(gear_b_x - gear_a_x) == pytest.approx(expected_distance, abs=0.5)


def test_coincident_face_attachments_are_auto_separated_not_silently_stacked(tiny_bundle_path):
    """Mission-brief concern: "the request for gussets on opposite sides
    produces identical instances in the same position." Two sibling
    gussets attached to the SAME target+face with the IDENTICAL
    offset_u/offset_v (the LLM emitted the relationship twice without
    differentiating it -- a confirmed real failure mode) must not land
    on top of each other, excluded-by-collision as a last resort, or
    silently merged -- assembly_plan.separate_coincident_face_attachments
    must auto-separate them using their own real built extents, and both
    must survive into the final combined mesh at genuinely distinct
    positions."""
    from assembly_plan import Relationship

    points = [
        ActionablePoint(description="a box 60mm x 50mm x 6mm", position_mm=(0.0, 0.0, 0.0), id="base"),
        ActionablePoint(
            description="a triangular gusset, 20mm x 20mm, 4mm thick", position_mm=(0.0, 0.0, 0.0), id="gusset_a",
            relationship=Relationship(kind="attached_to_face", target_id="base", face="+y"),
        ),
        ActionablePoint(
            description="a triangular gusset, 20mm x 20mm, 4mm thick", position_mm=(0.0, 0.0, 0.0), id="gusset_b",
            relationship=Relationship(kind="attached_to_face", target_id="base", face="+y"),  # IDENTICAL -- the bug case
        ),
    ]
    result = build_assembly_from_points(points, tiny_bundle_path, max_workers=2)
    by_id = {p.point.id: p for p in result.points}
    assert by_id["gusset_a"].status == "ok"
    assert by_id["gusset_b"].status == "ok"
    # neither was excluded to resolve a collision -- both survive as requested
    assert by_id["gusset_a"].excluded_from_assembly is False
    assert by_id["gusset_b"].excluded_from_assembly is False
    assert result.collision_warnings == []
    # genuinely distinct positions now, not the identical offset both started with
    pos_a, pos_b = by_id["gusset_a"].point.position_mm, by_id["gusset_b"].point.position_mm
    assert pos_a != pos_b
    assert "auto-separated" in (by_id["gusset_a"].error or "")
    assert "auto-separated" in (by_id["gusset_b"].error or "")
    assert result.mesh is not None


def test_invalid_relationship_is_rejected_before_building_anything(tiny_bundle_path):
    """A dangling target_id is a structurally invalid plan -- every point
    must come back "rejected" with a clear reason, and nothing should be
    built at all (not even the independent "plate" point), matching this
    project's standing rule that a malformed plan is caught before
    geometry execution, not discovered partway through."""
    from assembly_plan import Relationship

    points = [
        ActionablePoint(description="a box 100mm x 60mm x 8mm", position_mm=(0.0, 0.0, 0.0), id="plate"),
        ActionablePoint(
            description="a cylinder 10mm diameter, 15mm tall", position_mm=(0.0, 0.0, 0.0), id="boss",
            relationship=Relationship(kind="attached_to_face", target_id="nonexistent", face="+z"),
        ),
    ]
    result = build_assembly_from_points(points, tiny_bundle_path, max_workers=2)
    assert result.mesh is None
    assert all(p.status == "rejected" for p in result.points)
    assert any("RelationshipError" in (p.error or "") for p in result.points)


def test_relationship_cycle_is_rejected_before_building_anything(tiny_bundle_path):
    from assembly_plan import Relationship

    points = [
        ActionablePoint(description="a box 40mm x 40mm x 10mm", position_mm=(0.0, 0.0, 0.0), id="a",
                         relationship=Relationship(kind="concentric_with", target_id="b", axis="z")),
        ActionablePoint(description="a box 40mm x 40mm x 10mm", position_mm=(0.0, 0.0, 0.0), id="b",
                         relationship=Relationship(kind="concentric_with", target_id="a", axis="z")),
    ]
    result = build_assembly_from_points(points, tiny_bundle_path, max_workers=2)
    assert result.mesh is None
    assert all(p.status == "rejected" for p in result.points)


def test_three_level_dependency_chain_builds_correctly(tiny_bundle_path):
    """plate -> boss -> shaft, mirroring the spec's own illustrative
    mounting-plate-with-boss example -- three dependency waves, each
    resolved from the previous wave's REAL built geometry."""
    from assembly_plan import Relationship

    points = [
        ActionablePoint(description="a box 120mm x 80mm x 8mm", position_mm=(0.0, 0.0, 0.0), id="plate"),
        ActionablePoint(description="a cylinder 20mm diameter, 15mm tall", position_mm=(0.0, 0.0, 0.0), id="boss",
                         relationship=Relationship(kind="attached_to_face", target_id="plate", face="+z")),
        ActionablePoint(description="a cylinder 6mm diameter, 25mm long", position_mm=(0.0, 0.0, 0.0), id="shaft",
                         relationship=Relationship(kind="concentric_with", target_id="boss", axis="z")),
    ]
    result = build_assembly_from_points(points, tiny_bundle_path, max_workers=2)
    by_id = {p.point.id: p for p in result.points}
    assert all(by_id[pid].status == "ok" for pid in ("plate", "boss", "shaft"))
    # shaft must share boss's x/y exactly (concentric) -- boss's own x/y
    # came from being centered on the plate's face, not the origin
    boss_pos, shaft_pos = by_id["boss"].point.position_mm, by_id["shaft"].point.position_mm
    assert shaft_pos[0] == pytest.approx(boss_pos[0], abs=0.5)
    assert shaft_pos[1] == pytest.approx(boss_pos[1], abs=0.5)
    assert result.mesh is not None and result.mesh.is_watertight


def test_point_build_result_carries_a_real_correctness_to_spec_check(tiny_bundle_path):
    """ACCURACY_AUDIT.md section 3: status="ok" alone never meant "matches
    the request." This confirms PointBuildResult.dimension_match_rate is
    actually computed end to end (spec_validation.py wired into the real
    build loop), not just that the module exists in isolation -- each of
    these 3 points has real numeric claims in its own description, so
    dimension_match_rate must be populated (not None) for all of them."""
    points = [
        ActionablePoint(description="a box 120mm x 80mm x 8mm", position_mm=(0.0, 0.0, 0.0), id="plate"),
        ActionablePoint(description="a cylinder 20mm diameter, 15mm tall", position_mm=(0.0, 0.0, 0.0), id="boss"),
    ]
    result = build_assembly_from_points(points, tiny_bundle_path, max_workers=2)
    for p in result.points:
        if p.status == "ok":
            assert p.dimension_match_rate is not None
            assert 0.0 <= p.dimension_match_rate <= 1.0


def test_wrong_hole_count_cannot_hide_behind_a_high_match_score(tiny_bundle_path):
    """Mission-brief concern: "the base can receive a 75% matching score
    despite an incorrect hole count." validate_hole_count_against_description
    is already wired into the same per-point status downgrade dimension_match_rate
    uses (build_assembly_from_points's own "problems" list) -- this confirms
    that chain actually holds for a real built mesh: a base claiming 4 holes
    but only 3 actually drilled (program_override bypasses prediction
    entirely, so this is the REAL geometry, not a guess) must come back
    status="suspect" with the mismatch disclosed in its own error, and the
    whole assembly must be INCOMPLETE, never a clean PASS just because the
    other dimensions happen to match."""
    points = [ActionablePoint(
        description="a box 140mm x 100mm x 10mm with a corner array of four 8mm diameter holes, each 12mm from a corner",
        position_mm=(0.0, 0.0, 0.0),
        program_override=[
            {"op": "add_box", "args": {"length": 140, "width": 100, "height": 10, "center": [0, 0, 0]}},
            {"op": "drill_hole_pattern", "args": {"target": 0, "pattern": "corner_array", "count": 3, "radius": 4.0, "depth": 20, "margin": 12, "axis": "z"}},
            {"op": "finalize", "args": {"target": 1}},
        ],
    )]
    result = build_assembly_from_points(points, tiny_bundle_path, max_workers=1)
    point = result.points[0]
    assert point.claimed_hole_count == 4
    assert point.detected_hole_count == 3
    assert point.status == "suspect"
    assert "4 hole" in point.error and "3" in point.error
    assert result.overall_status == "INCOMPLETE"


def test_collision_warning_flags_two_siblings_resolved_to_the_same_position(tiny_bundle_path):
    """Confirmed real gap this closes: nothing previously caught two
    independently-placed parts ending up stacked on top of each other --
    the exact shape of the gear-mesh-spacing bug this project already
    fixed once elsewhere (see llm_decompose._correct_gear_mesh_spacing).
    Two cylinders with NO relationship connecting them, both at the
    identical position, must be flagged -- AND, per "it can't just give
    up", actually resolved: the later one ('b') is excluded from the
    combined mesh rather than left overlapping, disclosed via both the
    warning text and its own PointBuildResult.excluded_from_assembly."""
    points = [
        ActionablePoint(description="a cylinder 20mm diameter, 60mm long", position_mm=(0.0, 0.0, 0.0), id="a"),
        ActionablePoint(description="a cylinder 20mm diameter, 60mm long", position_mm=(0.0, 0.0, 0.0), id="b"),
    ]
    result = build_assembly_from_points(points, tiny_bundle_path, max_workers=2)
    assert any("'a'" in w and "'b'" in w for w in result.collision_warnings)
    by_id = {p.point.id: p for p in result.points}
    assert by_id["a"].excluded_from_assembly is False
    assert by_id["b"].excluded_from_assembly is True
    assert by_id["b"].status == "suspect"
    assert "excluded from the combined assembly" in by_id["b"].error
    # the assembly still finishes -- a complete model missing the
    # conflicting part, not no model at all
    assert result.mesh is not None


def test_collision_warning_does_not_false_positive_on_correctly_spaced_parts(tiny_bundle_path):
    points = [
        ActionablePoint(description="a cylinder 20mm diameter, 60mm long", position_mm=(0.0, 0.0, 0.0), id="a"),
        ActionablePoint(description="a cylinder 20mm diameter, 60mm long", position_mm=(200.0, 0.0, 0.0), id="b"),
    ]
    result = build_assembly_from_points(points, tiny_bundle_path, max_workers=2)
    assert result.collision_warnings == []


def test_collision_warning_excludes_relationship_connected_parts(tiny_bundle_path):
    """A shaft concentric_with a boss's bore is SUPPOSED to sit almost
    entirely inside the boss's own bounding box -- must never be flagged,
    since that's this project's own most common, intentional placement
    pattern, not a bug."""
    from assembly_plan import Relationship

    points = [
        ActionablePoint(description="a cylinder 20mm diameter, 10mm tall", position_mm=(0.0, 0.0, 0.0), id="boss"),
        ActionablePoint(
            description="a cylinder 10mm diameter, 70mm long", position_mm=(0.0, 0.0, 0.0), id="shaft",
            relationship=Relationship(kind="concentric_with", target_id="boss", axis="y"),
        ),
    ]
    result = build_assembly_from_points(points, tiny_bundle_path, max_workers=2)
    assert result.collision_warnings == []


def test_relationship_connected_walks_transitive_chains_in_either_direction():
    from assembly_plan import Relationship
    from parallel_builder import _relationship_connected

    relationships = {
        "housing": None,
        "boss": Relationship(kind="attached_to_face", target_id="housing", face="+y"),
        "shaft": Relationship(kind="concentric_with", target_id="boss", axis="y"),
        "gear": Relationship(kind="concentric_with", target_id="shaft", axis="y"),
        "unrelated": None,
    }
    assert _relationship_connected("gear", "housing", relationships) is True  # 3 hops
    assert _relationship_connected("housing", "gear", relationships) is True  # same chain, other direction
    assert _relationship_connected("boss", "shaft", relationships) is True
    assert _relationship_connected("gear", "unrelated", relationships) is False


def test_find_and_resolve_collisions_uses_the_calibrated_threshold():
    import trimesh

    from parallel_builder import _find_and_resolve_collisions

    stacked_a = trimesh.creation.box([20, 20, 20])
    stacked_b = trimesh.creation.box([20, 20, 20])
    far = trimesh.creation.box([20, 20, 20])
    far.apply_translation([200, 0, 0])

    warnings, excluded = _find_and_resolve_collisions(
        {"a": stacked_a, "b": stacked_b, "c": far}, relationships={"a": None, "b": None, "c": None}
    )
    assert len(warnings) == 1
    assert "'a'" in warnings[0] and "'b'" in warnings[0]
    # resolved, not just reported: the later of the two colliding parts is excluded
    assert excluded == {"b"}


def test_find_and_resolve_collisions_excludes_nothing_when_nothing_collides():
    import trimesh

    from parallel_builder import _find_and_resolve_collisions

    far_a = trimesh.creation.box([20, 20, 20])
    far_b = trimesh.creation.box([20, 20, 20])
    far_b.apply_translation([200, 0, 0])

    warnings, excluded = _find_and_resolve_collisions({"a": far_a, "b": far_b}, relationships={"a": None, "b": None})
    assert warnings == []
    assert excluded == set()


def test_program_override_skips_prediction_and_builds_the_exact_hand_edited_program(tiny_bundle_path):
    """New editable "Program steps" section (frontend/workspace.html's
    Prompt tab) -- ActionablePoint.program_override, when set, must bypass
    compositional_decomposer/program_model prediction ENTIRELY and build
    exactly the hand-edited program instead, still through the same
    Sandbox.execute()/repair/validation pipeline a predicted program goes
    through (confirmed here via a real, measurably different built mesh,
    not just a resolution_method label). Description deliberately has no
    "NNmm" claims of its own (just "a box") -- a description claiming
    different dimensions than the edited program actually builds is a
    SEPARATE, correctly-firing concern (spec_validation.py's dimension
    check, confirmed elsewhere), not what this test is isolating."""
    points = [ActionablePoint(
        description="a box", position_mm=(0.0, 0.0, 0.0),
        program_override=[
            {"op": "add_box", "args": {"length": 99.0, "width": 99.0, "height": 99.0, "center": [0.0, 0.0, 0.0]}},
            {"op": "finalize", "args": {"target": 0}},
        ],
    )]
    result = build_assembly_from_points(points, tiny_bundle_path, max_workers=1)
    p = result.points[0]
    assert p.resolution_method == "user_edited"
    assert p.status == "ok"
    # the built geometry reflects the EDITED program (99mm), not whatever
    # the original description alone would otherwise have predicted
    assert p.mesh.extents == pytest.approx([99.0, 99.0, 99.0])


def test_program_override_none_predicts_exactly_as_before(tiny_bundle_path):
    """The default (no override) path must be byte-for-byte unaffected by
    this feature existing -- confirms resolution_method is never
    "user_edited" when program_override wasn't set at all."""
    points = [ActionablePoint(description="a box 40mm x 40mm x 10mm", position_mm=(0.0, 0.0, 0.0))]
    result = build_assembly_from_points(points, tiny_bundle_path, max_workers=1)
    assert result.points[0].resolution_method != "user_edited"


def test_overall_status_pass_when_every_point_is_clean(tiny_bundle_path):
    points = [ActionablePoint(description="a box 40mm x 40mm x 10mm", position_mm=(0.0, 0.0, 0.0))]
    result = build_assembly_from_points(points, tiny_bundle_path, max_workers=1)
    assert result.overall_status == "PASS"


def test_overall_status_incomplete_when_built_envelope_violates_the_spec(tiny_bundle_path):
    """Mission-brief Test F: a required 60x50x60mm envelope that actually
    built as 79.0x97.1x93.5mm must not come back PASS just because the
    mesh itself is a structurally valid, watertight solid. This is
    already wired end to end (spec_validation.py's dimension_match_rate
    downgrades a mismatched point's own status to "suspect", which
    overall_status's existing suspect-check already catches) -- this test
    confirms that chain actually holds at the ASSEMBLY level, not just
    that the per-point field gets populated."""
    points = [ActionablePoint(
        description="a box 60mm x 50mm x 60mm", position_mm=(0.0, 0.0, 0.0),
        program_override=[
            {"op": "add_box", "args": {"length": 79.0, "width": 97.1, "height": 93.5, "center": [0, 0, 0]}},
            {"op": "finalize", "args": {"target": 0}},
        ],
    )]
    result = build_assembly_from_points(points, tiny_bundle_path, max_workers=1)
    assert result.points[0].status == "suspect"
    assert result.points[0].dimension_match_rate < 1.0
    assert result.overall_status == "INCOMPLETE"


# ---------------- _diagnose_envelope_overrun (direct, not just via the full
# build pipeline above) -- confirmed real gap this closes: this function had
# NO test at all despite being the exact mechanism the mission brief's own
# "251.2x124.0x108.6mm vs ~150x100mm" regression traces through. ----------

def test_diagnose_envelope_overrun_identifies_the_component_that_actually_overruns():
    """Two components define the combined bbox's two opposite extremes on
    X -- by construction, "distance from the assembly's own center" is
    then EXACTLY equal for both of them (each sits exactly half the total
    span from the midpoint, which is what makes it the midpoint), so this
    deliberately isolates the case where only ONE component -- gear_b --
    is the far outlier on BOTH sides, with 'base' nested entirely within
    its reach, so the attribution is unambiguous rather than a tie."""
    import trimesh

    from parallel_builder import _diagnose_envelope_overrun

    base = trimesh.creation.box([150.0, 100.0, 10.0])  # exactly matches the claimed footprint
    gear_b = trimesh.creation.box([200.0, 20.0, 20.0])
    gear_b.apply_translation([10.0, 0.0, 0.0])  # spans x=[-90, 110] -- reaches further than base on BOTH sides

    warnings = _diagnose_envelope_overrun((150.0, 100.0), {"base": base, "gear_b": gear_b})
    assert len(warnings) == 1
    assert "X" in warnings[0]
    assert "'gear_b'" in warnings[0]  # the specifically-blamed id (repr-quoted in the message)
    assert "'base'" not in warnings[0]


def test_diagnose_envelope_overrun_is_quiet_within_the_declared_tolerance():
    """A claim is explicitly "approximate" (see extract_claimed_envelope_mm's
    own docstring) -- a small, few-percent overrun must not be flagged as
    if the claim were a hard maximum."""
    import trimesh

    from parallel_builder import _diagnose_envelope_overrun

    base = trimesh.creation.box([155.0, 100.0, 10.0])  # 3% over the claimed 150mm -- well inside the 20% default tolerance
    warnings = _diagnose_envelope_overrun((150.0, 100.0), {"base": base})
    assert warnings == []


def test_diagnose_envelope_overrun_checks_x_and_y_independently():
    import trimesh

    from parallel_builder import _diagnose_envelope_overrun

    base = trimesh.creation.box([150.0, 300.0, 10.0])  # X fine, Y wildly over
    warnings = _diagnose_envelope_overrun((150.0, 100.0), {"base": base})
    assert len(warnings) == 1
    assert "Y" in warnings[0] and "X" not in warnings[0]


def test_diagnose_envelope_overrun_is_a_noop_with_no_placed_meshes():
    from parallel_builder import _diagnose_envelope_overrun

    assert _diagnose_envelope_overrun((150.0, 100.0), {}) == []


def test_semantic_size_conflict_is_disclosed_but_explicit_dimension_still_wins(tiny_bundle_path):
    """Non-negotiable principle: "explicit numerical dimensions override
    semantic defaults associated with names such as 'NEMA 17'." A
    description naming NEMA 17 (standard 42.3mm square face) but
    explicitly stating 50mm x 50mm must: (1) disclose the conflict via
    semantic_size_conflicts, (2) still BUILD and VALIDATE against the
    explicit 50mm (not the standard's 42.3mm) -- a disclosed conflict is
    not itself a defect, so status must stay "ok" and overall_status
    "PASS" when the built geometry correctly honors the explicit value."""
    points = [ActionablePoint(
        description="a NEMA 17 stepper motor mount, 50mm x 50mm x 10mm thick",
        position_mm=(0.0, 0.0, 0.0),
        program_override=[
            {"op": "add_box", "args": {"length": 50, "width": 50, "height": 10, "center": [0, 0, 0]}},
            {"op": "finalize", "args": {"target": 0}},
        ],
    )]
    result = build_assembly_from_points(points, tiny_bundle_path, max_workers=1)
    point = result.points[0]
    assert len(point.semantic_size_conflicts) == 1
    assert "NEMA 17" in point.semantic_size_conflicts[0]
    assert "42.3" in point.semantic_size_conflicts[0]
    assert "50" in point.semantic_size_conflicts[0]
    assert "explicit" in point.semantic_size_conflicts[0]
    # built correctly to the EXPLICIT 50mm -- the conflict is disclosed, not a defect
    assert point.status == "ok"
    assert point.dimension_match_rate == 1.0
    assert result.overall_status == "PASS"


def test_wrong_bore_diameter_cannot_hide_behind_a_high_match_score(tiny_bundle_path):
    """Mission-brief concern (boss bore diameter): a claimed 12mm bore
    that was actually built as 24mm (e.g. a radius/diameter mix-up) must
    not come back a clean "ok" just because the overall cylinder
    dimensions happen to match -- spec_validation.py's
    validate_hole_diameters_against_description measures the ACTUAL
    built hole from its own fitted-circle radius, independent of what
    the program's own args claimed to be drilling."""
    points = [ActionablePoint(
        description="a cylinder 30mm diameter, 20mm tall with a 12mm diameter bore through it",
        position_mm=(0.0, 0.0, 0.0),
        program_override=[
            {"op": "add_cylinder", "args": {"radius": 15, "height": 20, "center": [0, 0, 0], "axis": "z"}},
            {"op": "drill_hole", "args": {"target": 0, "radius": 12.0, "depth": 30, "center": [0, 0, 0], "axis": "z"}},  # wrong: radius=12 means a 24mm hole, not the claimed 12mm
            {"op": "finalize", "args": {"target": 1}},
        ],
    )]
    result = build_assembly_from_points(points, tiny_bundle_path, max_workers=1)
    point = result.points[0]
    assert point.unmatched_hole_diameters_mm == [12.0]
    assert point.status == "suspect"
    assert "12mm" in point.error
    assert result.overall_status == "INCOMPLETE"


def test_overall_status_fail_when_a_required_part_is_rejected(tiny_bundle_path):
    """Confirmed real gap this closes: a part that completely fails to
    build (no usable mesh at all) must make the WHOLE assembly FAIL, not
    just disclose it in a per-part list a caller might not inspect."""
    points = [ActionablePoint(
        description="a box", position_mm=(0.0, 0.0, 0.0),
        program_override=[{"op": "finalize", "args": {"target": 99}}],  # references a handle that was never created
    )]
    result = build_assembly_from_points(points, tiny_bundle_path, max_workers=1)
    assert result.points[0].status == "rejected"
    assert result.overall_status == "FAIL"


def test_overall_status_incomplete_when_a_repair_drops_a_step(tiny_bundle_path):
    """A hole that fully consumes its target used to recover only to
    status="partial" (the last good checkpoint, with the drill dropped
    implicitly by Sandbox.execute itself). Since repair.py's
    TRUNCATE_TO_EARLIER_STEP strategy was added ("it can't just give up"),
    execute_with_bounded_repair now actively re-finalizes at that last
    good checkpoint too, so this point reaches status="ok" -- a real,
    finished box. But a step (drill_hole) WAS genuinely dropped to get
    there, which repair_dropped_a_step must disclose, and which
    overall_status must not hide behind a clean PASS."""
    points = [ActionablePoint(
        description="a box", position_mm=(0.0, 0.0, 0.0),
        program_override=[
            {"op": "add_box", "args": {"length": 10, "width": 10, "height": 10, "center": [0, 0, 0]}},
            {"op": "drill_hole", "args": {"target": 0, "radius": 50, "depth": 50, "center": [0, 0, 0], "axis": "z"}},
            {"op": "finalize", "args": {"target": 1}},
        ],
    )]
    result = build_assembly_from_points(points, tiny_bundle_path, max_workers=1)
    assert result.points[0].status == "ok"
    assert result.points[0].repair_applied is True
    assert result.points[0].repair_dropped_a_step is True
    assert any("drill_hole" in a for a in result.points[0].repair_attempts)
    assert result.overall_status == "INCOMPLETE"


def test_overall_status_incomplete_on_a_real_collision(tiny_bundle_path):
    """A collision between two unrelated parts is no longer an automatic
    FAIL: _find_and_resolve_collisions excludes one of the two from the
    combined mesh instead ("it can't just give up" applied at the
    assembly level), so the assembly still produces a usable result --
    just not a fully clean one, hence INCOMPLETE rather than FAIL."""
    points = [
        ActionablePoint(description="a cylinder 20mm diameter, 60mm long", position_mm=(0.0, 0.0, 0.0)),
        ActionablePoint(description="a cylinder 20mm diameter, 60mm long", position_mm=(0.0, 0.0, 0.0)),
    ]
    result = build_assembly_from_points(points, tiny_bundle_path, max_workers=2)
    assert result.overall_status == "INCOMPLETE"
    assert result.mesh is not None
    assert any(p.excluded_from_assembly for p in result.points)


def test_overall_status_incomplete_for_an_empty_assembly():
    from parallel_builder import AssemblyBuildResult

    assert AssemblyBuildResult(mesh=None, points=[], fell_back_to_single_point=False).overall_status == "INCOMPLETE"
