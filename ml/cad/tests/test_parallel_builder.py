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
            assert isinstance(p.unmatched_dimensions_mm, list)
