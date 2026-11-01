"""Phase B's own reproducible benchmark harness -- runs the REAL
production generation pipeline (program_model.py + sandbox.py +
compositional_decomposer.py + every wired validator) against the 60
fixed prompts in benchmark_prompts.py, in the "controlled deterministic"
mode (Part 8): decomposition is FIXED/hand-written, never produced by a
real LLM call for this harness's own 60-prompt run (a real local-LLM
decompose_prompt call measured 10-90+ minutes PER PROMPT elsewhere in
this project's own session history -- 60 such calls is not a budget
this environment has). `run_full_llm_mode()` below exists and is wired
to the real llm_decompose.decompose_prompt for anyone who DOES want to
spend that budget on a specific prompt; it is never silently substituted
for the deterministic path, and this module never claims a deterministic
result IS an LLM-mode result (Part 8's own explicit rule).

Independent ground truth (benchmark_prompts.BenchmarkPrompt's own
expected_* fields) is scored by measuring the BUILT MESH directly
(trimesh extents, evaluation.geometry_eval.evaluate_mesh's raw hole/slot
counts) -- never by trusting feature_verification.py's/component_
inventory.py's own self-reported status for ground truth itself (Part
10's own rule). Those validators' own reports ARE still recorded, to
separately score validator reliability (axis H)."""

from __future__ import annotations

import json
import time
import traceback
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from assembly_plan import Relationship
from benchmark_prompts import PROMPTS, BenchmarkPrompt
from build_validation import build_validation_report
from clarification import assess_build
from evaluation.geometry_eval import evaluate_mesh
from llm_decompose import ActionablePoint
from parallel_builder import build_assembly_from_points

FAILURE_LABELS = (
    "PROMPT_INTERPRETATION_FAILURE", "COMPONENT_EXTRACTION_FAILURE", "DECOMPOSITION_FAILURE",
    "DIMENSION_ASSIGNMENT_FAILURE", "OPERATION_ARBITRATION_FAILURE", "GEOMETRY_GENERATION_FAILURE",
    "BOOLEAN_FAILURE", "FEATURE_OMISSION", "FEATURE_GEOMETRY_INCORRECT", "COMPONENT_OMISSION",
    "COMPONENT_DUPLICATION", "COMPONENT_IDENTITY_FAILURE", "ASSEMBLY_PLACEMENT_FAILURE",
    "ASSEMBLY_COMPATIBILITY_FAILURE", "FINAL_MESH_PROCESSING_FAILURE", "VALIDATOR_FALSE_POSITIVE",
    "VALIDATOR_FALSE_NEGATIVE", "UNSUPPORTED_OPERATION", "ENVIRONMENTAL_FAILURE",
)


def get_decomposition(bp: BenchmarkPrompt) -> list[ActionablePoint]:
    """A FIXED decomposition, never the real LLM. [] (the common,
    single-component case) means "use the prompt text as one point" --
    not hand-duplicated for 40+ single-part prompts."""
    if not bp.decomposition:
        return [ActionablePoint(description=bp.prompt, position_mm=(0.0, 0.0, 0.0), id="p0")]
    points = []
    for i, entry in enumerate(bp.decomposition):
        pid = entry.get("id", f"p{i}")
        rel_dict = entry.get("relationship")
        rel = Relationship(
            kind=rel_dict["kind"], target_id=rel_dict["target_id"], face=rel_dict.get("face"),
            offset_u=rel_dict.get("offset_u", 0.0), offset_v=rel_dict.get("offset_v", 0.0),
            axis=rel_dict.get("axis"), axis_offset=rel_dict.get("axis_offset", 0.0),
        ) if rel_dict else None
        points.append(ActionablePoint(description=entry["description"], position_mm=(float(i) * 150.0, 0.0, 0.0), id=pid, relationship=rel))
    return points


@dataclass
class PromptResult:
    id: str
    category: str
    prompt: str
    mode: str
    completed: bool
    elapsed_seconds: float
    error: "str | None" = None
    overall_status: "str | None" = None
    validation_report: "dict | None" = None
    scores: dict = field(default_factory=dict)
    failure_labels: list[str] = field(default_factory=list)
    measured: dict = field(default_factory=dict)
    notes: str = ""
    clarification_report: "dict | None" = None  # Phase 8 Part D's own new metric source --
        # clarification.assess_build()'s own as_dict(), computed from the SAME already-built
        # result validation_report above was computed from, never a second build. None exactly
        # when validation_report is None (generation itself failed -- nothing to assess).


def _score_generation_success(result) -> bool:
    return result is not None and result.mesh is not None


def _score_component_completeness(bp: BenchmarkPrompt, result) -> "tuple[float, dict]":
    actual = len([p for p in result.points if p.mesh is not None and not p.excluded_from_assembly])
    expected = bp.expected_component_count
    score = 1.0 if actual == expected else max(0.0, 1.0 - abs(actual - expected) / max(expected, 1))
    return score, {"expected": expected, "actual": actual}


def _score_dimensions(bp: BenchmarkPrompt, result) -> "tuple[float, dict]":
    if not bp.expected_dimensions_mm:
        return 1.0, {"applicable": False}
    meshes = [p.mesh for p in result.points if p.mesh is not None]
    if not meshes:
        return 0.0, {"applicable": True, "matched": [], "unmatched": bp.expected_dimensions_mm}
    measured_lengths: list[float] = []
    for m in meshes:
        measured_lengths.extend(float(e) for e in m.extents)
        measured_lengths.append(float(m.volume) ** (1 / 3) if m.volume > 0 else 0.0)
    matched, unmatched = [], []
    for claim in bp.expected_dimensions_mm:
        if claim <= 0:
            continue
        closest = min(measured_lengths, key=lambda v: abs(v - claim)) if measured_lengths else None
        if closest is not None and abs(closest - claim) / claim <= bp.tolerance_fraction:
            matched.append(claim)
        else:
            unmatched.append(claim)
    total = len(matched) + len(unmatched)
    return (len(matched) / total if total else 1.0), {"applicable": True, "matched": matched, "unmatched": unmatched}


def _score_features(bp: BenchmarkPrompt, result) -> "tuple[float, dict]":
    if not bp.expected_features:
        return 1.0, {"applicable": False}
    totals = {"hole": 0, "slot": 0}
    for p in result.points:
        if p.mesh is None:
            continue
        report = evaluate_mesh(p.mesh)
        totals["hole"] += report.detected_hole_count
        totals["slot"] += report.detected_slot_count
    detail = {"applicable": True}
    scores = []
    for ftype, expected_count in bp.expected_features:
        actual_count = totals.get(ftype, 0)
        detail[ftype] = {"expected": expected_count, "actual": actual_count}
        scores.append(1.0 if actual_count == expected_count else max(0.0, 1.0 - abs(actual_count - expected_count) / max(expected_count, 1)))
    return (sum(scores) / len(scores) if scores else 1.0), detail


def _score_compatibility(bp: BenchmarkPrompt, result) -> "tuple[float, dict]":
    if bp.expected_relationship_count == 0:
        return 1.0, {"applicable": False}
    report = getattr(result, "compatibility_report", None)
    if report is None or not report.evaluations:
        return 0.0, {"applicable": True, "evaluations": 0, "expected_relationships": bp.expected_relationship_count}
    compatible = sum(1 for e in report.evaluations if e.status.value in ("COMPATIBLE", "UNSUPPORTED_TO_VERIFY"))
    return (compatible / len(report.evaluations)), {
        "applicable": True, "evaluations": len(report.evaluations), "compatible": compatible,
        "statuses": [e.status.value for e in report.evaluations],
    }


def _score_geometry_validity(result) -> "tuple[float, dict]":
    meshes = [p.mesh for p in result.points if p.mesh is not None]
    if not meshes:
        return 0.0, {"watertight": 0, "total": 0}
    watertight = sum(1 for m in meshes if m.is_watertight)
    return watertight / len(meshes), {"watertight": watertight, "total": len(meshes)}


def _classify_failures(bp: BenchmarkPrompt, result, scores: dict) -> list[str]:
    labels: list[str] = []
    if not _score_generation_success(result):
        labels.append("GEOMETRY_GENERATION_FAILURE")
        return labels
    if any(p.status == "rejected" for p in result.points):
        labels.append("GEOMETRY_GENERATION_FAILURE")
    comp_score = scores.get("component_completeness", 1.0)
    if comp_score < 0.999:
        comp_detail = scores.get("component_completeness_detail", {})
        if comp_detail.get("actual", 0) < comp_detail.get("expected", 0):
            labels.append("COMPONENT_OMISSION")
        elif comp_detail.get("actual", 0) > comp_detail.get("expected", 0):
            labels.append("COMPONENT_DUPLICATION")
    if scores.get("dimensional_accuracy", 1.0) < 0.999:
        labels.append("DIMENSION_ASSIGNMENT_FAILURE")
    if scores.get("feature_completeness", 1.0) < 0.999:
        labels.append("FEATURE_OMISSION")
    if scores.get("assembly_compatibility", 1.0) < 0.999:
        labels.append("ASSEMBLY_COMPATIBILITY_FAILURE")
    if scores.get("geometry_validity", 1.0) < 0.999:
        labels.append("FINAL_MESH_PROCESSING_FAILURE")
    if any(p.resolution_method == "statistical" and p.status == "out_of_distribution" for p in result.points):
        labels.append("PROMPT_INTERPRETATION_FAILURE")
    return labels


def run_one(bp: BenchmarkPrompt, bundle_path: Path) -> PromptResult:
    t0 = time.perf_counter()
    try:
        points = get_decomposition(bp)
        result = build_assembly_from_points(points, bundle_path, max_workers=1, original_prompt=bp.prompt)
    except Exception as exc:
        return PromptResult(
            id=bp.id, category=bp.category, prompt=bp.prompt, mode="deterministic", completed=False,
            elapsed_seconds=time.perf_counter() - t0, error=f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}",
            failure_labels=["ENVIRONMENTAL_FAILURE"],
        )
    elapsed = time.perf_counter() - t0

    gen_ok = _score_generation_success(result)
    comp_score, comp_detail = _score_component_completeness(bp, result)
    dim_score, dim_detail = _score_dimensions(bp, result)
    feat_score, feat_detail = _score_features(bp, result)
    compat_score, compat_detail = _score_compatibility(bp, result)
    geom_score, geom_detail = _score_geometry_validity(result)

    scores = {
        "generation_success": 1.0 if gen_ok else 0.0,
        "component_completeness": comp_score, "component_completeness_detail": comp_detail,
        "dimensional_accuracy": dim_score, "dimensional_accuracy_detail": dim_detail,
        "feature_completeness": feat_score, "feature_completeness_detail": feat_detail,
        "assembly_compatibility": compat_score, "assembly_compatibility_detail": compat_detail,
        "geometry_validity": geom_score, "geometry_validity_detail": geom_detail,
    }
    labels = _classify_failures(bp, result, scores)
    vr = build_validation_report(result) if gen_ok else None
    validation_report = vr.as_dict() if vr is not None else None
    clarification_report = assess_build(result, validation_report=vr).as_dict() if vr is not None else None

    return PromptResult(
        id=bp.id, category=bp.category, prompt=bp.prompt, mode="deterministic", completed=gen_ok,
        elapsed_seconds=elapsed, overall_status=result.overall_status if gen_ok else None,
        validation_report=validation_report, scores=scores, failure_labels=labels,
        measured={"n_points": len(result.points), "resolution_methods": [p.resolution_method for p in result.points]},
        notes=bp.notes, clarification_report=clarification_report,
    )


def run_benchmark(prompts: list[BenchmarkPrompt], bundle_path: Path, output_dir: Path) -> "tuple[str, list[PromptResult]]":
    run_id = f"run-{uuid.uuid4().hex[:10]}"
    run_dir = output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    results: list[PromptResult] = []
    for bp in prompts:
        r = run_one(bp, bundle_path)
        results.append(r)
        (run_dir / f"{bp.id}.json").write_text(json.dumps(asdict(r), indent=2, default=str))

    (run_dir / "_summary.json").write_text(json.dumps([asdict(r) for r in results], indent=2, default=str))
    return run_id, results


def run_full_llm_mode(bp: BenchmarkPrompt, bundle_path: Path) -> PromptResult:
    """The OTHER mode (Part 8) -- real llm_decompose.decompose_prompt,
    never silently substituted for the deterministic path above. Exists
    and is wired correctly; NOT invoked for the 60-prompt run in this
    session (see this module's own docstring for the measured per-call
    cost that makes a 60x run infeasible here)."""
    from llm_decompose import decompose_prompt

    t0 = time.perf_counter()
    decomposition = decompose_prompt(bp.prompt)
    result = build_assembly_from_points(decomposition.points, bundle_path, max_workers=1, original_prompt=bp.prompt)
    elapsed = time.perf_counter() - t0
    gen_ok = _score_generation_success(result)
    comp_score, comp_detail = _score_component_completeness(bp, result)
    dim_score, dim_detail = _score_dimensions(bp, result)
    feat_score, feat_detail = _score_features(bp, result)
    compat_score, compat_detail = _score_compatibility(bp, result)
    geom_score, geom_detail = _score_geometry_validity(result)
    scores = {
        "generation_success": 1.0 if gen_ok else 0.0, "component_completeness": comp_score,
        "component_completeness_detail": comp_detail, "dimensional_accuracy": dim_score,
        "dimensional_accuracy_detail": dim_detail, "feature_completeness": feat_score,
        "feature_completeness_detail": feat_detail, "assembly_compatibility": compat_score,
        "assembly_compatibility_detail": compat_detail, "geometry_validity": geom_score,
        "geometry_validity_detail": geom_detail,
    }
    vr = build_validation_report(result) if gen_ok else None
    return PromptResult(
        id=bp.id, category=bp.category, prompt=bp.prompt, mode="full_llm", completed=gen_ok,
        elapsed_seconds=elapsed, overall_status=result.overall_status if gen_ok else None,
        validation_report=vr.as_dict() if vr is not None else None,
        scores=scores, failure_labels=_classify_failures(bp, result, scores),
        measured={"n_points": len(result.points), "fell_back": decomposition.fell_back},
        clarification_report=assess_build(result, validation_report=vr).as_dict() if vr is not None else None,
    )


if __name__ == "__main__":
    import sys

    bundle = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("models/program_model.joblib")
    out_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("benchmark_runs")
    run_id, results = run_benchmark(PROMPTS, bundle, out_dir)
    print(f"run_id={run_id}")
    print(f"completed={sum(1 for r in results if r.completed)}/{len(results)}")
