"""Tests for decompose_eval.py -- llm_decompose.py's missing accuracy
harness. decompose_prompt itself is mocked throughout (no real LLM calls in
the automated suite, matching this project's established convention) --
a real-model run is a separate, explicitly-invoked script
(scripts/run_decomposition_eval.py), not part of `pytest tests/`."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from assembly_plan import Relationship
from decompose_eval import (
    DecompositionLabel,
    evaluate_one,
    load_decomposition_labels,
    run_decomposition_eval,
)
from llm_decompose import ActionablePoint, DecomposeResult


def _fake_result(points: list[ActionablePoint], fell_back: bool = False) -> DecomposeResult:
    return DecomposeResult(points=points, raw_llm_output="...", fell_back=fell_back)


def test_load_decomposition_labels_parses_the_real_fixture_file():
    labels = load_decomposition_labels(Path(__file__).resolve().parent.parent / "data" / "eval_decomposition.jsonl")
    assert len(labels) >= 10
    ids = {l.id for l in labels}
    assert "d01_single_primitive" in ids
    first = labels[0]
    assert first.prompt
    assert isinstance(first.expected_relationship_kinds, tuple)


def test_evaluate_one_passes_on_a_correct_single_part_decomposition():
    label = DecompositionLabel(
        id="t1", prompt="a cylinder 30mm diameter, 80mm tall",
        expected_component_count=1, required_substrings=("30mm", "80mm"),
    )
    with patch("decompose_eval.decompose_prompt", return_value=_fake_result(
        [ActionablePoint(description="a cylinder 30mm diameter, 80mm tall", position_mm=(0, 0, 0))]
    )):
        result = evaluate_one(label)
    assert result.overall_pass is True
    assert result.component_count_correct is True
    assert result.required_substrings_present is True


def test_evaluate_one_fails_on_wrong_component_count():
    label = DecompositionLabel(id="t2", prompt="x", expected_component_count=2)
    with patch("decompose_eval.decompose_prompt", return_value=_fake_result(
        [ActionablePoint(description="a box", position_mm=(0, 0, 0))]
    )):
        result = evaluate_one(label)
    assert result.component_count_correct is False
    assert result.overall_pass is False


def test_evaluate_one_fails_on_missing_required_substring():
    label = DecompositionLabel(id="t3", prompt="x", expected_component_count=1, required_substrings=("65mm",))
    with patch("decompose_eval.decompose_prompt", return_value=_fake_result(
        [ActionablePoint(description="a cylinder 40mm diameter", position_mm=(0, 0, 0))]
    )):
        result = evaluate_one(label)
    assert result.required_substrings_present is False
    assert "65mm" in result.missing_required_substrings
    assert result.overall_pass is False


def test_evaluate_one_checks_relationship_kind_usage():
    label = DecompositionLabel(id="t4", prompt="x", expected_component_count=2, expected_relationship_kinds=("attached_to_face",))
    points_with_rel = [
        ActionablePoint(description="a plate", position_mm=(0, 0, 0), id="plate"),
        ActionablePoint(
            description="a boss", position_mm=(0, 0, 0), id="boss",
            relationship=Relationship(kind="attached_to_face", target_id="plate", face="+z"),
        ),
    ]
    with patch("decompose_eval.decompose_prompt", return_value=_fake_result(points_with_rel)):
        result = evaluate_one(label)
    assert result.relationships_found == ("attached_to_face",)
    assert result.relationship_kinds_satisfied is True
    assert result.overall_pass is True

    points_without_rel = [
        ActionablePoint(description="a plate", position_mm=(0, 0, 0)),
        ActionablePoint(description="a boss", position_mm=(0, 0, 10)),
    ]
    with patch("decompose_eval.decompose_prompt", return_value=_fake_result(points_without_rel)):
        result2 = evaluate_one(label)
    assert result2.relationship_kinds_satisfied is False
    assert result2.overall_pass is False


def test_evaluate_one_detects_invented_dimensions_when_expected_none():
    label = DecompositionLabel(id="t5", prompt="x", expected_component_count=1, expect_no_dimensions=True)
    with patch("decompose_eval.decompose_prompt", return_value=_fake_result(
        [ActionablePoint(description="an L-shape bracket, 80mm long", position_mm=(0, 0, 0))]
    )):
        result = evaluate_one(label)
    assert result.has_invented_dimensions is True
    assert result.overall_pass is False

    with patch("decompose_eval.decompose_prompt", return_value=_fake_result(
        [ActionablePoint(description="an L-shape bracket", position_mm=(0, 0, 0))]
    )):
        result2 = evaluate_one(label)
    assert result2.has_invented_dimensions is False
    assert result2.overall_pass is True


def test_evaluate_one_fails_when_llm_falls_back_regardless_of_other_checks():
    label = DecompositionLabel(id="t6", prompt="x", expected_component_count=1)
    with patch("decompose_eval.decompose_prompt", return_value=_fake_result(
        [ActionablePoint(description="x", position_mm=(0, 0, 0))], fell_back=True
    )):
        result = evaluate_one(label)
    assert result.fell_back is True
    assert result.overall_pass is False


def test_component_count_check_is_skipped_when_label_leaves_it_unset():
    label = DecompositionLabel(id="t7", prompt="x", expected_component_count=None)
    with patch("decompose_eval.decompose_prompt", return_value=_fake_result(
        [ActionablePoint(description="a", position_mm=(0, 0, 0)), ActionablePoint(description="b", position_mm=(0, 0, 0))]
    )):
        result = evaluate_one(label)
    assert result.component_count_correct is None
    assert result.overall_pass is True  # nothing else required by this label, so it passes


def test_run_decomposition_eval_aggregates_pass_rate_and_fallback_rate():
    labels = [
        DecompositionLabel(id="a", prompt="x", expected_component_count=1),
        DecompositionLabel(id="b", prompt="x", expected_component_count=1),
    ]
    results = [
        _fake_result([ActionablePoint(description="a", position_mm=(0, 0, 0))]),  # passes (count=1)
        _fake_result([ActionablePoint(description="a", position_mm=(0, 0, 0))], fell_back=True),  # fails (fell back)
    ]
    with patch("decompose_eval.decompose_prompt", side_effect=results):
        summary = run_decomposition_eval(labels)
    assert summary.pass_rate == 0.5
    assert summary.fallback_rate == 0.5
    assert summary.component_count_accuracy == 1.0  # both actually had count==1; fallback doesn't affect this metric
