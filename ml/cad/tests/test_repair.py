"""Tests for repair.py -- section 8's bounded repair loop.

HONEST NOTE on test strategy: manifold3d (this project's boolean kernel)
proved more robust than expected during this work -- a deliberate attempt
to construct a naturally-failing boolean case (touching boxes, exact
tangent cylinders, coincident boxes, a thin-sliver overlap) did not
produce one real failure to repair. Rather than skip testing the repair
LOGIC itself, these tests prove its selection/swap/bookkeeping behavior
is correct via controlled failure injection (Sandbox.execute mocked to
simulate a fail-then-succeed sequence) -- the part of this module that is
actually new code, as opposed to Sandbox.execute's own boolean-op
reliability, which is tested elsewhere (test_sandbox.py) and was not
itself modified by this repair mechanism."""

from unittest.mock import patch

from sandbox import ProgramExecutionResult, ToolCall
from repair import (
    RepairedExecutionResult,
    _find_last_commutative_boolean_before,
    _swap_boolean_operands,
    execute_with_bounded_repair,
)


def _fake_result(status, failing_step=None, mesh=None):
    return ProgramExecutionResult(status=status, mesh=mesh, executed_steps=1, failing_step=failing_step, error="x" if status != "ok" else None)


def test_swap_boolean_operands_swaps_a_and_b_only():
    program = [
        ToolCall(op="add_box", args={"length": 10, "width": 10, "height": 10, "center": [0, 0, 0]}),
        ToolCall(op="add_box", args={"length": 10, "width": 10, "height": 10, "center": [5, 0, 0]}),
        ToolCall(op="boolean_union", args={"a": 0, "b": 1}),
    ]
    repaired = _swap_boolean_operands(program, 2)
    assert repaired[2].args == {"a": 1, "b": 0}
    assert repaired[0] == program[0] and repaired[1] == program[1]  # other steps untouched
    assert program[2].args == {"a": 0, "b": 1}  # original program never mutated


def test_find_last_commutative_boolean_skips_boolean_difference():
    """boolean_difference is NOT commutative -- must never be selected for
    an operand swap, since swapping would change what the operation means
    (A-B vs B-A), not just how it's computed."""
    program = [
        ToolCall(op="add_box", args={}),
        ToolCall(op="boolean_difference", args={"a": 0, "b": 1}),
        ToolCall(op="add_box", args={}),
        ToolCall(op="boolean_union", args={"a": 2, "b": 3}),
    ]
    assert _find_last_commutative_boolean_before(program, failing_step=None) == 3


def test_find_last_commutative_boolean_returns_none_when_there_is_no_boolean():
    program = [ToolCall(op="add_box", args={}), ToolCall(op="shell", args={"target": 0, "wall_thickness": 2})]
    assert _find_last_commutative_boolean_before(program, failing_step=None) is None


def test_execute_with_bounded_repair_does_nothing_on_a_successful_program():
    program = [ToolCall(op="finalize", args={"target": 0})]
    with patch("repair.Sandbox.execute", return_value=_fake_result("ok")) as mock_exec:
        outcome = execute_with_bounded_repair(program)
    assert outcome.result.status == "ok"
    assert outcome.attempts == []
    assert outcome.was_repaired is False
    mock_exec.assert_called_once()  # no retry attempted


def test_execute_with_bounded_repair_recovers_via_operand_swap():
    """The core mechanism: first execute() call fails (suspect), the
    repaired (operand-swapped) program's execute() call succeeds (ok) --
    the final result must be the IMPROVED one, and the attempt must be
    logged as improved=True."""
    program = [
        ToolCall(op="add_box", args={"length": 10, "width": 10, "height": 10, "center": [0, 0, 0]}),
        ToolCall(op="add_box", args={"length": 10, "width": 10, "height": 10, "center": [5, 0, 0]}),
        ToolCall(op="boolean_union", args={"a": 0, "b": 1}),
        ToolCall(op="finalize", args={"target": 2}),
    ]
    results = [_fake_result("suspect"), _fake_result("ok")]
    with patch("repair.Sandbox.execute", side_effect=results) as mock_exec:
        outcome = execute_with_bounded_repair(program)
    assert outcome.result.status == "ok"
    assert outcome.was_repaired is True
    assert len(outcome.attempts) == 1
    assert outcome.attempts[0].strategy == "swap_commutative_boolean_operands"
    assert outcome.attempts[0].step_index == 2
    assert outcome.attempts[0].improved is True
    assert mock_exec.call_count == 2
    # the actual call that produced the "ok" result must have had swapped operands
    repaired_program_passed = mock_exec.call_args_list[1].args[0]
    assert repaired_program_passed[2].args == {"a": 1, "b": 0}


def test_execute_with_bounded_repair_logs_a_failed_attempt_honestly():
    """If the repair doesn't help, this must be reported, not hidden --
    the original (unimproved) result stays authoritative."""
    program = [
        ToolCall(op="add_box", args={"length": 10, "width": 10, "height": 10, "center": [0, 0, 0]}),
        ToolCall(op="add_box", args={"length": 10, "width": 10, "height": 10, "center": [5, 0, 0]}),
        ToolCall(op="boolean_union", args={"a": 0, "b": 1}),
        ToolCall(op="finalize", args={"target": 2}),
    ]
    results = [_fake_result("suspect"), _fake_result("suspect")]  # repair doesn't help
    with patch("repair.Sandbox.execute", side_effect=results):
        outcome = execute_with_bounded_repair(program)
    assert outcome.result.status == "suspect"
    assert outcome.was_repaired is False
    assert len(outcome.attempts) == 1
    assert outcome.attempts[0].improved is False


def test_execute_with_bounded_repair_stops_when_nothing_is_repairable():
    """A failure with no commutative boolean anywhere in the program (e.g.
    a rejected shell() call) must not retry anything -- there's nothing
    this strategy can act on, and section 8 forbids guessing beyond what's
    actually a known, safe repair."""
    program = [ToolCall(op="add_box", args={}), ToolCall(op="shell", args={"target": 0, "wall_thickness": 2})]
    with patch("repair.Sandbox.execute", return_value=_fake_result("rejected")) as mock_exec:
        outcome = execute_with_bounded_repair(program)
    assert outcome.attempts == []
    assert mock_exec.call_count == 1  # only the original attempt -- no blind retry


def test_execute_with_bounded_repair_is_bounded_by_max_attempts():
    program = [
        ToolCall(op="add_box", args={}), ToolCall(op="add_box", args={}),
        ToolCall(op="boolean_union", args={"a": 0, "b": 1}), ToolCall(op="finalize", args={"target": 2}),
    ]
    # every call returns "suspect" -- repair never helps -- confirms it stops after max_attempts, not forever
    with patch("repair.Sandbox.execute", return_value=_fake_result("suspect")) as mock_exec:
        outcome = execute_with_bounded_repair(program, max_attempts=1)
    assert mock_exec.call_count == 2  # original + exactly 1 bounded retry, never more
    assert len(outcome.attempts) == 1
