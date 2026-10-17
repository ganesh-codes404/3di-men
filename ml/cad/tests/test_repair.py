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
    _repair_by_truncation,
    _swap_boolean_operands,
    _truncate_and_refinalize,
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
    the original (unimproved) result stays authoritative. Since the
    operand swap fails to improve anything, truncate_to_earlier_step is
    also tried (the "it can't just give up" fallback) -- also logged,
    also honestly marked as not having helped, since Sandbox.execute is
    mocked to always return "suspect" regardless of program content."""
    program = [
        ToolCall(op="add_box", args={"length": 10, "width": 10, "height": 10, "center": [0, 0, 0]}),
        ToolCall(op="add_box", args={"length": 10, "width": 10, "height": 10, "center": [5, 0, 0]}),
        ToolCall(op="boolean_union", args={"a": 0, "b": 1}),
        ToolCall(op="finalize", args={"target": 2}),
    ]
    with patch("repair.Sandbox.execute", return_value=_fake_result("suspect")):
        outcome = execute_with_bounded_repair(program)
    assert outcome.result.status == "suspect"
    assert outcome.was_repaired is False
    # 1 operand-swap attempt + 3 truncation tries (keep=3,2,1 -- non_finalize_len is 3)
    assert len(outcome.attempts) == 4
    assert all(a.improved is False for a in outcome.attempts)
    assert outcome.attempts[0].strategy == "swap_commutative_boolean_operands"
    assert all(a.strategy == "truncate_to_earlier_step" for a in outcome.attempts[1:])


def test_execute_with_bounded_repair_stops_swap_but_still_tries_truncation():
    """A failure with no commutative boolean anywhere in the program (e.g.
    a rejected shell() call) must not retry the operand-swap strategy --
    there's nothing it can act on, and section 8 forbids guessing beyond
    what's actually a known, safe repair. But truncation is a SEPARATE,
    last-resort strategy and must still get its chance ("it can't just
    give up") rather than the whole repair loop giving up just because
    its first strategy found nothing to do."""
    program = [ToolCall(op="add_box", args={}), ToolCall(op="shell", args={"target": 0, "wall_thickness": 2})]
    with patch("repair.Sandbox.execute", return_value=_fake_result("rejected")) as mock_exec:
        outcome = execute_with_bounded_repair(program)
    assert all(a.strategy == "truncate_to_earlier_step" for a in outcome.attempts)
    assert len(outcome.attempts) == 2  # non_finalize_len=2 (no trailing finalize step here) -- keep=2, then keep=1
    assert outcome.was_repaired is False
    assert mock_exec.call_count == 3  # original + 2 truncation tries, no swap attempt


def test_execute_with_bounded_repair_is_bounded_by_max_attempts():
    program = [
        ToolCall(op="add_box", args={}), ToolCall(op="add_box", args={}),
        ToolCall(op="boolean_union", args={"a": 0, "b": 1}), ToolCall(op="finalize", args={"target": 2}),
    ]
    # every call returns "suspect" -- repair never helps -- confirms the
    # operand-swap strategy stops after max_attempts, not forever (truncation
    # has its own separate bound, max_tries, asserted on its own below)
    with patch("repair.Sandbox.execute", return_value=_fake_result("suspect")) as mock_exec:
        outcome = execute_with_bounded_repair(program, max_attempts=1)
    swap_attempts = [a for a in outcome.attempts if a.strategy == "swap_commutative_boolean_operands"]
    truncation_attempts = [a for a in outcome.attempts if a.strategy == "truncate_to_earlier_step"]
    assert len(swap_attempts) == 1  # exactly 1 bounded retry, never more
    assert len(truncation_attempts) == 3  # non_finalize_len=3 -- keep=3,2,1
    assert mock_exec.call_count == 1 + 1 + 3  # original + swap retry + truncation tries


def test_truncate_and_refinalize_keeps_first_n_steps_and_refinalizes():
    program = [
        ToolCall(op="add_box", args={"length": 1}),
        ToolCall(op="add_box", args={"length": 2}),
        ToolCall(op="add_box", args={"length": 3}),
        ToolCall(op="finalize", args={"target": 2}),
    ]
    truncated = _truncate_and_refinalize(program, keep_steps=2)
    assert len(truncated) == 3
    assert truncated[0] == program[0] and truncated[1] == program[1]
    assert truncated[2] == ToolCall(op="finalize", args={"target": 1})


def test_execute_with_bounded_repair_rescues_a_bad_finalize_target():
    """Real (unmocked) integration case: a program whose finalize target
    is simply wrong (e.g. a stale handle left over from a dropped later
    step) is the simplest case TRUNCATE_TO_EARLIER_STEP exists for -- keep
    every real step, just correct the finalize target itself. No
    commutative boolean exists here, so operand-swap can't help; this
    exercises only the new fallback."""
    program = [
        ToolCall(op="add_box", args={"length": 40, "width": 30, "height": 20, "center": [0, 0, 0]}),
        ToolCall(op="finalize", args={"target": 99}),  # bad reference
    ]
    outcome = execute_with_bounded_repair(program)
    assert outcome.result.status == "ok"
    assert outcome.was_repaired is True
    assert outcome.result.mesh is not None
    assert sorted(outcome.result.mesh.extents) == [20.0, 30.0, 40.0]
    truncation_attempts = [a for a in outcome.attempts if a.strategy == "truncate_to_earlier_step"]
    assert len(truncation_attempts) == 1  # succeeds immediately at keep=non_finalize_len -- nothing dropped
    assert truncation_attempts[0].improved is True
    assert "kept every real step" in truncation_attempts[0].description


def test_execute_with_bounded_repair_drops_a_broken_trailing_step_and_still_completes():
    """A later step that structurally fails (bad handle reference caught
    by Sandbox.validate()) has no commutative boolean to swap and nothing
    else the operand-swap strategy can act on -- truncation is the only
    thing standing between this and a bare rejection. Confirms a real,
    finished solid still comes out, missing only the broken step, and the
    drop is disclosed via RepairAttempt.description, never silent."""
    program = [
        ToolCall(op="add_box", args={"length": 40, "width": 30, "height": 20, "center": [0, 0, 0]}),
        ToolCall(op="translate", args={"target": 99, "dx": 1, "dy": 0, "dz": 0}),  # bad handle
        ToolCall(op="finalize", args={"target": 1}),
    ]
    outcome = execute_with_bounded_repair(program)
    assert outcome.result.status == "ok"
    assert outcome.was_repaired is True
    assert outcome.result.mesh is not None
    truncation_attempts = [a for a in outcome.attempts if a.strategy == "truncate_to_earlier_step"]
    successful = [a for a in truncation_attempts if a.improved]
    assert len(successful) == 1
    assert "translate" in successful[0].description  # discloses exactly what got dropped


def test_execute_with_bounded_repair_reports_honestly_when_truncation_cannot_help():
    """When nothing -- not operand swap, not any amount of truncation --
    fixes the failure, the original (unimproved) result must stay
    authoritative, and every attempt must still be logged, never hidden."""
    program = [
        ToolCall(op="add_box", args={}),
        ToolCall(op="add_box", args={}),
        ToolCall(op="finalize", args={"target": 1}),
    ]
    with patch("repair.Sandbox.execute", return_value=_fake_result("rejected")):
        outcome = execute_with_bounded_repair(program)
    assert outcome.result.status == "rejected"
    assert outcome.was_repaired is False
    assert len(outcome.attempts) == 2  # non_finalize_len=2 -- keep=2, then keep=1, neither helps
    assert all(a.improved is False for a in outcome.attempts)
    assert all(a.strategy == "truncate_to_earlier_step" for a in outcome.attempts)


def test_repair_by_truncation_is_bounded_by_max_tries():
    """Even with far more steps available to drop, truncation must stop
    at max_tries -- it's a bounded repair loop, not an unbounded search."""
    program = [
        ToolCall(op="add_box", args={}), ToolCall(op="add_box", args={}),
        ToolCall(op="add_box", args={}), ToolCall(op="add_box", args={}),
        ToolCall(op="add_box", args={}),
        ToolCall(op="finalize", args={"target": 4}),
    ]
    with patch("repair.Sandbox.execute", return_value=_fake_result("rejected")):
        outcome = execute_with_bounded_repair(program)
    truncation_attempts = [a for a in outcome.attempts if a.strategy == "truncate_to_earlier_step"]
    assert len(truncation_attempts) == 3  # bounded by max_tries, not non_finalize_len (5)


def test_repair_by_truncation_never_goes_below_one_surviving_step():
    """An empty program has nothing to finalize at all -- truncation must
    never reduce `keep` to 0."""
    program = [ToolCall(op="add_box", args={}), ToolCall(op="finalize", args={"target": 0})]
    outcome_holder = RepairedExecutionResult(result=_fake_result("rejected"))
    with patch("repair.Sandbox.execute", return_value=_fake_result("rejected")):
        _repair_by_truncation(program, _fake_result("rejected"), outcome_holder)
    assert all(a.step_index >= 0 for a in outcome_holder.attempts)  # step_index = keep - 1, keep never 0
