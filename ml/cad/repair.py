"""A bounded, evidence-driven repair loop -- section 8's explicit
requirement, connecting Sandbox.execute()'s already-typed failure statuses
to an actual recovery attempt instead of stopping at "reported, not
dropped" (the behavior every prior piece of this project's own work has
stopped at -- see MISSION_STATUS.md's repeatedly-flagged "next highest-
impact improvement").

Deliberately narrow in scope, not a general auto-fixer: section 8's own
repair constraints rule out almost every tempting "fix" a program's
execution failure might suggest -- "Never change explicit user dimensions
without permission" rules out adjusting shell()'s wall_thickness on a
rejected shell; "Never silently remove a requested feature to recover
from failure" rules out dropping a fillet/round-edges step that failed
its box precondition. What's left, that's genuinely safe under both
constraints simultaneously, is narrower: retrying a BOOLEAN operation
with its operands in the opposite order. boolean_union/intersection are
mathematically commutative (A union B == B union A), so this changes
NEITHER the requested dimensions NOR the requested feature -- it only
gives manifold3d's own CSG implementation a different numerical path
through the same operation, which is a well-known, legitimate robustness
technique for boundary-representation booleans (confirmed as the kind of
operand-order sensitivity that causes real boolean failures -- see
sandbox.py's own boolean_union/difference/intersection docstrings and
Sandbox.execute()'s own immediate-validation comment on why this class of
failure is real, not hypothetical).

boolean_difference is NOT commutative (A - B != B - A) and is
deliberately excluded -- swapping its operands would silently change what
the operation MEANS (which solid is being cut from which), not just how
it's computed, which is exactly the kind of requirement-weakening repair
section 8 forbids.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from sandbox import ProgramExecutionResult, Sandbox, ToolCall

_COMMUTATIVE_BOOLEAN_OPS = {"boolean_union", "boolean_intersection"}
_RETRYABLE_STATUSES = {"suspect", "partial", "rejected"}


@dataclass
class RepairAttempt:
    strategy: str
    step_index: int
    description: str
    improved: bool  # True only if the repaired program's result status strictly improved


@dataclass
class RepairedExecutionResult:
    result: ProgramExecutionResult
    attempts: list[RepairAttempt] = field(default_factory=list)

    @property
    def was_repaired(self) -> bool:
        return any(a.improved for a in self.attempts)


_STATUS_RANK = {"rejected": 0, "partial": 1, "suspect": 2, "ok": 3}  # higher is strictly better


def _swap_boolean_operands(program: list[ToolCall], step_index: int) -> list[ToolCall]:
    repaired = list(program)
    call = repaired[step_index]
    repaired[step_index] = ToolCall(op=call.op, args={**call.args, "a": call.args["b"], "b": call.args["a"]})
    return repaired


def _find_last_commutative_boolean_before(program: list[ToolCall], failing_step: int | None) -> int | None:
    """The boolean step most likely responsible for a failure at
    `failing_step` (or at the very end, if failing_step is None -- a
    whole-program-level "suspect"/empty-mesh outcome) is the LAST
    commutative boolean op that ran before it, not necessarily the
    failing step itself (Sandbox.execute()'s own immediate per-step
    validation, added this mission, means the step that actually produced
    bad geometry IS normally the failing step -- but a whole-program
    "suspect" with no specific failing_step can only be attributed to
    whatever boolean op ran last)."""
    upper_bound = failing_step if failing_step is not None else len(program)
    for i in range(min(upper_bound, len(program)) - 1, -1, -1):
        if program[i].op in _COMMUTATIVE_BOOLEAN_OPS:
            return i
    return None


def execute_with_bounded_repair(program: list[ToolCall], max_attempts: int = 1) -> RepairedExecutionResult:
    """Runs Sandbox.execute() normally; on a non-"ok" result, attempts up
    to `max_attempts` bounded repairs (default 1, matching section 8's
    "default to at most two repair attempts per failed step" guidance,
    kept at the conservative end since this repair is narrow and a second
    attempt on the SAME boolean step after an operand swap has no further
    legitimate variation to try). Returns the best result found and a
    full, honest log of every attempt -- including attempts that did NOT
    improve anything, never hidden."""
    result = Sandbox.execute(program)
    outcome = RepairedExecutionResult(result=result)
    if result.status == "ok" or result.status not in _RETRYABLE_STATUSES:
        return outcome

    current_program = program
    current_result = result
    for _ in range(max_attempts):
        step = _find_last_commutative_boolean_before(current_program, current_result.failing_step)
        if step is None:
            break  # nothing this repair strategy can act on -- stop, don't guess further
        repaired_program = _swap_boolean_operands(current_program, step)
        repaired_result = Sandbox.execute(repaired_program)
        improved = _STATUS_RANK[repaired_result.status] > _STATUS_RANK[current_result.status]
        outcome.attempts.append(RepairAttempt(
            strategy="swap_commutative_boolean_operands", step_index=step,
            description=f"retried step {step} ({current_program[step].op}) with operands swapped "
                        f"(commutative -- same requested geometry, different CSG numerical path)",
            improved=improved,
        ))
        if improved:
            current_program, current_result = repaired_program, repaired_result
            outcome.result = repaired_result
        if repaired_result.status == "ok":
            break
    return outcome
