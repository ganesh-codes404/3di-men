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

A second, more aggressive strategy was added afterward, by explicit
request: the operand-swap repair alone still gives up on anything it
doesn't apply to (a bad handle reference caught by Sandbox.validate(),
a precondition failure on a non-boolean step, two unrelated points
resolved to a colliding position) -- "it can't just give up" is the
literal requirement this closes. TRUNCATE_TO_EARLIER_STEP removes the
program's own trailing step(s) and re-finalizes at whatever the last
surviving step produced, so a part that's missing one late feature still
completes as a real, finished solid instead of failing outright. This
DOES mean a requested feature can end up dropped -- the one thing this
module's own first strategy was built to avoid -- but it is never done
silently: every attempt (used or not) is logged in RepairAttempt exactly
like the existing strategy, and a repaired-by-truncation result is
disclosed via resolution/repair fields the same way a repaired boolean
swap already is. This is a deliberate, requested trade-off (a complete
model with one feature missing, clearly disclosed, over no model at all),
not a silent regression of the "never drop a requested feature" principle.
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


def _truncate_and_refinalize(program: list[ToolCall], keep_steps: int) -> list[ToolCall]:
    """Keeps only the first `keep_steps` non-finalize steps, re-finalizing
    at the last one's own produced handle. Safe regardless of op category
    (create/consumes_one/consumes_two/consumes_target): every non-finalize
    step calls Sandbox._new_handle() exactly once, so step i always
    produces handle i -- the same assumption program_model.py's own
    handle_offset encoding and every other part of this codebase already
    relies on."""
    kept = list(program[:keep_steps])
    kept.append(ToolCall(op="finalize", args={"target": keep_steps - 1}))
    return kept


def _repair_by_truncation(
    program: list[ToolCall], current_result: ProgramExecutionResult, outcome: "RepairedExecutionResult", max_tries: int = 3,
) -> tuple[list[ToolCall], ProgramExecutionResult]:
    """Last-resort fallback, tried after the operand-swap strategy above
    has nothing left to offer: progressively drops the program's own
    trailing step(s) and re-finalizes earlier, so a part that's missing
    one late feature still completes as a real, finished solid rather
    than failing outright ("it can't just give up"). Bounded to
    `max_tries` attempts and never truncates below one surviving step
    (an empty program has nothing to finalize at all)."""
    non_finalize_len = len(program) - 1 if program and program[-1].op == "finalize" else len(program)
    current_program, current = program, current_result
    # Starts at non_finalize_len itself (keep every real step, just
    # re-finalize at the correct last handle) -- not non_finalize_len - 1
    # -- since the ORIGINAL failure can be nothing more than a bad
    # finalize target itself (every real call site computes this
    # correctly, but a predicted/hand-edited program is not guaranteed
    # to); only decrease further than that if the problem is actually in
    # an earlier step, not the finalize call itself.
    keep = non_finalize_len
    tries = 0
    while keep >= 1 and tries < max_tries and current.status != "ok":
        candidate = _truncate_and_refinalize(program, keep)
        candidate_result = Sandbox.execute(candidate)
        improved = _STATUS_RANK[candidate_result.status] > _STATUS_RANK[current.status]
        dropped_steps = program[keep:non_finalize_len]
        if dropped_steps:
            dropped_ops = ", ".join(c.op for c in dropped_steps)
            description = (
                f"dropped trailing step(s) [{dropped_ops}] and re-finalized at step {keep - 1} "
                f"({program[keep - 1].op}) instead -- the model still completes, just without "
                f"whatever those dropped steps were adding"
            )
        else:
            description = (
                f"kept every real step and re-finalized at step {keep - 1} ({program[keep - 1].op}) -- "
                f"the original failure may have been nothing more than a bad finalize target"
            )
        outcome.attempts.append(RepairAttempt(
            strategy="truncate_to_earlier_step", step_index=keep - 1,
            description=description,
            improved=improved,
        ))
        if improved:
            current_program, current = candidate, candidate_result
        keep -= 1
        tries += 1
    return current_program, current


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

    # Last resort: the operand-swap strategy either doesn't apply (no
    # commutative boolean in the program at all -- e.g. a bad handle
    # reference, or a non-boolean step's own precondition failure) or
    # didn't fully fix it. Never give up here -- a complete model missing
    # one late feature is still a usable result, which bare rejection is
    # not.
    if current_result.status != "ok":
        current_program, current_result = _repair_by_truncation(current_program, current_result, outcome)
        outcome.result = current_result
    return outcome
