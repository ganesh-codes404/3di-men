"""Consistency tests for capability_registry.py -- section 10's "machine-
readable capability registry" requirement. The critical property: this
registry must never silently drift from sandbox.py's REAL tool roster
(TOOL_OPS), in either direction -- a tool with no registry entry would be
invisible to anything that consults the registry before trusting an op
name, and a registry entry for a nonexistent tool would let a planner
"invent" an op that isn't actually implemented."""

from sandbox import TOOL_OPS
from capability_registry import full_registry, get_capability, is_known_tool


def test_every_tool_op_has_a_registry_entry():
    registry = full_registry()
    missing = set(TOOL_OPS) - set(registry.keys())
    assert not missing, f"TOOL_OPS entries with no capability_registry row: {sorted(missing)}"


def test_every_registry_entry_is_a_real_tool_op():
    registry = full_registry()
    extra = set(registry.keys()) - set(TOOL_OPS)
    assert not extra, f"capability_registry rows naming a nonexistent tool: {sorted(extra)}"


def test_is_known_tool_matches_tool_ops_exactly():
    for op in TOOL_OPS:
        assert is_known_tool(op), f"{op} is a real tool but is_known_tool() says no"
    assert not is_known_tool("not_a_real_tool_xyz")


def test_every_entry_has_a_real_purpose_and_valid_fidelity_and_risk_class():
    for op, cap in full_registry().items():
        assert cap.op == op
        assert cap.purpose, f"{op}: empty purpose"
        assert cap.fidelity in ("exact", "kernel_dependent", "approximate")
        assert cap.risk_class in ("additive", "subtractive", "transformative", "topology_changing", "replacement")


def test_get_capability_returns_none_for_unknown_op():
    assert get_capability("not_a_real_tool_xyz") is None
    assert get_capability("add_box") is not None


def test_approximate_tools_all_document_a_known_failure_mode():
    """An "approximate" fidelity classification without any documented
    failure mode would be an unexplained admission, not useful
    information -- every approximate tool must say WHY."""
    for op, cap in full_registry().items():
        if cap.fidelity == "approximate":
            assert cap.known_failure_modes, f"{op} is marked approximate but documents no known failure mode"


def test_replacement_risk_tools_document_their_precondition():
    """Section 6's safety rules single out replacement/destructive
    operations for stronger guarantees -- every tool in this risk class
    must document what precondition protects against misuse (even if that
    precondition is "none yet," which must be stated, not omitted)."""
    for op, cap in full_registry().items():
        if cap.risk_class == "replacement":
            assert cap.preconditions, f"{op} is risk_class=replacement but documents no precondition"
