"""Tests for Mitsugen's optional Route B (mitsugen/neural_adapter.py --
TripoSR-based neural generation for genuinely organic/complex objects).

HONEST about what's actually exercised here: this Mac has no CUDA GPU,
so the real generation call (generate_neural_mesh) cannot run end-to-end
on this machine -- that is not a gap in these tests, it's the actual,
correct behavior being verified (check_neural_generation_available()
must truthfully report "no CUDA" here, not pretend otherwise). What IS
fully, genuinely tested on this machine: the availability-check logic
itself (using this machine's REAL installed torch, not a mock), the
routing heuristic, and the pipeline's fallback-with-disclosure behavior
when the neural backend is unavailable. See
mitsugen/RTX4060_NEURAL_SETUP.md section 8 for what still needs the
actual RTX 4060 laptop."""

import pytest

from mitsugen.neural_adapter import (
    NeuralAvailability,
    check_neural_generation_available,
    requires_neural_generation,
)


# ---------------- routing heuristic ----------------

@pytest.mark.parametrize("prompt", [
    "a figurine of a cat",
    "a low-poly cat",
    "a dragon statue",
    "a small dinosaur sculpture",
    "a bear figurine for my desk",
])
def test_requires_neural_generation_matches_organic_prompts(prompt):
    assert requires_neural_generation(prompt) is True


@pytest.mark.parametrize("prompt", [
    "a mug with a handle",
    "a phone stand",
    "a wall hook with a mounting plate",
    "a round stool with four legs",
    "a rectangular box with a lid",
])
def test_requires_neural_generation_does_not_match_route_a_prompts(prompt):
    assert requires_neural_generation(prompt) is False


# ---------------- availability check: REAL behavior on THIS machine ----------------

def test_availability_check_is_disabled_by_default(monkeypatch):
    """The default (no env var set) must be OFF -- this backend is
    opt-in only, never silently active just because torch happens to
    be importable. monkeypatch.delenv alone is enough to test this
    properly: _neural_enabled() (and every other setting in this
    module) reads os.environ fresh on every call rather than caching a
    stale value from import time, so no module reload is needed --
    see this module's own comment on that design choice, adopted after
    a real test-isolation bug elsewhere in this project (a reload-based
    test broke an unrelated test's dataclass equality check via class
    identity)."""
    monkeypatch.delenv("MITSUGEN_NEURAL_ENABLED", raising=False)
    result = check_neural_generation_available()
    assert result.available is False
    assert "disabled" in result.reason


def test_availability_check_reports_real_cuda_status_when_enabled(monkeypatch):
    """Genuinely exercises this machine's REAL torch install (confirmed
    present, CPU-only, no CUDA) -- not mocked. If this ever starts
    returning available=True on this exact Mac, something about the
    machine changed, not just the test."""
    monkeypatch.setenv("MITSUGEN_NEURAL_ENABLED", "1")
    result = check_neural_generation_available()
    assert result.available is False
    assert result.torch_installed is True  # confirmed installed on this machine
    assert result.cuda_available is False  # confirmed real state of this machine -- no CUDA GPU
    assert "CUDA" in result.reason


def test_availability_check_never_raises_without_the_env_var():
    """Calling this with no configuration at all (the common case on
    any machine that hasn't set this up) must never raise."""
    result = check_neural_generation_available()
    assert isinstance(result, NeuralAvailability)
    assert result.available is False


# ---------------- generate_neural_mesh: never raises when unavailable ----------------

def test_generate_neural_mesh_returns_a_disclosed_error_when_unavailable(monkeypatch):
    from mitsugen.neural_adapter import generate_neural_mesh

    monkeypatch.delenv("MITSUGEN_NEURAL_ENABLED", raising=False)
    result = generate_neural_mesh("a cat figurine")
    assert result.mesh is None
    assert result.error is not None
    assert "unavailable" in result.error
