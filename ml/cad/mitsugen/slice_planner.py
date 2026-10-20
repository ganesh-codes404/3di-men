"""Stage B: deterministic slice planning. Given a MitsugenShapeSpec's
radius-vs-height waypoints and its max_slices budget, chooses an ACTUAL
slice count N (1 <= N <= max_slices) and places each slice where the
radius curve's own curvature is highest -- never equally spaced,
never automatically N == max_slices (a plain cylinder needs 2 slices
regardless of how high max_slices is set).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mitsugen.spec import MitsugenShapeSpec

_CURVE_SAMPLES = 400  # fine-resolution sampling of the TRUE radius curve, used only to measure
                        # curvature/approximation error -- never the slice count itself


def _smoothstep(t: np.ndarray) -> np.ndarray:
    return 3 * t * t - 2 * t * t * t


def radius_at(waypoints_mm: list[tuple[float, float]], height_fraction: float) -> float:
    """The TRUE (continuous) radius at a given height fraction -- smoothstep-
    blended between whichever pair of consecutive waypoints bracket it,
    same tangent-continuous convention sandbox.py's add_shouldered_cylinder/
    add_loft already use (reused as a technique, not as a shared function --
    see this package's own __init__.py for why no code is imported from
    there)."""
    fractions = [w[0] for w in waypoints_mm]
    if height_fraction <= fractions[0]:
        return waypoints_mm[0][1]
    if height_fraction >= fractions[-1]:
        return waypoints_mm[-1][1]
    for i in range(len(waypoints_mm) - 1):
        f0, r0 = waypoints_mm[i]
        f1, r1 = waypoints_mm[i + 1]
        if f0 <= height_fraction <= f1:
            local_t = 0.0 if f1 == f0 else (height_fraction - f0) / (f1 - f0)
            return float(r0 + (r1 - r0) * _smoothstep(np.array([local_t]))[0])
    return waypoints_mm[-1][1]  # unreachable for a well-formed, sorted waypoint list


def _sample_curve(waypoints_mm: list[tuple[float, float]], n: int = _CURVE_SAMPLES) -> tuple[np.ndarray, np.ndarray]:
    fracs = np.linspace(0.0, 1.0, n)
    radii = np.array([radius_at(waypoints_mm, float(f)) for f in fracs])
    return fracs, radii


def _approximation_error_fraction(waypoints_mm: list[tuple[float, float]], slice_fracs: np.ndarray) -> float:
    """Builds a piecewise-LINEAR approximation through exactly the
    proposed slice positions, compares it against the TRUE (smoothstep)
    curve at _CURVE_SAMPLES fine points, and returns the worst-case
    deviation as a fraction of the overall radius RANGE (not of each
    local radius, which would blow up near a near-zero radius) -- the
    actual, measurable thing "does N slices represent this shape within
    tolerance" means."""
    true_fracs, true_radii = _sample_curve(waypoints_mm)
    slice_radii = np.array([radius_at(waypoints_mm, float(f)) for f in slice_fracs])
    approx_radii = np.interp(true_fracs, slice_fracs, slice_radii)
    radius_range = max(float(true_radii.max() - true_radii.min()), 1e-6)
    return float(np.max(np.abs(approx_radii - true_radii)) / radius_range)


def _curvature_weighted_positions(waypoints_mm: list[tuple[float, float]], n_interior: int) -> np.ndarray:
    """Places `n_interior` interior slice positions (endpoints are always
    included separately) proportional to the TRUE curve's own local
    curvature -- a flat/linear stretch gets few or none, a sharply
    bending stretch (near a waypoint transition) gets more. Implemented
    as inverse-CDF sampling over a curvature-derived density, the
    standard deterministic way to turn "more slices where it bends more"
    into actual positions rather than a vague heuristic."""
    fracs, radii = _sample_curve(waypoints_mm)
    d2 = np.gradient(np.gradient(radii, fracs), fracs)
    density = np.abs(d2) + 1e-6  # a small floor so a perfectly flat stretch still gets SOME coverage
    cdf = np.cumsum(density)
    cdf = cdf / cdf[-1]
    targets = (np.arange(1, n_interior + 1) - 0.5) / n_interior  # midpoints of n_interior equal CDF-mass buckets
    positions = np.interp(targets, cdf, fracs)
    return np.clip(positions, 1e-6, 1.0 - 1e-6)


@dataclass
class PlannedSlice:
    slice_id: int
    order: int
    height_fraction: float
    position_mm: float  # along the construction axis, from the base
    radius_mm: float


@dataclass
class SlicePlan:
    slices: list[PlannedSlice]
    requested_max_slices: int
    actual_slice_count: int
    approximation_error_fraction: float
    within_tolerance: bool
    limitation: str | None  # None when within_tolerance; otherwise an explicit, actionable explanation


def plan_slices(spec: MitsugenShapeSpec) -> SlicePlan:
    """Chooses the SMALLEST N (2 <= N <= spec.max_slices) whose
    curvature-weighted placement keeps the piecewise-linear
    approximation within spec.tolerance_fraction of the true curve --
    never jumps straight to max_slices "to be safe" (explicitly
    forbidden: "do not automatically use N_max for every object"). If
    even N == max_slices can't meet tolerance, still returns that best
    attempt, but with within_tolerance=False and a specific, actionable
    `limitation` message -- never silently accepted as a success."""
    max_n = max(2, int(spec.max_slices))
    best_fracs = None
    best_error = None
    chosen_n = max_n
    within_tolerance = False

    for n in range(2, max_n + 1):
        if n == 2:
            fracs = np.array([0.0, 1.0])
        else:
            interior = _curvature_weighted_positions(spec.waypoints_mm, n - 2)
            fracs = np.concatenate([[0.0], np.sort(interior), [1.0]])
        error = _approximation_error_fraction(spec.waypoints_mm, fracs)
        best_fracs, best_error, chosen_n = fracs, error, n
        if error <= spec.tolerance_fraction:
            within_tolerance = True
            break

    limitation = None
    if not within_tolerance:
        limitation = (
            f"the requested shape's radius profile cannot be represented within "
            f"{spec.tolerance_fraction:.0%} tolerance using the configured maximum of {max_n} slices "
            f"(best achievable: {best_error:.1%} deviation at {chosen_n} slices) -- increase the maximum "
            f"construction slice count, or simplify the requested shape (fewer/less abrupt transitions)"
        )

    planned = [
        PlannedSlice(
            slice_id=i, order=i, height_fraction=float(f), position_mm=float(f) * spec.height_mm,
            radius_mm=radius_at(spec.waypoints_mm, float(f)),
        )
        for i, f in enumerate(best_fracs)
    ]
    return SlicePlan(
        slices=planned, requested_max_slices=max_n, actual_slice_count=len(planned),
        approximation_error_fraction=best_error, within_tolerance=within_tolerance, limitation=limitation,
    )
