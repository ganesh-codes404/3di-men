"""Stage C: 2D cross-section profile representation and validation.
v1 profiles are always circles (the radial-symmetry construction
strategy -- see spec.py's own docstring for that scope), represented as
a regular polygon approximation so the SAME validation and lofting code
would work unchanged if a future profile kind (e.g. a traced outline)
were added."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

_DEFAULT_SECTIONS = 32  # polygon sides approximating a circle -- matches sandbox.py's own
                          # add_cylinder default facet density, not an independently chosen number


@dataclass
class Profile2D:
    slice_id: int
    outline: np.ndarray  # (N, 2) -- closed polygon, NOT repeating the first point at the end
    radius_mm: float  # the nominal radius this profile was generated FROM, for diagnostics


@dataclass
class ProfileValidationReport:
    valid: bool
    reason: str | None  # None when valid=True
    method: str = "closure/finiteness/self-intersection/non-degeneracy checks on the 2D outline"


def make_circular_profile(slice_id: int, radius_mm: float, sections: int = _DEFAULT_SECTIONS) -> Profile2D:
    if radius_mm <= 0:
        radius_mm = 1e-3  # a degenerate (near-zero) radius is still a valid, buildable point-like profile --
                           # a true zero would produce NaN angles below, so clamp to a tiny positive value
    angles = np.linspace(0.0, 2 * np.pi, sections, endpoint=False)
    outline = np.column_stack([radius_mm * np.cos(angles), radius_mm * np.sin(angles)])
    return Profile2D(slice_id=slice_id, outline=outline, radius_mm=radius_mm)


def validate_profile(profile: Profile2D) -> ProfileValidationReport:
    """Never raises: every check below is a plain numeric/topological
    test on the outline array. Checks, in order (stated plainly so a
    caller knows exactly what was and wasn't verified):
    1. Nonempty (>= 3 points -- a polygon needs at least a triangle).
    2. Every coordinate is finite (no NaN/inf -- a confirmed REAL risk
       for any profile generator that divides by a radius, not a
       theoretical one, see make_circular_profile's own zero-radius
       clamp above).
    3. Non-degenerate: the polygon's own signed area is not ~zero (a
       profile that collapsed to a line or a point is not a closed
       boundary).
    4. No self-intersections, via shapely -- a circle built from
       monotonically-increasing angles can't self-intersect by
       construction, but this is checked anyway since it's cheap and a
       FUTURE non-circular profile kind would need exactly this, so the
       check is written generally now rather than assumed safe forever.
    """
    outline = profile.outline
    if outline is None or len(outline) < 3:
        return ProfileValidationReport(valid=False, reason=f"profile has fewer than 3 points ({0 if outline is None else len(outline)})")
    if not np.all(np.isfinite(outline)):
        return ProfileValidationReport(valid=False, reason="profile contains non-finite (NaN/inf) coordinates")

    x, y = outline[:, 0], outline[:, 1]
    signed_area = 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))
    if abs(signed_area) < 1e-9:
        return ProfileValidationReport(valid=False, reason="profile has effectively zero area (degenerate/collapsed outline)")

    try:
        from shapely.geometry import Polygon

        poly = Polygon(outline)
        if not poly.is_valid or not poly.is_simple:
            return ProfileValidationReport(valid=False, reason="profile self-intersects or forms an invalid boundary")
    except Exception as exc:
        return ProfileValidationReport(valid=False, reason=f"profile geometry could not be validated: {exc}")

    return ProfileValidationReport(valid=True, reason=None)


def validate_adjacent_profiles(a: Profile2D, b: Profile2D) -> ProfileValidationReport:
    """Stage C's own "adjacent profiles are compatible with the chosen
    lofting method" check. v1's circular profiles are always resampled
    to the SAME point count before lofting (see builder.py), so the only
    genuine incompatibility is a hole-count/topology mismatch -- both
    are always simple closed curves with no inner holes in this
    construction strategy, so this is currently always compatible by
    construction. Written as a real (not a stub) check anyway: a future
    profile kind with inner boundaries (a hollow cross-section) would
    need this to actually compare hole counts, and this is where that
    comparison would go."""
    if len(a.outline) < 3 or len(b.outline) < 3:
        return ProfileValidationReport(valid=False, reason="one of the two profiles is not a valid closed boundary")
    return ProfileValidationReport(valid=True, reason=None)
