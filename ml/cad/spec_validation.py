"""Correctness-to-spec validation -- the single most-named gap in this
project's own accuracy audit (ACCURACY_AUDIT.md section 3: "validation
measures structural validity, never correctness"). `Sandbox.execute()`'s
`status="ok"` only ever required a watertight mesh with positive volume;
nothing anywhere in the pipeline checked whether the BUILT geometry's
actual dimensions match what the request's own description text claimed.

This is a deliberately SCOPED first version, not the full multi-layer
system a complete correctness-to-spec validator would need (dimensional
accuracy is "Layer 2" of a larger taxonomy; feature completion, spatial
constraints, and negative constraints are separate, harder problems this
module does not attempt). What it DOES do: extract every numeric "NNmm"
dimension claim from a part's own description text (the text this
project's own LLM-decomposition system prompt already requires to be
"plain, dimension-first" -- see llm_decompose.py's STEP 3), and check that
the BUILT mesh has some real measured dimension close to each claim.

Matching claimed numbers to measured mesh properties is intentionally NOT
axis-aware (it doesn't try to determine "this claim is the X extent,
that one is Z") -- for a part whose description reads
"a cylinder 30mm diameter, 80mm tall," the built mesh's extents are
[30, 30, 80] purely from the geometry of a cylinder (diameter governs two
axes), so a fixed claims-to-axes ORDER mapping would not even be
meaningful. Instead, each claim is matched against whichever measured
quantity (any extent, or a volume-derived characteristic length) is
closest to it. This catches the dominant real failure mode -- "the
request said 65mm and nothing on the built part is anywhere near 65mm" --
without pretending to a semantic precision ("which number was the hole
diameter") this version does not have the information to provide.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import trimesh

# Deliberately scoped to millimeters, matching every unit this project's
# own system prompt and tool vocabulary already use exclusively (see
# llm_decompose.py's _SYSTEM_PROMPT -- every worked example is in mm, no
# other unit ever appears in this project's own generated text).
_DIMENSION_CLAIM_RE = re.compile(r"(\d+(?:\.\d+)?)\s*mm")


def extract_claimed_dimensions_mm(description: str) -> list[float]:
    """Every "NNmm"/"NN.Nmm" numeric claim in the text, in the order they
    appear. Deliberately permissive (matches a count, like "four holes,
    20mm from a corner," just as readily as a size) -- false positives
    here just mean one extra claim checked against the mesh, which either
    finds a real match (harmless) or doesn't (reported as unmatched, which
    is still an honest, inspectable result, not a wrong pass/fail)."""
    return [float(m.group(1)) for m in _DIMENSION_CLAIM_RE.finditer(description)]


@dataclass
class DimensionClaimResult:
    claimed_mm: float
    matched: bool
    closest_measured_mm: float | None
    deviation_fraction: float | None  # None when matched is False and nothing was even close


@dataclass
class DimensionValidationReport:
    claims: list[DimensionClaimResult] = field(default_factory=list)

    @property
    def all_matched(self) -> bool:
        return all(c.matched for c in self.claims)

    @property
    def match_rate(self) -> float:
        return sum(1 for c in self.claims if c.matched) / len(self.claims) if self.claims else 1.0

    @property
    def unmatched_claims_mm(self) -> list[float]:
        return [c.claimed_mm for c in self.claims if not c.matched]


def _measured_reference_lengths(mesh: trimesh.Trimesh) -> list[float]:
    """Every real, measured length-like quantity this scoped check is
    willing to compare a claim against: the 3 bounding-box extents (covers
    "a box LxWxH," "a cylinder's height," a diameter showing up as two
    equal extents) and the volume's cube root (a rough characteristic
    size, catches a claim that's meant as an overall "size" rather than a
    specific axis -- e.g. a sphere's diameter, which IS one of the extents
    already, so this mostly adds redundancy for sphere-like shapes rather
    than new false-accepts)."""
    extents = [float(e) for e in mesh.extents]
    characteristic = float(mesh.volume) ** (1.0 / 3.0) if mesh.volume > 0 else 0.0
    return extents + [characteristic]


def validate_dimensions_against_description(
    description: str, mesh: trimesh.Trimesh, tolerance_fraction: float = 0.15,
) -> DimensionValidationReport:
    """Never raises: a mesh with no valid geometry or a description with no
    numeric claims both degrade to an honest, empty-or-unmatched report,
    never an exception -- this runs inside the same build path as every
    other geometry-validation step in this project, which has a standing
    "never crash on a validation question" convention (see
    evaluation/geometry_eval.py, mesh_quality.py)."""
    claims_mm = extract_claimed_dimensions_mm(description)
    report = DimensionValidationReport()
    if not claims_mm:
        return report

    try:
        reference_lengths = _measured_reference_lengths(mesh)
    except Exception:
        reference_lengths = []

    for claimed in claims_mm:
        if not reference_lengths or claimed <= 0:
            report.claims.append(DimensionClaimResult(claimed, False, None, None))
            continue
        closest = min(reference_lengths, key=lambda measured: abs(measured - claimed))
        deviation = abs(closest - claimed) / claimed
        report.claims.append(DimensionClaimResult(
            claimed_mm=claimed, matched=deviation <= tolerance_fraction,
            closest_measured_mm=closest, deviation_fraction=deviation,
        ))
    return report
