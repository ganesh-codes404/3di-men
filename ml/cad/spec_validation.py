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
#
# The negative lookahead excludes a "NNmm from {the/a} {word} edge/corner/
# face" phrase -- a PLACEMENT offset (how far a feature sits from a
# reference edge/corner), not a built SIZE. Confirmed real bug this
# closes: "a slot 10mm from the left edge" used to get checked against the
# mesh's overall extents exactly like a stated dimension, producing a
# false "unmatched: 10mm" warning on a perfectly correct build (the
# project's own existing "Nmm from a corner" hole-margin phrasing, e.g.
# compositional_decomposer.py's corner-array pattern, has always had this
# same false-positive, just never surfaced as confusingly before).
_DIMENSION_CLAIM_RE = re.compile(
    r"(\d+(?:\.\d+)?)\s*mm(?!\s*(?:wide|deep|long|tall|thick)?\s*from\s+(?:the\s+|an?\s+)?(?:\w+\s+)?(?:edges?|corners?|faces?)\b)",
    re.IGNORECASE,
)


def extract_claimed_dimensions_mm(description: str) -> list[float]:
    """Every "NNmm"/"NN.Nmm" numeric claim in the text that reads as a
    built SIZE, in the order they appear -- a "NNmm from an edge/corner/
    face" placement offset is deliberately excluded (see the regex's own
    comment above). Still deliberately permissive beyond that (e.g. a bare
    count like "four holes" with no "from an edge" qualifier would still
    be picked up if phrased as "4mm" somewhere unrelated) -- a remaining
    false positive here just means one extra claim checked against the
    mesh, which either finds a real match (harmless) or doesn't (reported
    as unmatched, still an honest, inspectable result, not a wrong
    pass/fail)."""
    return [float(m.group(1)) for m in _DIMENSION_CLAIM_RE.finditer(description)]


# Assembly-level envelope claim -- a different kind of claim than any
# per-part dimension above: "keep the overall footprint close to 150mm x
# 100mm" describes the WHOLE assembly's intended X/Y footprint, not any one
# part's own size, and is stated in the ORIGINAL PROMPT text (never a per-
# part description). Deliberately permissive about the connecting words
# ("close to"/"of roughly"/"around"/nothing at all) since real prompts
# phrase this loosely by design (an approximate target, not a part spec).
#
# Confirmed real gap (live-reported "dimension issues" that kept recurring
# despite _diagnose_envelope_overrun existing): the ORIGINAL version of
# this regex required the literal word "footprint" -- a prompt phrasing
# the exact same claim as "overall size 150mm x 100mm" (confirmed directly:
# extract_claimed_envelope_mm returned None for it) never even registered
# a claim to check the built assembly against, so the overrun diagnostic
# silently never ran for that whole class of ordinary phrasing. "footprint"
# vs "overall size"/"overall dimensions" now both match -- still requires
# an EXPLICIT assembly-level claim in the prompt's own words (never
# inferred from a single part's own stated size, which is a different,
# unrelated, much riskier guess this deliberately does not make).
_ENVELOPE_CLAIM_RE = re.compile(
    # "footprint" is unambiguous on its own (always means the whole
    # assembly's own footprint, "overall" prefix optional); "size"/
    # "dimensions" are far more generic words that just as often describe
    # a SINGLE part (e.g. "the plate's size is 150mm x 100mm") -- those
    # two only count as an assembly-level claim when "overall" makes that
    # explicit, so this never infers an envelope claim from an ordinary
    # per-part dimension statement.
    r"(?:(?:overall\s+)?footprint|overall\s+(?:size|dimensions?))\s*"
    r"(?:close\s+to|of\s+(?:roughly|approximately)?|around|approximately)?\s*"
    r"(\d+(?:\.\d+)?)\s*(?:mm)?\s*x\s*(\d+(?:\.\d+)?)\s*mm",
    re.IGNORECASE,
)


def extract_claimed_envelope_mm(prompt: str) -> tuple[float, float] | None:
    """The overall assembly's claimed X/Y footprint from the ORIGINAL
    prompt text (not a per-part description) -- e.g. "keep the overall
    footprint close to 150mm x 100mm." None when the prompt makes no such
    claim at all (most prompts don't -- an individual part's own stated
    dimensions are a completely separate, already-covered concern, see
    extract_claimed_dimensions_mm). Only the FIRST match is used: a
    footprint claim is a property of the whole assembly, not a per-
    component list, so multiple matches would be ambiguous, not additive,
    unlike a hole-count claim."""
    m = _ENVELOPE_CLAIM_RE.search(prompt)
    if not m:
        return None
    return (float(m.group(1)), float(m.group(2)))


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


# Feature validation (hole count) -- same scoped-honesty discipline as the
# dimension check above, not a general feature verifier. Added because
# GeometryReport.detected_hole_count (evaluation/geometry_eval.py) was
# already computed by Sandbox.execute() on every build but discarded by
# every caller (confirmed by grep: nothing outside sandbox.py itself ever
# read it) -- this wires the data that already existed into an actual
# check, rather than adding new measurement machinery.
_HOLE_COUNT_WORDS = {
    "a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
}
# "<count word/digit> [NNmm [diameter]] [a descriptive word] hole(s)" --
# matches this project's own established phrasings ("a 6mm diameter hole
# through it", "a corner array of four holes", "four 6.6mm diameter
# holes, each...", "four 8mm diameter through-holes", "two 6mm diameter
# mounting holes"). The trailing descriptive-word group is a SMALL,
# EXPLICIT list, not a generic "\w+" filler -- confirmed real regression
# from an earlier, more permissive version of this fix: a bare "any
# word" filler can swallow the actual count word itself ("a corner
# array of four holes" matched group(1)="array", not "four", because
# "of"+"four" got absorbed as 2 generic filler words before "holes").
# Restricting the filler to real descriptive/connector words that
# actually precede "hole(s)" in practice avoids that ambiguity --
# "of"/"four"/"array" are never in this list, so they can never be
# mistaken for filler.
_HOLE_DESCRIPTOR_WORDS = r"(?:through|mounting|pilot|clearance|locating|alignment|access|drain|vent|screw|bolt|counterbore|countersink|tapped|threaded)"
# "equally/evenly spaced" is a SPECIFIC, explicit connector phrase
# (never a generic "\w+" filler, same discipline as _HOLE_DESCRIPTOR_
# WORDS's own comment above) allowed to sit between the count word and
# the rest -- confirmed real gap this closes (adversarial regression,
# this project's own brief's worked example): "six equally spaced 5mm
# holes" previously matched nothing at all (extract_claimed_hole_count
# returned None, not even a wrong count), because the regex had no
# tolerance for ANY word between the count and the dimension/descriptor
# -- "six" and "5mm" are not adjacent here. Dropping it without the
# phrase (just "six 5mm holes") already matched fine.
_SPACED_CONNECTOR = r"(?:(?:equally|evenly)\s+spaced\s+)?"
_HOLE_COUNT_RE = re.compile(
    rf"\b(\w+)\s+{_SPACED_CONNECTOR}(?:[\d.]+\s*mm\s*(?:diameter\s*)?)?(?:{_HOLE_DESCRIPTOR_WORDS}-?\s*){{0,2}}holes?\b", re.IGNORECASE,
)


def extract_claimed_hole_count(description: str) -> int | None:
    """Sums every "<count> ... hole(s)" phrase found (a description can
    mention more than one group -- a bolt circle AND a separate mounting
    hole). Returns None when the text mentions no holes at all, distinct
    from a found total of zero."""
    total = 0
    found = False
    for m in _HOLE_COUNT_RE.finditer(description):
        word = m.group(1).lower()
        count = _HOLE_COUNT_WORDS.get(word)
        if count is None:
            try:
                count = int(word)
            except ValueError:
                continue
        total += count
        found = True
    return total if found else None


@dataclass
class HoleCountValidationReport:
    claimed_count: int | None  # None when the description made no hole claim at all
    detected_count: int | None  # None when no GeometryReport was available to check against
    matched: bool  # True whenever there was nothing to check (vacuously) OR counts agree exactly

    # detected_hole_count's own disclosed scope (evaluation/geometry_eval.py:
    # tries all 3 world axes, but still axis-ALIGNED only, on largely
    # watertight, extruded parts) -- carried through here so a caller never
    # treats a mismatch as more certain than it actually is.
    detector_limitation: str = (
        "hole detection recognizes cylindrical features running along any of the 3 world axes "
        "(x/y/z) on largely axis-aligned, watertight parts, but not an arbitrarily tilted hole "
        "axis -- a mismatch may reflect that scope, not necessarily a wrong build"
    )


def validate_hole_count_against_description(
    description: str, geometry_report: dict | None,
) -> HoleCountValidationReport:
    """Never raises, same convention as validate_dimensions_against_description.
    Exact-count comparison (no tolerance -- a hole count claim is a whole
    number, there's no meaningful "15% off" for "four holes")."""
    claimed = extract_claimed_hole_count(description)
    detected = geometry_report.get("detected_hole_count") if geometry_report else None
    if claimed is None or detected is None:
        return HoleCountValidationReport(claimed_count=claimed, detected_count=detected, matched=True)
    return HoleCountValidationReport(claimed_count=claimed, detected_count=detected, matched=claimed == detected)


# Feature validation (hole/bore DIAMETER) -- same wiring principle as hole
# count above: GeometryReport.hole_diameters_mm (evaluation/geometry_eval.py)
# was already measurable from each detected hole's own fitted-circle radius
# (_classify_wall_component computes it to score circularity), but that
# radius was discarded the moment circularity was read off it -- nothing
# anywhere checked whether a CLAIMED bore/hole diameter (e.g. "a 12mm bore")
# matches what was actually built, only that SOME hole exists at all
# (hole count) and looks round (circularity). More precise than the general
# dimension check above: that one can only compare a claim against the
# mesh's overall bounding-box extents/characteristic size, which can produce
# a misleading "match" by coincidence (some unrelated extent happens to be
# close to the claimed number) without ever checking the specific FEATURE
# the claim was actually about.
_HOLE_DIAMETER_RE = re.compile(r"(\d+(?:\.\d+)?)\s*mm\s*(?:diameter\s*)?(?:hole|bore)s?\b", re.IGNORECASE)


def extract_claimed_hole_diameters_mm(description: str) -> list[float]:
    """Every "NNmm [diameter] hole(s)/bore(s)" claim, in the order they
    appear. Deliberately permissive like extract_claimed_dimensions_mm --
    a false positive just means one extra claim checked, reported as
    unmatched if nothing detected lines up, never a silently wrong
    pass/fail."""
    return [float(m.group(1)) for m in _HOLE_DIAMETER_RE.finditer(description)]


@dataclass
class HoleDiameterClaimResult:
    claimed_mm: float
    matched: bool
    closest_detected_mm: float | None
    deviation_fraction: float | None  # None when matched is False and nothing was even close


@dataclass
class HoleDiameterValidationReport:
    claims: list[HoleDiameterClaimResult] = field(default_factory=list)
    # Same disclosed scope as HoleCountValidationReport.detector_limitation --
    # carried through here so a caller never treats a diameter mismatch as
    # more certain than it actually is.
    detector_limitation: str = (
        "hole diameter is measured by fitting a circle to each detected axis-aligned cylindrical "
        "feature (any of the 3 world axes) -- the same scope limitation as detected_hole_count "
        "(evaluation/geometry_eval.py); a mismatch may reflect that scope, not necessarily a wrong build"
    )

    @property
    def all_matched(self) -> bool:
        return all(c.matched for c in self.claims)

    @property
    def unmatched_claims_mm(self) -> list[float]:
        return [c.claimed_mm for c in self.claims if not c.matched]


def validate_hole_diameters_against_description(
    description: str, geometry_report: dict | None, tolerance_fraction: float = 0.15,
) -> HoleDiameterValidationReport:
    """Never raises, same convention as validate_dimensions_against_description.
    Each claimed diameter is matched against whichever DETECTED hole
    diameter (not just any mesh extent) is closest to it -- independent of
    which detected hole index it "should" correspond to, same
    not-index-anchored matching principle as the general dimension check."""
    claims_mm = extract_claimed_hole_diameters_mm(description)
    report = HoleDiameterValidationReport()
    if not claims_mm:
        return report

    detected_mm = (geometry_report or {}).get("hole_diameters_mm") or []
    for claimed in claims_mm:
        if not detected_mm or claimed <= 0:
            report.claims.append(HoleDiameterClaimResult(claimed, False, None, None))
            continue
        closest = min(detected_mm, key=lambda measured: abs(measured - claimed))
        deviation = abs(closest - claimed) / claimed
        report.claims.append(HoleDiameterClaimResult(
            claimed_mm=claimed, matched=deviation <= tolerance_fraction,
            closest_detected_mm=closest, deviation_fraction=deviation,
        ))
    return report


# Semantic-name-vs-explicit-dimension conflict detection -- a confirmed,
# previously entirely-unaddressed gap (grepped the whole repo for "NEMA":
# zero hits before this). This project's own non-negotiable principle:
# "explicit numerical dimensions override semantic defaults associated
# with names such as 'NEMA 17'" -- that override already happens
# MECHANICALLY (extract_claimed_dimensions_mm reads whatever explicit
# number the text states, never a name-implied default), but nothing
# ever made the CONFLICT ITSELF visible when a description names a known
# standard size AND ALSO states a dimension that disagrees with it. A
# silent override is still a silent decision -- the point of this
# function is disclosure, not a new enforcement mechanism (the existing
# dimension check already enforces the explicit value).
#
# Deliberately scoped to NEMA stepper-motor mounting faces specifically
# (the brief's own named example), not a general "semantic name" lookup
# for arbitrary terms like "bicycle" -- a broad lookup table of informal
# names to assumed sizes would itself be exactly the kind of fabricated-
# default risk this project avoids elsewhere (see llm_decompose.py's
# STEP 0: realistic sizing is only ever applied for a short, explicitly
# enumerated list of well-known object types, never invented generally).
# Values are the real, standard NEMA mounting-face dimensions (the square
# faceplate a motor of that size actually has), not estimates.
_NEMA_STANDARD_FACE_MM: dict[int, float] = {
    8: 20.3, 11: 28.2, 14: 35.2, 17: 42.3, 23: 57.2, 34: 86.0, 42: 110.0,
}
_NEMA_NAME_RE = re.compile(r"\bNEMA\s*-?\s*(\d+)\b", re.IGNORECASE)


@dataclass
class SemanticSizeConflict:
    semantic_name: str  # e.g. "NEMA 17"
    standard_value_mm: float  # the real standard's own dimension
    explicit_value_mm: float  # what the description ALSO explicitly states
    resolved_value_mm: float  # always == explicit_value_mm -- see this module's own docstring block above


def detect_semantic_size_conflicts(description: str, tolerance_fraction: float = 0.1) -> list[SemanticSizeConflict]:
    """Finds a NEMA-size name in `description` and checks whether the
    SAME description also states an explicit dimension that disagrees
    with that standard's own real mounting-face size. Returns one
    SemanticSizeConflict per disagreeing explicit claim -- empty if no
    NEMA name is present, or if every explicit claim either matches the
    standard or there are no explicit claims to compare at all (nothing
    to disclose a conflict about).

    Only considers a claim that appears AT LEAST TWICE in the
    description (e.g. "50mm x 50mm") -- a NEMA mounting face is always
    square, so a repeated value is the actual signal that a face-size
    claim is being made at all. Confirmed necessary, not a hypothetical
    tightening: an UNREPEATED claim elsewhere in the same sentence (e.g.
    "...50mm x 50mm x 10mm thick") almost always describes a DIFFERENT
    attribute entirely (thickness, length) that was never claiming to be
    the mounting face size in the first place -- comparing it against
    the NEMA standard produced a real false-positive ("10mm" flagged as
    "conflicting" with a 42.3mm faceplate) during this function's own
    development, confirmed by direct testing, not assumed.

    `resolved_value_mm` is always the explicit value, never the
    semantic one -- this function reports a conflict, it does not
    create ambiguity about which value wins; see this module's own
    docstring block above for why that resolution needs no new
    enforcement code of its own."""
    nema_match = _NEMA_NAME_RE.search(description)
    if nema_match is None:
        return []
    size = int(nema_match.group(1))
    standard_mm = _NEMA_STANDARD_FACE_MM.get(size)
    if standard_mm is None:
        return []  # a NEMA size this project doesn't have a verified standard value for -- don't guess

    all_claims = [c for c in extract_claimed_dimensions_mm(description) if c > 0]
    repeated_values = {c for c in all_claims if all_claims.count(c) >= 2}

    conflicts = []
    for claimed in sorted(repeated_values):
        deviation = abs(claimed - standard_mm) / standard_mm
        if deviation > tolerance_fraction:
            conflicts.append(SemanticSizeConflict(
                semantic_name=f"NEMA {size}", standard_value_mm=standard_mm,
                explicit_value_mm=claimed, resolved_value_mm=claimed,
            ))
    return conflicts
