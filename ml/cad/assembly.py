"""Minimal, sequential-only assembly coordinator: turns a compound prompt
describing several components ("a base plate with four mounting brackets, a
shaft in the center, and a top cover with ventilation holes") into one
combined, positioned mesh.

Explicitly NOT the full multi-instance/parallel-execution/shared-world-model
system from later roadmap phases -- this runs in one process, one thread,
sequentially: decompose -> parse each component with the existing single-part
parser -> place deterministically -> build -> combine -> validate. No
external LLM API anywhere in this module.

Placement is deterministic and order/keyword-based, NOT physical or learned
spatial reasoning -- satellites are never reoriented to face the base, and
"missing overlap" is a diagnostic signal, not a guarantee (see
validate_assembly). State this honestly rather than implying more than it
does, the same way nlp/prompt_parser.py documents its own limitations.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import trimesh

from evaluation.geometry_eval import GeometryReport, evaluate_mesh
from nlp.prompt_parser import (
    _BRACKET_KEYWORDS,
    _ENCLOSURE_KEYWORDS,
    _FLANGE_KEYWORDS,
    _GEAR_KEYWORDS,
    _PLATE_KEYWORDS,
    _SHAFT_KEYWORDS,
    _WORD_NUMBERS,
)
from part_families import get_part_family

# Families that can act as a base/cover a satellite gets placed relative to.
# flange is deliberately excluded from this set for v1 -- always treated as a
# satellite, never a base -- a documented simplification, not an oversight.
_BASE_LIKE_FAMILIES = {"plate", "enclosure"}

_ALL_FAMILY_KEYWORDS: dict[str, list[str]] = {
    "flange": _FLANGE_KEYWORDS,
    "plate": _PLATE_KEYWORDS,
    "bracket": _BRACKET_KEYWORDS,
    "shaft": _SHAFT_KEYWORDS,
    "enclosure": _ENCLOSURE_KEYWORDS,
    "gear": _GEAR_KEYWORDS,
}

# Splitting on comma, " and ", AND " with " (not just comma/and) is what lets
# "a base plate with four mounting brackets" become TWO segments -- the base
# and its satellite count+family -- rather than one segment whose single
# rightmost keyword ("brackets") would swallow the "plate" mention entirely.
# This is safe for genuine per-component detail too (e.g. "flange ... with a
# 25mm bore"): a segment with no family keyword of its own just reattaches to
# the currently-open component instead of starting a new one.
#
# This comma-splitting ONLY makes sense for a compact, single-paragraph
# prompt where commas really do separate sibling components. A verbose,
# multi-paragraph engineering-spec prompt (one blank-line-separated
# paragraph per component, e.g. "The base plate should be 300 mm long,
# 220 mm wide, and 15 mm thick. Add four mounting holes...") uses commas
# purely as dimension-list separators WITHIN one component's own paragraph
# -- splitting on them there severs a component's dimensions from the
# component itself, and (worse) can misread an embedded dimension number
# ("300 mm") as an instance count. See decompose_prompt()/_MAX_SANE_COUNT
# below: a real prompt with several such embedded dimensions once produced
# a parsed count of 300 for one component, which cascaded into building
# and concatenating over a thousand part instances and hung the process
# for minutes with no resource bound -- this paragraph-aware path plus the
# hard count cap both exist directly because of that.
_SEGMENT_SPLIT_RE = re.compile(r",\s*|\s+and\s+|\s+with\s+", re.IGNORECASE)
_PARAGRAPH_SPLIT_RE = re.compile(r"\n\s*\n+")

# A parsed instance count above this is treated as a misread dimension
# number rather than a real "build N of these" request -- no legitimate
# single-assembly component count in this system's scope should ever need
# to exceed this. Acts as a hard backstop even if the count/dimension
# disambiguation below still lets something bad through on some future
# phrasing -- the goal is that a parsing mistake can degrade to "wrong
# geometry," never "the process hangs building 1000+ parts."
_MAX_SANE_COUNT = 20

# A bare digit immediately followed by one of these is a DIMENSION ("300
# mm", "14 mm", "25 deg"), never an instance count -- must not be picked up
# by the count-word scan below. Spelled-out count words ("four", "six")
# are unambiguous and don't need this check.
_MEASUREMENT_UNIT_WORDS = {
    "mm", "cm", "m", "in", "inch", "inches", "deg", "degrees", "degree",
    "kg", "g", "lb", "lbs", "percent",
}

# A paragraph with no numeric measurement in it at all is treated as
# connective/summary text (an intro sentence previewing several
# components, a closing bullet list of qualitative geometric
# requirements, an instruction to "generate the assembly...") rather than
# a component's own description, and is dropped in multi-paragraph mode --
# see _decompose_multi_paragraph. A real per-component paragraph in this
# style of prompt is, in practice, dense with dimensions; a genuinely
# terse dimensionless component paragraph ("Add a shaft.") is an accepted,
# disclosed gap of this heuristic, not silently handled.
_MEASUREMENT_RE = re.compile(r"\d+(?:\.\d+)?\s*(?:mm|cm|m|in|inch|inches|deg|degrees?)\b", re.IGNORECASE)

ParseFn = Callable[[str], tuple[str, dict]]  # text -> (part_family, params_dict) -- same shape as parser_eval.ParseFn


@dataclass
class PlacementTransform:
    """Bare 4x4 matrix wrapper, local to ml/cad -- NOT the same class as
    ml/threedimen's own Transform (same name, different, unrelated project;
    avoid confusing the two)."""

    matrix: np.ndarray

    @classmethod
    def identity(cls) -> "PlacementTransform":
        return cls(np.eye(4))

    @classmethod
    def translation(cls, xyz) -> "PlacementTransform":
        m = np.eye(4)
        m[:3, 3] = xyz
        return cls(m)

    def compose(self, other: "PlacementTransform") -> "PlacementTransform":
        """self applied AFTER other, i.e. result = self.matrix @ other.matrix."""
        return PlacementTransform(self.matrix @ other.matrix)

    def apply_to_mesh(self, mesh: trimesh.Trimesh) -> trimesh.Trimesh:
        result = mesh.copy()
        result.apply_transform(self.matrix)
        return result


@dataclass(frozen=True)
class ComponentSpec:
    component_id: str
    family: str
    clause_text: str
    params: Any
    count: int = 1


@dataclass(frozen=True)
class AssemblySpec:
    prompt: str
    components: tuple[ComponentSpec, ...]


@dataclass
class PlacedComponent:
    spec: ComponentSpec
    instance_index: int
    transform: PlacementTransform
    local_mesh: trimesh.Trimesh  # PRE-placement, builder's own local frame
    world_mesh: trimesh.Trimesh  # local_mesh with transform applied


@dataclass
class AssemblyResult:
    mesh: trimesh.Trimesh | None
    placed: tuple[PlacedComponent, ...]
    geometry_reports: dict[str, GeometryReport]  # component_id -> report, evaluated on local_mesh
    overlapping_pairs: tuple[tuple[str, str], ...]  # AABB-overlap diagnostic, not a hard failure


def _segment_count_and_anchor(segment: str) -> tuple[int, str | None]:
    lowered = segment.lower()

    matches: list[tuple[int, str]] = []
    for family, keywords in _ALL_FAMILY_KEYWORDS.items():
        for kw in keywords:
            if kw == "mount" and re.search(r"\bholes?\b", lowered):
                # "mount" is a deliberately loose plate keyword (it has to
                # match "mounting"/"mounted" too), and "mounting holes" is
                # by far the single most common phrase that contains it --
                # but a hole is a FEATURE, never a standalone part family,
                # so "with four mounting holes" must never anchor a new
                # component the way "with four mounting brackets" correctly
                # does. Confirmed as a real bug, not hypothetical: even the
                # canonical single-part example "rectangular aluminum
                # mounting plate, ..., with four 10 mm mounting holes, ..."
                # was being split into a fake 2-component assembly (the
                # plate, plus a mystery second "component" for its own
                # holes) before this exclusion existed. Any OTHER keyword
                # (the literal word "bracket", "shaft", "flange", etc.)
                # still anchors normally -- this is scoped to the one
                # keyword that's actually ambiguous.
                continue
            idx = lowered.find(kw)
            if idx != -1:
                matches.append((idx, family))
    if not matches:
        return 1, None
    matches.sort(key=lambda m: m[0])
    anchor_char_idx, anchor_family = matches[0]  # leftmost: the introducing noun phrase ("two vertical
    # support brackets"), not a later incidental mention of the same or a different family elsewhere in
    # the segment/paragraph (e.g. "...along the length of the base plate" inside a BRACKET paragraph).

    # A count word only counts if it appears BEFORE the anchor keyword --
    # "four" in "four mounting brackets" is how many brackets; "four" deep
    # inside "...Apply a 5 mm fillet... Add four mounting holes..." (a
    # PLATE paragraph) is a hole count, an attribute of one plate, not "how
    # many plates." Restricting the scan window this way is what makes it
    # safe to look at an entire multi-sentence paragraph, not just a short
    # single-clause segment.
    words_before_anchor = re.findall(r"[a-z0-9]+", lowered[:anchor_char_idx])
    count = 1
    for i, w in enumerate(words_before_anchor):
        if w in _WORD_NUMBERS:
            count = _WORD_NUMBERS[w]
            break
        if w.isdigit():
            next_word = words_before_anchor[i + 1] if i + 1 < len(words_before_anchor) else ""
            if next_word in _MEASUREMENT_UNIT_WORDS:
                continue  # a dimension, not a count -- keep scanning
            count = int(w)
            break
    return min(count, _MAX_SANE_COUNT), anchor_family


def decompose_prompt(prompt: str) -> list[dict]:
    """Groups the prompt's segments into per-component clause groups. Returns
    [{"count": int, "clauses": [str, ...]}, ...] -- family isn't decided here;
    plan_assembly() gets that from actually running the real single-part
    parser on each group's reassembled text, so decomposition never
    duplicates (and risks diverging from) the parser's own classification
    logic -- it only decides where segment boundaries fall.

    A prompt with a blank line anywhere is treated as the verbose,
    multi-paragraph style (one component per paragraph) and routed to
    _decompose_multi_paragraph instead of the compact-prompt comma-splitting
    below -- see that function's docstring for why the two styles need
    different splitting rules entirely, not just a tweaked regex."""
    paragraphs = [p.strip() for p in _PARAGRAPH_SPLIT_RE.split(prompt) if p.strip()]
    if len(paragraphs) > 1:
        return _decompose_multi_paragraph(paragraphs)

    raw_segments = [s.strip() for s in _SEGMENT_SPLIT_RE.split(prompt) if s.strip()]
    # ", and X" only has ONE separator (the comma) consumed by the split regex
    # -- "and" survives as a literal leading word on the next segment ("X, Y,
    # and Z" is the common case). Strip it as cleanup rather than complicating
    # the split regex to special-case comma-then-and.
    raw_segments = [re.sub(r"^and\s+", "", s, flags=re.IGNORECASE) for s in raw_segments]
    groups: list[dict] = []
    for segment in raw_segments:
        count, anchor_family = _segment_count_and_anchor(segment)
        if anchor_family is not None or not groups:
            groups.append({"count": count, "clauses": [segment]})
        else:
            groups[-1]["clauses"].append(segment)
    return groups


def _decompose_multi_paragraph(paragraphs: list[str]) -> list[dict]:
    """Verbose, engineering-spec-style prompts put ONE component's full
    description in its own blank-line-separated paragraph, with every
    comma inside it just separating dimension/detail clauses -- NOT
    separating sibling components the way a compact single-paragraph
    prompt does. Splitting on commas there (like the compact path does)
    fragments one component's paragraph into many spurious groups and
    severs dimension clauses from the component they describe -- confirmed
    concretely: a real prompt like this produced 16 groups instead of 4
    real components, with most components' actual dimensions landing in
    the WRONG group and defaulting instead.

    So each paragraph becomes exactly one group (no further splitting),
    anchored on its leftmost family keyword. A paragraph with no numeric
    measurement anywhere in it (_MEASUREMENT_RE) is dropped instead of
    becoming a group -- it's connective/summary text (an intro sentence
    previewing several components, a closing bullet list of qualitative
    requirements like "the shaft must be horizontal," a final "generate
    the assembly..." instruction), not a component's own description.
    Without this check, an intro sentence naming all N components up front
    would spuriously produce an extra, duplicate, dimensionless group for
    whichever family it happens to mention first."""
    groups: list[dict] = []
    for paragraph in paragraphs:
        if not _MEASUREMENT_RE.search(paragraph):
            continue
        count, anchor_family = _segment_count_and_anchor(paragraph)
        if anchor_family is not None:
            groups.append({"count": count, "clauses": [paragraph]})
    return groups


def plan_assembly(prompt: str, parse_fn: ParseFn) -> AssemblySpec:
    groups = decompose_prompt(prompt)
    components = []
    for i, group in enumerate(groups):
        clause_text = ", ".join(group["clauses"])
        family, params_dict = parse_fn(clause_text)
        params = get_part_family(family).params_cls(**params_dict)
        components.append(
            ComponentSpec(
                component_id=f"{family}_{i}", family=family, clause_text=clause_text,
                params=params, count=group["count"],
            )
        )
    return AssemblySpec(prompt=prompt, components=tuple(components))


def _rect_perimeter_positions(length: float, width: float, n: int, margin_fraction: float = 0.15) -> list[tuple[float, float]]:
    if n <= 0:
        return []
    if n == 1:
        return [(0.0, 0.0)]
    hx, hy = length / 2 * (1 - margin_fraction), width / 2 * (1 - margin_fraction)
    if n <= 4:
        return [(sx * hx, sy * hy) for sx in (-1, 1) for sy in (-1, 1)][:n]
    return [(hx * np.cos(2 * np.pi * i / n), hy * np.sin(2 * np.pi * i / n)) for i in range(n)]


def _circle_perimeter_positions(diameter: float, n: int, margin_fraction: float = 0.15) -> list[tuple[float, float]]:
    if n <= 0:
        return []
    if n == 1:
        return [(0.0, 0.0)]
    r = diameter / 2 * (1 - margin_fraction)
    return [(r * np.cos(2 * np.pi * i / n), r * np.sin(2 * np.pi * i / n)) for i in range(n)]


def build_assembly(spec: AssemblySpec) -> AssemblyResult:
    placed: list[PlacedComponent] = []
    base_seen = False
    base_shape: tuple[float, float] | None = None  # (length, width) of the base's rectangular footprint
    running_max_z = 0.0

    for component in spec.components:
        part_spec = get_part_family(component.family)
        local_mesh = part_spec.builder(component.params)
        extent_z = float(local_mesh.bounds[1][2] - local_mesh.bounds[0][2])
        is_base_like = component.family in _BASE_LIKE_FAMILIES
        # Defensive, not redundant with decompose_prompt's own cap: this is
        # the last line of defense against building/copying/concatenating
        # an unbounded number of part instances for ANY ComponentSpec this
        # function is handed, however it was constructed -- see
        # _MAX_SANE_COUNT's docstring for the concrete incident (a parsed
        # count of 300 hung the process for minutes with no bound) this
        # guards against.
        count = min(component.count, _MAX_SANE_COUNT)

        if is_base_like and not base_seen:
            base_seen = True
            base_shape = (component.params.length_mm, component.params.width_mm)
            positions = [(0.0, 0.0)] * count
            target_z = 0.0
        elif is_base_like and base_seen:
            # second plate/enclosure = cover, stacked above every already-placed
            # component's running max Z, not just the base's own height
            positions = [(0.0, 0.0)] * count
            target_z = running_max_z
        else:
            # satellite: distribute around the base's perimeter (or a default
            # spread if no base has been placed yet). No reorientation --
            # satellites keep their local, identity rotation.
            positions = _rect_perimeter_positions(*base_shape, count) if base_shape else _circle_perimeter_positions(100.0, count)
            target_z = 0.0

        for instance_idx, (x, y) in enumerate(positions):
            # generic ground correction: every builder centers its solid
            # differently in its own local frame (plate: -t/2, cylinder:
            # [-h/2, h/2]) -- translate by (target_z - local min Z) rather
            # than hardcoding a per-family offset, so this covers all
            # families uniformly, including future ones.
            z_translation = target_z - float(local_mesh.bounds[0][2])
            transform = PlacementTransform.translation([x, y, z_translation])
            world_mesh = transform.apply_to_mesh(local_mesh)
            placed.append(
                PlacedComponent(
                    spec=component, instance_index=instance_idx,
                    transform=transform, local_mesh=local_mesh, world_mesh=world_mesh,
                )
            )
        running_max_z = max(running_max_z, target_z + extent_z)

    combined = trimesh.util.concatenate([p.world_mesh for p in placed]) if placed else None
    return AssemblyResult(mesh=combined, placed=tuple(placed), geometry_reports={}, overlapping_pairs=())


def _aabb_overlap(a: trimesh.Trimesh, b: trimesh.Trimesh) -> bool:
    a_min, a_max = a.bounds
    b_min, b_max = b.bounds
    return bool(np.all(a_min <= b_max) and np.all(b_min <= a_max))


def validate_assembly(result: AssemblyResult) -> AssemblyResult:
    """Adds geometry_reports (per component's LOCAL, pre-placement mesh --
    evaluating the world-transformed mesh would confound "is this feature
    detectable" with "did the assembly's placement move it") and
    overlapping_pairs (pairwise AABB overlap -- a diagnostic/regression
    signal only; there is no collision repair in this version)."""
    reports: dict[str, GeometryReport] = {}
    for p in result.placed:
        key = f"{p.spec.component_id}[{p.instance_index}]"
        reports[key] = evaluate_mesh(p.local_mesh)

    overlaps: list[tuple[str, str]] = []
    for i in range(len(result.placed)):
        for j in range(i + 1, len(result.placed)):
            a, b = result.placed[i], result.placed[j]
            if _aabb_overlap(a.world_mesh, b.world_mesh):
                key_a = f"{a.spec.component_id}[{a.instance_index}]"
                key_b = f"{b.spec.component_id}[{b.instance_index}]"
                overlaps.append((key_a, key_b))

    result.geometry_reports = reports
    result.overlapping_pairs = tuple(overlaps)
    return result


def run_assembly(prompt: str, parse_fn: ParseFn) -> AssemblyResult:
    spec = plan_assembly(prompt, parse_fn)
    result = build_assembly(spec)
    return validate_assembly(result)
