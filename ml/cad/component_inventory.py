"""Component Inventory -- tracks every requested PHYSICAL COMPONENT from
the ORIGINAL user prompt (before decomposition can silently discard
information) through decomposition, generation, and final assembly.

Confirmed real root cause this exists to fix (adversarial audit, this
project's own standing "always re-measure, never trust the planner"
principle extended one level up): `parallel_builder.build_assembly_from_
prompt` calls `decompose_prompt(prompt)` and then passes ONLY
`decomposition.points` into `build_assembly_from_points` -- the ORIGINAL
PROMPT TEXT is discarded immediately after (used for nothing but
`extract_claimed_envelope_mm`, an unrelated footprint check). Nothing
anywhere re-checks the decomposed POINT LIST's own completeness against
what the user actually asked for: program_model.py, sandbox.py,
feature_verification.py, and assembly_compatibility.py all operate on
whatever points they were HANDED, never on the request that was supposed
to produce them. A component the LLM's own decomposition drops,
duplicates, or merges away is structurally invisible to every existing
downstream check.

Deliberately regex/keyword-based EXTRACTION (same discipline as
op_requirements.py/spec_validation.py/dimension_requirements.py -- no
additional LLM call, see this module's own Part 13 performance
constraints), but the completeness check itself -- reconciling an
independently-derived expectation against whatever decomposition
actually produced -- is real structure, not "more regex."""

from __future__ import annotations

import dataclasses
import re
from dataclasses import dataclass, field
from enum import Enum
from itertools import count as _counter

from dimension_requirements import RequirementStatus

_id_counter = _counter(1)


def _next_id(prefix: str) -> str:
    return f"{prefix}-{next(_id_counter):03d}"


# keyword -> component_type -- the same small, explicit, deterministic
# keyword-to-category style op_requirements._BASE_GEOMETRY_KEYWORDS
# already uses, generalized to mechanical PART nouns (Part 3's own list)
# rather than base-shape words. Deliberately excludes "boss" -- Part 1's
# own STEP 1 and this brief's own Part 3 both treat a boss as a FEATURE
# of another component (op_requirements.py already tracks it that way),
# never a standalone component; counting it here too would double-count
# exactly the case this brief's own Test H warns against ("one component,
# four bosses, no false count of five components").
_COMPONENT_TYPE_KEYWORDS: dict[str, str] = {
    "bracket": "bracket", "plate": "plate", "panel": "plate",
    "shaft": "shaft", "rod": "shaft",
    "gear": "gear", "bearing": "bearing", "housing": "housing",
    "motor": "motor", "mount": "mount",
    "fastener": "fastener", "bolt": "fastener", "screw": "fastener",
    "spacer": "spacer", "standoff": "standoff", "bushing": "bushing",
    "flange": "flange", "coupling": "coupling", "crossbar": "crossbar",
    "frame": "frame", "enclosure": "enclosure",
    "support": "support", "leg": "support",
    "cover": "cover", "lid": "cover",
}
# component_type -> every keyword that maps to it (the inverse of the
# above, built once) -- used by reconciliation to check whether a
# DECOMPOSED point's own free-text description names this type at all.
_KEYWORDS_BY_TYPE: dict[str, list[str]] = {}
for _kw, _ctype in _COMPONENT_TYPE_KEYWORDS.items():
    _KEYWORDS_BY_TYPE.setdefault(_ctype, []).append(_kw)

_COUNT_WORDS = {
    "a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
}
_VAGUE_COUNT_WORDS = r"(?:a\s+few|several|some|multiple|a\s+number\s+of|a\s+couple\s+of)"

_TYPE_ALTERNATION = "|".join(sorted(_COMPONENT_TYPE_KEYWORDS, key=len, reverse=True))
_NOUN_RE = re.compile(rf"\b({_TYPE_ALTERNATION})s?\b", re.IGNORECASE)
_COUNT_WORD_RE = re.compile(r"\b(\d+|a|an|one|two|three|four|five|six|seven|eight|nine|ten)\b", re.IGNORECASE)
_VAGUE_WORD_RE = re.compile(rf"\b{_VAGUE_COUNT_WORDS}\b", re.IGNORECASE)
_PAIR_OF_RE = re.compile(r"\bpair\s+of\b", re.IGNORECASE)
# A clause boundary -- comma, semicolon, period, or "and" -- bounds how
# far a count word is allowed to reach to claim a noun, same "one
# clause's own context never claims a number/word from a different
# clause" discipline as dimension_requirements.py's own _CLAUSE_BOUNDARY_RE
# (confirmed independently necessary here too, not copied blindly: "two
# brackets, one crossbar" must never let "two" reach across the comma to
# also claim "crossbar").
_CLAUSE_BOUNDARY_RE = re.compile(r"[,;.]|\band\b", re.IGNORECASE)
_MAX_COUNT_GAP_CHARS = 30


class ReconciliationStatus(Enum):
    MATCHED = "MATCHED"
    MISSING_FROM_PLAN = "MISSING_FROM_PLAN"
    COUNT_MISMATCH = "COUNT_MISMATCH"
    DUPLICATED_IN_PLAN = "DUPLICATED_IN_PLAN"
    IDENTITY_AMBIGUOUS = "IDENTITY_AMBIGUOUS"
    UNSUPPORTED = "UNSUPPORTED"
    UNRESOLVED = "UNRESOLVED"


@dataclass
class ComponentTypeRequirement:
    """One distinct, expected component TYPE from the ORIGINAL prompt --
    e.g. "bracket_type_A," requested twice. `instance_ids` are the
    decomposed points' own stable ids (see parallel_builder._unique_
    point_ids) once reconciled -- this module never invents a SEPARATE
    identity scheme; it reuses the one the rest of the pipeline already
    keys everything off of (Part 5's own "do not use mesh order... as
    identity" rule, satisfied by reuse rather than new machinery)."""

    component_id: str
    component_type: "str | None"  # None only for a genuinely UNRESOLVED entry (Part 3's own ambiguous-noun case)
    component_name: str
    source_text: str
    explicitness: str = "EXPLICIT"
    mandatory: bool = True
    requested_count: int = 1
    generated_count: int = 0
    instance_ids: list[str] = field(default_factory=list)
    status: RequirementStatus = RequirementStatus.EXTRACTED
    reconciliation: "ReconciliationStatus | None" = None
    failure_reason: str = ""


def _clause_spans(text: str) -> list[tuple[int, int]]:
    spans, start = [], 0
    for m in _CLAUSE_BOUNDARY_RE.finditer(text):
        spans.append((start, m.start()))
        start = m.end()
    spans.append((start, len(text)))
    return spans


def _clause_containing(spans: list[tuple[int, int]], pos: int) -> tuple[int, int]:
    for s, e in spans:
        if s <= pos <= e:
            return s, e
    return spans[-1]


def extract_component_requirements(prompt: str) -> list[ComponentTypeRequirement]:
    """Never raises, never invents a count it cannot determine (an
    UNRESOLVED entry for that case instead -- Part 3's own rule). For
    each recognized noun occurrence, finds the NEAREST preceding count
    phrase (a definite number/word, "pair of," or a vague quantity word)
    within the SAME clause -- bounded-gap, clause-aware nearest match,
    the same discipline dimension_requirements._nearest_keyword already
    uses, independently confirmed necessary here: a naive "count word
    immediately before the noun" regex matches NEITHER of this brief's
    own two worked examples ("two identical L-brackets connected by ONE
    100mm CROSSBAR" -- a dimension clause sits between the count and the
    noun; "ONE MOUNTING plate" -- a descriptive adjective does) --
    verified directly: both previously extracted nothing at all for the
    second component. Sums every count-phrase found for the SAME
    component type across the whole prompt (same "multiple groups can
    refer to the same feature" convention as spec_validation.
    extract_claimed_hole_count)."""
    clause_spans = _clause_spans(prompt)
    count_spans = [m.span() for m in _COUNT_WORD_RE.finditer(prompt)]
    vague_spans = [m.span() for m in _VAGUE_WORD_RE.finditer(prompt)]
    pair_of_spans = [m.span() for m in _PAIR_OF_RE.finditer(prompt)]

    totals: dict[str, int] = {}
    source_text_by_type: dict[str, str] = {}
    unresolved: list[ComponentTypeRequirement] = []

    # Adjacent recognized nouns ("motor housing," "bearing housing,"
    # "motor mount") are a single compound noun, not two components --
    # the LAST word is the true head noun, the first is acting as its
    # own modifier (confirmed real gap: "motor housing" is this
    # project's own established phrasing, see llm_decompose.py's system
    # prompt, and would otherwise double-count a "motor" AND a
    # "housing" from one physical part). Keep only the rightmost noun
    # in any chain of matches separated by nothing but whitespace.
    noun_matches = list(_NOUN_RE.finditer(prompt))
    kept_matches = []
    for i, m in enumerate(noun_matches):
        if i + 1 < len(noun_matches):
            nxt = noun_matches[i + 1]
            if prompt[m.end():nxt.start()].strip() == "":
                continue  # a later noun immediately follows -- this one is a modifier, not the head
        kept_matches.append(m)

    for m in kept_matches:
        noun_start, noun_end = m.span()
        ctype = _COMPONENT_TYPE_KEYWORDS[m.group(1).lower()]
        clause_start, clause_end = _clause_containing(clause_spans, noun_start)

        def _nearest_before(spans: list[tuple[int, int]]) -> "tuple[int, int] | None":
            candidates = [s for s in spans if clause_start <= s[0] and s[1] <= noun_start and (noun_start - s[1]) <= _MAX_COUNT_GAP_CHARS]
            return max(candidates, key=lambda s: s[1]) if candidates else None

        pair_span = _nearest_before(pair_of_spans)
        count_span = _nearest_before(count_spans)
        vague_span = _nearest_before(vague_spans)

        # Whichever quantity signal sits CLOSEST to the noun wins -- a
        # vague word closer than any definite count must win too (e.g.
        # "two of the several available brackets" is a contrived
        # counterexample this isn't trying to solve; realistically a
        # vague word closer than a definite one means the definite one
        # belongs to something else in the same clause).
        best = max(
            [("pair", pair_span), ("count", count_span), ("vague", vague_span)],
            key=lambda kv: kv[1][1] if kv[1] else -1,
        )
        kind, span = best
        if span is None:
            continue  # a recognized noun with no nearby quantity signal at all -- not this extractor's job to assume 1

        source = prompt[span[0]:noun_end]
        if kind == "vague":
            unresolved.append(ComponentTypeRequirement(
                component_id=_next_id("COMP-UNRESOLVED"), component_type=None, component_name=m.group(0),
                source_text=source, status=RequirementStatus.UNRESOLVED,
                failure_reason=f"{source!r} names a quantity this module cannot resolve to a definite count -- preserved as ambiguous, never guessed",
            ))
            continue
        if kind == "pair":
            count = 2
        else:
            word = prompt[span[0]:span[1]].lower()
            count = _COUNT_WORDS.get(word)
            if count is None:
                try:
                    count = int(word)
                except ValueError:
                    continue
        totals[ctype] = totals.get(ctype, 0) + count
        source_text_by_type.setdefault(ctype, source)

    out = [
        ComponentTypeRequirement(
            component_id=f"{ctype}_type_{chr(65 + i)}", component_type=ctype, component_name=ctype,
            source_text=source_text_by_type[ctype], requested_count=count,
        )
        for i, (ctype, count) in enumerate(totals.items())
    ]
    out.extend(unresolved)
    return out


def reconcile_component_inventory(
    expected: list[ComponentTypeRequirement], points: list, point_ids: "list[str] | None" = None,
) -> list[ComponentTypeRequirement]:
    """Never mutates `expected` in place -- returns NEW
    ComponentTypeRequirement objects (same convention as dimension_
    assignment.assign_requirements/feature_verification.verify_
    requirements: a reconciliation stage returns a new result, never a
    silent in-place edit). `points` is whatever decomposition actually
    produced (ActionablePoint or anything duck-typed the same way --
    .description); `point_ids` are their already-assigned stable ids
    (parallel_builder._unique_point_ids's own output) -- passed in
    rather than re-derived, so this module's own identity list is
    ALWAYS the one the rest of the pipeline is already keyed by, never
    a second, parallel id scheme (Part 5's own rule)."""
    if point_ids is None:
        point_ids = [getattr(p, "id", "") or f"_idx{i}" for i, p in enumerate(points)]

    expected_types = {c.component_type for c in expected if c.component_type is not None}
    point_match_types: dict[str, list[str]] = {}
    matches_by_type: dict[str, list[str]] = {t: [] for t in expected_types}
    for pid, p in zip(point_ids, points):
        desc_l = (getattr(p, "description", "") or "").lower()
        matched = [t for t in expected_types if any(re.search(rf"\b{kw}\b", desc_l) for kw in _KEYWORDS_BY_TYPE.get(t, []))]
        point_match_types[pid] = matched
        for t in matched:
            matches_by_type[t].append(pid)

    out: list[ComponentTypeRequirement] = []
    for comp in expected:
        if comp.component_type is None:
            out.append(comp)  # already UNRESOLVED at extraction time -- nothing to reconcile
            continue

        all_matching = matches_by_type.get(comp.component_type, [])
        unique_matching = [pid for pid in all_matching if len(point_match_types[pid]) == 1]
        ambiguous_matching = [pid for pid in all_matching if len(point_match_types[pid]) > 1]

        if ambiguous_matching and len(unique_matching) < comp.requested_count:
            out.append(dataclasses.replace(comp,
                status=RequirementStatus.UNRESOLVED, reconciliation=ReconciliationStatus.IDENTITY_AMBIGUOUS,
                generated_count=len(all_matching), instance_ids=list(all_matching),
                failure_reason=(
                    f"{len(ambiguous_matching)} candidate point(s) also match another expected component type -- "
                    f"cannot confidently assign them to {comp.component_type!r} without risking double-counting "
                    "a point that may actually satisfy a DIFFERENT requirement"
                ),
            ))
            continue

        if not all_matching:
            out.append(dataclasses.replace(comp,
                status=RequirementStatus.FAILED, reconciliation=ReconciliationStatus.MISSING_FROM_PLAN,
                generated_count=0, failure_reason=f"no decomposed point's own description names a {comp.component_type!r} at all -- requested {comp.requested_count}, found 0",
            ))
            continue

        usable = unique_matching if len(unique_matching) >= comp.requested_count else all_matching
        if len(usable) == comp.requested_count:
            out.append(dataclasses.replace(comp,
                status=RequirementStatus.GEOMETRY_VERIFIED, reconciliation=ReconciliationStatus.MATCHED,
                generated_count=len(usable), instance_ids=list(usable), failure_reason="",
            ))
        elif len(usable) > comp.requested_count:
            recon = ReconciliationStatus.DUPLICATED_IN_PLAN if comp.requested_count == 1 else ReconciliationStatus.COUNT_MISMATCH
            out.append(dataclasses.replace(comp,
                status=RequirementStatus.FAILED, reconciliation=recon,
                generated_count=len(usable), instance_ids=list(usable),
                failure_reason=f"requested {comp.requested_count}, decomposition produced {len(usable)} distinct {comp.component_type!r} point(s)",
            ))
        else:
            out.append(dataclasses.replace(comp,
                status=RequirementStatus.FAILED, reconciliation=ReconciliationStatus.COUNT_MISMATCH,
                generated_count=len(usable), instance_ids=list(usable),
                failure_reason=f"requested {comp.requested_count}, decomposition produced only {len(usable)} distinct {comp.component_type!r} point(s)",
            ))
    return out


def components_complete(inventory: "list[ComponentTypeRequirement] | None") -> bool:
    """COMPONENTS_COMPLETE (Part 11's own truthful criterion): true only
    if every MANDATORY expected component type reconciled as MATCHED.
    An inventory of None (no original prompt was available -- e.g. the
    manual/hand-edited points path, which has no "original request" to
    check against) is vacuously complete, never a fabricated failure."""
    if inventory is None:
        return True
    return all(c.reconciliation is ReconciliationStatus.MATCHED for c in inventory if c.mandatory)
