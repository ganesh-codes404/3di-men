"""A single, narrowly-scoped question: roughly how many distinct physical
COMPONENTS (not features -- see _COMPONENT_NOUNS below, which deliberately
excludes holes/slots/patterns, exactly matching llm_decompose.py's own
STEP 1 part-vs-feature distinction) does the ORIGINAL prompt text seem to
ask for? Used as a cheap, independent cross-check against how many of
decompose_prompt()'s own points actually made it into the final built
assembly (see parallel_builder.AssemblyBuildResult.built_component_count) --
catching a component the LLM's decomposition silently dropped, which
nothing else in this pipeline checks (see PILOT_FAILURE_INVESTIGATION.md
section 5).

This is NOT a re-implementation of the component-inventory/reconciliation
system from the (reverted) 2026-11-01 commit b8b74a0 -- no per-type
identity tracking, no clause-boundary NLP, no separate dataclass hierarchy.
It answers one narrower question (a single total count, not "which specific
types/instances"), on purpose: good enough to flag "the build produced
fewer components than the prompt implied," not a claim to know WHICH
component is missing or re-derive the decomposition itself.

Deliberately conservative in both directions it can go wrong:
- Returns None (never 0, never a guess) whenever the prompt doesn't look
  like a multi-component request at all -- no recognized component noun
  found anywhere. This is what keeps free-form / common-object prompts
  ("a water bottle", "a coffee mug") OUT of this check entirely, the same
  way a hole-count or dimension claim with nothing to check against
  returns None elsewhere in this project (spec_validation.py's own
  convention) -- not a special case, just what the extraction naturally
  produces when there's nothing to find.
- Only counts a component noun when it's freshly INTRODUCED (an indefinite
  article "a"/"an", or an explicit cardinal/plural), never when referred
  back to with a definite article ("the") -- "an L-shaped mounting
  bracket... The horizontal base... The vertical plate..." is ONE
  component (a bracket) described by its own two integral faces, not
  three separate parts; "the" is what marks that back-reference. Confirmed
  against all 4 pilot prompts in PILOT_FAILURE_INVESTIGATION.md: this
  rule is what keeps P3 (a true single-part L-bracket) from being
  falsely flagged as missing two components, while still correctly
  counting P1/P2/P4's genuinely separate "a bearing housing", "a shaft",
  "a cylindrical coupling", "a cylindrical standoff", "a spacer", "a
  collar", etc.
- Two recognized nouns directly adjacent ("bearing housing", "base
  plate") are a single compound noun -- only the LAST (head) noun counts,
  so "bearing" doesn't also separately count alongside "housing".
- A noun with no identifiable preceding cardinal defaults to 1, not 0 --
  "a shaft" is one shaft. An explicit cardinal ("two parallel horizontal
  shafts") multiplies that occurrence instead.
- Known, accepted blind spot: "two bearing blocks PER SHAFT" (two shafts)
  is counted as 2 (the literal cardinal on that one occurrence) rather
  than the true 4 (2 blocks x 2 shafts) -- this module does not attempt
  per-X multiplication; a noun directly preceded by "per" is instead
  excluded entirely rather than guessed at, so it neither adds a phantom
  extra shaft NOR tries and fails to multiply.
  This is a safe direction to be wrong in: it can only make the estimate
  too LOW, which can only make this check too lenient (silent on some
  true gaps), never too strict (never a false "missing component" flag
  from this specific limitation).
- A BARE plural with no article or cardinal at all ("All shafts,
  bearings, gears and pulleys must be separate identifiable components")
  is excluded, not defaulted to 1 the way a bare SINGULAR is -- a
  qualifier-less plural reads as a generic recap of already-described
  categories, not a fresh countable instance; confirmed against P4's own
  closing sentence, which would otherwise phantom-count all 4 already-
  counted types a second time.
- A component noun immediately followed by "assembly"/"mechanism"/
  "system"/"stack"/"unit" is excluded -- it's modifying the whole
  request's own TITLE ("a mechanical shaft support assembly"), not
  naming a standalone part; confirmed against P1, whose own title
  phrase would otherwise phantom-count an extra "shaft".
"""

from __future__ import annotations

import re

# Deliberately small and conservative -- real, distinct, standalone
# mechanical PART nouns only (llm_decompose.py's own STEP 1 vocabulary:
# "a plate, a bracket, a shaft, a gear, a bushing, a housing, a cover, a
# standoff, a fastener boss"), never a feature word (hole, slot, pocket,
# pattern, fillet, chamfer -- those are STEP 1's own "part of the SAME
# entry" features, not separate components, and are correctly invisible
# to this extractor simply by not appearing in this list).
#  "mount"/"support" deliberately excluded despite being real part nouns
# in isolation: in practice, in this kind of prompt, "mount" is used as a
# VERB far more often than a noun ("Mount a bearing housing..."), and
# "support" overwhelmingly shows up inside the ASSEMBLY'S OWN TITLE
# ("a mechanical shaft support assembly"), not naming a separate part --
# both confirmed as real false-positive sources against the 4 pilot
# prompts, not a theoretical concern.
_COMPONENT_NOUNS = (
    "bracket", "plate", "panel", "shaft", "rod", "gear", "bearing",
    "housing", "motor", "fastener", "bolt", "screw", "spacer",
    "standoff", "bushing", "flange", "coupling", "collar", "pulley",
    "wheel", "tire", "rim", "wing", "cover", "lid", "block",
    "arm", "base", "frame", "enclosure", "bar", "pin", "axle", "hinge",
)

_COUNT_WORDS = {
    "a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
}

_NOUN_ALTERNATION = "|".join(sorted(_COMPONENT_NOUNS, key=len, reverse=True))
# A run of one or more recognized nouns separated only by whitespace (the
# compound-noun case, e.g. "bearing housing") -- matched greedily so
# "bearing housing" is one noun-run, not two.
_NOUN_RUN_RE = re.compile(rf"\b(?:(?:{_NOUN_ALTERNATION})s?\s+)*(?:{_NOUN_ALTERNATION})s?\b", re.IGNORECASE)
_QUALIFIER_RE = re.compile(r"\b(the|a|an|one|two|three|four|five|six|seven|eight|nine|ten)\b", re.IGNORECASE)
# A clause boundary -- comma, semicolon, or period -- bounds how far back a
# noun-run is allowed to look for its own qualifying article/cardinal: "a
# cylindrical standoff... Place a spacer" must never let an earlier
# clause's "a" attach to this clause's "spacer" (it already has its own).
# Deliberately NOT "and" (unlike an earlier version of this function) --
# "the shaft and bearing" must still resolve "bearing" as covered by the
# SAME "the" as "shaft" (its nearest qualifier within the clause, with no
# closer "a"/cardinal of its own), not as a fresh, unqualified mention;
# "a gear... and a gear" still comes out right too, since each gear finds
# its OWN nearer "a" rather than an earlier, unrelated "the" several
# words back in the same comma-free sentence (see the NEAREST-qualifier
# logic below, which only ever looks at the single closest qualifier to
# each noun, never "anywhere in the clause").
_CLAUSE_BOUNDARY_RE = re.compile(r"[,;.]")
_TITLE_WORD_RE = re.compile(r"\s*(?:\w+\s+)?(?:assembly|mechanism|system|stack|unit)\b", re.IGNORECASE)


def estimate_expected_component_count(prompt: str) -> "int | None":
    """None when no recognized component noun appears anywhere in the
    prompt (the free-form/common-object case -- nothing to check). An
    integer >= 1 otherwise: this module's best-effort estimate of how
    many distinct physical components the prompt's own text asks for.
    Never raises."""
    clause_starts = [0] + [m.end() for m in _CLAUSE_BOUNDARY_RE.finditer(prompt)]

    def _clause_start_before(pos: int) -> int:
        start = 0
        for cs in clause_starts:
            if cs <= pos:
                start = cs
            else:
                break
        return start

    total = 0
    found_any = False
    for m in _NOUN_RUN_RE.finditer(prompt):
        found_any = True

        # "per shaft", "per axle": a rate/distribution phrase, not a fresh
        # countable instance -- excluded outright (see this module's own
        # docstring on why multiplying through "per" is out of scope).
        preceding_word = prompt[:m.start()].rstrip().rsplit(None, 1)[-1] if prompt[:m.start()].rstrip() else ""
        if preceding_word.lower() == "per":
            continue

        # "a shaft support ASSEMBLY", "a mechanical transmission SYSTEM":
        # the noun-run is modifying the request's own title, not naming a
        # standalone part.
        if _TITLE_WORD_RE.match(prompt, m.end()):
            continue

        clause_start = _clause_start_before(m.start())
        qualifier_matches = list(_QUALIFIER_RE.finditer(prompt, clause_start, m.start()))
        nearest = qualifier_matches[-1] if qualifier_matches else None
        qualifier = nearest.group(1) if nearest else None
        if qualifier and qualifier.lower() == "the":
            continue  # a direct, definite back-reference to an already-introduced part/face
        if qualifier and qualifier.lower() != "a" and qualifier.lower() != "an":
            # a cardinal ("two") immediately preceded by "the" ("the two
            # plates") is STILL a definite back-reference to an already-
            # introduced pair/group, not a fresh count -- checked only for
            # the NEAREST qualifier itself, so an unrelated, earlier "the"
            # several words further back in a long, comma-free sentence
            # ("...on the first shaft and a 30-tooth spur gear...") never
            # wrongly reaches forward to block a later, different part's
            # own closer "a".
            before_cardinal = prompt[clause_start:nearest.start()].rstrip()
            prev_word = before_cardinal.rsplit(None, 1)[-1] if before_cardinal else ""
            if prev_word.lower() == "the":
                continue
        if qualifier is None and m.group(0).rstrip().endswith("s"):
            continue  # a bare, qualifier-less PLURAL ("shafts, bearings, gears...") reads as a
                      # generic recap of already-described categories, not a fresh countable instance
        total += _COUNT_WORDS.get((qualifier or "a").lower(), 1)
    return total if found_any else None
