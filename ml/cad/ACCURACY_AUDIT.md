# Why 3D-model accuracy and quality are limited — a root-cause audit

Written in direct response to "why are the accuracy and quality of the 3D
models so low." Every claim below is backed by something checked this
session — a measured number, a line of code, or a first-party comment
already in the repo — not impression. Ordered roughly by how much of the
observed quality ceiling each one actually explains.

## TL;DR

The system was **explicitly, deliberately scoped** to not attempt
open-ended generalization — that's root cause #1, and it's not a bug.
Past that honest ceiling, the real, fixable contributors are: a
three-stage pipeline where each stage can independently degrade the
result with no error-correction between stages; a validation layer that
checks "is this a valid mesh" but never "does this match what was asked
for"; several geometric operations that are disclosed approximations, not
exact constructions; training data whose phrasing doesn't match what the
LLM actually produces at inference time; and — until an hour ago — a
complete absence of any measurement of the LLM-decomposition step, the
first and most failure-prone stage in the whole chain.

> **UPDATE, same session:** a follow-on mission ("Build a Documentation-
> Grounded, Constraint-Aware, Self-Validating 3D Generation System")
> directly targeted sections 3, 4, 6, and 10 of this audit. Real,
> implemented, tested fixes now exist for: the bounding-box-replacement
> bug (#4), boolean reliability + a real transactional checkpoint (#2/#4),
> shell()'s approximation now being measured and rejected when too far
> off instead of silently returned (#4), a first correctness-to-spec
> dimensional check wired into the real pipeline (#3), a machine-readable
> capability registry (new), and a bounded repair loop (#6). See
> `MISSION_STATUS.md`'s Addendum 5 for the full, verified account —
> **this file's own numbered sections below are left as originally
> written** (the historical record of what was true when this audit was
> written), with a short status note added at the top of each section
> this follow-on mission actually addressed.

---

## 1. An architectural ceiling, by design — not a bug

**Status: unchanged.** This remains true and was not a target of the
follow-on mission (it's a scope decision, not a bug to fix).

`data/program_dataset.py`'s own module docstring states this plainly,
written well before this conversation:

> "true open-ended compositional generalization (correctly building an
> arbitrary novel object) is not realistically achievable under this
> project's constraints (CPU-only, scikit-learn, no LLM API, a few-
> hundred-example generic corpus) ... The honest target here is: reliable
> reproduction of the 6 known shapes through a uniform tool-call
> representation, modest and honestly-measured competence on simple
> 2-3-primitive novel combinations ... and visible, typed failure beyond
> that -- never silent degradation."

This is the single biggest explanatory factor for "why isn't a complex,
novel request built accurately": it was never the design target. The
6 known families (plate/flange/bracket/shaft/enclosure/gear) and simple
2-3-tool combinations are where real competence was invested;
anything past that is expected to fail honestly (a rejected/suspect
status), not silently produce a wrong shape. If the actual expectation is
"handle any arbitrary CAD request well," that expectation is ahead of
what this architecture was built to do — and closing that gap is a
different, much larger project (a large pretrained sequence model trained
on a huge CAD corpus — explicitly named in the same docstring as the kind
of system this is NOT).

## 2. A three-stage pipeline with no error-correction between stages

**Status: partially addressed.** `sandbox.py`'s own layer now has a real
bounded repair loop (`repair.py`) and a real transactional checkpoint
(`Sandbox.execute()`'s redesign) -- see Addendum 5. The cross-stage gap
(LLM → classifier) is unchanged; the fixes are within-stage, not across.

```
prompt → llm_decompose.py (LLM) → program_model.py (classifier) → sandbox.py (executor)
```

Each stage can independently produce a wrong result, and nothing checks a
stage's output against the ORIGINAL prompt — only against the previous
stage's output:

- If the LLM decomposes a request into the wrong number of parts, or
  invents/drops a dimension, `program_model.py` will faithfully predict a
  tool sequence for that WRONG description — a perfect prediction at
  stage 2 cannot undo an error at stage 1.
- If `program_model.py` mispredicts the tool sequence for a CORRECT
  description, the geometry is wrong even though the LLM did its job
  right.
- `sandbox.py`'s own validation (`ProgramExecutionResult.status`) only
  checks whether the EXECUTED program produced a structurally valid mesh
  — it has no way to know whether that mesh matches the user's original
  request at all (see #3).

No stage gets a second attempt informed by why the previous one failed —
a single misprediction at any stage is terminal for that part (see #6).

## 3. Validation measures "is this a valid mesh," never "is this correct"

**Status: a first, scoped version now exists.** `spec_validation.py`
compares a part's description's own numeric "NNmm" claims against the
built mesh's real measured dimensions, wired into the live pipeline
(`PointBuildResult.dimension_match_rate`/`unmatched_dimensions_mm`,
surfaced in `/api/build`'s response). Confirmed live: a correctly-built
cylinder scores 1.0; a deliberately wrong one correctly flags the
mismatched claim. Honestly scoped to OVERALL dimensions only (not
feature-level details like hole positions) -- see Addendum 5 for the full
account of what this does and doesn't catch.

This is subtle but probably underrated as a cause of perceived low
quality: `Sandbox.execute()`'s `status="ok"` requires only
`report.watertight and mesh.volume > 0` (confirmed directly,
`sandbox.py:2382-2383`). `evaluate_mesh()`/`mesh_topology_stats()` check
watertightness, degenerate faces, planar-face fraction, hole/cylinder
detection, mirror symmetry — all **structural** properties.

**None of this checks the mesh's actual dimensions, proportions, or
feature placement against what the prompt asked for.** A build can
report `status="ok"` — the success case everywhere in this project's own
reporting — while being, say, 80mm when the user asked for 65mm, or
missing a feature the LLM silently dropped from its description. The
system's own "success rate" numbers (0.88–1.0 depending on bucket) measure
geometric validity, not correctness to spec. This is very likely a real
gap between "the numbers say it's working" and "the user looks at the
model and it's wrong."

## 4. Several geometric operations are disclosed approximations, not exact constructions

**Status: 2 of the 3 specific risks named below are now fixed, not just
documented.** `round_box_edges`/`chamfer_box_edges` now REJECT a non-box
target (a real precondition, `_is_axis_aligned_box`) instead of silently
replacing its geometry with its bounding box. `shell()` now MEASURES its
own wall-thickness deviation via real ray-casting and raises an explicit
error when it's too large (confirmed: a plain box scores 0% deviation and
passes; a gear profile scores 72% and a steeply-tapered vessel scores
40-60%, both now correctly rejected; this project's own actual
water-bottle pattern scores only 3-11% and still passes). The
axis-aligned-faces-only restriction on `fillet_edge`/`chamfer_edge` is
unchanged (already a typed rejection, not a silent approximation, so it
wasn't the risk this follow-on mission prioritized). See Addendum 5.

Found by grepping `sandbox.py` for its own disclosure comments (these are
pre-existing, honest admissions in the code, not new findings):

- **`shell()`** (hollowing): an axis-scaled-copy-and-subtract
  approximation, not a true constant-distance offset surface — "accurate
  for box-like/prismatic shapes... increasingly approximate for shapes
  with fine surface detail at a scale comparable to wall_thickness."
- **`round_box_edges`/`chamfer_box_edges`**: reads the target's bounding
  box and **replaces the whole shape** with a reconstructed rounded/
  chamfered box — correct only when the target genuinely IS a box; on any
  other shape it silently replaces the real geometry with its bounding
  box's rounded/chamfered reconstruction, discarding whatever the actual
  shape was. `program_model.py`'s classifier has no semantic understanding
  of what a handle currently contains, so nothing prevents this tool being
  predicted against a non-box target.
- **`add_knurl_pattern`**: "a grip-texture approximation... not a true
  diamond-knurl crosshatch" — explicitly a simplification.
- **`fillet_edge`/`chamfer_edge`**: only work on axis-aligned, mutually
  perpendicular faces — cannot fillet/chamfer a curved or non-axis-aligned
  edge at all; any request needing that fails outright, by design.
- **Mesh faceting introduces real, measured dimensional error even in the
  "working" tools**: an ellipse extrusion measured ~0.6% off its analytic
  volume; `add_sweep`'s arc-path case measured 0.8–3.4% off the
  Pappus's-theorem-predicted volume depending on the fixed `sections`/
  `path_segments` quality knobs (24/32/48 by default — not adaptive, not
  predicted, just hardcoded). This is a representation-level ceiling: the
  whole stack is faceted-mesh-based, not BREP/NURBS, so curved geometry is
  *always* a polygonal approximation of the true analytic shape, by
  construction.
- **Booleans** (`manifold3d`-backed) can produce non-watertight results on
  some real inputs — confirmed, not hypothetical: this is exactly why
  `status="suspect"` exists as its own category in `ProgramExecutionResult`.

## 5. Training-data phrasing doesn't match inference-time phrasing

**Status: unchanged** (not a target of the follow-on mission; the
earlier-session +8.3-point `hand_written` accuracy improvement, see
`MISSION_STATUS.md` Addendum 3, addressed a different gap -- missing
training examples for specific tools -- not this phrasing-style mismatch).

`data/program_dataset.py`'s `generic_composition` patterns — the ONLY
training exposure to composing primitives beyond the 6 fixed families —
are, by the file's own description, "mechanically rendered from the
program itself, one clause per tool call... telegraphic, not fluent
English." But the text `program_model.py` actually receives at inference
time comes from `llm_decompose.py`'s LLM, writing in a more natural
(if STEP-3-constrained) style. This is a real train/inference distribution
gap baked into the architecture's two-stage design — not a bug introduced
this session, but a standing structural tax on accuracy for anything
outside the 6 well-trodden families.

Compounding this: the `hand_written` eval bucket — the one bucket meant to
test real phrasing — is **24 examples**. Every percentage point reported
against it (0.750 → 0.833 this session) carries real statistical noise
(±1 example ≈ ±4.2 points). The numbers are real and the improvement is
real, but the confidence interval around them is wide.

## 6. No repair/retry loop — one misprediction is terminal

**Status: a first, narrow repair loop now exists.** `repair.py` connects
a failed boolean operation to ONE bounded, safe retry (swapping operands
on a commutative boolean -- union/intersection, never difference). This
is deliberately narrow: section 8 of the follow-on mission's own
constraints ("never change explicit user dimensions," "never silently
remove a requested feature") rule out most tempting auto-fixes (e.g.
auto-adjusting `shell()`'s wall_thickness), leaving operand-order retry as
the one mechanically safe repair under BOTH constraints simultaneously.
Honestly disclosed: a deliberate attempt to construct a naturally-failing
boolean case to prove real-world value did not succeed (manifold3d proved
more robust than expected) -- the mechanism is tested via controlled
failure injection instead, which proves the logic is correct, not that it
has already demonstrably fixed a real failure. See Addendum 5. The
knowledge-base-informed repair ("retrieve troubleshooting_procedure chunks
for the actual error") this section originally described as the
pending-but-flagged idea is still NOT implemented -- this is a narrower,
immediately-safe mechanism, not that fuller vision.

A `suspect`/`rejected` result is reported honestly (not silently papered
over), which is correct behavior — but nothing retries with different
information. `knowledge/chunking.py` already tags chunks as
`failure_mode`/`troubleshooting_procedure`, and `sandbox.py` already
returns a typed, specific error — the pieces for a bounded repair loop
("retry once, informed by the actual failure") exist but were never wired
together (flagged as the "next highest-impact improvement" in
`MISSION_STATUS.md` as far back as the RAG-build mission, still true now).
Until that exists, every misprediction anywhere in the pipeline is a
dead end for that part, with no mechanism to recover.

## 7. Hardware and model-capability ceiling

**Status: unchanged** (a hardware fact, not something this follow-on
mission's code changes could address).

Measured directly this session, not assumed: this machine's local LLM
(Qwen3-4B-Instruct) runs at roughly 0.1–0.4 tokens/sec without GPU
offload, and real `decompose_prompt()` calls take 13–20+ minutes even
*with* manual GPU offload enabled. Two direct consequences for accuracy:

- **The LLM itself is a generic, off-the-shelf 4B-parameter model, never
  fine-tuned on this project's own decomposition task.** All of its
  behavior is steered through prompt engineering (a ~4400-token system
  prompt) and grammar-constrained decoding — real, working levers, but
  with a lower ceiling than task-specific training would have.
- **The sheer per-call cost has sharply limited how much real,
  full-pipeline measurement has ever happened.** Across this entire
  session, only a handful of real `decompose_prompt()` calls have been
  run end-to-end (roughly half a dozen). The `program_model.py` numbers
  this report cites (0.833 hand_written, etc.) are real and well-measured
  — but they test only HALF the pipeline. The LLM-decomposition step's own
  accuracy had **zero formal measurement** until `decompose_eval.py` was
  built this session (see the next section) — "low quality" may partly
  reflect "almost no one has ever measured this half of the system at
  scale," not strictly "it's provably bad," though the structural reasons
  above (stages 2–6) make some real degradation there a near-certainty
  regardless.

## 8. Assembly placement outside 3 relationship kinds relies on LLM arithmetic — documented unreliable

**Status: unchanged** (the follow-on mission's section 9, "deterministic
constraint and placement engine," was not reached this pass -- see
Addendum 5's honest scope list).

`parallel_builder.py` combines built parts by translation. Positions come
from either (a) `assembly_plan.py`'s deterministic relationship resolution
(`attached_to_face`/`concentric_with`/`symmetric_about` — exact, resolved
from real built geometry) or (b) the LLM's own estimated `position_mm`
coordinates for anything else. Case (b) is **confirmed unreliable by this
project's own prior investigation**: `llm_decompose.py`'s
`_correct_gear_mesh_spacing()` exists specifically because a live test
showed the model "consistently copied a distance straight out of the
system prompt's own worked examples instead of recomputing for the
request's actual numbers, even with an explicit... instruction." Any
multi-part assembly whose spatial relationship isn't one of the 3
deterministic kinds is positioned by exactly the kind of LLM arithmetic
already shown not to work reliably — parts can end up overlapping,
floating apart, or simply mis-spaced.

---

## What this session actually did about it

Not purely diagnostic — concrete, measured improvements landed today:

- Closed a real training gap (6 tools totally unreachable from text) with
  a measured **+8.3-point accuracy improvement** on the hardest bucket.
- Built `decompose_eval.py` — the harness that makes section 5/7's "the
  LLM-decomposition step was never measured" gap closeable going forward.
- Found and fixed 3 real bugs via real acquired content (a missing `.rst`
  parser, a YAML front-matter corruption bug, a stack-trace-misread-as-
  heading bug) and one in the benchmark script itself (a topology
  key-name typo that silently reported 0% watertight when the real figure
  was 100%) — each found by not trusting a suspicious number at face
  value, which is itself the relevant discipline for catching more of
  what's described above.
- **Ran `decompose_eval.py` for real** against two of this project's own
  historically-fragile cases, immediately after writing this audit:
  - `d06` ("small aluminum L-bracket" — a terse, custom, non-everyday
    part with zero dimensions given): decomposed to a single entry, "an
    L-shape bracket," with **no invented dimensions and no spurious
    relationship** — **PASS**. 1157s (~19.3 min).
  - `d07` (two meshing spur gears, module 2mm, 15/20 teeth, each on its
    own 8mm/40mm shaft): decomposed to the correct 4 parts with every
    number right (module 2, 15 teeth, 20 teeth, 8mm, 40mm) and, critically,
    **zero relationships used** — exactly STEP 2's rule that meshing
    gears use plain position_mm, not `attached_to_face`/`concentric_with`
    — **PASS**. 254.5s (~4.2 min).

  n=2 is not proof of general reliability (section 5's honest caveat about
  small samples applies here too), but this is real, positive evidence on
  two of the SPECIFIC failure modes this project's own prior sessions
  fought hardest to fix (dimension invention, gear-mesh relationship
  misuse) — not cherry-picked successes, the first two real samples run.
  Notably, d07 took 4.5x less time than d06 despite producing 4x the
  output — plausibly because it closely resembles `_SYSTEM_PROMPT`'s own
  worked gearbox example, needing less generation "effort" than a genuinely
  novel one-line request, a concrete illustration of section 1's point
  that well-trodden patterns get real competence while novel ones cost
  more (here, in latency; elsewhere, in accuracy).

## Prioritized recommendations, if the goal is to move the needle further

1. **Make validation check correctness-to-spec, not just mesh validity**
   (#3) — the highest-leverage structural fix. Compare built dimensions
   against the dimensions actually stated in the prompt/description; flag
   a mismatch as a distinct, reportable status instead of folding it into
   a blanket "ok."
2. **Wire up the repair loop** (#6) — the pieces already exist; this is
   integration work, not new invention.
3. **Run `decompose_eval.py` at real scale** (#5/#7) — now that it exists,
   actually measuring the LLM-decomposition step across more than a
   handful of real calls would turn "probably degraded somewhere in
   stage 1" into a real, specific, fixable number.
4. **Fix `round_box_edges`/`chamfer_box_edges`'s bounding-box-replacement
   risk** (#4) — either make it refuse non-box targets explicitly (typed
   failure, consistent with this project's own stated principle) rather
   than silently substituting geometry, or implement a true edge-local
   version.
5. **Accept, don't fight, root cause #1** — if truly open-ended
   generalization is the actual goal, that is a different, larger project
   (the docstring's own words) than tuning this one further; if the
   6-family-plus-modest-composition scope is acceptable, the above 4 items
   are where real accuracy gains are still available within it.
