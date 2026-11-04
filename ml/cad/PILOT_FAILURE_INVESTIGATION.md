# Pilot Failure Root-Cause Investigation

Investigation-only. No production code was modified. Git state at the time of
this investigation: branch `main`, HEAD `ad239605c5b8971d49341a74de20c768d1390875`
("Merge branch 'feature/abutted-to-relationship'"), working tree clean except
for the evidence PDF (`IMG_2095 3.pdf`, untracked) this report is built from.

## 0. Evidence used

The pilot prompts, results, and UI chrome in this report come from
`IMG_2095 3.pdf` (4 screenshots of the live `/workspace` UI, added to the repo
root by the user during this investigation). No separate "evaluation report"
document, pilot log, or P1-P5 artifact exists anywhere in the repository, git
history (all commits on `main`, confirmed via `git log --all`), or the
filesystem (Desktop/Downloads/Documents searched) — I looked and could not
find one. Everything below is reconstructed from those 4 screenshots plus a
full trace of the actual production code (`server.py`, `llm_decompose.py`,
`parallel_builder.py`, `assembly_plan.py`, `sandbox.py`, `spec_validation.py`,
`mesh_quality.py`, `frontend/workspace.html`) at the current `HEAD`.

**P5 is not represented in the PDF at all** — only 4 screenshots exist, one
each for prompts I'm labeling P1-P4 below (the brief's own prose, e.g. "Use P1
and P2 as examples... coupling, spacer and collar", "P3's extra cylinders",
confirms this numbering). I have no evidence for P5 and make no claims about
it.

The four prompts, transcribed verbatim from the screenshots:

- **P1**: "Design a mechanical shaft support assembly consisting of a 100 mm
  × 80 mm × 12 mm rectangular base plate, with four 8 mm diameter mounting
  holes positioned 10 mm from each corner. Mount a bearing housing centrally
  on the plate. The housing should have a 40 mm outer diameter and a 20 mm
  central bore. Add a 20 mm diameter shaft, 100 mm long, passing
  concentrically through the bearing bore. At the positive end of the shaft,
  attach a cylindrical coupling of 30 mm diameter and 25 mm length, joined
  end-to-end with no axial gap. Keep the shaft and bearing concentric, and
  ensure the coupling does not overlap the shaft body."
- **P2**: "Create a vertical mechanical stack on a 60 mm × 60 mm × 10 mm
  square base. Place a cylindrical standoff of 20 mm diameter and 30 mm
  height centrally on the top face. Place a 20 mm diameter, 10 mm tall
  spacer directly on top of the standoff, with zero gap. Place a 20 mm
  diameter, 15 mm tall collar on top of the spacer, leaving an intentional
  3 mm gap between the spacer and collar. All three cylindrical components
  must share the same vertical axis. The base must remain at the bottom."
- **P3**: "Generate an L-shaped mounting bracket from 8 mm thick material.
  The horizontal base should measure 100 mm × 60 mm. The vertical plate
  should be 100 mm wide and 70 mm tall, mounted flush along the rear 100 mm
  edge of the base. Add four 8 mm through-holes to the horizontal base,
  arranged symmetrically with their centres 12 mm from the nearest edges.
  Add two 10 mm through-holes to the vertical plate, arranged symmetrically
  along its width and height. Keep the two plates perpendicular and ensure
  the holes pass completely through their respective plates."
- **P4**: "Build a compact mechanical transmission assembly on a 140 mm ×
  100 mm × 10 mm base plate. Include two parallel horizontal shafts, each
  12 mm in diameter and 80 mm long, supported by two bearing blocks per
  shaft. Position the shafts 60 mm apart, measured centre-to-centre. Mount a
  30-tooth spur gear with module 2 on the first shaft and a 30-tooth spur
  gear with module 2 on the second shaft. The gears must mesh correctly. Add
  a 25 mm diameter pulley at the exposed end of the first shaft and a 25 mm
  diameter pulley at the exposed end of the second shaft. Keep the pulleys
  aligned in the same plane. All shafts, bearings, gears and pulleys must be
  separate identifiable components."

All 4 screenshots are the `/workspace` UI (`frontend/workspace.html`), the
"Continue" prompt flow — not the manual Tool Console, not the landing-page
showcase.

A fresh, live re-run of P1 against this machine's current server was
attempted during this investigation as a concrete worked example. It is
still running as this report is written — see §6 for why, and §1 for what it
already tells us independent of its result.

## 1. Production-path architecture (confirmed from code)

The "Continue" button (`frontend/workspace.html:623`) calls
`requestClarification()` → `POST /api/clarify` (an LLM pre-check that may ask
up to 3 questions) → if none, `runBuild()` → `POST /api/build` →
`server.py:api_build()` (`server.py:202-355`) → `build_assembly_from_prompt()`
(`parallel_builder.py:787`), which:

1. `llm_decompose.decompose_prompt()` (`llm_decompose.py:628`) — the **only**
   file in the project that imports an LLM. A local, in-process
   `Qwen3-4B-Instruct-2507` (Q4_K_M, GGUF, via `llama-cpp-python`,
   `MODEL_NAME`/`QUANTIZATION` in `llm_decompose.py:39-41`), loaded as a lazy
   singleton (`llm_decompose.py:236-248`), called once per `/api/build`
   request with a long, carefully engineered system prompt. Its job is
   strictly bounded to deciding **what parts exist and how they relate**
   (`relationship` objects: `attached_to_face`, `concentric_with`,
   `parallel_to`, `symmetric_about`) — it never emits geometry or numbers
   beyond what the user already gave.
2. Each decomposed point is built independently by this project's own
   proprietary `program_model.py` + `sandbox.py` (statistical prediction or,
   for literally-parseable descriptions, `compositional_decomposer.py`'s
   deterministic path), via a `ProcessPoolExecutor`
   (`parallel_builder.py:354-644`, `build_assembly_from_points`).
3. Relationship-bearing points get their real position AND rotation resolved
   from the target's actual built mesh (`assembly_plan.resolve_position` /
   `resolve_rotation_deg`), in dependency-wave order.
4. Per-part dimension/hole-count/diameter validation runs
   (`spec_validation.py`), a collision check excludes unrelated overlapping
   parts (`parallel_builder._find_and_resolve_collisions`), and everything is
   concatenated into one mesh and exported to a uuid-named `.stl`
   (`server.py:37-40`).
5. `applyBuildResponse()` in the frontend loads that STL via `loadSTL()`
   (`frontend/workspace.html:826-917`).

This is a substantially more validated pipeline than a first look suggests —
see §5. It is also, as of `HEAD`, running a stripped-down version of itself:
commit `b8b74a0` (2026-11-01 19:58) added a much larger "engineering
validation pipeline" (component-inventory reconciliation, a unified build
validation report, `assembly_compatibility.py`, a clarification-decision
system) and commit `a28ac18` (2026-11-01 23:45, ~4 hours later) reverted it
back to the pre-pipeline `frontend` commit, explicitly "at the user's explicit
request." None of that reverted work is present at current `HEAD`. I don't
know whether the pilots were run before or after this rollback — the PDF has
no timestamp — but it matters: the reverted pipeline's "component inventory
reconciliation" is almost exactly the capability whose absence explains the
biggest gap found below (§5).

## 2. Investigation 1 — Coordinate systems and orientation

**Confirmed: yes, there is a real up-axis mismatch, and it is the single
most visually obvious thing wrong in all 4 screenshots.**

Traced end to end:

- **Generation** (`sandbox.py:464-470`, `add_box`): `add_box(length, width,
  height)` → `trimesh.creation.box(extents=[length, width, height])`. The
  three numbers map directly to local X, Y, Z. `height` → local **Z**.
- **Relationship semantics** (`llm_decompose.py`'s system prompt, STEP 2;
  `assembly_plan.py:445-449`, `_ALIGN_LOCAL_Z_TO`): the entire relationship
  system is built around **Z as the "natural" vertical/running axis** — a
  cap sits on a bottle via `face: "+z"` ("ON TOP"), a boss rises from a
  plate via `+z`, `_ALIGN_LOCAL_Z_TO["+z"] = (0,0,0)` (no rotation needed —
  Z is already "up"). This is a real, internally consistent convention:
  **the generator is Z-up.**
- **Mesh export** (`server.py:37-40`, `_export_mesh`): `mesh.export()` to
  `.stl` — raw triangle coordinates, no axis metadata (STL format has none),
  no transform applied.
- **Frontend load** (`frontend/workspace.html:826-917`, `loadSTL`):
  `new STLLoader().parse(buf)` — the raw vertex coordinates are parsed
  **directly** into the three.js scene. No `geometry.rotateX(...)`, no axis
  remap, anywhere in this function or its call sites. I read the whole
  function; there is no conversion step.
- **three.js scene setup** (`frontend/workspace.html:698`, `:799`):
  `camera.up.set(0, 1, 0)` (explicit Y-up) and `new THREE.GridHelper(300, ...)`
  (three.js's `GridHelper` always lies flat in the **XZ** plane by
  construction — Y is its normal). **The viewer is Y-up**, with the visible
  "ground" plane being XZ.

So: the generator treats a box's `height` argument as local **Z**, meaning
"the dimension that should point up, off the ground plane." The viewer
treats **Y** as up and XZ as the ground plane. **Nothing in this pipeline —
not export, not the API response, not `loadSTL` — ever converts between the
two.** A box built as "100mm × 80mm footprint, 12mm tall" (generator's
intent: X=100, Y=80, Z=12-thin-and-vertical) lands in the viewer with X=100
(a horizontal screen axis), Y=80 (the viewer's **vertical** axis), Z=12 (into
the screen). The result: an object meant to be a thin, wide, flat base lying
on the ground instead renders as an **80mm-tall vertical slab**, 100mm wide,
and only 12mm deep — i.e., looks like it's standing up like a wall.

This is exactly what all three "base plate" prompts show:

- **P1** (page 1): the "100×80×12mm rectangular base plate" renders as a
  tall vertical rectangle with the 4 corner holes visible on its face —
  consistent with Y (viewer-vertical) carrying the plate's 80mm dimension.
- **P2** (page 2): the "60×60×10mm square base" that's explicitly required
  to "remain at the bottom" of a "vertical...stack" instead renders as a
  vertical slab, with the cylindrical stack running sideways out of it
  horizontally — the exact opposite of what was asked for, and exactly what
  the Z→viewer-depth, Y-gets-a-horizontal-footprint-dimension mapping
  predicts.
- **P4** (page 4): the "140×100×10mm base plate" again renders standing
  vertically.
- **P3** (page 3) is the control case that confirms this isn't a camera
  artifact: an **L-bracket** is two plates meeting at 90°, so whichever axis
  is "wrong," at least one of its two faces reads correctly from most
  angles — and indeed the L-shape itself is recognizable in that screenshot,
  unlike P1/P2/P4's single flat plates, which have no such redundancy and so
  show the mismatch starkly.

**This is not a camera-angle illusion.** A camera looking at a genuinely
thin, flat, horizontal plate from a 3/4 angle would show a thin sliver, not
a tall rectangular face with visible frontal area roughly matching the
plate's two LARGEST stated dimensions (100×80, 140×100) — which is what's in
every screenshot. The only way to get that is for the mesh's actual
in-viewer bounding box to have two large extents on screen-horizontal and
screen-vertical axes and one small extent on the screen-depth axis, which
only happens if Y (one of the plate's two large footprint dimensions) is
genuinely occupying the viewer's vertical axis.

**Concrete coordinate example** (traced from code, not yet confirmed against
a live P1 re-run — see §6 for why): for P1's base plate, `compositional_decomposer`
or `program_model` would call `add_box(length=100, width=80, height=12)`
(or some permutation depending on how the LLM orders the dimensions in its
plain-text description, e.g. "a box 100mm x 80mm x 12mm" — the order the LLM
chooses to list the three numbers in literally determines length/width/height
here). Generator-space: X=100 or whichever axis gets the first number,
Y=second number, Z=12 (the explicitly-named "thick" dimension, almost always
listed last and so almost always Z). Exported STL: raw coordinates, vertices
spanning those same three axes, unchanged. Loaded into three.js: those same
raw X/Y/Z coordinates become three.js X/Y/Z directly. Displayed: whatever was
generator-Y (a large, ~80-100mm footprint dimension) is now screen-vertical;
whatever was generator-Z (the 12mm thickness) is now screen-depth. The plate
reads as "standing up."

**The viewer does not rotate the whole model, nor individual objects** — I
read `loadSTL` in full; there is no rotation call anywhere in it, and the
combined mesh arrives from the server already concatenated (one STL, one
`THREE.Mesh`), so "individual objects" isn't even a concept the viewer has at
render time.

**Does this explain the screenshots by camera orientation alone?** No —
camera orientation is a red herring here; the bug is upstream of the camera
entirely, baked into the raw mesh coordinates the camera is pointed at. No
camera fix would address it; only an axis conversion at export or load time
would.

## 3. Investigation 2 — Workspace isolation ("Continue" / scene lifecycle)

**Confirmed: "Continue" replaces, it does not append — both on the frontend
and the backend. This is NOT where P3's extra geometry comes from.**

- `runBuild()` (`frontend/workspace.html:1352-1380`) → `applyBuildResponse()`
  (`:1305-1350`) → `setPartsFromBuildResult(data.points)`
  (`:1161-1193`): `currentParts = points.map(...)` — a **wholesale
  reassignment**, not an append (`.push` is only ever used by the
  playground's manual "+ Add part" button, a deliberate user action,
  `:1195-1204`).
- `applyBuildResponse` → `loadSTL(data.mesh_url)` → `loadSTL`
  (`:826-917`) does `if (currentMesh) scene.remove(currentMesh); if
  (wireframeMesh) scene.remove(wireframeMesh);` **before** adding the new
  mesh — the old three.js objects are genuinely removed from the scene, not
  hidden or layered under the new one.
- `_export_mesh` (`server.py:37-40`) names every exported STL with a fresh
  `uuid.uuid4().hex` — no filename reuse, no risk of a stale cached mesh
  being served under an old or colliding URL.
- `decompose_prompt()` (`llm_decompose.py:628-687`) is a pure function of
  `(prompt, knowledge_context)`. The only persistent state across calls is
  the lazy-loaded `_llm` singleton (model weights) and two grammar objects —
  neither carries conversation history. `llama-cpp-python`'s
  `create_chat_completion` resets its own internal evaluation state at the
  start of each call by default (no `set_cache(...)` is ever called
  anywhere in `llm_decompose.py`), so there's no KV-cache bleed between
  requests on the same `Llama` object either.
- The server (`server.py:854`, `app.run(host=..., port=..., debug=False)`)
  passes neither `threaded=True` nor `processes=`, so even if a user
  double-clicked Continue or opened a second tab mid-build, there's no
  shared-mutable-state race to worry about in the way concurrent requests
  sharing one `Llama` context would create (though see §6 for the real,
  confirmed consequence of this: it does not scale to more than one request
  at a time, at all).

So: a genuinely fresh "Continue" click, on its own, does not leak state from
a previous build, on either side of the wire. **P3's extra cylinders are not
frontend/backend session contamination.**

What I could **not** rule out without a live re-run or the original
`raw_llm_output` (not present in `build_log.jsonl` — see §4): the LLM's
decomposition for that specific prompt genuinely producing extra parts. Two
concrete, code-grounded candidate mechanisms, both plausible, neither
confirmed:

1. **Plain hallucination** — the model associates an "L-shaped mounting
   bracket... with four 8mm through-holes" with fasteners/bolts/rollers it
   was never asked for. Nothing in the system prompt forbids this, and
   nothing downstream checks "every part in the output is traceable to the
   request" (see §5).
2. **A specific, real parsing behavior in `_extract_json_array`**
   (`llm_decompose.py:393-416`): it deliberately concatenates **every**
   schema-shaped JSON array found after the model's last `</think>` tag,
   not just the last one — a documented, intentional fix for a model that
   "frequently formats its final answer as several separate single-item
   arrays back to back" (a correct decomposition split across multiple
   arrays). But the same code path cannot distinguish that benign case from
   a model that second-guesses itself post-`</think>` and emits **two
   different full attempts** (a draft, then a correction) — both would
   parse as schema-shaped arrays and both would be concatenated, silently
   doubling or inflating the point list. I did not find a test in
   `tests/test_llm_decompose_isolation.py`'s remnants or elsewhere that
   distinguishes these two cases.

Both are hypotheses, not confirmed causes — see §6 for what a live P3 re-run
(inspecting the now-restored `raw_llm_output` logging) would settle
definitively.

## 4. Investigation 3 — Component identity and geometry

- Each decomposed point keeps a **stable identity** throughout the build:
  `ActionablePoint.id`, carried through `PointBuildResult`, `parallel_builder`'s
  `placed_meshes_by_id` dict, and back out as one entry per point in
  `/api/build`'s JSON response (`server.py:247-308`). Components are never
  silently fused into an opaque blob before the response is built.
- They ARE, however, concatenated into one combined mesh
  (`trimesh.util.concatenate`, `parallel_builder.py:634`) for `mesh_url` —
  **but the per-component meshes are not lost**: `PointBuildResult.mesh`
  holds each part's own mesh object server-side for the duration of the
  request (used for the per-part dimension/hole checks in step 5), and the
  per-point entries in the JSON response (`topology`, `dimension_match_rate`,
  `claimed_hole_count`/`detected_hole_count`, `program`) are all computed
  from that **individual** mesh, not the combined one. So: "is the final
  combined mesh the only representation available to validation" — **no**,
  per-part validation runs on per-part meshes; only the single exported
  `.stl` for the viewer is the fused one. Individual component meshes
  themselves are not separately exported/downloadable anywhere in this
  pipeline, though — `_export_mesh` is only ever called once per request, on
  the combined result.
- **Whether the coupling/spacer/collar exist in the generated plan, geometry
  and final output** for P1/P2: I cannot answer this from the screenshots
  alone with certainty, and I don't have the original `raw_llm_output` (see
  §4 below / §6). What the screenshots are consistent with:
  - **P1**: the combined-mesh panel shows `1435 verts, 2874 faces,
    watertight`, bounding box reading (partially obscured by the phone
    photo's left edge) approximately `100.0 × 80.0 × 100.0mm`. The spec
    implies a total "depth" along the shaft axis of plate-thickness (12mm) +
    shaft (100mm) + coupling (25mm, "no axial gap") ≈ 137mm if the coupling
    truly sits beyond the shaft's far end with no overlap. A ~100mm reading
    on that axis is short of 137mm by a wide margin — consistent with
    either the coupling overlapping the shaft (the exact negative constraint
    the prompt explicitly asked to avoid), the coupling being omitted
    entirely, or my own misreading of a blurry phone photo's partially
    cropped number. I flag this as a **discrepancy worth re-checking against
    a live run**, not a confirmed defect — the bounding-box panel's first
    digit is genuinely not legible in the source image.
  - **P2**: only **one** cylindrical body is visible in the render, where
    three (standoff, spacer, collar) were requested, each with distinct
    stated dimensions and an explicit 3mm gap between two of them. This is
    either (a) components genuinely missing from the decomposition, (b)
    components built but with such severe overlap/collision exclusion that
    only one survived in the combined mesh (`_find_and_resolve_collisions`
    — see §5 — would exclude a colliding sibling with no relationship
    explaining the overlap, and three stacked cylinders described in plain
    `position_mm` with no `concentric_with`/`attached_to_face` relationship
    between them is exactly the shape of plan that check targets), or (c)
    the up-axis mismatch (§2) making 3 distinct short stacked cylinders look
    like one blob from this particular camera angle. I can't distinguish
    these from the screenshot alone.
  - **P4**: only **one** gear is visible where two meshing gears, two
    shafts, bearing blocks, and two pulleys were requested — the same
    "fewer parts rendered than requested" pattern as P2, same inability to
    distinguish "never planned" from "planned, built, then excluded by
    collision resolution" from "planned, built, present, but occluded/hard
    to see" from a single static screenshot.

**The honest answer to Investigation 3's core question, given available
evidence: I cannot confirm from the screenshots alone whether missing
components were never planned (an LLM decomposition gap), planned but
collision-excluded (parallel_builder's own, documented, working-as-designed
behavior), or planned, built, and present but simply not visible at this
camera angle given the up-axis bug. These require the live per-point JSON
(§6) to distinguish, and I did not have it in time for this report.**

## 5. Investigation 4 — Actual assembly plans (table)

I cannot populate accurate "Planned / Built / Dimensions / Position /
Rotation / Final mesh present" cells for each requested component without
the actual decomposition JSON for each pilot run — `build_log.jsonl`
(`logs/build_log.jsonl`, gitignored, local-only) is the only place that data
would have been captured, and its last entry is from **2026-11-01 19:54**,
during the now-reverted validation-pipeline development window — **it has
no record of these 4 pilot runs**, regardless of when they happened. Two
explanations, both real, not mutually exclusive: (a) the pilots happened on
a server process that was later killed without ever appending further, or
(b) **`log_build()` is only called on the success path of `api_build()`**
(`server.py:234-235`: an exception inside the `try` block returns the error
JSON immediately, before `log_build` is ever reached) — so any pilot run
that errored out server-side (a real possibility given this pipeline's
documented fragility under this hardware's latency, §6) would leave **zero**
trace in the log, even on a freshly-restarted, correctly-logging server.
This is itself a real gap: **a failed build is invisible to the one logging
mechanism this project has**, which directly works against debugging
exactly the kind of pilot failure this investigation is for.

What I can state with confidence, independent of per-run data: every
component named in all 4 prompts is one the system's own vocabulary and
`_SYSTEM_PROMPT` worked examples explicitly cover (boxes, cylinders, holes,
gear profiles, hole patterns, `concentric_with`/`parallel_to` relationships
for shaft/gear pairs) — none of P1-P4 asks for anything outside the
documented tool roster. So a missing component, if confirmed, would not be
"the builder can't make that shape" but specifically a **decomposition or
assembly-level loss**, not a geometry-capability gap.

## 6. Investigation 5 — Validator audit

This is the most concretely confirmable investigation, directly from code,
independent of any live run.

**What IS validated, per-part, today** (`spec_validation.py`,
`parallel_builder.py:472-592`):
- Structural watertightness + positive volume (`Sandbox.execute`'s own
  `status="ok"` gate — unchanged, pre-existing).
- Stated dimensions ("NNmm" claims in the description text) vs. the built
  mesh's actual measured extents (`validate_dimensions_against_description`)
  — downgrades `status` from `"ok"` to `"suspect"` on mismatch, and this
  downgrade is surfaced in the API response and in `AssemblyBuildResult.overall_status`.
- Claimed hole count vs. detected hole count, and claimed hole/bore diameter
  vs. detected diameter (`validate_hole_count_against_description`,
  `validate_hole_diameters_against_description`) — same downgrade behavior.
- A semantic-name-vs-explicit-dimension conflict disclosure (e.g. "NEMA 17"
  named alongside a disagreeing explicit size) — disclosure only, not a
  failure.
- Unrelated-part collision: `_find_and_resolve_collisions`
  (`parallel_builder.py:670-727`) excludes one of two independently-placed
  (no relationship) parts whose world-space AABBs overlap by >30% of the
  smaller one's volume, and discloses this in `collision_warnings`.
- An aggregate `overall_status` (`PASS`/`INCOMPLETE`/`FAIL`,
  `parallel_builder.py:176-` onward) that rolls all of the above up into one
  field specifically so a caller doesn't have to inspect every point
  individually.

**What is explicitly, by design, NOT validated** (confirmed from
`spec_validation.py`'s own module docstring, lines 1-30): "feature
completion, spatial constraints, and negative constraints are separate,
harder problems this module does not attempt," and dimension matching is
"intentionally NOT axis-aware." Concretely, for these 4 prompts, nothing
checks:
- "the coupling does not overlap the shaft body" (P1, a negative spatial
  constraint).
- "the gears must mesh correctly" / "pulleys aligned in the same plane" (P4,
  spatial/relational constraints beyond simple dimension matching).
- Whether a part is oriented correctly in the real-world sense (§2's bug is
  invisible to every check above — a mesh can be perfectly watertight, the
  right dimensions, the right hole count, AND be rotated 90° from where a
  human would expect it, and nothing here flags that).

**The confirmed, highest-impact gap: nothing anywhere in the live
`/api/build` path checks that the NUMBER of components the request implies
were actually planned.** I verified this by checking for any live
"component count" / "component inventory" logic reachable from
`server.py` — there is none. The only thing in the repository that scores
decomposition **component count** is `decompose_eval.py`, which is an
**offline, developer-run benchmark** against a labeled dataset
(`llm_decompose.decompose_prompt` called directly, scored against
`expected_component_count`) — it is never imported by `server.py` and has no
bearing on what a live pilot build reports back. `AssemblyBuildResult.overall_status`
can only ever be `PASS`/`INCOMPLETE`/`FAIL` relative to whatever points the
LLM's decomposition **did** produce — a decomposition that silently drops
the "collar" from P2's 3-component stack, builds the remaining 2 perfectly,
and returns them both as `status="ok"` gets `overall_status: "PASS"`, with
nothing anywhere in the response disclosing that a component was ever lost.
**This is a confirmed false-pass mechanism, not a hypothesis** — it follows
directly from reading `overall_status`'s own implementation, which only
iterates over `self.points` (whatever the decomposition produced), never
compares against the original prompt.

This is precisely the capability the reverted `b8b74a0` commit
("Component inventory reconciliation: tracks every requested physical
component from the original prompt through decomposition, detecting
missing/duplicated/ambiguous components rather than silently dropping them")
existed to close, and precisely the gap re-opened when it was rolled back on
2026-11-01.

## 7. Confirmed vs. unconfirmed

**Confirmed (traced directly in code, or directly measurable from the
screenshots with no interpretation required):**
- The generator is Z-up; the viewer is Y-up; no conversion exists anywhere
  in the export/API/load path. (§2)
- "Continue" replaces scene/parts state on both frontend and backend; no
  session/cache leak exists in the reachable code. (§3)
- Per-part dimension/hole-count/diameter validation and unrelated-part
  collision exclusion are real and wired into the live path. (§5)
- No live check exists anywhere for "does the built component count match
  what the request implied" — `overall_status` cannot see a silently-dropped
  component. (§5)
- A failed (exception-raising) `/api/build` call is never logged by
  `build_logger.py` — only successes are. (§4)
- The component-inventory system that would have closed the count-checking
  gap was built, then explicitly reverted at the user's own request on
  2026-11-01, ~4 hours after being added. (§1, §5)
- Local LLM decomposition on this machine is severe enough that a single
  request ran 15+ minutes of CPU time with no result in this investigation's
  own live test. (§6)

**Unconfirmed hypotheses (plausible, grounded in code, but requiring a live
re-run or the original raw LLM output to settle):**
- Whether P1's coupling is actually present, overlapping, or omitted (the
  ~100mm vs. ~137mm bounding-box discrepancy, §4).
- Whether P2/P4's missing-looking components were never planned by the LLM,
  planned-then-collision-excluded, or planned-and-present-but-occluded by
  the up-axis bug. (§4)
- Whether P3's extra cylinders are an LLM hallucination for that specific
  prompt, or an artifact of `_extract_json_array`'s array-concatenation
  logic double-counting a self-corrected answer. (§3)

**Cannot be confirmed without more data (listed per the brief's own
request):**
- Anything requiring the original `raw_llm_output` for these specific 4
  runs, or the exact decomposition JSON — not present in `build_log.jsonl`,
  not in the PDF, not recoverable from the STL files alone (none of the
  exported `.stl` files in `webapp_output/` are dated anywhere near when
  these screenshots were taken — all existing ones predate this
  investigation).
- Whether the pilots ran before or after the 2026-11-01 23:45 validation-
  pipeline rollback (no timestamp in the PDF).
- The precise P1 bounding-box third dimension (phone photo, left edge
  cropped) without a live re-run.

## 8. Required final response

- **Is there an up-axis mismatch?** Yes, confirmed: the generator is Z-up
  (`add_box`'s `height`→Z, every relationship's "+z is up" convention), the
  three.js viewer is Y-up (`camera.up.set(0,1,0)`, `GridHelper` flat in XZ),
  and no conversion exists anywhere between mesh export and the viewer. This
  is the single most visually obvious defect across P1/P2/P4.
- **Does Continue append to the previous scene?** No — confirmed replace,
  both frontend (`currentParts` reassignment, `scene.remove()` before adding
  the new mesh) and backend (`decompose_prompt` is a pure function with no
  cross-request state).
- **Are components fused before validation?** No — per-part validation
  (dimensions, hole count/diameter) runs on each part's own individual mesh
  before the final `trimesh.util.concatenate` fuses them for export. Only
  the viewer's single STL is fused; it's not the only thing validation sees.
- **Does the validator verify prompt-level engineering requirements?**
  Partially, and the gaps are specific and confirmed: per-part dimension/
  hole-count/diameter checks exist and gate `status`; whole-assembly
  **negative/spatial constraints** ("doesn't overlap," "meshes correctly,"
  "aligned in the same plane") and **component-count-against-the-request**
  checks do not exist anywhere in the live path — the latter was built and
  then explicitly reverted on 2026-11-01.
- **Are components missing from the LLM plan, geometry construction, or
  final rendering?** Can't be isolated to one stage from the available
  evidence — plausible at any of the three for P2/P4's apparently-missing
  parts, and a live re-run with the response JSON (not just a screenshot)
  is the only way to tell them apart. What's ruled out: it's not a frontend
  rendering/scene-state bug (§3).
- **Single highest-priority fix:** the up-axis conversion (§2). It is the
  cheapest fix of everything found (one rotation, applied once, either at
  mesh export in `server.py:_export_mesh` or at load in
  `loadSTL`/`STLLoader.parse`), it is 100% confirmed (not a hypothesis), and
  it is very likely responsible for most of the *visual* severity of P1/P2/P4
  independent of whatever the component-count gap turns out to explain —
  fixing it alone would make the next round of pilot testing dramatically
  easier to evaluate correctly, including distinguishing real missing-
  component bugs from "it's actually there, just sideways."

## 9. Prioritized remediation plan (smallest safe change first)

Investigation-only — none of this was implemented. Ordered by risk/cost,
cheapest first:

1. **Up-axis conversion** (§2, §8). Smallest possible change: one fixed
   rotation applied once, either when `server.py:_export_mesh` writes the
   STL, or when `frontend/workspace.html:loadSTL` parses it (the latter is
   probably safer — it leaves the generator's own internal Z-up convention,
   which `assembly_plan.py`'s whole relationship system is built around,
   completely untouched, and only changes how the already-correct geometry
   is displayed). Needs checking against `frontend/index.html`'s showcase
   viewer and `print_lab.html` too, for the same STLLoader pattern, before
   calling this done everywhere it matters.
2. **Log failures, not just successes**, in `build_logger.py` /
   `server.py:api_build`'s exception path (§4). A one-line change (log
   before returning the error response, not instead of it) that would make
   the *next* pilot failure immediately debuggable instead of invisible.
3. **Re-add component-count reconciliation against the original prompt**
   (§5, §8) — the specific capability `b8b74a0` already built and that was
   reverted. Given it already existed once, the lowest-risk path is
   reviewing that reverted diff (still fully reachable at tag
   `backup-before-rollback-2026-11-01` per the rollback commit's own
   message) rather than re-designing from scratch — but re-introducing it
   is a bigger, riskier change than #1/#2 and was explicitly reverted once
   already, so this needs the user's own sign-off on scope before any of it
   comes back, not just a mechanical re-apply.
4. **A live re-run of all 4 (ideally with `answers` pre-filled to skip the
   clarify round-trip) with the response JSON captured**, to convert §4/§7's
   unconfirmed hypotheses into confirmed findings — specifically P1's
   coupling-overlap question and whether P2/P4's missing-looking parts were
   never planned vs. collision-excluded vs. occluded. Blocked, in practice,
   by #5 below.
5. **Build latency** (§6): not a correctness bug, but confirmed severe
   enough (15+ minutes of CPU time for P1 alone, on this machine, with no
   result) to be a real operational blocker to *any* further pilot testing
   or live debugging, including #4 above. Worth its own investigation
   (model size/quantization choice, `N_GPU_LAYERS` — confirmed unused
   despite a Metal-capable GPU being present per `llm_decompose.py`'s own
   comments, §deployment config) separate from this report's scope.
