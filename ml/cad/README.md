# ml/cad — local, GPU-free CAD prototype

Read `docs/cad-finetuning-technical-report.md` (repo root) first, especially
**Section 10**, which records why this ended up rule-based instead of a
fine-tuned Hunyuan3D pipeline: the project owner asked for a prototype that
runs entirely on this laptop (Intel integrated graphics, 8GB RAM, no CUDA),
without cloud spend, and "not too heavy." Hunyuan3D-2.1 needs an NVIDIA GPU
(10–29GB VRAM) no matter how it's fine-tuned, so there is no version of it
that runs here. This instead demonstrates the same "prompt → engineering
intent → CAD geometry" idea using a rule-based parser and a deterministic
mesh generator — no neural network anywhere in the loop.

## Quickstart

```bash
cd ml/cad
pip3 install -r requirements.txt

python3 inference.py --prompt "rectangular aluminum mounting plate, 120 mm x 80 mm x 8 mm, with four 10 mm mounting holes, 20 mm from each edge, 5 mm corner fillets" --out plate.stl
python3 evaluate.py --meshes . --out eval_report.json
python3 benchmark.py --n 20
python3 scripts/preview_mesh.py --mesh plate.stl --out plate.png --elev 89 --azim -90
```

(Use `python3`/`pip3`, not `python`/`pip` — this machine has no `python` alias.)

### Web UI

```bash
python3 server.py --port 5050
# open http://127.0.0.1:5050 — type a prompt, click Generate, rotate/zoom the result
```

This is a thin Flask wrapper (`server.py`) around the exact same
`generate_from_prompt()` that `inference.py`'s CLI uses — same parser, same
generator, same two part families. The page (`frontend/index.html`) is a
single static HTML file using [three.js](https://threejs.org) (loaded from a
CDN) to render the returned `.stl` interactively in-browser via `STLLoader`;
nothing is installed for this beyond Flask (already in `requirements.txt`).
Generated meshes are written to `webapp_output/` (gitignored, not the
`examples/` folder).

### Want to see real Hunyuan3D output instead of the rule-based prototype?

`kaggle/hunyuan3d_baseline.ipynb` runs the actual Hunyuan3D-2.1 model on a
free Kaggle GPU (shape-generation only, no local GPU needed) — see
`kaggle/README.md` for exact upload/run/download steps.

---

## How it actually works

There is no AI model anywhere in this pipeline. It's three deterministic
stages, each one plain Python:

```
"rectangular aluminum mounting plate,     nlp/prompt_parser.py       data/procedural_generator.py       trimesh
 120mm x 80mm x 8mm, four 10mm     ──▶    regex extraction      ──▶  dataclass params ──▶ trimesh    ──▶ .export()
 mounting holes, 5mm fillets"             + keyword classify         primitives + boolean difference       .stl / .glb / .obj
```

### 1. Prompt → structured parameters (`nlp/prompt_parser.py`)

`parse_prompt(text)` does two things, both plain regex/keyword matching — no
tokenizer, no embeddings, no model weights:

**a. Decide the part family.** It scans the lowercased prompt for keywords, first
match wins, checked in this order:

- contains `flange`, `housing`, `pump`, `bearing`, `pulley`, or `bore` → **flange**
- contains `plate`, `bracket`, `mount`, or `panel` → **plate**
- none of the above → defaults to **plate** (recorded as `part_type_inferred: true` in the output, so it's visible when the guess was a guess)

**b. Extract every numeric field** with its own regex, each with a hardcoded
fallback if the prompt doesn't mention it:

| Plate field | Regex trigger | Default if absent |
|---|---|---|
| length/width/thickness | `"120mm x 80mm x 8mm"` pattern, or separate `"... mm long"` / `"... mm wide"` / `"... mm thick"` | 100 × 60 × 6 mm |
| hole_count | a number word (`"two"`–`"twelve"`) or digit before `"holes"` — snapped to 2 or 4 (nearest) | 4 |
| hole_diameter_mm | `"N mm ... hole(s)"`, or an `M`-thread size (`M8` → 8mm, via a fixed lookup table for M3–M16) | 8 mm |
| hole_edge_margin_mm | `"N mm from each/the edge"` | 15 mm |
| fillet_radius_mm | `"N mm (corner) fillet"` | 5 mm |
| material | one of `aluminum/steel/polycarbonate/titanium/plastic/brass` | aluminum |

| Flange field | Regex trigger | Default if absent |
|---|---|---|
| outer_diameter_mm | `"N mm outer diameter"` | 120 mm |
| bore_diameter_mm | `"N mm (central) bore"` | 40 mm |
| thickness_mm | `"N mm (wall) thick(ness)"` | 8 mm |
| bolt_hole_count | number word before `"mounting holes"` | 4 |
| bolt_hole_diameter_mm | same as plate's hole_diameter (mm or M-thread) | 8 mm |
| bolt_circle_diameter_mm | `"N mm bolt circle"` | 75% of outer diameter |
| material | one of `steel/aluminum/cast iron/titanium/brass` | steel |

Every field ends up in either `parsed_fields` (came from the prompt) or
`defaulted_fields` (guessed) — that split is printed by `inference.py` so you
can always see exactly what the parser understood versus filled in.

### 2. Parameters → mesh (`data/procedural_generator.py`)

The parsed dataclass (`MountingPlateParams` or `FlangeParams`) drives plain
`trimesh` geometry construction:

**Mounting plate:**
1. Build a 2D rounded-rectangle outline by hand — 4 straight corners replaced
   with `fillet_radius`-sized circular arcs (8 line segments per corner), as an
   (N,2) point array.
2. `trimesh.creation.extrude_polygon()` extrudes that 2D outline into a solid
   slab of height `thickness_mm` (this triangulates the flat cap faces
   internally via `shapely`/`earcut` — that's the fan pattern visible in
   preview renders, it's just how a filled polygon becomes triangles).
3. For each hole position (4 corners, or 2 along the long axis), build a
   cylinder taller than the plate and **boolean-subtract** it from the slab
   (`mesh.difference(hole, engine="manifold")`, using the `manifold3d`
   library). This is what actually punches a clean, correctly-shaped hole
   through the solid, rather than just cutting the outline.

**Flange:** same idea — a solid cylinder disk (`outer_diameter_mm`), minus a
central bore cylinder, minus one bolt-hole cylinder per `bolt_hole_count`,
evenly spaced by angle (`np.linspace(0, 2π, count, endpoint=False)`) on the
bolt circle radius. All boolean subtraction, same engine.

Both builders return a `trimesh.Trimesh` — a plain in-memory triangle mesh
(vertex array + face index array). No CAD kernel, no B-rep, no NURBS — it's
computed once as an exact analytic shape and triangulated directly, which is
why dimensions and hole positions come out exact rather than approximated.

### 3. Mesh → file

`inference.py` calls `mesh.export(args.out)` — this is `trimesh`'s own writer;
the output format is chosen entirely by the file extension you pass to
`--out` (`.stl`, `.glb`, `.obj`, `.ply` all work, since the pipeline never
made assumptions about the format). `.stl` (what the CLI defaults/examples
use) is a plain binary triangle-soup format: every triangle's 3 vertex
coordinates plus a face normal, nothing else — no color, no material, no
hierarchy. That's a deliberate byproduct of stage 2 outputting a raw
`Trimesh`, not a limitation added at export time.

### PNG generation is a separate, optional tool

`scripts/preview_mesh.py` is **not** part of the generation pipeline — it's a
debugging/demo convenience that exists only because "what does the output
look like" is hard to answer from an STL file alone. It does:

1. `trimesh.load(path, force="mesh")` — reads the STL back into vertices/faces.
2. `mesh.vertices[mesh.faces]` — gathers the actual 3D triangles.
3. Hands those triangles to `matplotlib`'s `Poly3DCollection` in a 3D axes
   object, sets the camera (`--elev`/`--azim`), and calls `savefig()`.

This uses `matplotlib` specifically because it needs no OpenGL/display context
(works headless, over SSH, etc.) — at the cost of the internal triangulation
lines being visible (matplotlib doesn't do proper hidden-line removal the way
a real 3D viewer does), which is why a top-down view (`--elev 89 --azim -90`)
reads more clearly than the default isometric angle for these particular
shapes.

### Evaluation (`evaluate.py` → `evaluation/geometry_eval.py`)

Loads a mesh back with `trimesh` and runs pure-geometry checks: watertightness,
manifoldness, a planar-face-area fraction (via `trimesh`'s coplanar-facet
grouping), and a hole/cylinder detector that fits circles to clusters of
near-vertical mesh faces and classifies them as holes (concave, full 360°
loop) vs. convex features like fillets (partial arc) via the sign of the face
normal relative to the fitted circle's center. Validated against the known
generation parameters of the procedural dataset — see report Section 8.

---

## Max scope of this build, precisely

This is a narrow, honest prototype, not a general CAD generator. Exactly what
it can and can't do:

**Part families: exactly two.**
- Rectangular mounting plate — rounded corners (single uniform fillet radius),
  flat, uniform thickness, 2 or 4 through-holes only (any other requested
  count snaps to whichever of 2/4 is closer).
- Circular flange — flat, uniform thickness, one central bore, N bolt holes
  evenly spaced on a single bolt circle (N is whatever the prompt says, no
  cap).

**Nothing else exists as geometry**, so a prompt for a gear, shaft, bearing,
pulley, bracket-with-ribs, threaded hole, chamfer, slot, or any assembly of
multiple bodies gets forced onto one of the two shapes above via the keyword
table (e.g. "gear" isn't in either keyword list today, so it falls through to
the plate default) — it will not silently produce the wrong *dimensions*, but
it can produce the wrong *kind of object* entirely. This is the single
biggest scope boundary right now.

**Geometric constraints, even within the two supported families:**
- Everything is axis-aligned and extruded/revolved about a single vertical
  axis — no compound curves, no draft angles, no non-planar surfaces.
- All holes are simple circular through-holes — no counterbores,
  countersinks, blind holes, or actual helical threads (an `M8` in the prompt
  only sets the hole's nominal diameter to 8mm, it does not cut thread
  geometry).
- One fillet radius per plate, applied uniformly to all 4 corners; no
  chamfers anywhere; no ribs.
- One material label per part, used only as a text string in the description
  — there's no color, texture, or physical-property data attached to the mesh.
- Units are always millimeters; the parser does not recognize inches or any
  other unit and will silently treat any bare number as mm.

**Prompt understanding is regex, not language understanding.** It will miss
phrasings outside the patterns in the table above (e.g. "holes spaced 20mm
apart" instead of "20mm from each edge" won't be picked up) and fall back to
the default for that field rather than erroring — always check
`parsed_fields`/`defaulted_fields` in the output to know which happened.

**Speed/footprint:** ~40–110ms per generation on this machine, pure CPU, no
GPU/VRAM used at all (`benchmark.py` measures this directly) — because there
is no model inference happening, just geometry construction.

## Layout

```
data/          procedural_generator.py — parametric plate + flange builders (trimesh)
nlp/           prompt_parser.py — regex-based prompt -> engineering-intent extraction
evaluation/    geometry_eval.py — CAD-aware mesh evaluation, validated against known ground truth
scripts/       generate_dataset.py, run_eval.py, preview_mesh.py — CLI glue + PNG rendering
examples/      sample .stl/.png outputs (mounting_plate, pump_housing_flange)
configs/       cad_lora_v1.yaml — placeholder for the *future* neural path (Section 4), unused today
inference.py   prompt -> mesh, the real working entry point
evaluate.py    mesh -> geometry/CAD report
benchmark.py   latency benchmark over a prompt set
train.py       no-op by design; explains why in its docstring
```
