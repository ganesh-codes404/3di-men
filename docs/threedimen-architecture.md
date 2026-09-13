# 3di-men Core Architecture: Surface Representation, Teacher Pipeline, Dataset

**Status:** Foundation implemented (`ml/threedimen/`), no model trained yet. This
is Phase 1 infrastructure — see the roadmap at the end.

This is a different, broader system than `ml/cad/` (a narrow CAD-parametric
prototype built earlier — see `docs/cad-finetuning-technical-report.md`).
`ml/cad/` stays exactly as it was; nothing here depends on it or replaces it.

## 1. What the 3di-men surface representation is

`threedimen.geometry.representation.SurfaceRepresentation` is the internal
geometry object every mesh in this system gets converted into. It's not "an
OBJ file with extra steps" — it's a composition of one or more
`SurfacePatch` instances plus shared context (a `Transform`, a list of
`MaterialInfo`, a list of `SemanticPart` labels, free-form metadata).

`MeshPatch` (a plain triangle mesh) is the only `SurfacePatch` implementation
today. Every patch kind, present or future, must implement the same five
methods (`to_mesh_data`, `bounding_box`, plus three serialization hooks), and
patches self-register by `kind` string. That's the entire contract a future
`NurbsPatch` or `BRepFace` would need to satisfy — nothing in ingestion,
dataset storage, quality filtering, or the ML interfaces would need to
change to support one.

Serialization is split: small/structural data goes to a `.json` file
(human-diffable), bulk numeric arrays go to a companion `.npz` (avoids
re-embedding large arrays in a text format). `SurfaceRepresentation.save()`/
`.load()` round-trip both together.

## 2. Why a custom representation

A representation built directly around "it's just a mesh" can't grow into
parametric surfaces, NURBS, or B-rep later without a breaking rewrite of
every downstream consumer (dataset format, evaluation, ML encoders). The
layered design (patches + composition) means adding a new geometry kind is a
new `SurfacePatch` subclass, not a redesign of the system around it. Full
NURBS/B-rep implementation is explicitly *not* in this milestone — the
abstraction just avoids painting the project into a corner before that work
starts.

## 3. Hunyuan3D's role as a teacher/data generator

`threedimen.teacher.TeacherModel` is the interface (`generate(prompt) ->
MeshArtifact`); `Hunyuan3DTeacher` is one implementation, not a hardcoded
assumption. It speaks the exact HTTP contract already implemented and
debugged in `ml/cad/kaggle/hunyuan3d_api_server.ipynb` (`POST /generate` ->
job id -> poll `/jobs/{id}` -> download from `mesh_url`) — that notebook is
the reference server implementation; `Hunyuan3DTeacher` is only the client.
Configuration (the tunnel endpoint, generation mode) is never hardcoded —
it's read from `THREEDIMEN_HUNYUAN3D_ENDPOINT`/`THREEDIMEN_HUNYUAN3D_MODE` or
constructor args, and a clear error is raised if it isn't set.

Swapping to another text-to-3D model, an image-to-3D model, a local model,
or a future proprietary teacher means implementing `TeacherModel` — nothing
in `geometry/`, `dataset/`, or `quality/` imports a specific teacher by
name. `MockTeacher` is a second, real implementation (deterministic
procedural primitives, no network/GPU) that exists specifically so the rest
of the pipeline is fully testable without Hunyuan3D.

## 4. Dataset architecture

`threedimen.dataset.DatasetStore` manages an on-disk dataset:

```
<root>/manifest.json                        DatasetManifest (paths only)
<root>/samples/<id>/raw<ext>                 exactly what the teacher produced
<root>/samples/<id>/processed.glb            exported SurfaceRepresentation, for quick viewing
<root>/samples/<id>/surface.json / .npz      the SurfaceRepresentation itself (authoritative)
```

No mesh data is duplicated beyond what's needed: the manifest only ever
stores relative paths. Every `DatasetSample` carries its
`representation_version`, `pipeline_version`, and the dataset itself carries
a `dataset_version` — `DatasetManifest.from_dict` refuses to load a manifest
whose major version doesn't match the current code, rather than silently
misinterpreting an incompatible schema.

## 5. How to ingest a mesh

```bash
3dimen mesh inspect path/to/model.glb      # validate + summarize, non-destructive
3dimen mesh validate path/to/model.glb     # validation only, exits non-zero on error
3dimen mesh convert in.obj out.glb         # ingest (validate+normalize) + re-export
```

Or from Python:

```python
from threedimen.geometry.ingest import ingest_mesh

result = ingest_mesh("model.glb", strict=True)  # raises on invalid geometry
result.surface        # the SurfaceRepresentation
result.repairs_applied  # e.g. ["removed 3 degenerate face(s)", "rescaled by 0.02..."]
```

The pipeline: load -> validate (reject empty/NaN/out-of-range-index meshes)
-> normalize coordinate system (glTF/GLB is Y-up per spec, rotated to
3di-men's internal Z-up convention; other formats carry no mandated
convention and pass through unrotated) -> optional repair (degenerate faces,
inconsistent winding, each one logged) -> recenter + optionally rescale
(fully invertible via `surface.metadata["normalization"]`) -> convert to
`SurfaceRepresentation`. Nothing is silently modified.

## 6. How to generate a dataset sample

```bash
3dimen dataset add --prompt "compact centrifugal water pump with four mounting holes" \
    --model hunyuan3d --category pump --root data/threedimen_v1
```

(`--model mock` works identically without any GPU/network, for testing the
pipeline itself.) This calls the teacher, ingests the result (strict mode —
a genuinely broken mesh stops here rather than entering the dataset),
scores it with `threedimen.quality.assess_quality` (watertightness,
manifoldness, degenerate-face fraction, disconnected components,
non-manifold edges, bounding-box sanity, approximate near-duplicate
detection), and rejects it unless `--force` is passed.

For generating many samples in one run, `3dimen dataset build --n 200
--model hunyuan3d --root data/threedimen_v1` combines prompt generation
(Part 5) with the same per-sample teacher-call/ingest/quality-filter
sequence, but treats each sample independently: a single failed generation
(a dropped Kaggle tunnel, a timeout, a genuinely broken mesh) is recorded
and skipped rather than aborting the rest of the batch — a real
consideration once a single image-based Hunyuan3D generation takes minutes
and a batch covers dozens of them. It also carries duplicate-detection state
across the whole batch (seeded from the existing dataset's own quality
records, not just samples added in the current run). Prints per-sample
progress plus a final single-line JSON summary
(`{requested, accepted, rejected, failed, elapsed_seconds}`).

`3dimen dataset stats`/`validate` report on and sanity-check an existing
dataset.

Prompts for bulk generation come from `threedimen.prompts.generate_prompts`
(`3dimen prompt generate --n 100 --category pump,valve`) — eleven categories
today (pump, motor, bracket, pipe, valve, enclosure, ev_charger, tool,
furniture, vehicle, consumer_product), each a small independent template
spec, so adding a category is additive, not a rewrite.

## 7. How this eventually trains our own model

`threedimen.ml.SurfaceEncoder`/`SurfaceDecoder` are the interfaces a future
training loop is built around. `PointSampleEncoder` (surface -> sampled
points+normals) and `ConvexHullDecoder` (points -> a crude reconstructed
mesh) are real, working, *minimal* implementations — not stubs — that exist
to prove the interface plumbing end to end (encode -> decode -> a valid
`SurfaceRepresentation` comes back out) before any neural network exists.
Both are deliberately framework-agnostic (numpy, not `torch.Tensor`) so
point-based, mesh-token, graph, implicit, latent, or hybrid representations
all remain open — swapping in a trained encoder/decoder later doesn't
require touching `dataset/` or `geometry/`.

## 8. Future CAD fine-tuning roadmap

```
PHASE 1  Hunyuan-generated meshes -> 3di-men representation -> synthetic training data
         (this document — DatasetStore + TeacherModel + ingest pipeline, implemented)

PHASE 2  Train an initial 3di-men text-to-surface model
         A first version exists (`ml/threedimen/threedimen/model/`) as a CPU
         proof of concept: a small point-cloud autoencoder + bag-of-
         embeddings text encoder, trained end to end on the actual collected
         dataset. It validates that the full loop works (dataset -> tensors
         -> gradient descent -> a checkpoint that generates a new
         SurfaceRepresentation from an unseen prompt) -- it does NOT claim
         generative quality, since tens-to-hundreds of samples is far below
         what a generative model needs to generalize. A model that actually
         produces recognizable, prompt-faithful geometry still needs a much
         larger dataset (Phase 3) and, at that scale, real GPU compute --
         the "needs a rented GPU" constraint from
         docs/cad-finetuning-technical-report.md Section 3 applies to that
         later, larger-scale training run, not to this CPU-scale plumbing
         validator.

PHASE 3  Improve with real-world 3D assets
         A first version exists: `3dimen dataset import --source cap3d`
         (Cap3D captions + Objaverse meshes) and `--source trellis`
         (TRELLIS-500K's Objaverse-XL-sourced entries) pull a small, curated
         subset of real-world (caption, mesh) pairs through the exact same
         ingest/quality pipeline as generated samples -- deliberately not
         the full corpora (Cap3D ~1.6M objects, TRELLIS-500K ~500K), same
         "don't blindly download huge datasets" principle as the DeepCAD/
         Text2CAD recommendation below. Verified against live data before
         being called done: real objects are frequently decorative
         multi-part models (one test pull had 171 disconnected mesh
         components), which the quality filter's default thresholds --
         tuned for single-body generated meshes -- will reject unless
         adjusted; see ml/threedimen/README.md's "Importing existing
         datasets" section.

PHASE 4  Fine-tune/align using CAD data
         (Text2CAD/DeepCAD, per docs/cad-finetuning-technical-report.md
         Section 5 — that report's dataset findings carry forward directly)

PHASE 5  Generate structured CAD-compatible geometry
         (this is where a NurbsPatch/BRepFace SurfacePatch implementation
         and real B-rep export would get built, using the extension point
         Part 1 deliberately left open rather than building it now)
```

Nothing above Phase 1 has been started. `ml/cad/`'s rule-based prototype
remains the working end-to-end demo in the meantime — it is not superseded
by this document, just architecturally separate from it.
