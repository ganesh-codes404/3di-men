# threedimen

3di-men's core geometry representation, teacher/data pipeline, and dataset
tooling. See `../../docs/threedimen-architecture.md` for the full design
rationale, why-not-just-a-mesh reasoning, and the Phase 1-5 roadmap this
feeds into. This file is the practical "how to run it" reference.

Separate from `../cad/` (a narrower CAD-parametric prototype, untouched by
this package) — this is the general foundation described in the project's
roadmap: text prompt -> teacher model (Hunyuan3D, swappable) -> raw mesh ->
validate/normalize -> versioned `SurfaceRepresentation` -> dataset sample.

## Install

```bash
cd ml/threedimen
pip3 install -e ".[dev]"
```

## CLI

```bash
3dimen mesh inspect model.glb                    # validate + summarize
3dimen mesh validate model.glb                    # validate only, exit non-zero on error
3dimen mesh convert in.obj out.glb                # ingest + re-export

3dimen teacher generate "a compact bracket" --model mock       # --model hunyuan3d needs
                                                                  # THREEDIMEN_HUNYUAN3D_ENDPOINT set

3dimen dataset add --prompt "a centrifugal pump" --model mock --category pump --root data/v1
3dimen dataset build --n 50 --model mock --root data/v1     # generate N synthetic prompts + add everything that passes quality filtering
3dimen dataset stats --root data/v1
3dimen dataset validate --root data/v1

3dimen prompt generate --n 20 --category pump,valve --seed 0

# needs: pip install -e ".[train]"  (adds CPU PyTorch)
3dimen model train --root data/v1 --out model.pt --epochs 300 --device auto  # "auto" = GPU if available, else CPU
3dimen model generate "a compact aluminum bracket" --checkpoint model.pt --out generated.stl

# needs: pip install -e ".[external]"  (adds objaverse, huggingface_hub, pandas)
3dimen dataset import --source cap3d --n 20 --root data/v1     # real-world (caption, mesh) pairs from Cap3D/Objaverse
3dimen dataset import --source trellis --n 20 --root data/v1   # TRELLIS-500K's Objaverse-XL-sourced entries only
```

`--model hunyuan3d` requires a running `ml/cad/kaggle/hunyuan3d_api_server.ipynb`
tunnel (see `ml/cad/kaggle/README.md` for how to start one) and:

```bash
export THREEDIMEN_HUNYUAN3D_ENDPOINT=https://xxxx.trycloudflare.com
```

Without it, `--model mock` exercises the exact same pipeline with a
deterministic, network-free stand-in teacher — useful for developing/testing
everything downstream of "a mesh exists" without needing a GPU.

## Importing existing datasets (Cap3D, TRELLIS-500K)

`3dimen dataset import` pulls a small, curated subset of an existing public
text-3D corpus instead of generating new meshes — real-world (caption, mesh)
pairs rather than synthetic ones, for Phase 3 ("improve with real-world 3D
assets") in the architecture doc. Deliberately small-subset by design (`--n`
controls how many, not "import everything") — Cap3D covers ~1.6M objects and
TRELLIS-500K ~500K; downloading either in full is out of scope here, same
principle as this project's approach to DeepCAD/Text2CAD.

- **`--source cap3d`**: captions from [Cap3D](https://huggingface.co/datasets/tiange/Cap3D)
  (automated captions for Objaverse objects), meshes downloaded via the
  `objaverse` package given the caption's object UID.
- **`--source trellis`**: captions from [TRELLIS-500K](https://huggingface.co/datasets/JeffreyXiang/TRELLIS-500K)
  (Microsoft's curated training set for their TRELLIS model) — **only the
  Objaverse-XL-sourced entries** (~480K of ~500K total); the ABO/3D-FUTURE/
  HSSD/Toys4k-sourced entries use different hosting not implemented here.

Both make a best-effort attempt to skip non-permissively-licensed objects
via `check_license=True` (default) — Objaverse assets carry **mixed,
per-object licenses**, some not commercial-use-approved. This filter is a
convenience, not a legal guarantee; verify license suitability yourself
before using imported data beyond local experimentation. Pass
`--no-license-check` to skip it (faster, but you're on your own for license
review).

**Known real-world quirk, found by actually testing this against live
Cap3D/Objaverse data before writing docs about it**: unlike single-body
generated meshes, real Objaverse assets are frequently decorative multi-part
models (e.g. one test pull had **171 disconnected mesh components** — a
character model built from many separate small pieces). The quality
filter's default `max_disconnected_components: 8` (tuned for generated
single-body parts) will reject most of these. Either pass `--force` to
bypass the filter for external imports, or build a custom
`QualityThresholds(max_disconnected_components=...)` via the Python API
(`threedimen.quality`) if you're scripting this rather than using the CLI.

## Layout

```
threedimen/
  geometry/       SurfaceRepresentation, mesh I/O, ingestion pipeline
  teacher/        TeacherModel interface, Hunyuan3D + mock implementations
  dataset/        versioned on-disk dataset schema + store
  prompts/        synthetic prompt generator (11 categories)
  quality/        mesh-quality assessment / accept-reject filtering
  external/       importers for existing public datasets (Cap3D, TRELLIS-500K)
  ml/             SurfaceEncoder/SurfaceDecoder interfaces (prototype impls)
  model/          the first 3di-men model: a small text-to-shape proof of
                  concept (needs the optional `train` extra -- CPU or GPU PyTorch)
  cli/            the `3dimen` command
tests/            pytest suite, all fixtures generated procedurally (no
                  checked-in binary files); external-dataset tests never hit
                  the real network (they test the pure logic + the "extra not
                  installed" error path deterministically)
```

## Tests

```bash
pytest tests/ -v
```

97 tests, all offline (mock teacher + procedurally-generated fixture
meshes, no real network/GPU calls, ever). The `model/` tests are skipped
automatically (`pytest.importorskip("torch")`) if the `train` extra isn't
installed. Covers: representation save/load round-trips, ingestion of valid
and malformed meshes (empty, NaN, degenerate faces, out-of-range indices),
glTF axis-convention handling, normalization invertibility, dataset
read/write/versioning, quality filtering (including duplicate detection),
prompt generation determinism, the mock/Hunyuan3D teacher contracts, the ML
interface round-trip, device resolution (CPU/CUDA/auto), the training
bundle export, external-dataset import error handling, and every CLI
command — including, for the model, Chamfer distance correctness, network
output shapes, a full train→checkpoint→generate round trip, and a check
that reconstruction loss actually decreases with more training (not just
that the code runs without erroring).

Note: the Cap3D/TRELLIS download paths themselves were verified by hand
against the live network before being called done (a real 2-object Cap3D
pull, ingested and quality-scored) — that verification isn't automated into
the test suite (no real network calls in `pytest`), so re-verify by hand
again if these modules see significant changes later.

## The model — what it is and isn't

`3dimen model train` trains a small point-cloud autoencoder (a mini-PointNet
encoder/decoder pair) with a bag-of-embeddings text encoder trained to
predict the matching shape's latent vector — real gradient-based training on
your actual collected dataset, not a hardcoded/fake result.

**What it validates:** the full loop works end to end — dataset on disk ->
tensors -> training -> a checkpoint that can generate a new
`SurfaceRepresentation` from a text prompt it never saw during training.

**What it does not claim:** generative quality. Tens-to-hundreds of training
samples is far below what any generative 3D model needs to meaningfully
generalize — expect it to produce something that vaguely reflects the
dataset's overall shape distribution, not a recognizable "bracket" or "pump"
for a specific new prompt. That needs a much larger dataset (Phase 2/3 in
the architecture doc) and is a separate, later milestone from "does the
training loop actually work," which is what this proves today.

## Current limitations

- `MeshPatch` is the only `SurfacePatch` implementation — no NURBS/B-rep yet
  (by design; see the architecture doc's Part 1 rationale).
- The model (above) is a plumbing proof of concept, not a quality claim.
- `PointSampleEncoder`/`ConvexHullDecoder` remain minimal Part 7 interface
  implementations reused by the model, not something purpose-built for it.
- Quality filtering's near-duplicate detection is a simple heuristic
  (rounded bbox-aspect/vertex/face/volume key), not a perceptual hash or
  learned similarity metric.
- `Hunyuan3DTeacher` depends on a live tunnel from
  `ml/cad/kaggle/hunyuan3d_api_server.ipynb` being up — there's no persistent
  hosted endpoint yet.
- `--source trellis` only covers TRELLIS-500K's Objaverse-XL-sourced entries
  (~480K of ~500K) — ABO/3D-FUTURE/HSSD/Toys4k entries use different hosting
  not implemented here, and are silently skipped rather than attempted.
- External-dataset license filtering (`cap3d.py`/`trellis.py`) is
  best-effort (matches a license string against a small permissive-marker
  list); it is not a substitute for actually checking license suitability
  yourself before using imported data beyond local experimentation.
- The quality filter's `max_disconnected_components: 8` default is tuned
  for single-body generated meshes, not decorative multi-part real-world
  assets — expect most Cap3D/TRELLIS imports to need `--force` or a custom
  `QualityThresholds` (confirmed against live data, not assumed — see
  "Importing existing datasets" above).
