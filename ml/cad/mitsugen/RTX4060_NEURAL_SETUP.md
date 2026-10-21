# Mitsugen Route B (neural generation) on the RTX 4060 laptop

This sets up TripoSR (image-to-3D, MIT licensed) + a local text-to-image
step (sd-turbo), so Mitsugen's `mitsugen/neural_adapter.py` can actually
generate genuinely organic/complex objects (animals, figurines,
sculptures) instead of approximating them with primitives. See
`mitsugen/NEURAL_MODEL_RESEARCH.md` for the full evidence behind this
choice, researched on an Intel Mac that could not run any of this itself.

This is a SEPARATE, opt-in profile, same convention as
`QUICKSTART_RTX4060.md`/`.env.rtx4060.example` for the CAD-side LLM.
Nothing here changes any other machine's setup, and Mitsugen works
exactly as it already does (primitive composition + radial sweep) if
none of this is ever installed.

## 0. Detection commands -- run these first, don't assume

```
nvidia-smi                                    # confirms the GPU and driver are visible at all
python3 -c "import torch; print(torch.__version__, torch.cuda.is_available())"
python3 -c "import torch; print(torch.cuda.get_device_properties(0).total_memory / 1e9, 'GB VRAM')"
nvcc --version                                # CUDA toolkit version, if installed
```

If `torch.cuda.is_available()` is `False`, stop here and fix the driver/CUDA
install before doing anything below -- every later step assumes this already
works, same as this project's own `QUICKSTART_RTX4060.md`.

## 1. Supported OS / driver / CUDA assumptions

- Windows 11 + WSL2 Ubuntu (the same setup `QUICKSTART_RTX4060.md` already
  walks through for the CAD side), or native Linux with the NVIDIA driver
  installed. Native Windows + CUDA works too, but WSL2 is what this
  project's own existing setup already assumes.
- A CUDA-enabled PyTorch build matching whatever CUDA version step 0's
  `nvcc --version` reports. **Do not guess this** -- installing a PyTorch
  build for the wrong CUDA major version is the single most common real
  failure point here.

## 2. Python environment

Reuse the SAME venv `QUICKSTART_RTX4060.md` step 5 already set up for
`ml/cad` -- don't create a second one.

```
cd ml/cad
source venv/bin/activate   # or venv/Scripts/activate on native Windows
```

## 3. Install packages

```
# PyTorch -- pick the command for YOUR detected CUDA version from
# https://pytorch.org/get-started/locally/ -- do not copy this blindly:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121   # example only, cu121 shown -- match step 0's nvcc output

pip install diffusers transformers accelerate rembg xatlas
```

## 4. Clone TripoSR (not a pip package -- a research-repo checkout)

```
mkdir -p models
git clone https://github.com/VAST-AI-Research/TripoSR.git models/TripoSR
cd models/TripoSR
git log -1 --format=%H   # pin and record this commit -- the repo has no tagged releases
pip install -r requirements.txt
cd ../..
```

`torchmcubes` (one of TripoSR's own requirements) needs to compile against
CUDA. If this step fails, it's almost always the CUDA-toolkit-vs-PyTorch
version mismatch step 0 exists to catch -- re-check `nvcc --version` against
whichever CUDA build of PyTorch you installed in step 3.

## 5. Download weights -- directly onto THIS machine, never via the Mac

Do not download these anywhere else and copy them over -- `huggingface_hub`
caches them automatically on first use, so the first real generation call
IS the download step. If you'd rather pre-fetch:

```
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('stabilityai/TripoSR')
snapshot_download('stabilityai/sd-turbo')
"
```

Expected disk usage: a few GB total for both checkpoints (TripoSR's own
reconstruction model plus sd-turbo's smaller diffusion model) on top of
the TripoSR checkout itself (small, it's just code). Check free disk
space first (`df -h .` on Linux/WSL2) -- this project's own standing rule
for every machine, not just the constrained Mac.

## 6. Configure Mitsugen's own environment variables

None of these are required to just run Mitsugen as it already works --
they ONLY matter if you want Route B enabled.

```
export MITSUGEN_NEURAL_ENABLED=1
export MITSUGEN_TRIPOSR_CHECKOUT=$(pwd)/models/TripoSR     # default already matches this if you followed step 4 exactly
export MITSUGEN_TRIPOSR_PRETRAINED=stabilityai/TripoSR     # default
export MITSUGEN_TEXT_TO_IMAGE_MODEL=stabilityai/sd-turbo   # default
export MITSUGEN_TRIPOSR_MC_RESOLUTION=256                  # marching-cubes resolution; lower if you hit VRAM limits, see troubleshooting
```

## 7. Minimal inference command -- verify BEFORE starting the server

```
python3 -c "
from mitsugen.neural_adapter import check_neural_generation_available, generate_neural_mesh

avail = check_neural_generation_available()
print('available:', avail.available, '|', avail.reason)
assert avail.available, 'fix the reason above before continuing'

result = generate_neural_mesh('a small cat figurine')
print('error:', result.error)
if result.mesh is not None:
    print('watertight:', result.mesh.is_watertight)
    print('extents:', result.mesh.extents)
"
```

This is the FIRST time this exact code path will have ever executed --
it was written directly from TripoSR's own official `run.py` (verified
against the real repository source on the research machine) but could
not be run there (no CUDA GPU). If it fails, the traceback is the actual
diagnostic -- this is intentionally not wrapped in extra error handling
beyond what `generate_neural_mesh` already does, so you see the real error.

## 8. Start Mitsugen with the model enabled

```
# with the env vars from step 6 still exported in this shell:
python3 server.py --port 5050 --host 127.0.0.1
```

Then open `/mitsugen` and try the test prompts from section 9 below.
A prompt matching `mitsugen/neural_adapter.py`'s own `ORGANIC_SHAPE_KEYWORDS`
(animal names, "figurine", "statue", "sculpture", "creature", ...) will
route to Route B automatically; everything else keeps using the existing,
proven primitive-composition/radial-sweep paths exactly as before.

## 9. Required test prompts (per the mission brief's own list)

Run each of these through `/mitsugen` and record: generation time, peak
VRAM (`nvidia-smi` in a second terminal during generation), whether the
mesh is watertight, its extents, and whether it exported correctly:

- "a cup" (Route A -- confirms nothing regressed)
- "a phone stand" (Route A)
- "a low-poly cat" (Route B)
- "a house" (Route A -- multi-feature primitive composition)
- "a hook" (Route A)
- "a lidded box" (Route A)

For each Route B result, confirm `construction_mode` in the response is
`"neural_generation"` and that `fallback_reason`-style disclosure is
ABSENT (if present, Route B failed and silently fell back -- check the
finding text for why, never treat that as a successful neural result).

## 10. Troubleshooting

- **CUDA out-of-memory**: lower `MITSUGEN_TRIPOSR_MC_RESOLUTION` (256 ->
  128 first; TripoSR's own troubleshooting notes 128 can produce a
  coarser but still real mesh). If it still OOMs, check `nvidia-smi` for
  other processes holding VRAM (a stale Python process from a previous
  crashed run is the usual culprit) and kill them first.
- **`torchmcubes` build failure**: almost always the CUDA-toolkit-vs-
  PyTorch version mismatch step 0/3 exist to catch. Re-verify both
  versions match before trying anything else.
- **Missing weights / 401 from Hugging Face**: `stabilityai/TripoSR` and
  `stabilityai/sd-turbo` were both openly accessible (no gated-repo
  login) as of this research; if that's changed, create a free Hugging
  Face token and `huggingface-cli login` first, same pattern
  `kaggle/hunyuan3d_baseline.ipynb` already documents for a gated model.
- **`check_neural_generation_available()` reports "diffusers is not
  installed" or "torch is not installed" despite step 3**: confirm the
  venv from step 2 is actually activated in the SAME shell running
  `server.py` -- a second terminal without the venv activated is the
  most common cause.
- **Everything reports available, but every Route B prompt still falls
  back to primitive composition with an error**: the disclosed
  `fallback_reason`/finding text IS the real error from
  `generate_neural_mesh` -- it is deliberately not swallowed or
  generalized, see this module's own "never raises, but never hides the
  real reason either" discipline.
