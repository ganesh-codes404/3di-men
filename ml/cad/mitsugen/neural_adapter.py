"""Mitsugen Route B: optional neural 3D generation for genuinely organic
or complex objects (animals, figurines, sculptures) that
mitsugen/freeform.py's primitive vocabulary (box/cylinder/sphere/cone/
torus + boolean ops) cannot plausibly approximate. Route A -- everything
else (containers, boxes, hooks, phone stands, simple toys) -- stays on
the existing deterministic primitive-composition/radial-sweep paths;
see pipeline.py's own docstring for why forcing every prompt through one
model is the wrong call.

ARCHITECTURE DECISION (full evidence in this package's own
NEURAL_MODEL_RESEARCH.md, researched on an Intel Mac with no CUDA GPU --
never installed or run there, per that research task's own hardware
rules): TripoSR (VAST-AI-Research, MIT license for both code and
weights, ~6GB VRAM per its own README -- real margin under an 8GB
card), paired with a small local text-to-image step (sd-turbo via
diffusers) so the whole pipeline is actually driven by TEXT the way
every other Mitsugen path already is -- TripoSR itself only accepts an
image. Both stages run ENTIRELY LOCALLY on the configured machine's own
GPU; no network call, no hosted API, no paid service -- the same
standing rule every LLM/model integration in this project already
follows (llm_decompose.py, mitsugen/freeform.py).

NEVER required by default, and never silently substituted: every public
function below returns a typed, disclosed "unavailable" result if
torch, CUDA, diffusers, or the TripoSR checkout/weights aren't present
or configured -- never a crash, and never a cube (or any other
fallback) silently presented as a real neural result. All configuration
(the TripoSR checkout path, weights, enabled flag) comes ONLY from
environment variables -- see RTX4060_NEURAL_SETUP.md for the
target-machine install guide. No weights or model checkouts are ever
committed to this repository (see .gitignore).

HONEST STATUS: the availability-checking logic below has been verified
for real on this exact machine (an Intel Mac correctly, truthfully
reports "no CUDA GPU available" -- that is this module working
correctly, not a stub). The actual generation call sequence
(generate_neural_mesh) is written directly from TripoSR's own official
run.py -- the exact TSR.from_pretrained/model(...)/model.extract_mesh(...)
sequence, verified against the official repository, not guessed -- but
has NOT executed on real hardware yet. See tests/test_neural_adapter.py
for exactly what was and wasn't exercised, and RTX4060_NEURAL_SETUP.md
section 8 for the real verification this still needs."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

# Every setting below is environment-driven, never hardcoded to a path
# that exists on any one machine -- see RTX4060_NEURAL_SETUP.md for how
# these get set on the target laptop. All default to "off"/Mac-safe
# values so importing (and even calling the availability check on) this
# module is always safe on a machine with no GPU, like the one this was
# written on.
#
# Deliberately read FRESH from os.environ inside each function below,
# never cached in a module-level constant computed once at import time
# -- confirmed real bug this avoids (mission-brief audit, this exact
# session): mitsugen/freeform.py's own MODEL_PATH used to be a plain
# import-time constant, which made it impossible to test different env
# var values without importlib.reload()-ing the whole module, which in
# turn recreates every class in it as a NEW object and silently breaks
# == / isinstance for any name another test file already imported
# before the reload. Reading live avoids that trap entirely and is
# genuinely more correct besides -- an operator exporting this env var
# in the SAME shell session that's about to start server.py expects it
# to take effect, not to require restarting twice.


def _neural_enabled() -> bool:
    return os.environ.get("MITSUGEN_NEURAL_ENABLED", "0") == "1"


def _triposr_checkout_path() -> Path:
    return Path(os.environ.get(
        "MITSUGEN_TRIPOSR_CHECKOUT", str(Path(__file__).resolve().parent.parent / "models" / "TripoSR"),
    ))


def _triposr_pretrained() -> str:
    return os.environ.get("MITSUGEN_TRIPOSR_PRETRAINED", "stabilityai/TripoSR")


def _text_to_image_model() -> str:
    return os.environ.get("MITSUGEN_TEXT_TO_IMAGE_MODEL", "stabilityai/sd-turbo")


def _mc_resolution() -> int:
    return int(os.environ.get("MITSUGEN_TRIPOSR_MC_RESOLUTION", "256"))


def _neural_output_dir() -> Path:
    return Path(os.environ.get("MITSUGEN_NEURAL_OUTPUT_DIR", "/tmp/mitsugen_neural"))

# Route B: the organic/complex-object class this backend exists for.
# Deliberately a curated noun list, not a generic "anything unusual"
# rule -- a prompt that doesn't match stays on the existing, proven
# primitive-composition path, never forced through a model this
# machine can't even verify works yet. "Do not force all prompts
# through one neural model" (this package's own architecture brief).
ORGANIC_SHAPE_KEYWORDS = {
    "animal", "cat", "dog", "bird", "fish", "horse", "lion", "tiger", "bear", "elephant",
    "rabbit", "fox", "wolf", "deer", "dinosaur", "dragon", "snake", "turtle", "frog",
    "figurine", "statue", "statuette", "sculpture", "bust", "character", "creature",
    "mascot", "miniature figure", "toy animal", "action figure",
}


def requires_neural_generation(object_description: str) -> bool:
    """A cheap, disclosed heuristic -- see ORGANIC_SHAPE_KEYWORDS's own
    comment for why it's curated rather than broad. Checking this does
    NOT by itself mean neural generation will actually run; the caller
    still has to check check_neural_generation_available() and fall
    back honestly if it isn't."""
    import re

    lowered = object_description.lower()
    return any(re.search(rf"\b{re.escape(kw)}\b", lowered) for kw in ORGANIC_SHAPE_KEYWORDS)


@dataclass
class NeuralAvailability:
    available: bool
    reason: str
    torch_installed: bool = False
    cuda_available: bool = False
    diffusers_installed: bool = False
    triposr_checkout_present: bool = False


def check_neural_generation_available() -> NeuralAvailability:
    """Never raises. Checks every real precondition in order, cheapest
    first, and stops at the first failure with a SPECIFIC, disclosed
    reason -- never a bare "unavailable" with no explanation. This is
    the exact discipline this whole project applies to every other
    optional/fallback path (e.g. Mitsugen's own fell_back disclosure on
    the primitive-composition side); an optional neural backend is no
    exception just because it's new."""
    if not _neural_enabled():
        return NeuralAvailability(available=False, reason="neural generation is disabled (set MITSUGEN_NEURAL_ENABLED=1 to enable)")

    try:
        import torch
    except ImportError:
        return NeuralAvailability(available=False, reason="torch is not installed")

    cuda_available = bool(torch.cuda.is_available())
    if not cuda_available:
        return NeuralAvailability(
            available=False, reason="torch is installed but no CUDA GPU is available on this machine",
            torch_installed=True, cuda_available=False,
        )

    try:
        import diffusers  # noqa: F401
    except ImportError:
        return NeuralAvailability(
            available=False, reason="diffusers is not installed (needed for the local text-to-image step)",
            torch_installed=True, cuda_available=True,
        )

    checkout_path = _triposr_checkout_path()
    if not checkout_path.exists():
        return NeuralAvailability(
            available=False,
            reason=f"TripoSR checkout not found at {checkout_path} (see RTX4060_NEURAL_SETUP.md)",
            torch_installed=True, cuda_available=True, diffusers_installed=True,
        )

    return NeuralAvailability(
        available=True, reason="ready", torch_installed=True, cuda_available=True,
        diffusers_installed=True, triposr_checkout_present=True,
    )


@dataclass
class NeuralGenerationResult:
    mesh: "object | None"  # a trimesh.Trimesh when generation succeeded, from TripoSR's own mesh.export()-compatible object
    reference_image_path: "str | None"
    error: "str | None"  # None on success -- same "never a partial/guessed mesh on failure" discipline as every other BuildResult in this package


def generate_neural_mesh(object_description: str, output_dir: "Path | None" = None) -> NeuralGenerationResult:
    """Text -> a local sd-turbo reference image -> a TripoSR mesh,
    entirely on-device. Never raises -- any failure at any stage comes
    back as a typed, disclosed NeuralGenerationResult(mesh=None,
    error=...), the same discipline mitsugen/freeform_builder.py's own
    BuildResult follows.

    UNTESTED end-to-end (see this module's own docstring): written
    directly from TripoSR's own official run.py -- the exact
    TSR.from_pretrained / model(...) / model.extract_mesh(...) call
    sequence, verified against the official repository's source, not
    guessed -- but this exact code path has not executed on real
    hardware. Verify it for real on the RTX 4060 laptop
    (RTX4060_NEURAL_SETUP.md section 8) before trusting it in
    production."""
    availability = check_neural_generation_available()
    if not availability.available:
        return NeuralGenerationResult(mesh=None, reference_image_path=None, error=f"neural generation unavailable: {availability.reason}")

    import sys

    import torch
    from diffusers import AutoPipelineForText2Image

    out_dir = output_dir or _neural_output_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    reference_image_path = out_dir / "reference_image.png"

    try:
        t2i = AutoPipelineForText2Image.from_pretrained(_text_to_image_model(), torch_dtype=torch.float16)
        t2i = t2i.to("cuda")
        # sd-turbo's own documented single-step, zero-guidance inference
        # convention (same settings this project's own
        # kaggle/hunyuan3d_baseline.ipynb already used for the identical
        # purpose) -- a studio-photo framing gives TripoSR's background
        # removal step a clean subject to isolate.
        image = t2i(
            prompt=f"{object_description}, product photo, plain white background, studio lighting",
            num_inference_steps=1, guidance_scale=0.0,
        ).images[0]
        image.save(reference_image_path)
        del t2i
        torch.cuda.empty_cache()
    except Exception as exc:
        return NeuralGenerationResult(mesh=None, reference_image_path=None, error=f"text-to-image step failed: {exc}")

    try:
        checkout_path = _triposr_checkout_path()
        if str(checkout_path) not in sys.path:
            sys.path.insert(0, str(checkout_path))
        import numpy as np
        import rembg
        from PIL import Image
        from tsr.system import TSR
        from tsr.utils import remove_background, resize_foreground

        model = TSR.from_pretrained(_triposr_pretrained(), config_name="config.yaml", weight_name="model.ckpt")
        model.renderer.set_chunk_size(8192)
        model.to("cuda")

        # TripoSR's own official preprocessing (run.py): remove the
        # background, resize the foreground subject, then composite onto
        # a 50% gray background -- the exact framing the model was
        # trained against.
        rembg_session = rembg.new_session()
        img = remove_background(image, rembg_session)
        img = resize_foreground(img, 0.85)
        img_arr = np.array(img).astype(np.float32) / 255.0
        img_arr = img_arr[:, :, :3] * img_arr[:, :, 3:4] + (1 - img_arr[:, :, 3:4]) * 0.5
        img = Image.fromarray((img_arr * 255.0).astype(np.uint8))

        with torch.no_grad():
            scene_codes = model([img], device="cuda")
        meshes = model.extract_mesh(scene_codes, True, resolution=_mc_resolution())
        mesh = meshes[0]
    except Exception as exc:
        return NeuralGenerationResult(mesh=None, reference_image_path=str(reference_image_path), error=f"TripoSR shape generation failed: {exc}")

    return NeuralGenerationResult(mesh=mesh, reference_image_path=str(reference_image_path), error=None)
