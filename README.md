# 3di-men

> **Generate production-ready 3D assets from natural language.**

## Overview

3di-men is an AI-powered platform for creating and editing high-quality 3D assets.

Generate production-ready models from natural language or images, then seamlessly switch to an interactive editing playground when you need finer control. Instead of repeatedly tweaking prompts, creators can directly modify geometry, materials, textures, lighting, and other properties in a visual workspace.

Our goal is to make 3D creation as intuitive as writing a sentence—and as flexible as using professional 3D software.

---

## Vision

Traditional 3D creation is time-consuming and requires specialized expertise.

3di-men combines generative AI with a powerful editing environment, allowing creators to move effortlessly between AI-assisted generation and manual refinement. Whether you're building game assets, product visualizations, animations, or AR/VR experiences, the platform adapts to your workflow instead of forcing you into one.

---

## Features (Planned)

* Text-to-3D generation
* Image-to-3D generation
* Automatic mesh optimization
* PBR material generation
* Texture generation
* Multiple export formats (GLB, FBX, OBJ, USD)
* Game-engine-ready assets
* 3D playground
* API for developers
* Asset versioning
* AI-powered editing and refinement

## Current Status

This project is currently in active development.

The initial focus is on building a robust text-to-3D generation pipeline before expanding into editing, animation, and enterprise workflows.

**What's implemented so far:**

- [`ml/threedimen/`](ml/threedimen/) — the core foundation: a versioned surface
  representation, a swappable teacher-model interface (Hunyuan3D-2.1 today),
  mesh ingestion/validation, a versioned dataset format, synthetic prompt
  generation, quality filtering, and the ML-facing encoder/decoder
  interfaces a future 3di-men model will train against. See
  [`docs/threedimen-architecture.md`](docs/threedimen-architecture.md) for
  the design rationale and the Phase 1–5 roadmap.
- [`ml/cad/`](ml/cad/) — a separate, narrower prototype: rule-based
  prompt-to-CAD-part generation (mounting plates, flanges), entirely
  CPU-only, plus a working Hunyuan3D-2.1 integration (Kaggle-hosted, dual
  text-direct/image-based modes) and a local web viewer. See
  [`ml/cad/README.md`](ml/cad/README.md) and
  [`docs/cad-finetuning-technical-report.md`](docs/cad-finetuning-technical-report.md).

No model has been trained yet — both are data/inference infrastructure, not
a trained 3di-men model.

---

## Long-Term Goals

* Build the easiest way to create 3D assets with AI.
* Enable creators without traditional 3D modeling experience.
* Support real-time generation.
* Deliver production-ready assets instead of prototypes.
* Power games, films, AR/VR, robotics, e-commerce, and digital twins.

---

## Contributing

Contributions are welcome. As the project matures, contribution guidelines, issue templates, and documentation will be added.

---

## Contact

For questions, suggestions, or collaborations, please open an issue or start a discussion in this repository.
