# Tech Stack Proposal — Text-to-3D Generation Startup

> **Goal:** Ship a working prototype fast by leveraging proven open-source components instead of building everything from scratch. Responsibilities are split according to our strengths:
>
> - **Python:** Machine Learning, training, preprocessing, data pipelines
> - **Rust:** Inference server, APIs, plugins, deployment

---

# Roadmap

| Month | Focus | Milestone |
|--------|-------|-----------|
| **Month 1** | Phase 1 — Data & Base Model | Clean niche dataset + base model running locally |
| **Month 2** | Phase 2 — Fine-Tuning Loop | Fine-tuned model measurably outperforms base model |
| **Month 3** | Phase 3 — Inference & Plugin | Blender/Unity plugin generating assets from our model |
| **Month 4** | Phase 4 — Feedback Loop & Early Users | Friends actively using it, first retraining cycle from user feedback |
| **Month 5+** | Phase 5 — Environment & Simulation Exploration | Prototype static scene generation (not customer-facing) |
| **Month 5+ (Parallel)** | Phase 6 — Space Tech Track | Feasibility study on precision/tolerance-aware generation |

> **Important:**  
> Phases **3** and **4** intentionally overlap. The moment the plugin works, start collecting prompts, outputs, and ratings.
>
> Phases **5** and **6** only begin after the gaming asset workflow has proven itself. Otherwise we'll end up with two unfinished products.

---

# Phase 1 — Data & Base Model (Month 1)

## Objective

Build a clean dataset and get an open-source model generating assets locally.

| Need | Tool | Why |
|------|------|-----|
| Base Model | **Hunyuan3D 2.1** | Best open-source quality today, actively maintained |
| Model Hosting | Hugging Face Hub | Standard for downloading and versioning weights |
| Dataset | Objaverse / Objaverse-XL | Largest open collection of labeled 3D assets |
| Mesh Processing | trimesh + Blender (`bpy`) | Cleaning, normalization, batch rendering, re-export |
| Dataset Versioning | DVC or Hugging Face Datasets | Track dataset evolution and future retraining |

### Deliverable

- Local inference working
- Clean niche dataset
- Thousands of high-quality `(text, mesh, texture)` triples

---

# Phase 2 — Fine-Tuning Loop (Month 2)

## Objective

Adapt the base model to our niche while minimizing compute costs.

| Need | Tool | Why |
|------|------|-----|
| Training | PyTorch | Native framework used by Hunyuan3D |
| Efficient Fine-Tuning | PEFT (LoRA / QLoRA) | Makes training practical for a two-person team |
| Multi-GPU | Hugging Face Accelerate | Easily scale from one GPU to rented hardware |
| Experiment Tracking | Weights & Biases *(or local CSV/JSON logging)* | Compare runs objectively |
| Compute | Google Colab / Kaggle GPU | Free GPU access for LoRA-scale experiments |
| Evaluation | Custom Python metrics | Chamfer Distance + CLIP similarity |

### Deliverable

A fine-tuned model that consistently outperforms the base model on evaluation metrics.

---

# Phase 3 — Inference Server & Plugins (Month 3)

## Objective

Serve the model through a fast Rust backend and integrate directly into creator tools.

| Need | Tool | Why |
|------|------|-----|
| Model Export | ONNX | Removes Python dependency during inference |
| Rust Runtime | ONNX Runtime (`ort`) | Fast, lightweight Rust inference |
| API Server | axum | Modern asynchronous Rust web framework |
| Deployment | Docker | Portable deployment anywhere |
| Blender Plugin | Python `bpy` addon | Native Blender extension |
| Unity Plugin | C# + `HttpClient` | Same backend, different frontend |

### Architecture

```text
User Prompt
      │
      ▼
Blender / Unity Plugin
      │
HTTP
      ▼
Rust (axum)
      │
ONNX Runtime
      │
Fine-tuned Hunyuan3D
      │
Generated Mesh
      │
Returned to Plugin
```

### Deliverable

A user can generate assets directly inside Blender or Unity without needing a browser.

This becomes our key differentiator compared to browser-first tools.

---

# Phase 4 — Feedback Loop (Month 4)

## Objective

Turn user interactions into continuous training data.

| Need | Tool | Why |
|------|------|-----|
| Logging | PostgreSQL *(SQLite initially)* | Store prompts, meshes, ratings |
| Dashboard | Streamlit | Rapid internal monitoring |
| Retraining | Manual Python scripts | Curate useful failures before automating |

### Stored Data

- Prompt
- Generated mesh
- Generation time
- User rating
- Failure reason
- Edited mesh (optional)

### Deliverable

Every testing session produces new training examples that improve future versions.

---

# Phase 5 — Environment & Simulation Exploration (Month 5+)

> Prototype only. Not part of the first customer release.

## Objective

Move beyond individual assets into complete environments.

| Need | Tool | Why |
|------|------|-----|
| Scene Layout | Blender Python API (`bpy`) | Assemble generated assets into scenes |
| Physics | Blender Physics / PyBullet | Evaluate physical plausibility |

### Deliverable

Generate rough environments using assets from previous phases.

---

# Phase 6 — Space Tech Research Track (Month 5+)

> Parallel exploratory research.

## Objective

Investigate whether the same pipeline can generate precision engineering components.

| Need | Tool | Why |
|------|------|-----|
| Precision Dataset | GrabCAD, NASA Open Models | Dimensionally accurate CAD data |
| Fine-Tuning | Same LoRA pipeline | Optimize for dimensional accuracy rather than visual realism |
| Export | STL initially, STEP later | STEP is the long-term CAD standard |

### Key Difference

Game assets prioritize visual quality.

Space and engineering assets prioritize manufacturing tolerances.

Those are fundamentally different optimization targets.

---

# High-Level Architecture

```text
                ┌─────────────────────┐
                │     User Prompt     │
                └──────────┬──────────┘
                           │
                           ▼
                 Blender / Unity Plugin
                           │
                           ▼
                  Rust API (axum)
                           │
                           ▼
                  ONNX Runtime (ort)
                           │
                           ▼
             Fine-Tuned Hunyuan3D Model
                           │
                           ▼
                  Generated 3D Mesh
                           │
                           ▼
                  User Feedback Logged
                           │
                           ▼
                    PostgreSQL / SQLite
                           │
                           ▼
               Retraining Dataset Builder
                           │
                           ▼
                    LoRA Fine-Tuning
```

---

# Repository Structure

```text
text-to-3d/
│
├── ml/
│   ├── training/
│   ├── finetuning/
│   ├── evaluation/
│   ├── preprocessing/
│   └── inference/
│
├── rust-server/
│   ├── api/
│   ├── inference/
│   └── plugins/
│
├── blender-addon/
│
├── unity-plugin/
│
├── datasets/
│
├── docker/
│
└── docs/
```

---

# Technology Summary

## Python

- PyTorch
- PEFT (LoRA / QLoRA)
- Hugging Face
- Accelerate
- trimesh
- Blender (`bpy`)
- Streamlit

---

## Rust

- axum
- ONNX Runtime (`ort`)
- tokio
- serde
- Docker

---

## Infrastructure

- Hugging Face Hub
- PostgreSQL / SQLite
- Docker
- DVC (optional)

---

## Datasets

- Objaverse
- Objaverse-XL
- User-generated feedback dataset
- GrabCAD *(future)*
- NASA Open 3D Models *(future)*

---

# Long-Term Vision

## Phase 1

Generate high-quality individual 3D assets.

↓

## Phase 2

Specialize for our niche through fine-tuning.

↓

## Phase 3

Generate directly inside Blender and Unity.

↓

## Phase 4

Continuously improve from real user feedback.

↓

## Phase 5

Generate complete environments.

↓

## Phase 6

Expand into precision engineering and space technology.

---

# Philosophy

- **Ship quickly using proven open-source models.**
- **Collect proprietary data early.**
- **Fine-tune instead of rebuilding from scratch.**
- **Integrate directly into creators' workflows.**
- **Use feedback as the primary competitive advantage.**
- **Expand into larger markets only after the core product succeeds.**
