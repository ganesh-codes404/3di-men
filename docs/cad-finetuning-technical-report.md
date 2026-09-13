# Hunyuan3D → CAD Fine-Tuning: Technical Report (Phase 1)

**Date:** 2026-08-26
**Status:** Pre-implementation findings. No training has been run. No model weights or large datasets have been downloaded.

## 0. TL;DR

- **This repository currently contains no code.** It is `README.md` + `InitialPlan.md` (a roadmap doc) only. There is no Hunyuan3D checkout, no inference pipeline, no data pipeline, no training code anywhere in this repo to "inspect." Everything below about Hunyuan3D's architecture is from its public repo/paper, not from code present here.
- **This machine cannot run Hunyuan3D at all.** It's a MacBook with 8 GB RAM, integrated Intel HD Graphics 6000 (1.5 GB VRAM, no CUDA), 19 GB free disk. Hunyuan3D-2.1 needs 10–29 GB of **NVIDIA VRAM** depending on which stages run. Inference, LoRA training, and dataset processing at any real scale all require rented cloud GPU — this was already anticipated in `InitialPlan.md` ("Compute: Google Colab / Kaggle GPU").
- **Hunyuan3D-2.1's shape generator is image-conditioned, not text-conditioned.** Text prompts never reach the 3D diffusion transformer directly; a separate text-to-image model produces a reference image first, and *that* image conditions shape generation. This is the single most important architectural fact for this project — it means "fine-tune Hunyuan3D on (text, mesh) pairs" as literally stated is not how the system takes input, and changes the recommended strategy (Section 4).
- **A relevant dataset already exists** — Text2CAD/DeepCAD (170K parametric CAD models, 660K text annotations, NeurIPS 2024) — rather than needing to build a procedural generator from zero for v1 (Section 5).
- **The task brief's "MVP now" framing conflicts with this team's own roadmap.** `InitialPlan.md` explicitly places precision/CAD work in **Phase 6, Month 5+**, run only *after* the gaming-asset pipeline is proven, specifically to avoid "two unfinished products." This is a real scope decision, not something to silently override — flagged at the end of this report.

---

## 1. Architecture Analysis (Hunyuan3D-2.1, from public sources)

Since no Hunyuan3D code exists in-repo, this is based on the [Hunyuan3D-2.1 GitHub repo](https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1), [arXiv:2506.15442](https://arxiv.org/pdf/2506.15442), and [arXiv:2501.12202](https://arxiv.org/pdf/2501.12202) (2.0 paper, shared shape-gen lineage).

The pipeline has two largely independent stages:

```
Text prompt ──▶ [external T2I model] ──▶ reference image
                                              │
                                              ▼
                          ┌───────────────────────────────────┐
                          │  Stage 1: Shape Generation         │
                          │  DINOv2-Giant image encoder (518²) │
                          │  → 21-layer flow-matching DiT       │
                          │    (cross-attn image conditioning, │
                          │     MoE layer)                     │
                          │  → ShapeVAE latent → mesh (marching │
                          │     cubes / FlexiCubes-style)       │
                          └───────────────────────────────────┘
                                              │
                                              ▼  untextured mesh + input image
                          ┌───────────────────────────────────┐
                          │  Stage 2: Texture / PBR Painting   │
                          │  Multi-view diffusion conditioned   │
                          │  on geometry + reference image     │
                          │  → albedo/metallic/roughness maps  │
                          └───────────────────────────────────┘
                                              │
                                              ▼
                                     Textured 3D asset (GLB)
```

Key points:

- **Mesh representation:** explicit triangle mesh via a VAE + flow-matching diffusion transformer over a latent shape code, decoded through an implicit field (occupancy/SDF-like) into a mesh. Not a raw point cloud, not a voxel grid, not a NeRF — it's closer to "diffuse in latent space, decode to explicit surface."
- **Conditioning modality: image only** for the shape DiT. There is no text cross-attention branch in the shape generator itself. Any "text-to-3D" demo is really **text-to-image-to-3D** — the T2I model (typically Hunyuan-DiT or similar, swappable) does all the language understanding; Hunyuan3D-DiT only ever sees a picture.
- **Texture stage** is conditioned on the same reference image plus the generated geometry — also image-driven, not text-driven.
- **VRAM (from Tencent's own docs):** ~10 GB for shape generation alone, ~21 GB for texture generation alone, ~29 GB for the full pipeline running together, with a documented "low VRAM mode" for constrained GPUs. All figures assume an NVIDIA CUDA GPU; there is no meaningful CPU or Apple Silicon inference path in the reference implementation (some community ports exist, e.g. an MLX fork, but they are unofficial and not what a production pipeline should depend on).
- **License:** Tencent Hunyuan3D 2.1 Community License. Commercial use is allowed with a required attribution line ("Created with Hunyuan 3D-2.1"), **excludes the EU, UK, and South Korea as a territory**, and requires requesting a separate license from Tencent above 1M MAU. Worth a legal check before shipping a paid product on top of it, particularly the territory exclusion.

## 2. Current Inference Pipeline (in this repo)

There isn't one. `git log` shows a single "first commit" plus two roadmap-doc commits. No `ml/`, no `rust-server/`, no model weights, no requirements file existed before this session. I created the empty `ml/cad/{data,training,models,inference,evaluation,scripts,configs}` skeleton in this session per the engineering-requirements section of the brief, but it contains no working inference code yet, and won't until we've settled the compute question in Section 6.

## 3. Training Feasibility Analysis

| Constraint | Reality |
|---|---|
| Local GPU | None. Intel HD 6000 integrated, 1.5 GB VRAM, no CUDA/ROCm. |
| Local RAM | 8 GB total — too small to even hold Hunyuan3D-2.1's weights in fp32 CPU inference comfortably, let alone train. |
| Local disk | 19 GB free. Hunyuan3D-2.1's combined checkpoints (shape + texture/PBR models) run tens of GB — **won't fit** alongside anything else on this machine as-is. |
| Network/dataset scale | DeepCAD/Text2CAD and Objaverse are multi-GB to multi-TB. Downloading either in full to this machine is not viable and wasn't attempted. |

**Conclusion:** every real step past "write code and reason about it" — running baseline Hunyuan3D inference, processing a real dataset, training any LoRA, benchmarking latency/VRAM — requires rented GPU compute (Colab/Kaggle/RunPod/Lambda, exactly as `InitialPlan.md` already scoped for Phase 2). Nothing in this report should be read as "trained" or "benchmarked" — it's architecture and code that's ready to run the moment GPU access exists.

Given that constraint, of the fine-tuning options in the brief:

- **LoRA** — the right choice, and the *only* one that makes sense to attempt before proving the concept. Feasible on a single rented consumer/prosumer GPU.
- **QLoRA** — unnecessary at this stage; LoRA on a frozen fp16/bf16 backbone already fits comfortably in the VRAM tiers above without needing 4-bit quantization overhead and its inference-time dequant cost.
- **Full fine-tuning** — explicitly ruled out per the brief's own constraints, and there's no evidence yet that LoRA is insufficient.

## 4. Recommended Fine-Tuning Strategy

The image-only conditioning fact from Section 1 changes which "option" in the brief's Phase 4 is actually correct.

**Fine-tuning the Hunyuan3D shape DiT directly on (text, mesh) pairs is not possible as stated** — it has no text input path. There are three real ways to get language into the loop, in order of cost/risk:

1. **(Recommended for MVP) LoRA the upstream text-to-image model to draw CAD-style reference images**, and leave Hunyuan3D-DiT + the texture stage completely frozen and untouched. Teach the T2I model to produce clean, canonical, mechanically-legible renders — orthographic-ish framing, correct hole/bore/flange placement, consistent thickness, visible symmetry — from prompts like the ones in Phase 6. Hunyuan3D's shape reconstruction is already good at turning a clean, well-composed image into a matching mesh; the failure mode today is that a generic T2I model doesn't know what a flange's bolt circle should look like, not that Hunyuan3D can't reconstruct one if shown it properly.
   - This is the brief's "Option B" in spirit, just placed *before* Hunyuan3D instead of after it.
   - Trainable params: a rank-16–32 LoRA on an SDXL/Flux-class UNet or DiT is on the order of 10–50M parameters — trains on a single rented GPU (even a free-tier Colab T4/A100) in hours on a few thousand image/text pairs.
   - **Inference overhead is close to zero**: the LoRA merges into the T2I model's existing weights; Hunyuan3D-DiT and the texture model are byte-identical to baseline. This satisfies the brief's speed requirement ("as little latency as possible") almost by construction, since nothing new runs on the expensive part of the pipeline.
2. **(v2, only if v1's ceiling is measured and insufficient) LoRA on Hunyuan3D-DiT's cross-attention/MoE layers**, still image-conditioned, to bias shape reconstruction toward CAD-like topology (flat planar patches, clean cylindrical bores, symmetric hole patterns) even when the input image is imperfect. This is the brief's "Option A," done at the right layer. Higher cost (10 GB+ VRAM training run, needs paired mechanical (image, mesh) data), real risk of degrading general-purpose shape quality — needs the eval suite from Section 7 to gate it.
3. **(v2/v3) A lightweight non-learned CAD-snapping post-process** on the output mesh: RANSAC-fit planes/cylinders, detect near-circular boundary loops and regularize them into true circles, detect approximate mirror/rotational symmetry and enforce it. This is not a neural adapter — it's classical geometry processing (a few hundred ms on CPU via trimesh/scipy), so it adds negligible latency and directly targets the "CAD characteristics" eval criteria (planarity, cylindricality, symmetry, hole quality) without touching either neural stage. Cheap enough to prototype **locally, right now, with no GPU**, which is why it's implemented in this session (Section 7).

**Do not** attempt to bolt a text cross-attention branch onto the frozen shape DiT — that's a genuine architecture rewrite of Hunyuan3D internals, explicitly against the brief's "don't unnecessarily rewrite Hunyuan3D" constraint, and there's no evidence option 1+3 is insufficient yet.

## 5. Dataset Strategy

The brief assumes we need to build a procedural CAD generator from scratch. Before doing that, it's worth noting **Text2CAD** ([arXiv:2409.17106](https://arxiv.org/abs/2409.17106), NeurIPS 2024) already exists and covers most of Phase 3's spec:

- Built on **DeepCAD** (~170K parametric sketch-and-extrude CAD models).
- **660K text annotations** per model, ranging from abstract ("generate two concentric cylinders") to precise, dimensioned instructions — i.e., exactly the "language → engineering intent → geometry" pairing the brief asks for, already produced by a validated annotation pipeline (LLaVA-NeXT + Mixtral), not something we'd need to re-invent.
- Parametric CAD sequences convert to meshes trivially (extrude/sketch operations → B-rep → triangulate), and multi-view renders of those meshes are exactly the reference images the T2I LoRA in Section 4 needs to imitate.

Recommended v1 dataset plan:

1. Pull a curated slice of Text2CAD covering mechanically-relevant categories (plates, flanges, brackets, housings — filterable by DeepCAD's shape metadata) rather than the full 170K — a few thousand examples is enough for a LoRA.
2. Render canonical multi-view images per mesh (consistent lighting/background) — this is the (text, image) pair the T2I LoRA trains on.
3. **Supplement, don't replace, with a small procedural generator** for the cases DeepCAD/Text2CAD is weak on: explicit real-world dimensions in mm, standard hole/thread callouts (M8, etc.), and — critically — as **ground truth for the evaluation suite**, since we know exact hole count/spacing/fillet radius by construction and can check the eval code against it before trusting it on model outputs.
4. Before bulk use, check Text2CAD/DeepCAD's license terms for the intended commercial use (not yet done in this session — flagged as an open item, same category as the Hunyuan3D license territory question).

A procedural generator matching the brief's example (parametrized mounting plate) is implemented and runnable today, entirely on CPU, in `ml/cad/data/` — see Section 8.

## 6. CAD-Specific Evaluation Strategy

Implemented as pure-CPU code (`trimesh` + `scipy`, no GPU/network) in `ml/cad/evaluation/`, so it can be validated against known-ground-truth procedural parts *before* any generative model exists:

- **Geometry validity:** watertight, manifold (edge-radial consistency), no self-intersections (approximate, via bounding-volume + normal-consistency checks), Euler characteristic, triangle count.
- **CAD characteristics:** planar-surface count/quality (RANSAC plane fitting over face clusters), cylindrical-surface count/quality (fit cylinders to curved patches, check radius consistency), hole count (genus / boundary-loop detection on non-watertight regions, circularity of hole boundaries), mirror/rotational symmetry score (ICP against a reflected/rotated copy of the mesh), bounding-box dimensions.
- **Prompt adherence:** parse requested counts/dimensions out of the (structured or natural-language) prompt and diff against measured geometry — only meaningful once there's a model to evaluate, but the parsing + diffing logic doesn't depend on one and is written now.

This directly gives Phase 8/9's benchmark table its "CAD Quality" and "Mesh Validity" columns as actual numbers instead of subjective judgment.

## 7. Speed Optimization Strategy

Because the recommended v1 architecture (Section 4, option 1) never modifies Hunyuan3D-DiT or the texture model, the honest baseline expectation is: **fine-tuned end-to-end latency ≈ baseline Hunyuan3D latency + T2I generation time (already part of the baseline pipeline for text prompts) + a few hundred ms of optional CPU-side geometry snapping.** There is no new neural network in the expensive part of the critical path. This should be *measured*, not assumed, once GPU access exists — but there's no architectural reason to expect a meaningful regression, which is the opposite risk profile of Option A/C in the brief (fine-tuning or wrapping the 3D DiT itself, which would add real latency and VRAM).

Standard levers (fp16/bf16, reduced diffusion steps, compiled graphs, attention kernels) all still apply and should be benchmarked against the untouched baseline once running on real hardware — no changes are proposed to them here since the baseline hasn't even been run yet.

## 8. What Was Actually Built This Session

Everything below runs with no GPU, no network dataset download, and no Hunyuan3D checkout — verified locally on this machine:

- `ml/cad/data/procedural_generator.py` — parametric generator for the brief's mounting-plate example (thickness/length/width/hole_count/hole_diameter/hole_spacing/fillet_radius/chamfer), emitting a mesh + a matching natural-language description per sample, plus its exact parameter JSON (needed as eval ground truth).
- `ml/cad/evaluation/geometry_eval.py` — the geometry-validity + CAD-characteristics checks from Section 6.
- `ml/cad/configs/cad_lora_v1.yaml` — config skeleton for the Section 4 T2I-LoRA strategy (paths, LoRA rank, base model slot left as a placeholder pending the decision in Section 9).
- `ml/cad/scripts/generate_dataset.py`, `ml/cad/scripts/run_eval.py` — CLI entry points tying the above together.

These were smoke-tested against each other (generate N procedural plates → run the evaluator → confirm detected hole count/planarity/symmetry match the known generation parameters) before being called "working."

## 9. Open Decisions (need the user, not a default)

- **Roadmap conflict:** `InitialPlan.md` puts this exact work in Phase 6, Month 5+, explicitly *after* the gaming-asset pipeline, to avoid splitting focus across two unfinished products. The task brief asks for it as an immediate MVP priority. Worth an explicit call on sequencing before sinking real GPU budget into it.
- **Cloud GPU provider/budget:** nothing here can move past "code that's ready to run" without a rented GPU (RunPod/Lambda/Colab Pro/etc.) and a decision on how much to spend on the first LoRA experiment.
- **Base T2I model choice** for the Section 4 LoRA (SDXL vs. Flux-dev vs. Hunyuan-DiT itself) — affects license terms and training cost; not picked yet.
- **Dataset licensing** for Text2CAD/DeepCAD under the intended commercial use, and the Hunyuan3D Community License's EU/UK/South Korea exclusion — both need a real legal read, not an assumption, before this goes into a paid product.

## 10. Decision Record (2026-08-26, post-review with project owner)

Section 9's open questions were put to the project owner directly. Answers:

1. **Roadmap sequencing: "Feasibility spike only."** This work stays at the report + local proof-of-concept stage; deeper implementation (Section 4's LoRA plan, real Hunyuan3D usage) is deferred to `InitialPlan.md`'s existing Phase 6 (Month 5+), not pursued now as an MVP priority.
2. **Compute: "No cloud or GPU. Make sure it runs well on the local laptop first... use the intel graphics in this laptop if possible... don't make it too heavy or unusable without proper compute."** This rules out Hunyuan3D-2.1 entirely for the actual prototype build — its shape and texture stages need 10-29GB of NVIDIA VRAM (Section 1) regardless of fine-tuning strategy; there is no version of it that runs on Intel integrated graphics or fits comfortably in 8GB of system RAM.

**Resulting scope change:** rather than stub out a neural pipeline that can't run here, the actual prototype implemented in `ml/cad/` (Section 8) uses the deterministic half of the brief's own Phase 6 architecture diagram —

```
natural language → CAD intent extraction → structured representation → mesh
```

— with the last two arrows served by a plain regex-based parser (`ml/cad/nlp/prompt_parser.py`) and a parametric mesh generator (`ml/cad/data/procedural_generator.py`, now covering rectangular mounting plates and circular flanges), instead of the fine-tuned-model arrow the original brief pictured there. This is not a smaller version of the Hunyuan3D fine-tuning plan — it's a different pipeline that happens to satisfy the same demo-facing requirement ("prompt → intelligent CAD geometry → usable 3D model, generated quickly") within the stated constraint of zero GPU and zero cloud spend. Measured on this machine: ~40-70ms per generation, CPU only, no VRAM used (`ml/cad/benchmark.py`).

Sections 1-9 above are left intact as-is: they're the accurate record of what Hunyuan3D-2.1 actually is and what it would take to fine-tune it, and remain the right starting point if/when the project revisits Phase 6 with real GPU budget. Nothing in this decision record changes those facts — it just means they weren't acted on yet.

`ml/cad/README.md` documents the resulting code layout and its honest limitations (two part families, exact-by-construction dimensions, evaluator tuned to axis-aligned vertically-drilled parts).
