# Kaggle: real Hunyuan3D-2.1 baseline

`hunyuan3d_baseline.ipynb` runs the actual Hunyuan3D-2.1 model (not the local
rule-based prototype in `ml/cad/`) on a free Kaggle GPU, so you can see real
neural 3D generation for comparison. Scope is deliberately small for a first
run: text → quick reference image (`sd-turbo`) → Hunyuan3D-2.1 **shape only**
(no texture/PBR painting — that needs ~21GB VRAM on its own and doesn't
reliably fit a single Kaggle GPU; shape-only needs ~10GB and fits comfortably
in Kaggle's 16GB).

I have no GPU in my own environment, so I could not run this end to end
myself — every install step and API call is transcribed directly from the
official [Hunyuan3D-2.1 repo](https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1)
(its root README quick-start and `hy3dshape/minimal_demo.py`), but you'll be
the first to actually execute it.

## How to put it on Kaggle and run it

1. Go to [kaggle.com](https://kaggle.com) → **Create** → **New Notebook**.
2. In the new notebook's editor: **File → Import Notebook** → upload
   `hunyuan3d_baseline.ipynb` from this folder. (Or drag-and-drop the file
   onto the import dialog.)
3. In the right-hand settings panel:
   - **Accelerator** → `GPU T4 x2` (or `P100`, whichever is available/faster
     for your account).
   - **Internet** → `On`. This requires phone-verifying your Kaggle account
     once, under kaggle.com/settings, if you haven't already — pip installs
     and the model download in this notebook need it, and it's off by
     default for new notebooks.
4. Run the cells top to bottom (**Run All**, or step through with
   Shift+Enter). The first code cell will fail loudly and tell you exactly
   what to fix if the GPU isn't attached.
5. Edit the `PROMPT` variable in the "text prompt → reference image" cell to
   whatever part you want to try.
6. If the model download in step 3 of the notebook fails with a "gated
   repo"/401 error, go back to the optional Hugging Face login cell (1b) and
   follow its instructions — create a free HF account + token, add it as a
   Kaggle Secret named `HF_TOKEN`, rerun that cell, then retry.

Kaggle's free tier gives ~30 GPU-hours/week; a single shape-only generation
takes well under a minute of that once the model is downloaded, so this
notebook barely touches the quota.

## Getting the output back to your local machine

The notebook's last cell zips everything into `/kaggle/working/hunyuan3d_output.zip`
(reference image, `.glb`, `.stl`, and a quick preview PNG). Two ways to grab it:

- **During the session:** left sidebar file browser → `/kaggle/working/` →
  `...` menu on `hunyuan3d_output.zip` → **Download**.
- **After the fact / to keep it:** **Save Version** → **Save & Run All** at
  top right, then open the notebook's **Output** tab once it finishes — every
  file under `/kaggle/working/` is downloadable there, and it stays even
  after the session ends.

Once `output.stl` is on your machine, point this repo's own tools at it —
same evaluator, same viewer, as the local prototype's output:

```bash
cd ml/cad
python3 scripts/preview_mesh.py --mesh ~/Downloads/output.stl --out compare.png --elev 89 --azim -90
python3 evaluate.py --meshes ~/Downloads --out kaggle_eval.json
```

That comparison — real Hunyuan3D output vs. the local rule-based prototype,
on the same kind of prompt, scored by the same evaluator — is the natural
next data point for deciding whether/how to pursue Section 4's fine-tuning
plan.
