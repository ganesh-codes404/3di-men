# Kaggle notebooks

Three notebooks, two different jobs. **Which one to use for what changed** —
read this before picking one:

- **`threedimen_api_server.ipynb`** — the **live backend for the frontend**.
  Serves 3di-men's own trained model (`ml/threedimen/threedimen/model/`)
  exclusively — no Shap-E, no Hunyuan3D-2.1 running here anymore. This is
  what `ml/cad`'s web UI's **Backend URL** field should point at now. No GPU
  needed at all (the model is ~1MB, runs on CPU in milliseconds) — set
  Accelerator to **None**. See its own intro cell for the checkpoint-upload
  steps (you have to upload your locally-trained `.pt` file as a Kaggle
  Dataset first, since Kaggle can't reach your local disk).
- **`hunyuan3d_baseline.ipynb`** / **`hunyuan3d_api_server.ipynb`** — the
  real Hunyuan3D-2.1 model, kept **only as a data-generation teacher** for
  growing `ml/threedimen`'s training dataset (`3dimen dataset build --model
  hunyuan3d ...`), **not for live user-facing generation anymore**. Still
  needs a real GPU (Accelerator → T4/P100) and real Kaggle GPU-hours, same
  as before.

So: run a Hunyuan3D notebook when you want more/better training data, run
`threedimen_api_server.ipynb` when you want the frontend to actually respond
to prompts.

Scope is deliberately small for a first run: text → quick reference image
(`sd-turbo`) → Hunyuan3D-2.1 **shape only** (no texture/PBR painting — that
needs ~21GB VRAM on its own and doesn't reliably fit a single Kaggle GPU;
shape-only needs ~10GB and fits comfortably in Kaggle's 16GB).

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

## Running `hunyuan3d_api_server.ipynb` as a data-generation teacher

(Not for the frontend anymore — see `threedimen_api_server.ipynb` below for
that. This is for `3dimen dataset build --model hunyuan3d ...` / `3dimen
teacher generate ... --model hunyuan3d`.)

Same import/GPU/Internet setup as above (steps 1-3), then:

1. **Run All.** Unlike the baseline notebook, this one doesn't finish —
   the last cell (the tunnel) is meant to keep running indefinitely. It'll
   print a block like:
   ```
   PUBLIC BACKEND URL: https://some-random-words.trycloudflare.com
   ```
2. Copy that URL. On your own machine:
   ```bash
   cd ml/cad
   python3 server.py
   ```
   open `http://127.0.0.1:5050`, paste the URL into the **Backend URL** field
   at the top of the page. The badge next to it should switch from `local`
   to `remote (Kaggle)`.
3. Type a prompt, click **Generate**. Expect it to take a few minutes, not
   seconds — the diffusion-sampling step alone (50 steps) runs ~2 minutes on
   a T4 regardless of settings, plus tens of seconds of volume decoding on
   top. The UI shows "Generating on Kaggle GPU… Ns elapsed" while it works —
   it's polling for the result rather than holding one request open,
   specifically because Cloudflare's free quick tunnels enforce a request
   timeout that a real generation (especially a cold first request) can run
   past; a single blocking request would finish
   successfully on the Kaggle side but the browser would still see it as a
   failed fetch. Polling sidesteps that regardless of how long it actually
   takes.
4. **Leave the Kaggle tab open and the tunnel cell running** for as long as
   you want the backend reachable. Closing it, the session idling out, or
   hitting Kaggle's quota all silently kill the tunnel — Generate will start
   failing with a fetch/connection error in that case; just rerun the tunnel
   cell (or the whole notebook, if the session itself died) for a new URL
   and paste the new one in.
5. To go back to the local rule-based backend, just clear the Backend URL
   field (or delete its text) — the badge switches back to `local`.

**Heads up on exposure:** the tunnel URL has no authentication — anyone with
the link can trigger generations on your GPU quota while it's up. Fine for
your own testing; don't post it publicly.

## Running `threedimen_api_server.ipynb` (the live frontend backend)

1. First train a checkpoint locally, if you haven't:
   ```bash
   cd ml/threedimen
   pip3 install -e ".[train]"
   3dimen model train --root data/threedimen_v1 --out checkpoints/threedimen_v1.pt --epochs 200
   ```
2. On kaggle.com: **Datasets → New Dataset → Upload**, pick that one `.pt`
   file, name it anything (e.g. "threedimen-checkpoint"), create it.
3. **Create → New Notebook** → **File → Import Notebook** → upload
   `threedimen_api_server.ipynb`.
4. In the notebook's right sidebar: **Add Input** → search for and attach
   the dataset from step 2.
5. Settings panel: **Accelerator → None** (this model needs no GPU at all —
   don't burn quota on it), **Internet → On** (same phone-verification
   requirement as before if you haven't already).
6. **Run All.** The checkpoint-loading cell will fail with a clear assertion
   error if it can't find your `.pt` file under `/kaggle/input/` — check
   step 4 if that happens. Like the other server notebook, the tunnel cell
   at the end doesn't finish on its own; leave it running.
7. Copy the printed `PUBLIC BACKEND URL`, paste it into `ml/cad`'s frontend
   **Backend URL** field, same as before.

Generation here is near-instant (well under a second — it's a ~1MB CPU
model), so unlike the Hunyuan3D backend you won't see much "elapsed" time in
the UI's polling status. **Quality is a genuine proof-of-concept, not a
Hunyuan3D-level result** — see `ml/threedimen/README.md`'s "what it is and
isn't" section before showing this to anyone expecting Hunyuan3D-grade
output.

To update the served model later (e.g. after growing the dataset and
retraining): upload a new version of the Kaggle Dataset in step 2, reattach
it, restart the notebook. There's no hot-reload.
