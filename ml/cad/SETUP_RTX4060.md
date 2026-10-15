# Running 3di-men on an RTX 4060 laptop (e.g. MSI)

This is an **optional, additional** hardware profile for a borrowed/temporary
machine with an NVIDIA RTX 4060 (laptop variant, typically 8GB VRAM). It does
not change anything about the project's default behavior.

**Safety guarantee:** nothing in `llm_decompose.py`, `requirements.txt`, or
`.env.example` was modified for this. The Mac (or any machine with no GPU
profile set up) keeps working exactly as it did before this file existed --
CPU-only, `N_GPU_LAYERS=0`. If this laptop goes back to its owner, there is
nothing to undo: just stop using `.env.rtx4060.example` and this guide.

## Why this is needed at all

`llm_decompose.py` already has GPU-offload plumbing (`N_GPU_LAYERS`,
`N_THREADS` env vars, read in `llm_decompose.py`'s config block). On this Mac,
the plain `pip install llama-cpp-python` happens to come with Metal (Apple
GPU) support built in, which is why that plumbing was never exercised here.
**On Windows/NVIDIA, the plain pip wheel has no CUDA kernels at all** --
setting `N_GPU_LAYERS=-1` against a CPU-only build does nothing (it just runs
on CPU, silently, no error). Getting real GPU offload requires installing a
CUDA-enabled build of `llama-cpp-python` specifically. That's the one real
platform-specific step below; everything else in the project is unchanged.

## Recommended: WSL2, not native Windows

Use **WSL2** (Windows Subsystem for Linux 2) with an Ubuntu distro, not native
Windows Python. Reasons:

- This project's tooling (`python3`/`pip3`, bash scripts, path conventions)
  is Unix-flavored throughout -- WSL2 matches it with zero translation.
- WSL2 has mature, well-documented NVIDIA CUDA passthrough (install the
  Windows NVIDIA driver only -- not a separate Linux driver -- then the CUDA
  toolkit inside WSL2).
- Several of this project's compiled dependencies (`manifold3d`, `triangle`,
  `rtree`) do not reliably publish native Windows (`win_amd64`) wheels; a
  manylinux wheel exists for Linux (confirmed buildable/packaged on Arch,
  Fedora, etc.), so WSL2 avoids needing a full Visual Studio C++ Build Tools
  setup just to compile them from source.

Native Windows Python *can* work, but expect to install Visual Studio Build
Tools ("Desktop development with C++") plus the CUDA Toolkit just to get a
C++ toolchain capable of building `llama-cpp-python` and possibly
`manifold3d`/`triangle` from source. WSL2 is the lower-risk path for a
short-lived experiment on borrowed hardware.

## Setup steps (WSL2 + Ubuntu)

1. Install the Windows NVIDIA driver (the regular GeForce driver -- it
   includes WSL2 CUDA support, no separate Linux driver needed).
2. Install WSL2 + Ubuntu: `wsl --install` in an elevated PowerShell, then
   reboot.
3. Inside the WSL2 Ubuntu shell, verify the GPU is visible: `nvidia-smi`
   should list the RTX 4060. If it doesn't, the Windows driver install is
   the thing to fix first -- nothing below will work without this.
4. Install the CUDA Toolkit inside WSL2 (NVIDIA's WSL-specific instructions;
   search "CUDA toolkit WSL2 Ubuntu" for the current apt-based install for
   your Ubuntu version) and confirm `nvcc --version` works.
5. Clone/copy this repo into the WSL2 filesystem (not `/mnt/c/...` -- native
   WSL2 filesystem paths are faster and avoid Windows/Linux file-permission
   friction).
6. Create a venv and install the project's normal CPU-only requirements
   first, unchanged:
   ```
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```
7. **Then** replace just `llama-cpp-python` with a CUDA-enabled build on top
   of that (this is the one RTX4060-specific step -- it only reinstalls this
   one package, nothing else in requirements.txt changes):
   ```
   CMAKE_ARGS="-DGGML_CUDA=on" FORCE_CMAKE=1 \
     pip install llama-cpp-python --force-reinstall --no-cache-dir
   ```
   This needs `cmake` and a C/C++ compiler available in WSL2
   (`sudo apt install -y cmake build-essential` if missing). The
   `--force-reinstall --no-cache-dir` matters -- pip will otherwise reuse the
   already-installed non-CUDA build from step 6.
8. Copy the model `.gguf` file onto this machine (same file this project
   already uses on the Mac -- `Qwen3-4B-Instruct-2507` Q4_K_M), and set
   `MODEL_PATH` (or `MODEL_NAME`/`QUANTIZATION`) accordingly if it's not at
   the default location.
9. Copy `.env.rtx4060.example` to `.env` and adjust `N_THREADS` to this
   laptop's real core count (see the comment in that file for how to check).

## Verify GPU offload is actually working

Don't just trust that it's on -- confirm it, the same way this project
confirmed Metal offload on the Mac earlier:

```
python3 -c "
from llama_cpp import Llama, llama_supports_gpu_offload
print('GPU offload supported:', llama_supports_gpu_offload())
llm = Llama(model_path='<path to the .gguf>', n_gpu_layers=-1, verbose=True)
"
```

`verbose=True` prints llama.cpp's own load-time log -- look for CUDA device
init lines (e.g. `ggml_cuda_init`, device name matching the RTX 4060) and a
layer-offload count matching the model's total layers, not `0`. If you only
see CPU-related init lines, the build in step 7 didn't actually pick up CUDA
-- re-check that `nvcc` was on `PATH` when that pip install ran.

Then restart `server.py` and do a live end-to-end check (per this project's
own standing rule: always verify backend changes with a real request, not
just unit tests) -- submit a build prompt via `/api/build` and compare wall-
clock decomposition time against the Mac to confirm the speedup is real.

## What NOT to do

- Don't change any default in `.env.example`, `requirements.txt`, or
  `llm_decompose.py` to point at RTX4060-specific values. Every Mac-facing
  default stays exactly as-is; this profile is additive only.
- Don't assume `N_GPU_LAYERS=-1` is safe to set globally "just in case" --
  on a CPU-only build it's a no-op, but on hardware with a weaker/shared GPU
  it could be actively worse than CPU. Treat it as RTX4060-specific, not a
  new project-wide norm.
- This laptop is borrowed/temporary. Don't commit anything here that assumes
  it will still be around later (hardcoded paths to this machine's
  filesystem, etc.) -- keep machine-specific values in `.env`, which is
  already gitignored, not in code.
