# RTX 4060 laptop: quick-start checklist

Pure step-by-step for getting this repo running with GPU offload on the MSI
laptop. For the "why" behind each step (WSL2 vs native Windows, why the CUDA
rebuild is needed, etc.), see [SETUP_RTX4060.md](SETUP_RTX4060.md) -- this
file is just the checklist version.

Nothing here changes the Mac setup. This is a separate, opt-in profile.

## 1. Get the repo on the laptop

- [ ] `git clone <repo-url>` (or pull, if already cloned)

## 2. Set up WSL2 + Ubuntu

- [ ] Install the regular Windows NVIDIA GeForce driver (not a separate Linux driver)
- [ ] In an elevated PowerShell: `wsl --install`, then reboot
- [ ] Open the WSL2 Ubuntu shell, run `nvidia-smi` -- confirm it lists the RTX 4060
  - If it doesn't show up, stop here and fix the Windows driver install first

## 3. Install the CUDA Toolkit inside WSL2

- [ ] Follow NVIDIA's WSL2-specific CUDA Toolkit install instructions for your Ubuntu version
- [ ] Confirm `nvcc --version` works in the WSL2 shell

## 4. Move the repo into the WSL2 filesystem

- [ ] Put the repo under WSL2's own filesystem (e.g. `~/3di-men`), **not** `/mnt/c/...`
  (faster, avoids Windows/Linux file-permission friction)

## 5. Set up the Python environment (same as the Mac, nothing special yet)

- [ ] `cd ml/cad`
- [ ] `python3 -m venv venv`
- [ ] `source venv/bin/activate`
- [ ] `pip install -r requirements.txt`

## 6. Rebuild `llama-cpp-python` with CUDA (the one RTX4060-specific step)

- [ ] `sudo apt install -y cmake build-essential`
- [ ] ```
      CMAKE_ARGS="-DGGML_CUDA=on" FORCE_CMAKE=1 \
        pip install llama-cpp-python --force-reinstall --no-cache-dir
      ```
- [ ] Watch the build output for CUDA-related compile steps (confirms it actually picked up `nvcc`)

## 7. Get the model file onto the laptop

- [ ] Copy the same `.gguf` used on the Mac (`Qwen3-4B-Instruct-2507`, Q4_K_M quantization)
  onto the laptop's filesystem, by any method (scp, USB, cloud drive)

## 8. Set up the env profile

- [ ] `cp .env.rtx4060.example .env`
- [ ] If the `.gguf` isn't at the default path, set `MODEL_PATH` in `.env` to point at it
- [ ] Run `nproc` in WSL2, set `N_THREADS` in `.env` to that number

## 9. Verify GPU offload is actually working

- [ ] Run:
      ```
      python3 -c "
      from llama_cpp import Llama, llama_supports_gpu_offload
      print('GPU offload supported:', llama_supports_gpu_offload())
      llm = Llama(model_path='<path to the .gguf>', n_gpu_layers=-1, verbose=True)
      "
      ```
- [ ] Confirm the verbose load log shows CUDA device init lines (e.g. `ggml_cuda_init`)
  and an offloaded-layer count that isn't `0`

## 10. Run it end-to-end

- [ ] Restart `server.py`
- [ ] Submit a real prompt via `/api/build`
- [ ] Compare wall-clock decomposition time against the Mac to confirm the speedup is real

---

If the laptop goes back to its owner: there's nothing to undo. Just don't use
`.env.rtx4060.example`/this checklist on whatever machine comes next -- the
Mac-safe defaults in `.env.example` are what every other machine already uses.
