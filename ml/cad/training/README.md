# Qwen3-4B decomposition fine-tuning pipeline

Fine-tunes `llm_decompose.py`'s Qwen3-4B-Instruct-2507 to be more reliable at
its ONE job: decomposing a natural-language CAD request into independently-
buildable part descriptions (`_SYSTEM_PROMPT`), and deciding when to ask
clarifying questions first (`_CLARIFY_SYSTEM_PROMPT`). It is NOT trained to
choose tools or generate geometry parameters -- that stays with this
project's own separate, proprietary `program_model.py` classifier. See
`llm_decompose.py`'s module docstring for why that split exists.

## Status

Dataset generation works today, on the local CPU-only machine. Training does
not run locally -- see `train.py`'s module docstring for why -- and has not
yet been run anywhere. This is a ready pipeline, not a trained result.

## Pipeline

```
prepare_dataset.py  (CPU, runs now)
        |
        v
training/data/{training,validation,test}/sft.jsonl
        |
        v
train.py  (GPU-ONLY -- rent one; fails fast here on purpose)
        |
        v
checkpoints/<experiment>/adapter/  (a LoRA adapter, not a merged model)
        |
        v
merge adapter into base weights, re-quantize to .gguf  (see below)
        |
        v
point MODEL_PATH at the new .gguf, restart server.py
        |
        v
evaluate.py / inference.py  (CPU, runs now, against base OR fine-tuned .gguf)
```

## 1. Build the dataset (works now)

```
python3 training/prepare_dataset.py --n-per-pattern 12 --seed 0
```

Writes `training/data/{raw,training,validation,test}/*.jsonl` plus a
`manifest.json` with exact counts and sources. Re-run after any change to
`data/program_dataset.py`'s pattern corpus or `sandbox.py`'s tool roster --
the dataset does not update itself. See `prepare_dataset.py`'s own docstring
for exactly what's synthetic-templated vs. hand-authored, and its disclosed
limitation (templated phrasing variety, not real user language).

`training/data/curated/` and `training/data/failures/` are intentionally
left empty by this script -- they're where real corrected tool sequences,
expert demonstrations, and logged production failures belong once this
system has real usage to learn from (see `server.py`'s logging, Section 18
of the original design brief). Mixing those in requires a manual curation
step this script does not attempt.

## 2. Fine-tune (GPU box only)

Rent a GPU (a single 16GB card comfortably fits a 4B QLoRA run). Copy over
`ml/cad/training/`, the `ml/cad/data/` and `ml/cad/part_families.py` modules
`prepare_dataset.py` imports, and the generated `training/data/` JSONL files
(or just re-run `prepare_dataset.py` there -- it's fast and reproducible).
Install the GPU-only block in `requirements.txt` (uncomment it there).

```
python3 training/train.py --config training/configs/qwen3_4b_decompose_lora_v1.yaml
```

Trains against the full-precision Hugging Face checkpoint
(`Qwen/Qwen3-4B-Instruct-2507`), NOT the local `.gguf` -- llama.cpp/gguf has
no LoRA training support. Saves a LoRA adapter (small, a few tens of MB) to
`output_dir/adapter`, not a merged model.

## 3. Merge + re-quantize back to GGUF (GPU box, or a machine with enough RAM)

```
python3 -c "
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
base = AutoModelForCausalLM.from_pretrained('Qwen/Qwen3-4B-Instruct-2507')
merged = PeftModel.from_pretrained(base, 'training/checkpoints/qwen3_4b_decompose_lora_v1/adapter').merge_and_unload()
merged.save_pretrained('merged-qwen3-4b-decompose-v1')
AutoTokenizer.from_pretrained('Qwen/Qwen3-4B-Instruct-2507').save_pretrained('merged-qwen3-4b-decompose-v1')
"
# then convert + quantize with llama.cpp's own conversion scripts:
python3 convert_hf_to_gguf.py merged-qwen3-4b-decompose-v1 --outfile qwen3-4b-decompose-v1-f16.gguf
./llama-quantize qwen3-4b-decompose-v1-f16.gguf qwen3-4b-decompose-v1-Q4_K_M.gguf Q4_K_M
```

Copy the resulting `.gguf` into `ml/cad/models/qwen/quantized/`, then either
rename it to match `MODEL_NAME`/`QUANTIZATION`'s expected filename or set
`MODEL_PATH` directly (see `.env.example`).

## 4. Evaluate (works now, against base OR fine-tuned)

```
python3 training/evaluate.py --split test --limit 5
python3 training/evaluate.py --split test --limit 5 --model-path models/qwen/quantized/qwen3-4b-decompose-v1-Q4_K_M.gguf
```

Never averages the two runs -- report them side by side, same principle as
`program_eval.py`'s 3-bucket rule. Each example costs ~10-15 minutes of real
inference on this hardware; `--limit` defaults small on purpose. See
`evaluate.py`'s own docstring for exactly what each metric measures and why
(structural/executable checks, not subjective quality).

## 5. Quick manual spot-check

```
python3 training/inference.py "a water bottle"
python3 training/inference.py "a bracket with some holes in it" --task clarify
```

## What this pipeline does NOT cover yet

- Real user requests / production failure logs feeding back into
  `training/data/curated/` and `failures/` (needs real usage first).
- A retry/error-recovery loop feeding a tool's structured error back to the
  LLM for self-correction (`parallel_builder.py` currently reports a
  failure once, it does not re-prompt).
- A 50+ case, difficulty-tiered (EASY/MEDIUM/HARD/AMBIGUOUS/ADVERSARIAL)
  hand-labeled benchmark -- `evaluate.py` scores the synthetic test split
  today, which is a different, narrower thing.
