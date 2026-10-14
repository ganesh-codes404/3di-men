#!/usr/bin/env python3
"""python training/train.py --config training/configs/qwen3_4b_decompose_lora_v1.yaml

LoRA/QLoRA supervised fine-tuning for llm_decompose.py's Qwen3-4B-Instruct-2507
model, on the dataset training/prepare_dataset.py builds.

GPU-ONLY, BY DESIGN: this machine (8GB RAM, no CUDA, dual-core CPU) cannot
run this job in any practical amount of time -- a 4B-parameter LoRA fine-tune
needs a real GPU. Rather than silently attempting a CPU run that would take
days, this fails fast with a clear message when no CUDA device is visible,
the same "fail-fast stub" convention requirements.txt's GPU-only dependency
block already documents. Run this on a rented GPU box (Colab/Kaggle/a cloud
instance) instead: copy this repo (or just ml/cad/training/ + the dataset +
the base model), install the GPU-only requirements block in requirements.txt,
and run this script there.

What this actually does once a GPU is present:
1. Loads the base model (transformers, optionally 4-bit via bitsandbytes for
   QLoRA) and tokenizer.
2. Applies a LoRA adapter (peft) per the config's `lora` section.
3. Tokenizes training/data/{training,validation}/sft.jsonl (chat-format,
   produced by prepare_dataset.py) via the model's own chat template.
4. Trains with a plain HF Trainer loop, checkpointing to output_dir.
5. Saves the adapter (not a merged model) to output_dir/adapter -- small,
   easy to version, and mergeable later or loadable directly via peft.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _require_gpu() -> None:
    try:
        import torch
    except ImportError:
        raise SystemExit(
            "torch is not installed. This is expected on the local CPU-only dev "
            "machine -- the GPU-only requirements block in requirements.txt is "
            "commented out deliberately. Install it on the GPU box, not here."
        )
    if not torch.cuda.is_available():
        raise SystemExit(
            "No CUDA GPU visible (torch.cuda.is_available() is False). "
            "Refusing to start a LoRA fine-tune on CPU -- a 4B-parameter "
            "model's adapter training would take an impractical amount of "
            "time on this hardware (this project's disclosed constraint: "
            "8GB RAM, no CUDA, dual-core CPU). Run this script on a rented "
            "GPU instance instead; nothing else about the command changes."
        )


def load_config(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_chat_jsonl(path: Path) -> list[dict]:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    args = ap.parse_args()

    _require_gpu()

    # Imports deferred past _require_gpu() so the fail-fast path above never
    # needs peft/bitsandbytes installed locally either.
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import (
        AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
        Trainer, TrainingArguments, DataCollatorForLanguageModeling,
    )

    config = load_config(args.config)
    base_model_name = config["base_model"]
    training_cfg = config["training"]
    lora_cfg = config["lora"]
    data_cfg = config["data"]
    output_dir = Path(config["output_dir"])

    quantization_config = None
    if config.get("qlora", {}).get("enabled"):
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=config["qlora"].get("quant_type", "nf4"),
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        quantization_config=quantization_config,
        torch_dtype=torch.bfloat16 if quantization_config is None else None,
        device_map="auto",
    )

    peft_config = LoraConfig(
        r=lora_cfg["rank"], lora_alpha=lora_cfg["alpha"], lora_dropout=lora_cfg["dropout"],
        target_modules=lora_cfg["target_modules"], bias="none", task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    train_records = load_chat_jsonl(Path(data_cfg["train_path"]))
    val_records = load_chat_jsonl(Path(data_cfg["val_path"]))

    def tokenize(records: list[dict]) -> list[dict]:
        out = []
        for r in records:
            text = tokenizer.apply_chat_template(r["messages"], tokenize=False, add_generation_prompt=False)
            enc = tokenizer(text, truncation=True, max_length=data_cfg.get("max_length", 4096))
            out.append(enc)
        return out

    train_dataset = tokenize(train_records)
    val_dataset = tokenize(val_records)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=training_cfg["batch_size"],
        gradient_accumulation_steps=training_cfg["gradient_accumulation_steps"],
        learning_rate=training_cfg["learning_rate"],
        max_steps=training_cfg["max_steps"],
        bf16=True,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=training_cfg.get("eval_steps", 50),
        save_steps=training_cfg.get("save_steps", 100),
        seed=training_cfg.get("seed", 0),
        report_to=[],
    )

    trainer = Trainer(
        model=model, args=training_args,
        train_dataset=train_dataset, eval_dataset=val_dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )
    trainer.train()

    adapter_dir = output_dir / "adapter"
    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    print(f"adapter saved to {adapter_dir}")


if __name__ == "__main__":
    main()
