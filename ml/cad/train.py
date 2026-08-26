#!/usr/bin/env python3
"""python train.py

There is currently nothing to train. Per the project owner's direction
(docs/cad-finetuning-technical-report.md Section 10), this prototype was
deliberately built as a rule-based prompt parser (nlp/prompt_parser.py) driving
a deterministic parametric mesh generator (data/procedural_generator.py) --
not a fine-tuned neural model -- specifically so it runs on a laptop with no
GPU. There is no LoRA, adapter, or weight file anywhere in this pipeline.

This file is kept (rather than deleted) so the CLI shape matches the
engineering spec, and so it's obvious where a learned component would plug in
if the project later moves to rented GPU compute and picks up the T2I-LoRA +
frozen-Hunyuan3D plan from Section 4:
  - The thing to train would be a LoRA on a text-to-image model, not on
    Hunyuan3D-2.1 itself (Hunyuan3D-DiT is image-conditioned, see Section 1).
  - configs/cad_lora_v1.yaml already has the shape that training would read.
Until that decision is made, extend nlp/prompt_parser.py's regex coverage and
data/procedural_generator.py's part library instead -- that's where this
prototype's "learning" actually happens right now (by hand, not by gradient).
"""
import sys


def main():
    print(__doc__, file=sys.stderr)
    sys.exit(1)


if __name__ == "__main__":
    main()
