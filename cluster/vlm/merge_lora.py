"""Merge a LoRA adapter into its base VLM and save a plain checkpoint (for vLLM, which cannot load LoRA on the
vision tower).
  python cluster/vlm/merge_lora.py BASE_DIR ADAPTER_DIR OUT_DIR
"""
import sys

import torch
from peft import PeftModel
from transformers import AutoModelForImageTextToText, AutoProcessor

base, adapter, out = sys.argv[1:4]
model = AutoModelForImageTextToText.from_pretrained(base, dtype=torch.bfloat16, device_map="cpu")
model = PeftModel.from_pretrained(model, adapter).merge_and_unload()
model.save_pretrained(out, max_shard_size="5GB")
AutoProcessor.from_pretrained(base).save_pretrained(out)
print("merged ->", out)
