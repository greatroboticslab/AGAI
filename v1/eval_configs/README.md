# `v1/eval_configs/`: inference config of the v1 report writer

Part of v1, the original system (2025 to April 2026). The config that loads v1's fine-tuned MiniGPT-v2 report writer
for inference. The application ([`../demo_v5.py`](../demo_v5.py)) and
[`../grounding/inference.py`](../grounding/inference.py) open it by the relative path
`eval_configs/minigptv2_eval.yaml`, so they run from `v1/`; the 2026 revival's scripts that run MiniGPT-v2 load the
model through it. Unchanged during the revival.

## Files

| File | What it is |
|---|---|
| [`minigptv2_eval.yaml`](minigptv2_eval.yaml) | MiniGPT-v2 (`arch: minigpt_v2`, `model_type: pretrain`) with LLaMA-2-7B-chat loaded in 8-bit (`low_resource: true`), LoRA r 64 / α 128, up to 500 text tokens, and the lab's fine-tuned checkpoint `output/minigpt_strawberry/checkpoint_best.pth`; a `strawberry_diagnostic` dataset block (448-pixel eval processor, the 2159 instruction records of [`../plant_diagnostic/datasets/`](../plant_diagnostic/datasets/) and the `train_aug` photographs) and eval-only `run:` settings. Every path in it is a lab-server path (`/data/AGAI/MiniGPT-4/...`) |

The lab notebook calls this checkpoint the April writer and notes that it is the last epoch of its training run,
because the runner never saved a best checkpoint in training mode
([`../../v2/EXPERIMENT_LOG.md`](../../v2/EXPERIMENT_LOG.md), 30 October 2026). [`../train.py`](../train.py) writes
that folder with either lab config in [`../train_configs/`](../train_configs/) (`job_name: minigpt_strawberry`); the
file's comment names a `train_v2.py`, which has never been in the repository.

## Who reads it

- **v1:** [`../demo_v5.py`](../demo_v5.py) (default `--cfg-path`),
  [`../grounding/inference.py`](../grounding/inference.py) (`load_minigpt()`),
  [`../hallucination_evaluation/config.py`](../hallucination_evaluation/config.py) (by its own location),
  [`../hallucination_evaluation/test_prompt_styles.py`](../hallucination_evaluation/test_prompt_styles.py) and
  [`../tests/test_model_loading.py`](../tests/test_model_loading.py) (by the absolute lab-server path).
- **v2**, through `load_minigpt()`: [`../../v2/vlm/latency_v1.py`](../../v2/vlm/latency_v1.py) runs the writer named
  here (v1 as deployed); [`../../v2/vlm/minigpt_reports.py`](../../v2/vlm/minigpt_reports.py),
  [`../../v2/vlm/minigpt_val_loss.py`](../../v2/vlm/minigpt_val_loss.py) and
  [`../../v2/benchmark/minigpt_diagnose.py`](../../v2/benchmark/minigpt_diagnose.py) build the model from this file
  and then replace its LoRA and projection weights with other checkpoints.

## Known problems

- **Paths.** The checkpoint, LLaMA weights, records, photographs and cache are named by absolute lab-server paths.
  Loading the file also resolves [`../configs/models/minigpt_v2.yaml`](../configs/models/minigpt_v2.yaml) and the
  `strawberry_diagnostic` dataset defaults under `/data/AGAI/MiniGPT-4/` ([`../configs/README.md`](../configs/README.md)).
  Replace the prefix or link that path to `v1/` ([Paths](../README.md#paths)).
- **The checkpoint must exist even when it is replaced.** `MiniGPTv2.from_config()` loads `ckpt` while building the
  model, so the v2 scripts that swap in other weights still need the file named here.
- **Missing files.** Neither the checkpoint nor the LLaMA weights ([`../llama_weights/`](../llama_weights/)) are in
  the repository.
