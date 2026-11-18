# `v1/configs/`: default MiniGPT-v2 model config

Part of v1, the original system (2025 to April 2026). `models/minigpt_v2.yaml` holds the model defaults that the
MiniGPT framework in [`../minigpt4/`](../minigpt4/) loads for `arch: minigpt_v2` with `model_type: pretrain`, which
every config in [`../eval_configs/`](../eval_configs/) and [`../train_configs/`](../train_configs/) asks for; the
2026 revival's retrain of the report writer read it as well. No file here was changed during the revival.

## Files

| File | What it is |
|---|---|
| [`models/minigpt_v2.yaml`](models/minigpt_v2.yaml) | the default config of `minigpt_v2` / `pretrain` (`PRETRAINED_MODEL_CONFIG_DICT` in [`../minigpt4/models/minigpt_v2.py`](../minigpt4/models/minigpt_v2.py)): a `model:` block (LLaMA-2-7B-chat in 8-bit, LoRA r 16 / α 32, prompt template `Human: {} Assistant: `, checkpoint `output/minigptv2_finetune/new/checkpoint_9.pth`), a `preprocess:` block (448-pixel train and val processors, 768-pixel eval), empty `datasets:` and `run: {task: image_text_pretrain}`; its header says it can also serve as a top-level `--cfg-path` file. Lab-server paths |
| [`models/minigpt_v2_pretrain.yaml`](models/minigpt_v2_pretrain.yaml) | a flat model config (`model_type: pretrain_v2`, LoRA r 64 / α 16, prompt `[INST] {} [/INST]`) that no code reads: `pretrain_v2` is not a model type of `minigpt_v2.py`, and without a `model:` key the file cannot be a `--cfg-path` config either |

[`../minigpt4/configs/models/minigpt_v2.yaml`](../minigpt4/configs/models/minigpt_v2.yaml) is a near-copy of
`models/minigpt_v2.yaml` (eval image size 672 instead of 768) that no code path reads.

## How the file is found and used

- `Config` ([`../minigpt4/common/config.py`](../minigpt4/common/config.py)) loads this file first and merges the
  top-level config's `model:` block over it, so a value set in the top-level config always wins and only the keys it
  leaves out come from here.
- The path is `<library root>/configs/models/minigpt_v2.yaml` (`get_abs_path` in
  [`../minigpt4/common/utils.py`](../minigpt4/common/utils.py)). No code registers the library root before `Config`
  is built ([`../train.py`](../train.py) registers its own folder only afterwards, and `minigpt4/` has no
  `__init__.py` that could do it), so the lookup falls back to `/data/AGAI/MiniGPT-4/configs/models/minigpt_v2.yaml`.
  Outside the lab server that path has to exist, or the prefix has to be replaced
  ([Paths](../README.md#paths)). The dataset defaults,
  [`../minigpt4/configs/datasets/strawberry_diagnostic.yaml`](../minigpt4/configs/datasets/strawberry_diagnostic.yaml),
  are found the same way.
- With the configs in this repository, the only value from this file that reaches the model is the prompt template
  of the three training configs, which set none: training wraps each instruction in `Human: {} Assistant: `. The
  configuration logged by the split v1 retrain shows it, together with this file's `use_lora`,
  `tune_mm_mlp_adapter`, `freeze_llama` and `preprocess:` entries
  ([`minigpt_splitv1_train_log.txt`](../../evaluation/paper_data/minigpt_splitv1_train_log.txt) in
  [`../../evaluation/paper_data/`](../../evaluation/paper_data/)).
  Those four entries have no effect: `MiniGPTv2.from_config()` does not read the first three, and no code reads the
  `preprocess:` block. [`../demo_v5.py`](../demo_v5.py) and [`../grounding/inference.py`](../grounding/inference.py)
  ask for `cfg.preprocess_cfg`, which `Config` does not define, and fall back to a 448-pixel processor; at inference
  they also use their own `[INST] ... [/INST]` conversation template.
