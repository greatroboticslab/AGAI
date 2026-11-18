# `v1/minigpt4/`: MiniGPT-v2 framework with the lab's changes

The model, dataset and training code behind v1's report writer, a LoRA-adapted MiniGPT-v2 on LLaMA-2-7B-chat (v1, 2025
to April 2026). It is a copy of [Vision-CAIR/MiniGPT-4](https://github.com/Vision-CAIR/MiniGPT-4), which builds on
Salesforce LAVIS (BSD-3-Clause), with the lab's changes; three of them were made on 31 October 2026 for the v2 revival,
which retrained the report writer on split v1 with this package.

## Contents

| Folder | Files | Contents | Changed or added by the lab |
|---|---:|---|---:|
| [`common/`](common/) | 23 | `Config` (loads the run YAML and merges the model and dataset defaults), the name registry, distributed and logging helpers, learning-rate schedulers (`optims.py`), `utils.get_abs_path`; upstream's `eval_utils.py`, `gradcam.py` and the official VQA toolkit (`vqa_tools/`), which no strawberry code uses | 3 |
| [`configs/`](configs/) | 30 | default model configs (`models/`), one default dataset config per builder (`datasets/`, including `strawberry_diagnostic.yaml`), `default.yaml` | 5 |
| [`conversation/`](conversation/) | 2 | `Chat` and the conversation templates (`CONV_VISION_minigptv2`) used at inference | 0 |
| [`datasets/`](datasets/) | 26 | dataset builders registered by name (`builders/`, including `strawberry_diagnostic`) and dataset classes (`datasets/`; the strawberry builder uses `LlavaConversationDataset`) | 10 |
| [`models/`](models/) | 8 | MiniGPT-v2 (`minigpt_v2.py` on `minigpt_base.py`), LLaMA loading with LoRA and 8-bit weights (`base_model.py`), the EVA-ViT-g image encoder, the LLaMA forward with a loss-reduction argument (`modeling_llama.py`), MiniGPT-4 and its Q-Former (not used by v1) | 8 |
| [`processors/`](processors/) | 4 | image processors (`blip2_image_train`, `blip2_image_eval`), text processors (`blip_caption`, `identity_text`), RandAugment | 1 |
| [`runners/`](runners/) | 2 | `RunnerBase`: training loop, validation, checkpoints | 1 |
| [`tasks/`](tasks/) | 3 | `setup_task`, `BaseTask` and `ImageTextPretrainTask` (`image_text_pretrain`, the task of every strawberry config) | 2 |

There is no `minigpt4/__init__.py`. Callers import the subpackages (`minigpt4.datasets.builders`, `minigpt4.models`,
`minigpt4.processors`, `minigpt4.runners`, `minigpt4.tasks`), whose `@registry.register_*` decorators fill the registry
that the YAML names (`arch: minigpt_v2`, `task: image_text_pretrain`, `strawberry_diagnostic`) are looked up in.

## The lab's changes

These 30 files carry the lab's changes (26 changed upstream files and 4 added files); the other 68 are upstream code.
"Jun 2025" marks changes already present in the repository's first commit (28 June 2025), made before the strawberry
work (a COCO-VQA fine-tune and training-loop edits); "Sep-Oct 2025" the strawberry edits of 29 September to 21 October
2025; "Jan 2026" compatibility edits of 21 January 2026 for newer PyTorch, timm and transformers; "31 Oct 2026" the v2
revival's edits for the split v1 retrain (commit `ee006d9`).

| File | Change | When |
|---|---|---|
| `common/config_override.py` (added) | `Config` subclass that loads the COCO-VQA dataset config for every dataset; nothing imports it | Jun 2025 |
| `common/registry.py` | prints the registered schedulers on every scheduler lookup | Jun 2025 |
| `common/utils.py` | `get_abs_path` falls back to the lab-server root `/data/AGAI/MiniGPT-4/` when `library_root` is not registered (below) | Jun 2025, Sep-Oct 2025 |
| `configs/default.yaml` | a COCO-VQA dataset block with lab-server paths | Jun 2025 |
| `configs/datasets/coco/defaults_vqa.yaml` | lab-server paths of the lab's COCO-VQA data | Jun 2025 |
| `configs/datasets/multitask_conversation/default.yaml` | points at the strawberry photographs and the 2,159 instruction records (lab-server paths) | 21 Oct 2025 |
| `configs/datasets/strawberry_diagnostic.yaml` (added) | defaults of the `strawberry_diagnostic` dataset: training records `plant_diagnostic/datasets/stage2_train_7class_fixed.json`, validation records `plant_diagnostic/jsondatasets/strawberry_stage2_val.json`, image folders `plant_diagnostic/data/train_aug` and `plant_diagnostic/data/holdout` (lab-server paths), image size 448, `blip_caption` text processor | Sep-Oct 2025 |
| `configs/models/minigpt_v2.yaml` | written as a complete inference config: lab-server checkpoint and LLaMA paths, LoRA rank 16 and alpha 32, image size 672 at inference; no code reads it (below) | Jun 2025, Sep-Oct 2025 |
| `datasets/__init__.py` | imports the builders, models, processors and tasks; adds `build_datasets(config_path)` | Jun 2025 |
| `datasets/builders/__init__.py` | exports `StrawberryDiagnosticBuilder` | Sep-Oct 2025 |
| `datasets/builders/base_dataset_builder.py` | warnings for missing annotation and image paths instead of downloads; `build()` passes lists of annotation files | Jun 2025 |
| `datasets/builders/image_text_pair_builder.py` | the `coco_vqa` builder gets an evaluation dataset; the `llava_conversation` builder renamed `llava_conversation_dead`, an unused `llava_conversation_dead2`, and a new `llava_conversation` builder with train and val splits | Jun 2025, Sep-Oct 2025 |
| `datasets/builders/strawberry_diagnostic_builder.py` (added) | builder `strawberry_diagnostic`: `LlavaConversationDataset` for train and val; its default config is `minigpt4/configs/datasets/strawberry_diagnostic.yaml` under `library_root` | Sep-Oct 2025 |
| `datasets/builders/vqa_builder.py` (added) | a second builder named `coco_vqa`; nothing imports it | Jun 2025 |
| `datasets/datasets/coco_vqa_datasets.py`, `datasets/datasets/vqa_datasets.py` | COCO-VQA datasets for the lab's fine-tune: several annotation files, image roots given as a list, image names built from COCO 2014 ids | Jun 2025 |
| `datasets/datasets/dataloader_utils.py` | a debug print; `MultiIterLoader.__iter__` | Jun 2025, Sep-Oct 2025 |
| `datasets/datasets/llava_dataset.py` | a record's photo path is tried as given if absolute, under the image root, under `<image root>/data/`, then relative to the working directory; records whose photo is not found are dropped and counted in a printed line | Sep-Oct 2025 |
| `models/base_model.py` | LLaMA loaded with LoRA adapters through `peft` and 8-bit weights through `BitsAndBytesConfig`; SDPA attention and `torch.load(weights_only=False)` | Sep-Oct 2025, Jan 2026 |
| `models/minigpt_base.py` | LLaMA frozen while its LoRA adapters stay trainable; **31 Oct 2026:** `reduction` is passed to the LLaMA forward only when it is not `"mean"` | Sep-Oct 2025, 31 Oct 2026 |
| `models/minigpt_v2.py` | non-reentrant gradient checkpointing; image embeddings cast to the projector's dtype; `torch.load(weights_only=False)` | Sep-Oct 2025, Jan 2026 |
| `models/eva_vit.py` | logging; `timm` import paths with fallbacks; `torch.load(weights_only=False)` | Jun 2025, Jan 2026 |
| `models/minigpt4.py` | a `LOCAL_RANK` line; `torch.load(weights_only=False)` | Sep-Oct 2025, Jan 2026 |
| `models/Qformer.py` | a `LOCAL_RANK` line; gradient-checkpoint call with `use_reentrant=False` (broken, see Known problems) | Sep-Oct 2025 |
| `models/__init__.py`, `models/modeling_llama.py` | a `LOCAL_RANK` line | Sep-Oct 2025 |
| `processors/blip_processors.py` | **31 Oct 2026:** text processor `identity_text`, which passes the text through unchanged (`blip_caption` lowercases it, replaces some punctuation and cuts it at 50 words); used by [`../train_configs/minigptv2_strawberry_splitv1.yaml`](../train_configs/minigptv2_strawberry_splitv1.yaml) | 31 Oct 2026 |
| `runners/runner_base.py` | rewritten: validation splits resolved from `run.evaluate` and `run.val_splits`, an optional automatic validation split (`auto_val_split_ratio`), output folder under `library_root`, device placement for 8-bit models, `checkpoint_best.pth` by the highest validation `agg_metrics`; `torch.load(weights_only=False)` | Jun 2025, Sep-Oct 2025, Jan 2026 |
| `tasks/base_task.py` | `train_step` with gradient accumulation, mixed precision and gradient clipping; the `torch.amp` API and a log line every 50 steps | Jun 2025, Sep-Oct 2025 |
| `tasks/image_text_pretrain.py` | `evaluation()` and `compute_validation_loss()`; **31 Oct 2026:** `evaluation()` no longer requires `run.evaluate` and returns the mean validation loss over at most `max_val_steps` batches, with `agg_metrics = exp(-loss)` so that the runner's highest-`agg_metrics` rule keeps the lowest loss | Sep-Oct 2025, 31 Oct 2026 |

## How configs are resolved

A run YAML (`--cfg-path`) names a model (`arch`, `model_type`) and datasets. `common/config.py` merges each with its
default config, which the model class and the dataset builder name by a relative path (`configs/models/minigpt_v2.yaml`;
`configs/datasets/...`; `minigpt4/configs/datasets/strawberry_diagnostic.yaml` for the strawberry builder).
`common/utils.get_abs_path` joins that path to the registry path `library_root`, or to `/data/AGAI/MiniGPT-4/` when
`library_root` is not registered:

- [`../train.py`](../train.py) registers `$MINIGPT4_LIB_ROOT`, or else its own folder (`v1/`), before it builds the
  config. The MiniGPT-v2 default is then [`../configs/models/minigpt_v2.yaml`](../configs/models/minigpt_v2.yaml), and
  the strawberry default is `configs/datasets/strawberry_diagnostic.yaml` in this folder.
- Every other entry point builds the config without registering it (`demo_v5.py`, `eval_only.py`, `eval_holdout.py`,
  `evaluation/eval_comprehensive.py`, [`../grounding/inference.py`](../grounding/inference.py) and through it the v2
  scripts, `hallucination_evaluation/evaluate.py` and `test_prompt_styles.py`). They read
  `/data/AGAI/MiniGPT-4/configs/models/minigpt_v2.yaml` and
  `/data/AGAI/MiniGPT-4/minigpt4/configs/datasets/strawberry_diagnostic.yaml`, so on a machine without that folder
  `Config` stops with a missing-file error. Link `/data/AGAI/MiniGPT-4` to `v1/` or replace the prefix
  ([`../README.md`](../README.md#paths)).
- `runners/runner_base.py` registers `$MINIGPT4_LIB_ROOT` or this folder only if nothing did, after the config is
  built, and writes checkpoints to `<library_root>/<run.output_dir>/<job_name or timestamp>/` (an absolute
  `output_dir` is used as given).

`configs/models/minigpt_v2.yaml` here differs from `v1/configs/models/minigpt_v2.yaml` only in the image size at
inference (672 here, 768 there), and only the latter is read. The other files in `configs/datasets/`,
`configs/models/minigpt4_*.yaml` and `configs/default.yaml` would be found only with `library_root` set to this folder,
which no entry point does before building the config.

## What uses it

- **v1.** `train.py`, `demo_v5.py`, `eval_only.py`, `eval_holdout.py`, `evaluation/eval_comprehensive.py`,
  `grounding/inference.py`, `hallucination_evaluation/evaluate.py` and `test_prompt_styles.py`, and
  `tests/test_model_loading.py` import it by package name, so `v1/` must be on `sys.path`: run them with `v1/` as the
  working directory. The configs in `v1/train_configs/` and `v1/eval_configs/` select its components.
- **v2.** `v2/vlm/latency_v1.py`, `v2/vlm/minigpt_reports.py`, `v2/vlm/minigpt_val_loss.py` and
  `v2/benchmark/minigpt_diagnose.py` load the report writer through `grounding.inference.load_minigpt`, with the
  working directory on `sys.path`.
  [`v2/slurm/train_minigpt_splitv1.sbatch`](../../v2/slurm/train_minigpt_splitv1.sbatch) runs `train.py` with
  `train_configs/minigptv2_strawberry_splitv1.yaml`. The retrain and `minigpt_val_loss.py` go through the
  31 October 2026 loss call in `models/minigpt_base.py`, and the retrain's config uses `identity_text`.

## Known problems

- **Lab-server paths.** Besides the `get_abs_path` fallback, `configs/datasets/strawberry_diagnostic.yaml`,
  `configs/models/minigpt_v2.yaml`, `configs/default.yaml`, `configs/datasets/coco/defaults_vqa.yaml` and
  `configs/datasets/multitask_conversation/default.yaml` name files under `/data/AGAI/MiniGPT-4/`.
- **No validation while training.** `RunnerBase` validates only when `run.evaluate` is true, and the same flag makes the
  run evaluation-only. A training run therefore saves `checkpoint_<epoch>.pth` every epoch and no `checkpoint_best.pth`;
  the split v1 retrain (`evaluate: false`) picked its epoch afterwards by validation loss with
  [`v2/vlm/minigpt_val_loss.py`](../../v2/vlm/minigpt_val_loss.py).
- **Missing photos shrink the data.** `LlavaConversationDataset` drops records whose photo it cannot find and only
  prints how many, so a wrong image root gives a smaller training or validation set instead of an error.
- **Broken or unused code.** `common/config_override.py` builds the path `minigpt4/minigpt4/configs/...`, which does not
  exist; `datasets/builders/vqa_builder.py` would raise `KeyError` when imported, because `image_text_pair_builder.py`
  already registers `coco_vqa`; `models/Qformer.py` passes `use_reentrant=False` to `create_custom_forward`, which takes
  one argument, so Q-Former gradient checkpointing would raise `TypeError` (MiniGPT-v2 builds no Q-Former). Nothing
  imports the first two.
- **Debug output.** `common/registry.py` prints the registered schedulers at every scheduler lookup.
