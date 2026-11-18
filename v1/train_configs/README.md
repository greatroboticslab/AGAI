# `v1/train_configs/`: MiniGPT-v2 training configs for `train.py`

LoRA fine-tuning configs for the report writer (MiniGPT-v2 on LLaMA-2-7B-chat), run with [`../train.py`](../train.py).
Two are v1 configs, from the original system (2025 to April 2026). `minigptv2_strawberry_splitv1.yaml` was added by
the 2026 revival on 31 October 2026 to retrain the same writer on split v1 (the revival's leak-free benchmark); it
runs with v1's `train.py`. In these names `minigptv2` is the MiniGPT-v2 model; the `_v2` of
`minigptv2_strawberry_v2.yaml` and the "Config v2" header of both lab files refer to the lab's second training
recipe.

## Files

| File | What it is |
|---|---|
| [`minigptv2_strawberry.yaml`](minigptv2_strawberry.yaml) | the lab's second recipe (its header lists the changes from the first: up to 500 text tokens, LoRA r 64 / α 128, learning rate 1e-5, 20 epochs): 2 GPUs × batch 2 × accumulation 8 (effective 32), 2000 iterations per epoch, 8-bit LLaMA, 448-pixel images, starting from `checkpoints/checkpoint_stage2.pth`; uses the 2159 instruction records with the `train_aug` photographs; writes `output/minigpt_strawberry/`; lab-server paths |
| [`minigptv2_strawberry_v2.yaml`](minigptv2_strawberry_v2.yaml) | identical to `minigptv2_strawberry.yaml` except `evaluate: true`, with which `train.py` validates and does not train |
| [`minigptv2_strawberry_splitv1.yaml`](minigptv2_strawberry_splitv1.yaml) | revival, 31 October 2026: the same recipe on one GPU (batch 2 × accumulation 16, 10 epochs of 2000 iterations), trained on the records of split v1 training photographs and their augmented copies, with the records of split v1 validation photographs as validation set; the training instruction is the application's prompt, passed unchanged by the `identity_text` text processor; paths of the cluster checkout (`/projects/.../agai/`); writes `minigpt_runs/minigpt_strawberry_splitv1/` |

## Running

```bash
cd v1
# the lab's recipe on two GPUs
python -m torch.distributed.run --nproc_per_node=2 train.py --cfg-path train_configs/minigptv2_strawberry.yaml
# the split v1 retrain on one GPU
python train.py --cfg-path train_configs/minigptv2_strawberry_splitv1.yaml
```

On the cluster the second command is
[`../../v2/slurm/train_minigpt_splitv1.sbatch`](../../v2/slurm/train_minigpt_splitv1.sbatch). The environment is the
MiniGPT-v2 one described in [`../README.md`](../README.md#setup).

## Inputs and outputs

- **Inputs, all named by absolute paths:** the LLaMA-2-7B-chat weights ([`../llama_weights/`](../llama_weights/)),
  the MiniGPT-v2 stage-2 checkpoint `checkpoints/checkpoint_stage2.pth` (not in the repository;
  [`../../v2/infra/hamilton_setup_stage1.sh`](../../v2/infra/hamilton_setup_stage1.sh) downloads it), and the
  records. The lab configs use the 2159 records of
  [`stage2_train_7class_fixed.json`](../plant_diagnostic/datasets/stage2_train_7class_fixed.json);
  the split v1 config uses `plant_diagnostic/datasets/split_v1/stage2_{train,val}_prompted.json`, which are not in the
  repository and are built by [`../../v2/vlm/filter_stage2_records.py`](../../v2/vlm/filter_stage2_records.py),
  [`../../v2/vlm/detect_parts.py`](../../v2/vlm/detect_parts.py) and
  [`../../v2/vlm/build_stage2_prompts.py`](../../v2/vlm/build_stage2_prompts.py).
- **Validation records.** The lab configs have none, so before training the runner moves a random 10 % of the 2159
  records into a validation split (`auto_val_split_ratio: 0.1`) and trains on the rest; training mode never uses that
  split (see Outputs). The split v1 config brings its own validation records.
- **Code the split v1 config needs:** the `identity_text` processor in
  [`../minigpt4/processors/blip_processors.py`](../minigpt4/processors/blip_processors.py) and the validation-loss
  `evaluation()` in [`../minigpt4/tasks/image_text_pretrain.py`](../minigpt4/tasks/image_text_pretrain.py), both added
  in the same commit.
- **Outputs:** the runner writes `<output_dir>/<job_name>/` with `checkpoint_<epoch>.pth` after every epoch. It
  validates only when `evaluate` is true; in training mode it ignores `val_splits` and saves no
  `checkpoint_best.pth` ([`../minigpt4/runners/runner_base.py`](../minigpt4/runners/runner_base.py)). For the split
  v1 run the epoch with the lowest validation loss was chosen afterwards by
  [`../../v2/vlm/minigpt_val_loss.py`](../../v2/vlm/minigpt_val_loss.py) and copied to `checkpoint_best.pth`
  ([`../../v2/slurm/select_and_report.sbatch`](../../v2/slurm/select_and_report.sbatch); result
  [`minigpt_splitv1_val_loss.json`](../../evaluation/paper_data/minigpt_splitv1_val_loss.json), training log
  [`minigpt_splitv1_train_log.txt`](../../evaluation/paper_data/minigpt_splitv1_train_log.txt), both in
  [`../../evaluation/paper_data/`](../../evaluation/paper_data/)).

## Known problems

- **Comments that do not match the code.** In `minigptv2_strawberry.yaml`, `val_splits` is commented "validation runs
  each epoch", and the header of `minigptv2_strawberry_splitv1.yaml` says the validation records select the
  checkpoint through `evaluation()`; with `evaluate: false` neither happens during training (lab notebook,
  30 October 2026). The split v1 header also names `cluster/build_stage2_prompts.py`, now
  [`../../v2/vlm/build_stage2_prompts.py`](../../v2/vlm/build_stage2_prompts.py).
- **Prompt template.** None of the three files sets `prompt_template`, so training wraps each instruction in
  `Human: {} Assistant: ` from [`../configs/models/minigpt_v2.yaml`](../configs/models/minigpt_v2.yaml); the
  application asks in its `[INST] ... [/INST]` conversation format ([`../configs/README.md`](../configs/README.md)).
- **Paths.** Every path is absolute: the lab server's `/data/AGAI/MiniGPT-4/...` in the two lab files, the cluster's
  `/projects/.../agai/...` in the split v1 file. Edit them before training elsewhere ([Paths](../README.md#paths)).
