# `v1/plant_diagnostic/scripts/`: ResNet-50 and instruction-data scripts

The 2025 scripts for v1's ResNet-50 classifier (training, evaluation, cross-validation, confusion audit) and for the
report writer's instruction data (v1, 2025 to April 2026). The lab server's April 2026 versions of three of the ResNet
scripts, and its multi-caption generator, are in the recovered copy of this folder,
[`v1/recovered/labbox_2026-10-19/plant_diagnostic/scripts/`](../../recovered/labbox_2026-10-19/plant_diagnostic/scripts/);
the v2 revival trained its ResNet-50 models with that folder's `train_resnet_straw5.py`.

## Contents

| File | What it does |
|---|---|
| [`resnetscripts/train_resnet_straw5.py`](resnetscripts/train_resnet_straw5.py) | fine-tunes an ImageNet ResNet-50 on class folders: head only, then selected layers; class-weighted loss or a weighted sampler, label smoothing, optional MixUp and EMA; random 90 to 100 % crops and random erasing in training, resize and centre crop for validation; fits a temperature on the validation set and saves it with the class list in the checkpoint |
| [`resnetscripts/eval_resnet_straw5.py`](resnetscripts/eval_resnet_straw5.py) | accuracy, per-class precision, recall and F1, and the confusion matrix of a checkpoint on a class-folder tree; prints them, writes no file |
| [`resnetscripts/audit_confusions.py`](resnetscripts/audit_confusions.py) | copies each misclassified photograph into `<out_dir>/<true>_to_<predicted>/` and prints the confusion matrix and per-class metrics |
| [`resnetscripts/crossval_resnet_straw.py`](resnetscripts/crossval_resnet_straw.py) | stratified k-fold training on one class-folder tree; writes `fold<j>.pth` and `summary.json` to `--out_dir`, with optional temperature fitting per fold |
| [`gptscripts/json_generation/generate_json_openai.py`](gptscripts/json_generation/generate_json_openai.py) | the instruction-data generator: sends every photograph under `--img-root/<class>/` with its class label to an OpenAI vision model (`--model`, default `o3` or `$OPENAI_VISION_MODEL`) and appends one image-report record per photograph to `--out` (JSONL); skips photographs already in the output, so an interrupted run resumes; writes a templated fallback report when a call fails; logs to `logs/generate_stage2_reports_7class.log` |
| [`gptscripts/json_generation/generate_stage2_jsonl.py`](gptscripts/json_generation/generate_stage2_jsonl.py) | empty file (0 bytes in every commit) |
| [`convert_jsonl_to_json.py`](convert_jsonl_to_json.py) | converts a JSONL file into a JSON array: `python convert_jsonl_to_json.py <input.jsonl> <output.json>` |

## Running

The scripts with options list them with `--help`. Defaults and the working directory they assume:

| Script | Working directory | Defaults |
|---|---|---|
| `resnetscripts/train_resnet_straw5.py` | `v1/` | `--train_root plant_diagnostic/data/train`, `--val_root` empty (a stratified `--val_frac 0.1` of the training root), `--out plant_diagnostic/models/resnet_straw5.pth`, `--img_size 256`, `--epochs_head 5`, `--epochs_full 15`, `--seed 42` |
| `resnetscripts/eval_resnet_straw5.py` | `v1/plant_diagnostic/` | `--ckpt models/resnet_straw5.pth`; `--data_root` required |
| `resnetscripts/audit_confusions.py` | `v1/plant_diagnostic/` | `--ckpt models/resnet_straw5.pth`, `--data_root data/holdout`, `--out_dir misclassified` |
| `resnetscripts/crossval_resnet_straw.py` | any | `--data_root` and `--out_dir` required; `--k 5` |
| `gptscripts/json_generation/generate_json_openai.py` | any | `--img-root` and `--out` required; `--sleep 0.6`; needs `OPENAI_API_KEY` in the environment (never in a file) |

The 2,159 records in [`../datasets/`](../datasets/) have the record format of `generate_json_openai.py`, run on
`data/train_aug/` with absolute paths and converted with `convert_jsonl_to_json.py` (commands in
[`v1/README.md`](../../README.md)); their reports start with a display name of the class (`Diagnosis: Gray mold.`) where
the committed generator writes the folder name (`Diagnosis: gray_mold.`). Checkpoints go to `models/`, which is not in
the repository.

## Known problems

- **Stale examples.** The usage example in `generate_json_openai.py` names
  `scripts/minigpttrainingscripts/json_generation/`, a folder that does not exist; the example in
  `crossval_resnet_straw.py` calls it `train_resnet_kfold.py` and uses lab-server paths.
- **Empty generator.** [`../jsondatasets/how_these_were_made`](../jsondatasets/how_these_were_made) points to
  `generate_stage2_jsonl.py`, which is empty.
- **Not the revival's recipe.** The recovered `train_resnet_straw5.py` resizes to a square without cropping and has no
  random erasing; results of the v2 revival's ResNet-50 runs do not come from the version here.
