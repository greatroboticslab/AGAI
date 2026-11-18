# `v1/plant_diagnostic/`: photographs, instruction data and classifier code of v1

The labeled strawberry photographs, the instruction data of the report writer and the 2025 scripts that trained and
evaluated v1's ResNet-50 classifier (v1, 2025 to April 2026). The v2 revival (October to November 2026) reads the
photographs and the instruction data from here.

## Contents

| Path | Contents |
|---|---|
| [`data/`](data/) | 3,719 labeled photographs: `train/` (1,540 in 7 classes), `train_aug/` (2,159: the `train/` photographs plus 620 augmented copies), `holdout/` (20 in 5 classes); `aug_report.json` |
| [`datasets/`](datasets/) | instruction data of the report writer: the 2,159 image-report records (`stage2_train_7class_fixed.json`, and the same records as `.jsonl`) and a 99-record question-answer set |
| [`jsondatasets/`](jsondatasets/) | earlier five-class instruction data (617 training and 68 validation records, and copies of them) |
| [`scripts/`](scripts/) | ResNet-50 training, evaluation, cross-validation and confusion audit (`resnetscripts/`); the instruction-data generator (`gptscripts/`); a JSONL-to-JSON converter |
| [`src/`](src/) | a second ResNet-50 trainer, folder prediction, an inference module and a dataset loader |
| [`tools/`](tools/) | `augment_dataset.py`, the offline augmentation whose output matches `data/train_aug/` |
| [`xtra/`](xtra/) | spreadsheets of web sources and a draft letter to farms; no code reads them |

Expected here but not in the repository: `models/` (classifier checkpoints; `demo_v5.py` loads
`models/resnet_strawberry.pth`, and git ignores `*.pth` files) and `datasets/split_v1/` (the report-writer records that
v2 builds for split v1 with `v2/vlm/filter_stage2_records.py`).

The April 2026 versions of several scripts, from the lab server, are in
[`../recovered/labbox_2026-10-19/plant_diagnostic/`](../recovered/labbox_2026-10-19/plant_diagnostic/); its
`train_resnet_straw5.py` trained every ResNet-50 that the v2 revival trained for its paper
([its README](../recovered/labbox_2026-10-19/README.md)).

## Working directory

The scripts resolve nothing from their own location, and their defaults assume different working directories:

- `v1/`: `scripts/resnetscripts/train_resnet_straw5.py` (defaults `plant_diagnostic/data/train`,
  `plant_diagnostic/models/resnet_straw5.pth`) and `src/resnet_classifier.py` (default
  `plant_diagnostic/models/resnet_straw7.pth`).
- `v1/plant_diagnostic/`: `src/train_resnet_straw5.py`, `src/predict_folder.py`,
  `scripts/resnetscripts/eval_resnet_straw5.py`, `scripts/resnetscripts/audit_confusions.py` and the smoke test of
  `src/strawberry_stage2_loader.py` (defaults `data/...` and `models/...`).
- No path defaults (all paths are arguments): `scripts/resnetscripts/crossval_resnet_straw.py`,
  `scripts/gptscripts/json_generation/generate_json_openai.py`, `scripts/convert_jsonl_to_json.py`,
  `tools/augment_dataset.py`. The examples in their docstrings use lab-server paths (`/data/AGAI/MiniGPT-4/...`) or old
  folder names.

## What uses it

- **v1.** `demo_v5.py` loads `models/resnet_strawberry.pth`. The MiniGPT configs
  (`minigpt4/configs/datasets/strawberry_diagnostic.yaml`, `eval_configs/minigptv2_eval.yaml`,
  `train_configs/minigptv2_strawberry.yaml` and `_v2.yaml`) name `datasets/stage2_train_7class_fixed.json`,
  `jsondatasets/strawberry_stage2_val.json`, `data/train_aug/` and `data/holdout/` by lab-server paths
  (`/data/AGAI/MiniGPT-4/plant_diagnostic/...`). `hallucination_evaluation/config.py` reads `datasets/`, `data/` and
  `models/` relative to `v1/`; the scripts in `v1/paper/scripts/` count and open the photographs in `data/`.
- **v2.** [`../../evaluation/split_v1.json`](../../evaluation/split_v1.json) and the other files in
  [`../../evaluation/`](../../evaluation/) name photographs as `plant_diagnostic/data/<folder>/<class>/<file>`, relative
  to `v1/` or to the folder that [`v2/infra/make_workdir.sh`](../../v2/infra/make_workdir.sh) builds. 292 of split v1's
  1,580 paths are not in this copy of the photographs ([`data/README.md`](data/README.md)).
  `v2/vlm/filter_stage2_records.py` and `v2/benchmark/minigpt_train_overlap.py` read
  `datasets/stage2_train_7class_fixed.json`; several v2 scripts load checkpoints from `models/`.
