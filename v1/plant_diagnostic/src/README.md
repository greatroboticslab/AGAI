# `v1/plant_diagnostic/src/`: ResNet-50 trainer, prediction and loaders

Four 2025 modules of v1's classifier work (v1, 2025 to April 2026): a ResNet-50 trainer, folder prediction, an inference
module and a dataset loader for the instruction data. The application and the v2 scripts use none of them: they import
[`v1/resnet_classifier.py`](../../resnet_classifier.py). The lab server's April 2026 versions of `predict_folder.py` and
`resnet_classifier.py` are in
[`v1/recovered/labbox_2026-10-19/plant_diagnostic/src/`](../../recovered/labbox_2026-10-19/plant_diagnostic/src/).

## Contents

| File | What it does |
|---|---|
| [`train_resnet_straw5.py`](train_resnet_straw5.py) | a variant of [`../scripts/resnetscripts/train_resnet_straw5.py`](../scripts/resnetscripts/train_resnet_straw5.py) with the same augmentation and two-phase fine-tune; it takes one `--data_root` and splits off `--val_frac` (default 0.1) for validation (the `scripts/` version also accepts a separate `--val_root`); saves the class list and a fitted temperature in the checkpoint |
| [`predict_folder.py`](predict_folder.py) | predicts one photograph or every photograph under a folder: four views (centre crops at three scales and a mirrored crop), temperature-scaled probabilities, and a decision `top1` (top-1 probability at least a per-class threshold, 0.78 by default), `top2` (top two together at least 0.86) or `unsure`; writes the rows to `--out` as JSON |
| [`resnet_classifier.py`](resnet_classifier.py) | inference module with `load_resnet` and `diagnose_or_none`, the function names of `v1/resnet_classifier.py`, but with other thresholds, device choice (`$RESNET_DEVICE`) and default checkpoint (`$RESNET_CKPT` or `plant_diagnostic/models/resnet_straw7.pth`); nothing imports this copy |
| [`strawberry_stage2_loader.py`](strawberry_stage2_loader.py) | PyTorch dataset over instruction JSON with `data/train/<class>/<file>` paths, such as [`../jsondatasets/strawberry_stage2.json`](../jsondatasets/strawberry_stage2.json): each item is the photograph, its label (the class folder in the path) and the conversation; `build_loader` adds optional class-weighted sampling |

## Running

Defaults assume `v1/plant_diagnostic/` as the working directory:

```bash
cd v1/plant_diagnostic
python src/train_resnet_straw5.py --data_root data/train --out models/resnet_straw5.pth   # the defaults
python src/predict_folder.py --ckpt models/<checkpoint>.pth --images data/holdout --out predictions.json
python src/strawberry_stage2_loader.py   # smoke test, see Known problems
```

`predict_folder.py` defaults to `--ckpt models/resnet_straw5_prod.pth` and `--out predictions.json`. Checkpoints are
not in the repository. The lab's session note
[`../../documentation/session_2024_12_17.md`](../../documentation/session_2024_12_17.md) runs the trainer from `v1/`
with explicit paths (`--data_root plant_diagnostic/data/train_aug --out plant_diagnostic/models/resnet_strawberry.pth`).

## Known problems

- The smoke test of `strawberry_stage2_loader.py` reads `datasets/strawberry_stage2.json`; the file is
  `jsondatasets/strawberry_stage2.json`, so the test stops with a missing-file error unless the path is edited.
- `resnet_classifier.py` here is not the module the application loads; changes to it have no effect on `demo_v5.py`.
