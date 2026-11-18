# `v1/evaluation/holdout/`: ResNet-50 results on the April 2026 holdout

Measured evaluations of two of v1's ResNet-50 checkpoints, v3 and v4, on the 70-photograph holdout (10 per class) of
the April 2026 technical report, committed on 25 April 2026 (v1, 2025 to April 2026). They are the inputs of the
report's classifier tables and figures, and the reference of the v2 revival's reproduction of ResNet-50 v4.

## Contents

| File | Checkpoint (lab server) | Top-1 | Top-2 | ECE | Temperature |
|---|---|---:|---:|---:|---:|
| [`resnet_v3_release10holdout_full_holdout.json`](resnet_v3_release10holdout_full_holdout.json) | `resnet_strawberry_v3_balanced20holdout.pth` | 60/70 (85.7 %) | 65/70 (92.9 %) | 0.048 | 0.781 |
| [`resnet_v4_release10holdout_full_holdout.json`](resnet_v4_release10holdout_full_holdout.json) | `resnet_strawberry_v4_release10holdout.pth` | 62/70 (88.6 %) | 65/70 (92.9 %) | 0.086 | 0.751 |

Both use flip test-time augmentation and the same 70 photographs. Each file holds `summary` (the numbers above),
`per_class` (support, top-1 and top-2 accuracy with Wilson 95 % intervals, mean top-1 probability of right and wrong
predictions, the predicted classes of the class's photographs), `class_order`, `confusion_matrix` (rows are the true
class), `reliability` (10 confidence bins), `predictions` (per photograph: path, true class, top-1 class and
probability, top-2), `checkpoint`, `holdout_root`, `evaluated_classes` and two `notes` from the lab. The script that
wrote them is not in the repository. They are regular git files, not Git LFS (a rule in `.gitattributes`).

## What uses them

- [`../../paper/scripts/`](../../paper/scripts/): `build_paper_assets.py`, `compute_extra_metrics.py`,
  `compute_threshold_sweep.py` and `render_misclass_grid.py` read the v4 file, and `compute_mcnemar.py` reads both.
  They find this folder as `<repository>/evaluation/holdout/` with the repository taken as two folders above the
  script, which is `v1/`.
- [`v2/benchmark/repro_resnet_v4.py`](../../../v2/benchmark/repro_resnet_v4.py) compares a re-run of v4 with the v4
  file (`--results`, default `evaluation/holdout/...` relative to the working directory; in the folder built by
  `v2/infra/make_workdir.sh`, `evaluation/holdout/` is a link to this folder). Its output,
  `evaluation/holdout/repro_resnet_v4_hamilton.json`, is not in the repository, although
  `v2/benchmark/minigpt_diagnose.py`, `v2/benchmark/minigpt_train_overlap.py` and `v2/vlm/zeroshot_qwen3vl.py` read
  it. The revival's reproduction and its reasons for replacing this holdout are in
  [`docs/BENCHMARK.md`](../../../docs/BENCHMARK.md).
- The recovered lab-server script
  [`run_minigpt_on_holdout.py`](../../recovered/labbox_2026-10-19/paper/scripts/run_minigpt_on_holdout.py) reads the v4
  file, and the report's reproducibility manifest (`v1/paper/sections/08_appendix.tex`) names it.

## Known problems

- **Lab-server paths.** `checkpoint`, `holdout_root` and the 70 photo paths are absolute paths on the lab server
  (`/data/AGAI/MiniGPT-4/...`). Only 16 of the 70 photographs exist at the corresponding place in this repository
  (`v1/plant_diagnostic/data/holdout/`, which holds 20 photographs in 5 classes), and the checkpoints are not in the
  repository. `render_misclass_grid.py` opens the photographs by these absolute paths.
- **Names and notes.** The v3 file's name says `release10holdout`, its checkpoint `balanced20holdout`. Both files carry
  the same two notes: that mold-class metrics of "the pre-carve checkpoint" are biased by leakage because it was
  trained before the mold holdout was set aside, and that the non-mold classes have very small holdout support. Which
  checkpoint the first note means is not stated; both files have 10 photographs in every class.
