# `v1/paper/scripts/`: scripts that write the April 2026 report's tables and figures

Seven Python scripts of v1's technical report (25 April 2026; v1, 2025 to April 2026). Each finds the repository as
`Path(__file__).resolve().parents[2]`, which in this layout is `v1/`: they read `v1/evaluation/holdout/`,
`v1/plant_diagnostic/` and `v1/rfdetr_parts/`, and write into [`../generated/`](../generated/) and
[`../figures/generated/`](../figures/generated/), whatever the working directory.

## Contents

| Script | Reads | Writes |
|---|---|---|
| [`build_paper_assets.py`](build_paper_assets.py) | `evaluation/holdout/resnet_v4_release10holdout_full_holdout.json`; `evaluation/stage2_pilot/dual_report_focus20_tight/dual_report_focus20_tight_summary.json` and `..._audit.jsonl` (not in the repository); the photographs in `plant_diagnostic/data/{train,train_aug,holdout}/<class>/` (counted; one holdout photograph per class shown) | figures `pipeline_overview`, `dataset_composition`, `holdout_examples_grid`, `resnet_v4_confusion_matrix`, `resnet_accuracy_calibration`, `dual_report_pilot_summary`; tables `dataset_counts_table`, `resnet_summary_table`, `resnet_per_class_table`, `status_table`, `dual_report_table` |
| [`compute_extra_metrics.py`](compute_extra_metrics.py) | the v4 holdout file | tables `bootstrap_ci_table`, `acceptance_gate_table`, `misclassification_table` |
| [`compute_threshold_sweep.py`](compute_threshold_sweep.py) | the v4 holdout file | figures `threshold_sweep`, `risk_coverage`; table `selective_accuracy_table` |
| [`compute_mcnemar.py`](compute_mcnemar.py) | the v3 and v4 holdout files | table `mcnemar_table` |
| [`render_misclass_grid.py`](render_misclass_grid.py) | the v4 holdout file, then each misclassified photograph at the absolute path stored in it | figure `misclassification_grid` |
| [`run_rfdetr_on_holdout.py`](run_rfdetr_on_holdout.py) | the photographs in `plant_diagnostic/data/holdout/<class>/` for all seven classes; `rfdetr_parts/output/checkpoint_best_total.pth` (not in the repository) | table `rfdetr_holdout_table`, `generated/rfdetr_holdout_summary.json` (not committed), figure `rfdetr_holdout_heatmap` |
| [`bench_resnet_latency.py`](bench_resnet_latency.py) | up to 70 photographs in `plant_diagnostic/data/holdout/`; `plant_diagnostic/models/resnet_strawberry_v4_release10holdout.pth` (not in the repository) | `generated/latency_summary.json` (not committed); not used by the report |

Figures are written as PDF and PNG; only the PDFs are committed.

## Running

From `v1/` (any working directory works). Environment: Python with numpy and matplotlib; Pillow for
`build_paper_assets.py` and `render_misclass_grid.py`; `compute_mcnemar.py` needs only the standard library;
`bench_resnet_latency.py` needs torch and torchvision; `run_rfdetr_on_holdout.py` needs the `rfdetr` package (the
`rfdetr` conda environment of v1).

```bash
cd v1
python paper/scripts/compute_extra_metrics.py
python paper/scripts/compute_threshold_sweep.py
python paper/scripts/compute_mcnemar.py
```

These three need only the two files in [`../../evaluation/holdout/`](../../evaluation/holdout/), and they reproduce
the five committed tables they write exactly (checked on 18 November 2026, in a copy of these folders). The full build
order is in [`../README.md`](../README.md).

## Known problems

- **Missing inputs.** `build_paper_assets.py` reads the dual-report pilot files first and stops, because
  `evaluation/stage2_pilot/` is not in the repository. `run_rfdetr_on_holdout.py` and `bench_resnet_latency.py` need
  checkpoints that are not in the repository.
- **Not the report's photographs.** `v1/plant_diagnostic/data/holdout/` holds 20 photographs in 5 classes; the report's
  holdout had 70 in 7. Even with the other inputs, `build_paper_assets.py` (holdout grid) and
  `run_rfdetr_on_holdout.py` would stop at the missing `gray_mold/` folder, and the counts behind
  `dataset_counts_table` and `dataset_composition` would differ from the report's.
- **Lab-server paths.** `render_misclass_grid.py` opens the photographs by the absolute paths in the v4 holdout file
  (`/data/AGAI/MiniGPT-4/plant_diagnostic/data/holdout/...`), so it runs only where that path exists.
- **Stale docstring.** `compute_threshold_sweep.py` lists `threshold_sweep_summary.tex` as an output; it writes
  `selective_accuracy_table.tex`.
- **Recovered companions.** Two more scripts of this folder from the lab server, `run_minigpt_on_holdout.py` and
  `score_minigpt_rubric.py`, are in
  [`../../recovered/labbox_2026-10-19/paper/scripts/`](../../recovered/labbox_2026-10-19/paper/scripts/); they also
  locate the repository as `parents[2]`, so they run only from this folder.
