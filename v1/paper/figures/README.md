# `v1/paper/figures/`: figures of the April 2026 technical report

The figures of v1's technical report (25 April 2026), all in `generated/` as PDFs written by the scripts in
[`../scripts/`](../scripts/) (v1, 2025 to April 2026). The sections include them as `figures/generated/<name>.pdf`.
Each script also writes a PNG copy, which is not committed.

## Contents

| File (in `generated/`) | Written by | Included in | Shows |
|---|---|---|---|
| [`pipeline_overview.pdf`](generated/pipeline_overview.pdf) | `build_paper_assets.py` | `04_system.tex` | diagram of the v1 pipeline |
| [`dataset_composition.pdf`](generated/dataset_composition.pdf) | `build_paper_assets.py` | `03_data_and_split.tex` | photographs per class in `train/`, `train_aug/` and `holdout/` (lab server, April 2026) |
| [`holdout_examples_grid.pdf`](generated/holdout_examples_grid.pdf) | `build_paper_assets.py` | `03_data_and_split.tex` | the first holdout photograph of each class, by file name |
| [`resnet_v4_confusion_matrix.pdf`](generated/resnet_v4_confusion_matrix.pdf) | `build_paper_assets.py` | `05_results_status.tex` | ResNet-50 v4 confusion matrix on the 70-photograph holdout |
| [`resnet_accuracy_calibration.pdf`](generated/resnet_accuracy_calibration.pdf) | `build_paper_assets.py` | `05_results_status.tex` | ResNet-50 v4 per-class top-1 accuracy and reliability diagram |
| [`threshold_sweep.pdf`](generated/threshold_sweep.pdf) | `compute_threshold_sweep.py` | `05_results_status.tex` | acceptance rate, and accuracy on the accepted photographs, for a uniform threshold on the top-1 probability |
| [`risk_coverage.pdf`](generated/risk_coverage.pdf) | `compute_threshold_sweep.py` | `05_results_status.tex` | risk-coverage curve of ResNet-50 v4 |
| [`rfdetr_holdout_heatmap.pdf`](generated/rfdetr_holdout_heatmap.pdf) | `run_rfdetr_on_holdout.py` | `05_results_status.tex` | RF-DETR Small part-detection rate per disease class |
| [`dual_report_pilot_summary.pdf`](generated/dual_report_pilot_summary.pdf) | `build_paper_assets.py` | `05_results_status.tex` | dual-report pilot: selection rules, report modes, self-reported overclaim risk |
| [`misclassification_grid.pdf`](generated/misclassification_grid.pdf) | `render_misclass_grid.py` | `08_appendix.tex` | the 8 holdout photographs ResNet-50 v4 gets wrong, with true and predicted class |
| [`resnet_v2_confusion_matrix.pdf`](generated/resnet_v2_confusion_matrix.pdf), [`resnet_v3_confusion_matrix.pdf`](generated/resnet_v3_confusion_matrix.pdf) | no script in the repository | not included | confusion matrices of earlier checkpoints, left from earlier builds |

## Known problems

- Most figures cannot be regenerated from this repository: `build_paper_assets.py` needs the dual-report pilot files
  and holdout photographs of all seven classes, `render_misclass_grid.py` the photographs at their lab-server paths,
  and `run_rfdetr_on_holdout.py` the detector checkpoint ([`../scripts/README.md`](../scripts/README.md)). Only
  `compute_threshold_sweep.py` (`threshold_sweep`, `risk_coverage`) runs on committed files.
