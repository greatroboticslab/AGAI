# `v1/paper/generated/`: LaTeX tables of the April 2026 technical report

The 11 tables of v1's technical report (25 April 2026), written by the scripts in [`../scripts/`](../scripts/) and
included by the sections with `\input{generated/...}` (v1, 2025 to April 2026). They are committed with the report's
sources, so building the PDF does not require the scripts.

## Contents

| File | Written by | Included in | Contents |
|---|---|---|---|
| [`acceptance_gate_table.tex`](acceptance_gate_table.tex) | `compute_extra_metrics.py` | `05_results_status.tex` | per class: photographs accepted by the runtime gate and their accuracy (ResNet-50 v4 on the holdout) |
| [`bootstrap_ci_table.tex`](bootstrap_ci_table.tex) | `compute_extra_metrics.py` | `05_results_status.tex` | bootstrap 95 % intervals of top-1 and top-2 accuracy |
| [`dataset_counts_table.tex`](dataset_counts_table.tex) | `build_paper_assets.py` | `03_data_and_split.tex` | photographs per class in `train/`, `train_aug/` and `holdout/` on the lab server in April 2026 (1,506 / 2,424 / 70) |
| [`dual_report_table.tex`](dual_report_table.tex) | `build_paper_assets.py` | `05_results_status.tex` | statistics of the dual-report pilot (20 photographs, 36 generated records) |
| [`mcnemar_table.tex`](mcnemar_table.tex) | `compute_mcnemar.py` | `05_results_status.tex` | McNemar test of ResNet-50 v3 against v4 on the same 70 photographs |
| [`misclassification_table.tex`](misclassification_table.tex) | `compute_extra_metrics.py` | `08_appendix.tex` | the 8 holdout photographs ResNet-50 v4 gets wrong |
| [`resnet_per_class_table.tex`](resnet_per_class_table.tex) | `build_paper_assets.py` | `05_results_status.tex` | ResNet-50 v4 per-class top-1 with Wilson 95 % intervals, and top-2 |
| [`resnet_summary_table.tex`](resnet_summary_table.tex) | `build_paper_assets.py` | `05_results_status.tex` | ResNet-50 v4 summary: top-1, top-2, ECE, temperature |
| [`rfdetr_holdout_table.tex`](rfdetr_holdout_table.tex) | `run_rfdetr_on_holdout.py` | `05_results_status.tex` | per disease class, the share of holdout photographs in which RF-DETR Small detects each plant part |
| [`selective_accuracy_table.tex`](selective_accuracy_table.tex) | `compute_threshold_sweep.py` | `05_results_status.tex` | top-1 accuracy on the most confident 50, 70, 85 and 100 % of the holdout |
| [`status_table.tex`](status_table.tex) | `build_paper_assets.py` | `08_appendix.tex` | project status in April 2026 (rows written in the script) |

The scripts also write JSON files here that are not committed: `rfdetr_holdout_summary.json`
(`run_rfdetr_on_holdout.py`) and `latency_summary.json` (`bench_resnet_latency.py`). `.gitattributes` would store JSON
files in this folder as regular git files, not Git LFS.

## Regenerating

From `v1/`, `python paper/scripts/compute_extra_metrics.py`, `compute_mcnemar.py` and `compute_threshold_sweep.py`
rewrite five of these tables from [`../../evaluation/holdout/`](../../evaluation/holdout/) and reproduce the committed
files exactly (checked on 18 November 2026). The other six need inputs that are not in the repository
([`../scripts/README.md`](../scripts/README.md)).
