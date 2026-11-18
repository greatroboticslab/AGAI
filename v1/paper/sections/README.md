# `v1/paper/sections/`: sections of the April 2026 technical report

The eight section files of v1's technical report (25 April 2026), input in this order by [`../main.tex`](../main.tex)
(v1, 2025 to April 2026). The tables and figures they include (`generated/...`, `figures/generated/...`) are paths
relative to `v1/paper/`, the folder LaTeX runs in.

## Contents

| File | Section | Includes |
|---|---|---|
| [`01_introduction.tex`](01_introduction.tex) | Introduction | |
| [`02_related_work.tex`](02_related_work.tex) | Related Work: plant disease classification, calibration, detection transformers and visible-part grounding, vision-language models and parameter-efficient adaptation, retrieval-augmented generation, hallucination, local inference | |
| [`03_data_and_split.tex`](03_data_and_split.tex) | Dataset and Split Protocol: classes and split roots, holdout by physical relocation, class-floor balancing, external-image audit | table `dataset_counts_table`; figures `dataset_composition`, `holdout_examples_grid` |
| [`04_system.tex`](04_system.tex) | System Architecture: classifier with test-time augmentation and calibration, acceptance gate, detector-grounded prompt, retrieval and hallucination retry, report-writer configuration, dual-report supervision policy | figure `pipeline_overview` |
| [`05_results_status.tex`](05_results_status.tex) | Experimental Results: the classifier on the 70-photograph holdout (acceptance gate, thresholds, v3 against v4), the RF-DETR detector, the dual-report pilot, pending experiments | tables `resnet_summary_table`, `bootstrap_ci_table`, `resnet_per_class_table`, `acceptance_gate_table`, `selective_accuracy_table`, `mcnemar_table`, `rfdetr_holdout_table`, `dual_report_table`; figures `resnet_v4_confusion_matrix`, `resnet_accuracy_calibration`, `threshold_sweep`, `risk_coverage`, `rfdetr_holdout_heatmap`, `dual_report_pilot_summary` |
| [`06_discussion.tex`](06_discussion.tex) | Discussion, including threats to validity | |
| [`07_conclusion.tex`](07_conclusion.tex) | Conclusion | |
| [`08_appendix.tex`](08_appendix.tex) | Appendix: project status, classifier, detector and report-writer training details, runtime decision rule, scope of the ambiguity policy, misclassified holdout photographs, reproducibility manifest | tables `status_table`, `misclassification_table`; figure `misclassification_grid` |

Tables are in [`../generated/`](../generated/) and figures in [`../figures/generated/`](../figures/generated/).

## Known problems

- The appendix's reproducibility manifest names paths relative to the lab server's working copy (the layout of `v1/`),
  a commit that is not in this repository (`6eaa403e...`) and files that are not in it (the two checkpoints and
  `evaluation/stage2_pilot/`); see [`../README.md`](../README.md#reproducibility-manifest).
