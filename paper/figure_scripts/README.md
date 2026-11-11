# Figure scripts for the 2026 paper

Every figure in the 2026 paper is drawn by one of these Python scripts with matplotlib. `agstyle.py` holds the shared style (fonts, colors, the colorblind-checked class palette) and the save helper, and each `fig_*.py` writes one figure as a PDF and a PNG.

| Script | Figure |
|---|---|
| `fig_system_architecture.py` | Fig. 1, system architecture |
| `fig_class_examples.py` | Fig. 2, example photographs of the seven conditions |
| `fig_leakage_audit.py` | Fig. 3, leakage audit and split v1 |
| `fig_classifier_comparison.py` | Fig. 4, classifier comparison |
| `fig_synthetic_robustness.py` | Fig. 5, synthetic data and robustness |
| `fig_honesty.py` | Fig. 6, label-swap honesty test |
| `fig_faithfulness.py` | Fig. 7, claim-level faithfulness and filters |
| `fig_training_pipelines.py` | Supplementary Fig. S1, training pipelines |
| `fig_dataset_composition.py` | Supplementary Fig. S2, dataset composition |
| `fig_annotation_workflow.py` | Supplementary Fig. S3, plant-part annotation workflow |
| `fig_part_annotations.py` | Supplementary Fig. S4, plant-part annotation examples |
| `fig_report_generation.py` | Supplementary Fig. S6, report generation |
| `fig_symptom_overlap.py` | Supplementary Fig. S7, overlap of visible signs |
| `fig_knowledge_base.py` | Supplementary Fig. S8, knowledge base |
| `fig_classifier_results.py` | Supplementary Fig. S9, calibration and confusion matrix of the SigLIP 2 recipe |
| `calib_to_eval.py` | helper for Fig. S9: pools the per-image files `evaluation/paper_data/calib_siglip2_recipe_test_seed*.json` into the input of `fig_classifier_results.py` |
| `fig_pending.py` | placeholder panels for results that are still running |

Some figures place photographs from the dataset (Figs. 1 to 3 and Supplementary Figs. S1, S4 and S6); everything else is drawn from the numbers in the scripts or from the result files they read. The paths in the scripts point to local copies of the lab data and results, so they need adjusting before the scripts run on another machine.
