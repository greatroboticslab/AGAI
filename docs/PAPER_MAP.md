[← README](../README.md) · [v1](../v1/README.md) · [v2](../v2/README.md) · [Documentation](README.md)

# Where v1 and v2 are in the journal manuscript

The 2026 journal manuscript compares two versions of AGAI on the leak-free split v1:

- **v1**, the original system ([`v1/`](../v1/README.md)): ResNet-50 classifier, RF-DETR Small part detector,
  MiniGPT-v2 report writer. In the manuscript it is the **baseline**, re-measured with the revival's code on split v1;
  none of its numbers come from the April 2026 technical report.
- **v2**, the revival's system ([`v2/`](../v2/README.md)): SigLIP 2 classifier, RF-DETR Large part detector,
  Qwen3-VL-8B student that can reject a wrong diagnosis, and the studies that chose them.

Every result in the manuscript was produced by the code in [`v2/`](../v2/README.md); the per-image result files are in
[`evaluation/paper_data/`](../evaluation/paper_data/README.md), and each run is in the lab notebook,
[`v2/EXPERIMENT_LOG.md`](../v2/EXPERIMENT_LOG.md). Figure and table numbers follow the manuscript build of
17 November 2026 (34 pages); they may shift as the draft on Overleaf changes. The figure scripts are in
[`paper/figure_scripts/`](../paper/figure_scripts/README.md).

## At a glance

| | Figures | Tables |
|---|---|---|
| **v1 results** (baseline) | Fig. 3 (leakage in v1's April evaluation), Fig. 4 and Fig. 5b (v1 ResNet-50 rows) | S4 and S5 (v1 ResNet-50 rows), S7 (v1 ResNet-50 calibration and gate), S9 (v1 RF-DETR Small), **S16** (v1 report writer), **S17** (v1 end to end) |
| **v2 results** | Fig. 4 (SigLIP 2 chosen), **Fig. 5**, **Fig. 6**, **Fig. 7**, **Fig. S9** | S4, S5, **S6**, S7 (SigLIP 2), **S8**, S9 (RF-DETR Large), **S10**, **S11**, **S12**, **S13**, **S14**, **S15**, **S17** (v2 end to end) |
| **v1 described** (how it works, no results) | Fig. 1, Figs. S1, S5, S6, S7, S8 | S2, S3 |
| **Data and context** | Fig. 2, Figs. S2, S3, S4 | Table 1, S1, S18 |

Bold: the item is only about that version. Section 3.9 (End-to-end evaluation), the abstract and the conclusion
compare v1 with v2 directly: accuracy, honesty, latency and memory (Table S17) and claim-level faithfulness.

## Every figure and table

v1 = reports v1 results; v2 = reports v2 results; both = puts v1 and v2 side by side.

### Figures

| Item | What it shows | Version | Code | Result files |
|---|---|---|---|---|
| Fig. 1 | architecture: classifier, gate, detector, report writer, consistency check; the v2 models named beside the v1 ones | both (description) | `paper/figure_scripts/fig_system_architecture.py` | none |
| Fig. 2 | example photographs of the seven conditions | data | `fig_class_examples.py` | none |
| Fig. 3 | leakage in v1's April 2026 evaluation (near-duplicates, test images used for checkpoint selection, warm starts) and split v1 | v1 audit; panel d is v2's benchmark | [`v2/benchmark/`](../v2/benchmark/README.md): `near_dupes.py`, `clean_eval_sets.py`, `make_split.py`; `fig_leakage_audit.py` | [`evaluation/split_v1.json`](../evaluation/split_v1.json), [`evaluation/label_audit.csv`](../evaluation/label_audit.csv) |
| Fig. 4 | test accuracy on split v1: the v1 ResNet-50 against fine-tuned VLMs, fine-tuned backbones (SigLIP 2, chosen for v2), ensembles and zero-shot VLMs | both | [`v2/classifier/`](../v2/classifier/README.md): `train_backbone.py`, `resnet_scores.py`, `ensemble.py`; [`v2/vlm/`](../v2/vlm/README.md): `train_vlm_lora.py`, `eval_vlm_scores.py`, `zeroshot_qwen3vl.py`; `fig_classifier_comparison.py` | `evaluation/paper_data/scores_resnet50_512_*` (resolution control); other per-image files stayed on the cluster (numbers in the notebook) |
| Fig. 5 | synthetic data and robustness of SigLIP 2 (clean and 128-pixel accuracy per training arm); panel b also the ResNet-50 with the lab's April synthetic set | v2 (panel b both) | [`v2/synthetic/`](../v2/synthetic/README.md), `v2/classifier/eval_lowq.py`, `lowq_table.py`, `eval_probe.py`; `fig_synthetic_robustness.py` | on the cluster (numbers in the notebook) |
| Fig. 6 | explanations that can reject a wrong diagnosis: the label-swap test of zero-shot VLMs and the distilled 8B student | v2 | `v2/vlm/annotate_vllm.py`, `score_honesty.py`, `train_vlm_reports.py`; `fig_honesty.py` | [`evaluation/paper_data/honesty_verified_comparisons.txt`](../evaluation/paper_data/honesty_verified_comparisons.txt) |
| Fig. 7 | claim-level faithfulness of four open captioners on 38 hard test photographs, and hallucination filters | v2 | `v2/vlm/caption_bakeoff.py`, `judge_claims.py`; `fig_faithfulness.py` | [`evaluation/bakeoff_set_v1.json`](../evaluation/bakeoff_set_v1.json); the rating tallies are with the authors (Section 2.15) |
| Fig. S1 | training pipelines of the three v1 models as retrained for split v1 | v1 (description) | `fig_training_pipelines.py` | none |
| Fig. S2 | composition of the collected data | data | `fig_dataset_composition.py` | none |
| Fig. S3 | plant-part annotation in Roboflow | data | `fig_annotation_workflow.py` | none |
| Fig. S4 | plant-part instance masks | data | `fig_part_annotations.py` | none |
| Fig. S5 | v1 inference for one photograph | v1 (description) | not in `paper/figure_scripts/` | none |
| Fig. S6 | v1 report generation (MiniGPT-v2, instruction data, prompt, consistency check) | v1 (description) | `fig_report_generation.py` | none |
| Fig. S7 | visible signs listed in the knowledge base | v1 (description) | `fig_symptom_overlap.py` | none |
| Fig. S8 | the knowledge base | v1 (description) | `fig_knowledge_base.py` | none |
| Fig. S9 | errors and calibration of the chosen v2 classifier (SigLIP 2 recipe), with the operating point of v1's gate thresholds | v2 | `v2/classifier/calibration_metrics.py`; `calib_to_eval.py`, `fig_classifier_results.py` | `evaluation/paper_data/calib_siglip2_recipe_*` |

### Tables

| Item | What it shows | Version | Code | Result files |
|---|---|---|---|---|
| Table 1 | conditions, causes and visible symptoms used as labeling criteria | data | none | none |
| S1 | likely origin of the real images per condition | data | none | none |
| S2 | training settings of the three v1 models in the split v1 re-evaluation | v1 (description) | `v1/recovered/.../train_resnet_straw5.py`, `v1/rfdetr_parts/rfdetr_config.yaml`, `v1/train_configs/minigptv2_strawberry_splitv1.yaml` | none |
| S3 | contents of the knowledge base | v1 (description) | `v1/knowledge_graph/disease_knowledge_base.json` | none |
| S4 | frozen backbones with a logistic-regression head, against the v1 ResNet-50 | both | `v2/classifier/probe_backbones.py` | on the cluster |
| S5 | fine-tuned classifiers and ensembles, including the v1 ResNet-50 (lab recipe) and the 512-pixel control | both | `v2/classifier/train_backbone.py`, `resnet_scores.py`, `ensemble.py`; `v2/vlm/train_vlm_lora.py` | `evaluation/paper_data/scores_resnet50_512_*` |
| S6 | the size shortcut of SigLIP 2 | v2 | `v2/classifier/eval_shortcut.py` | on the cluster |
| S7 | calibration and acceptance gate: v1 ResNet-50 and the SigLIP 2 recipe | both | `v2/classifier/calibration_metrics.py` | `evaluation/paper_data/calib_*` |
| S8 | synthetic data and robustness (SigLIP 2) | v2 | `v2/synthetic/`, `v2/classifier/eval_lowq.py`, `lowq_table.py` | on the cluster |
| S9 | plant-part detection on the clean detector test split: v1's RF-DETR Small (lab checkpoint and retrained) and v2's RF-DETR Large | both | [`v2/detector/`](../v2/detector/README.md): `make_detector_split.py`, `eval_parts.py` | `evaluation/paper_data/detector_split_v1.json`, `parts_*` |
| S10 | label-swap honesty: zero-shot VLMs, teachers and the distilled students | v2 | `v2/vlm/annotate_vllm.py`, `score_honesty.py`, `train_vlm_reports.py` | `evaluation/paper_data/honesty_verified_comparisons.txt` |
| S11 | captioners and their interim blind ratings | v2 | `v2/vlm/caption_bakeoff.py` | tallies with the authors (Section 2.15) |
| S12 | hallucination filters on 562 rated claims | v2 | `v2/vlm/judge_claims.py`, `caption_bakeoff.py` | tallies with the authors (Section 2.15) |
| S13 | majority voting over sampled zero-shot diagnoses | v2 | `v2/vlm/consensus_dx.py` | tallies with the authors (Section 2.15) |
| S14 | resolving disagreements between two zero-shot models | v2 | `v2/vlm/resolve_dx.py` | on the cluster |
| S15 | answer-format ablation for zero-shot diagnosis | v2 | `v2/vlm/consensus_dx.py` (`--prompt-style`) | on the cluster |
| S16 | detector on/off ablation of the v1 report writer, retrained on split v1 | v1 | `v2/vlm/filter_stage2_records.py`, `build_stage2_prompts.py`, `minigpt_val_loss.py`, `minigpt_reports.py`, `score_reports.py`, with v1's `train.py` | `evaluation/paper_data/reports_*`, `minigpt_splitv1_*` |
| S17 | latency and GPU memory of v1 and v2 end to end | both | `v2/vlm/latency_v1.py`, `latency_student.py`, `latency_summary.py`, `v2/classifier/latency_siglip.py`, `v2/detector/latency_rfdetr.py`, `v2/slurm/latency_v1_v2.sbatch` | `evaluation/paper_data/latency_*` |
| S18 | AGAI and related plant-health systems | context | none | none |

Section 3.9 also reports claim-level faithfulness of v1 and v2 reports on the same 49 photographs (unsupported visual
claims, one blind rater): `v2/vlm/make_faithfulness_claims.py` and `score_faithfulness.py`, result files
`evaluation/paper_data/faithfulness_v1v2_*`.
