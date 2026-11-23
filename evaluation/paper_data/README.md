# `evaluation/paper_data/`: result files behind the journal manuscript

The result files that the journal manuscript comparing v1 (the original system, 2025 to April 2026) with v2 (the 2026
revival) relies on, added from 31 October to 22 November 2026. All were produced during the revival: some measure v1
components on split v1 (the lab's ResNet-50 recipe, its RF-DETR Small checkpoint and April 2026 report writer, and v1
as deployed), the others measure v2 or compare the two. [`docs/MANUSCRIPT_QA.md`](../../docs/MANUSCRIPT_QA.md) gives
the numbers taken from them item by item, and the lab notebook [`v2/EXPERIMENT_LOG.md`](../../v2/EXPERIMENT_LOG.md)
records each run.

In the last column, section, table and figure numbers are the manuscript's, as docs/MANUSCRIPT_QA.md cites them;
"notebook (s)" marks a use that only entry (s) of the lab notebook records. The latency sample is 49 split v1 test
photographs (7 per condition, seed 0); the report sample is the 151 split v1 test photographs that carry part masks.

| File | Holds | Written by | Used for |
|---|---|---|---|
| `calib_resnet50_v1_summary.json` | calibration, acceptance-gate and per-class metrics of the lab's ResNet-50 recipe (the v1 classifier) on the 398 split v1 test photographs, per seed (5) and pooled: accuracy, macro-F1, ECE and AURC with and without temperature scaling, coverage and accuracy of the deployed v1 gate and of a gate chosen on validation, confusion matrices, risk-coverage curves, bootstrap intervals | [`v2/classifier/calibration_metrics.py`](../../v2/classifier/calibration_metrics.py) | 3.3; Table S7 (the v1 column) |
| `calib_resnet50_v1_test_seed0.json` to `calib_resnet50_v1_test_seed4.json` (5 files) | the per-photograph records behind it, one file per seed: label, prediction, probabilities with and without temperature, top-1 probability, both gate decisions | same | 3.3 |
| `calib_siglip2_recipe_summary.json` | the same metrics for the SigLIP 2 recipe (the v2 classifier), 3 seeds | same | 3.3; Table S7 (the chosen classifier) |
| `calib_siglip2_recipe_test_seed0.json` to `calib_siglip2_recipe_test_seed2.json` (3 files) | per-photograph records, one file per seed; [`paper/figure_scripts/calib_to_eval.py`](../../paper/figure_scripts/calib_to_eval.py) pools them for Fig. S9 | same | 3.3; Fig. S9 |
| `detector_split_v1.json` | the detector split that agrees with split v1: each of the 1,075 annotated images assigned to train (845), validation (67) or test (163) by perceptual-hash match to the split v1 photographs, with match distances and counts per part and condition | [`v2/detector/make_detector_split.py`](../../v2/detector/make_detector_split.py) | 2.6, 3.5; [Detector](../../docs/DETECTOR.md) |
| `faithfulness_v1splitv1_v2_claims.json` | round 2's blind claim set (lab notebook (aj), (am)): the 731 sentences of the retrained v1's reports (480) and of v2's reports (251, the same sentences as round 1), pooled and shuffled per photograph with the seed prefix `faith-v1splitv1-v2` | [`v2/vlm/make_faithfulness_claims.py`](../../v2/vlm/make_faithfulness_claims.py) `--v1 evaluation/paper_data/latency_v1_splitv1.json --seed-prefix faith-v1splitv1-v2` | the leak-free v1 comparison; ratings pending |
| `faithfulness_v1v2_claims.json` | the blind claim set: the 696 sentences of the v1 and v2 reports on the latency sample (v1 445, v2 251), pooled and shuffled per photograph | [`v2/vlm/make_faithfulness_claims.py`](../../v2/vlm/make_faithfulness_claims.py) `--claims-out` | 3.9, claim-level faithfulness |
| `faithfulness_v1v2_key.json` | the key that un-blinds the claims: claim to system and sentence, photograph to labels and the v2 student's verdict | same, `--key-out` | 3.9 |
| `faithfulness_v1v2_ratings.json` | 573 blind ratings of the claims by one rater, 4 to 12 November 2026 | export of the blind rating page's database; no script in the repository | 3.9, primary analysis |
| `faithfulness_v1v2_ratings_2026-11-17.json` | the same 573 ratings and 78 made on 17 November 2026 (651) | same | 3.9, sensitivity analysis |
| `faithfulness_v1v2_scores.txt` | primary scoring: the 41 photographs with every claim rated (`--complete-only`) | printed by [`v2/vlm/score_faithfulness.py`](../../v2/vlm/score_faithfulness.py) | 3.9, primary analysis |
| `faithfulness_v1v2_scores_all_ratings.txt` | scoring with every rating of 12 November (42 photographs) | same | notebook (w) |
| `faithfulness_v1v2_scores_2026-11-17.txt` | sensitivity scoring: the 46 photographs complete on 17 November (`--complete-only`) | same | 3.9, sensitivity analysis |
| `faithfulness_v1v2_scores_2026-11-17_all_ratings.txt` | scoring with all 651 ratings (47 photographs) | same | notebook (ac) |
| `honesty_verified_comparisons.txt` | label-swap honesty of the verified run of the 8B student and of seven other runs (the first run of the same student, the student from the base 8B and five zero-shot VLMs) on the split v1 test photographs, with paired McNemar tests | printed by [`v2/vlm/score_honesty.py`](../../v2/vlm/score_honesty.py) | [VLM honesty](../../docs/VLM_HONESTY.md) results table; Table S12 and 3.6 (notebook (s)) |
| `latency_rfdetr.json` | detector latency and GPU memory, in-process, on the latency sample: the lab's RF-DETR Small (v1) and RF-DETR Large retrained on the detector split (v2) | [`v2/detector/latency_rfdetr.py`](../../v2/detector/latency_rfdetr.py) | 3.9, latency table |
| `latency_siglip2.json` | latency, GPU memory and per-photograph predictions of the SigLIP 2 classifier (v2); the predictions are the labels handed to the student | [`v2/classifier/latency_siglip.py`](../../v2/classifier/latency_siglip.py) | 3.9, latency table |
| `latency_student.json` | latency, GPU memory and batched throughput of the distilled Qwen3-VL-8B student under vLLM (v2), with its 49 reports | [`v2/vlm/latency_student.py`](../../v2/vlm/latency_student.py) | 3.9, latency table; v2's reports for the faithfulness ratings |
| `latency_v1_splitv1.json` | v1's pipeline as deployed run with the models retrained on split v1 (ResNet-50, RF-DETR Small and report writer named in its `settings`): time per stage and the 49 reports of round 2 | [`v2/vlm/latency_v1.py`](../../v2/vlm/latency_v1.py) `--resnet ... --writer-ckpt ...` with `RFDETR_CHECKPOINT` (job 82899, `v2/slurm/v1_splitv1_reports.sbatch`) | the leak-free v1 comparison, notebook (aj), (am) |
| `latency_v1.json` | v1 as deployed (the path of `v1/demo_v5.py`): time per stage and GPU memory, with v1's 49 reports | [`v2/vlm/latency_v1.py`](../../v2/vlm/latency_v1.py) | 3.9, latency table; v1's reports for the faithfulness ratings |
| `latency_v1_v2.json` | the assembled table: median, 90th percentile and mean seconds per stage and pipeline, GPU memory per process, load times | [`v2/vlm/latency_summary.py`](../../v2/vlm/latency_summary.py) | 3.9, latency table; abstract and conclusion (22 s against 62 s per photograph) |
| `minigpt_splitv1_train_log.txt` | configuration and per-epoch training loss of the v1 report writer (MiniGPT-v2 with LoRA) retrained on the split v1 training photographs, 10 epochs | the run's `log.txt`, written by [`v1/train.py`](../../v1/train.py) with `v1/train_configs/minigptv2_strawberry_splitv1.yaml` ([`v2/slurm/train_minigpt_splitv1.sbatch`](../../v2/slurm/train_minigpt_splitv1.sbatch)) | 2.1, 2.7 |
| `minigpt_splitv1_val_loss.json` | validation loss of the 10 epoch checkpoints on the 212 validation records; the last epoch has the lowest | [`v2/vlm/minigpt_val_loss.py`](../../v2/vlm/minigpt_val_loss.py) | 2.1, 2.7 (checkpoint selection) |
| `mirror_dupes_v1.json` | mirror-aware near-duplicate audit of split v1 (pHash within 8 and dHash within 10 bits, of the image or its mirror, after EXIF orientation): 74 pairs; 10 test photographs with a near-copy in training or validation, 3 validation photographs with one in training | [`v2/benchmark/mirror_dupes.py`](../../v2/benchmark/mirror_dupes.py) | the manuscript's near-duplicate audit, notebook (s) |
| `parts_lab_small_on_splitv1_valid.json`, `parts_lab_small_on_splitv1_test.json` | per-part AP (COCO), image-level part presence and a threshold sweep of the lab's RF-DETR Small checkpoint (v1) on the detector split's 67 validation and 163 test images; test mAP@50:95 0.681, raised by near-copies of the test images in its training data | [`v2/detector/eval_parts.py`](../../v2/detector/eval_parts.py) | 3.5; 2.6 (validation files) |
| `parts_small_splitv1_valid.json`, `parts_small_splitv1_test.json` | the same for RF-DETR Small retrained on the detector split; test mAP@50:95 0.564 | same | Table S9, 3.5; 2.6 |
| `parts_large_splitv1_valid.json`, `parts_large_splitv1_test.json` | the same for RF-DETR Large retrained on the detector split (v2); test mAP@50:95 0.593 | same | Table S9, 3.5; 2.6 |
| `reports_april_writer.json` | 302 reports of the lab's April 2026 report writer (v1) on the report sample, each photograph with and without the detector's part lists in the prompt | [`v2/vlm/minigpt_reports.py`](../../v2/vlm/minigpt_reports.py) | Table S16 (the leaky reference row) |
| `reports_april_writer_scores.json` | the Table S16 measures of these reports per prompt condition: part-hallucination rate, condition consistency, regeneration rate, length, false parts | [`v2/vlm/score_reports.py`](../../v2/vlm/score_reports.py) | Table S16 |
| `reports_splitv1.json` | 302 reports on the same photographs and prompts from the writer retrained on split v1 | `v2/vlm/minigpt_reports.py` | Table S16 |
| `reports_splitv1_scores.json` | their Table S16 measures | `v2/vlm/score_reports.py` | Table S16 |
| `scores_resnet50_512_s<seed>_<runtime or squash>_split_v1_test.json` for seeds 1, 2, 3, 4 and 42 (10 files) | per-image class probabilities of ResNet-50 trained with the lab's recipe at 512 pixels, on the split v1 test photographs, under the deployed preprocessing (`runtime`: resize and centre crop) and the training one (`squash`); the resolution control | [`v2/classifier/resnet_scores.py`](../../v2/classifier/resnet_scores.py) `--img-size 512` (the validation files stayed on the cluster) | [Classifier](../../docs/CLASSIFIER.md), fine-tuned backbones table; 3.3 (notebook (t)) |
| `sensitivity_mirror_v1.txt` | accuracy and macro accuracy of 22 split v1 result files with and without the 10 test photographs that the mirror audit flags (post hoc) | printed by [`v2/benchmark/sensitivity_mirror.py`](../../v2/benchmark/sensitivity_mirror.py) | notebook (s) |
| `permutation_holm_v1.txt`, `permutation_holm_v1.json` | seed-averaged paired permutation tests of the classifier comparisons and synthetic-data arms, and Holm-adjusted P values within each family of the manuscript's comparisons (post hoc) | [`v2/benchmark/permutation_holm.py`](../../v2/benchmark/permutation_holm.py), run on the cluster | Supplementary Table S19, 2.8, 3.4, Table S5; notebook (ak) |
| `sensitivity_subsets_v1.txt` | ResNet-50 (v1 recipe, 5 seeds) against SigLIP 2 (recipe, 3 seeds) on the split v1 test set with and without the label-audit flags, the web-sourced file names and the 3 defective photographs, with seed-matched McNemar tests (post hoc) | printed by [`v2/benchmark/sensitivity_subsets.py`](../../v2/benchmark/sensitivity_subsets.py) from the `calib_*_test_seed*.json` records here | notebook (ah) |

## Where the files come from

The scripts write their results to `evaluation/<name>` relative to the working directory, so on the cluster they
landed in `evaluation/` of the working copy; the files kept for the manuscript were then copied into this folder (lab
notebook, entries of 31 October to 17 November 2026). The exceptions:

- `v2/vlm/latency_summary.py` writes `evaluation/paper_data/latency_v1_v2.json` itself. It too ran on the cluster, and
  its output was copied here with the four stage files (entry (l), 31 October 2026).
- The training log is the `log.txt` that the MiniGPT-v2 training code writes into the run's output folder on the
  cluster.
- `make_faithfulness_claims.py` writes the claim set and the key only to paths given with `--claims-out` and
  `--key-out`; the claim set was added here on 4 November 2026, the key with the ratings on 12 November 2026.
- The two ratings files are exports of the blind rating page's database; no script in the repository writes them.
- The `.txt` files other than the training log are saved printouts of the scripts named in the table.

## Using the files

Run the scripts in a folder built by `bash v2/infra/make_workdir.sh DEST` ([how](../README.md#using-the-files)).
There `evaluation/paper_data/` is a link to this folder, so whatever a script writes into it lands here.

- `v2/vlm/make_faithfulness_claims.py` reads `latency_v1.json` and `latency_student.json` here by default, and
  `--check` compares the claim set it rebuilds with `faithfulness_v1v2_claims.json`.
- The other scripts take these files as arguments: the claims, key and ratings for `score_faithfulness.py`; report
  files for `score_reports.py`, which writes `<input>_scores.json` next to its input, that is, here; per-image result
  files for `v2/benchmark/sensitivity_mirror.py` and `v2/benchmark/mcnemar_pair.py`; the calibration records for
  `paper/figure_scripts/calib_to_eval.py`.
- `v2/vlm/latency_summary.py` reads the four stage files from `evaluation/`, not from here, so copy them into
  `DEST/evaluation/` first; the script then rewrites `latency_v1_v2.json` here.

From the committed files these reproduce exactly (checked on 18 November 2026): the four
`faithfulness_v1v2_scores*.txt` files (`score_faithfulness.py`, with `--complete-only` for the two without
`all_ratings` in the name), the claim set (`make_faithfulness_claims.py --check`), both `reports_*_scores.json` files
(`score_reports.py`) and `latency_v1_v2.json` (`latency_summary.py`).

## Known problems

- Most inputs of these files are not in the repository: the per-image score files that the calibration summaries were
  computed from, the annotation runs behind `honesty_verified_comparisons.txt`, the 22 result files behind
  `sensitivity_mirror_v1.txt`, the part lists and report test set behind the reports, and every model checkpoint.
  Fields inside the files (`source`, `seeds`, `thresholds_from`, `run`, `weights`) give those inputs' paths on the
  cluster. Most of the files therefore cannot be regenerated from the repository alone; their per-image records can be
  rescored.
- The per-image files list photographs as `plant_diagnostic/data/...`, relative to `v1/`; 74 of the 398 split v1 test
  photographs are not in this repository's copy ([why](../README.md#photographs-not-in-this-repository)).
- The list of result files at the end of docs/MANUSCRIPT_QA.md names the 20 files of 31 October 2026. The 36 added
  from 4 November to 22 November are all named in the lab notebook; the Q&A's items also name the 8 calibration record
  files and some of the faithfulness and report files.
