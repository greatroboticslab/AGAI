# `v2/classifier/`: image-classifier studies on split v1

Part of v2, the lab's 2026 revival of AGAI (October-November 2026). Linear probes and cross-validation of frozen
vision backbones, fine-tunes with the lab's ResNet recipe (a fine-tuned SigLIP 2 is v2's classifier), ensembles, tests
on degraded and shrunk photographs, ResNet-50 scoring, calibration and acceptance gates, the mirrored-view ablation, and
the classifier's latency for the v1 versus v2 comparison.

Results: [Classifier](../../docs/CLASSIFIER.md), [Synthetic data](../../docs/SYNTHETIC_DATA.md) (low-quality
augmentation and the shortcut probe), the low-quality test suite in
[Benchmark](../../docs/BENCHMARK.md#low-quality-test-suite), and [Manuscript Q&A](../../docs/MANUSCRIPT_QA.md)
(calibration and the gate in 3.3, the mirrored view in 4, latency in 3.9). Every run is recorded in the lab notebook,
[`../EXPERIMENT_LOG.md`](../EXPERIMENT_LOG.md), under the scripts' flat names (`cluster/train_backbone.py`).

## Files

| File | Purpose | Environment | Inputs → outputs (defaults) |
|---|---|---|---|
| [`backbones.py`](backbones.py) | shared module, no command line: registry of 12 vision backbones (weight path, input size, normalization), `load()` returning the model and a pooled-feature function, `enable_grad_checkpointing()` | imported | weights under `/projects/$USER/agai/models/`, loaded offline |
| [`calibration_metrics.py`](calibration_metrics.py) | calibration (ECE, AURC, risk-coverage), the deployed v1 acceptance gate and a gate chosen on validation, per-class metrics and bootstrap intervals, with and without temperature, from per-image probability files | `vision` (inferred) | `--name` and `--pairs VAL_JSON:TEST_JSON ...` (one pair per seed, both required), `--reps 10000` → `evaluation/calib_<name>_test_seed<k>.json`, `evaluation/calib_<name>_summary.json`, table on stdout |
| [`cv_probes.py`](cv_probes.py) | 5-fold grouped, stratified cross-validation of linear probes on saved features over all 1,576 split v1 photographs (near-duplicate groups never straddle folds), equal-weight ensembles, paired McNemar | `vision` (inferred) | `--backbones` (required), `--ensembles`, `evaluation/split_v1.json`, `evaluation/near_dupes.json`†, `/projects/$USER/agai/features/<name>.npz` → `evaluation/cv5_probe_<name>.json` |
| [`ensemble.py`](ensemble.py) | equal-weight and validation-tuned probability ensembles, with exact McNemar against each member | `vision` (inferred) | positional `NAME=VAL_JSON,TEST_JSON ...` → stdout; with `--out NAME`: `evaluation/ensemble_<NAME>_{equal,tuned}_split_v1_test.json` |
| [`eval_lowq.py`](eval_lowq.py) | fine-tuned backbones on the split v1 test photographs degraded in five fixed ways (clean, low_res, very_low, defocus, noisy_dark) | `vision` | `--models` (required; `/projects/$USER/agai/models/<name>.pt`), `--split evaluation/split_v1.json`, `--set test` → `evaluation/lowq_<name>.json` |
| [`eval_probe.py`](eval_probe.py) | fine-tuned backbones on synthetic images they did not train on, broken down by the manifest's prompt fields (group, hand, quality, severity) | `vision` | `--models`, `--manifest` (both required; `evaluation/synth_<name>_manifest.json`†), `--only evaluation/split_<tree>.json`† → `evaluation/probe_<model>_<manifest name>[_heldout].json` |
| [`eval_shortcut.py`](eval_shortcut.py) | resolution-shortcut test: fine-tuned backbones on the split v1 test photographs as they are and shrunk to `--side` 275 px, with and without JPEG, keeping all class probabilities | `vision` (inferred) | `--models` (required), `--split evaluation/split_v1.json`, `--set test` → `evaluation/shortcut_<name>.json` |
| [`external_test.py`](external_test.py) | the external test of lab notebook (al): v1's ResNet-50 (application preprocessing, two views) and v2's SigLIP 2 recipe, each with v1's hand-set gate and its validation-chosen gate, on a COCO-annotated public set; groups from the annotation categories (gray mold and asymptomatic plants in scope, other diseases out of scope); recall, share accepted and accuracy among accepted per group | `vision` | `--images` folder with `_annotations.coco.json`, `models/resnet50_split_v1.pth`, `models/<--siglip>.pt`, `evaluation/calib_*_summary.json` → `--out` JSON |
| [`external_test_summary.py`](external_test_summary.py) | tables and prediction checks of the external test, read from its result file | stdlib | positional result JSON → stdout |
| [`latency_siglip.py`](latency_siglip.py) | latency and GPU memory of the fine-tuned SigLIP 2 classifier on the 49 seeded split v1 test photographs (7 per class) that all four `latency_*.py` scripts share; also records its predictions, which `vlm/latency_student.py` passes to the report writer | `vision` | `--model` (required; `/projects/$USER/agai/models/<model>.pt`), `--split-json evaluation/split_v1.json`, `--seed 0`, `--per-class 7` → `--out evaluation/latency_siglip2.json` |
| [`lowq_table.py`](lowq_table.py) | mean ± sd accuracy per degradation over seeds, and per-class accuracy at two levels, from `eval_lowq.py` outputs | stdlib | positional `BASE ARM ...` (each arm a comma-separated list of suffixes); `evaluation/lowq_<BASE><suffix>.json`† → stdout |
| [`probe_backbones.py`](probe_backbones.py) | linear probes on frozen features of each backbone (C chosen on validation macro accuracy); saves the features and per-image probabilities | `vision` (inferred) | `--only` (six backbones by default), `--split evaluation/split_v1.json` → `/projects/$USER/agai/features/<name>.npz`, `evaluation/probe_<name>_split_v1_{val,test}.json` |
| [`resnet_scores.py`](resnet_scores.py) | per-class probabilities of ResNet-50 checkpoints on split v1 validation and test under two preprocessings (runtime resize and centre crop; squash), with v1's two views and temperature; `--img-size N` for other input sizes | `minigptv` (inferred) | positional checkpoint paths, `evaluation/split_v1.json` → `evaluation/scores_<ckpt>_{runtime,squash}_split_v1_{val,test}.json` |
| [`train_backbone.py`](train_backbone.py) | fine-tunes a backbone and linear head on split v1 with the lab's ResNet recipe (v1's augmentations, layer-wise learning-rate decay, temperature fitted on validation, two-view test); `--train-root` swaps in another training tree, `--lowq-aug P` degrades training photographs | `vision` | `--backbone` (required), `--tag s0`, `--seed 0`, `--split evaluation/split_v1.json`, `--train-root` → `evaluation/ft_<backbone>_<tag>_split_v1_{val,test}.json`, `/projects/$USER/agai/models/ft_<backbone>_<tag>.pt` |
| [`tta_ablation.py`](tta_ablation.py) | one view against two views (photograph and mirror) for ResNet-50 checkpoints on split v1 test, with the predictions the mirror fixes and breaks | `minigptv` (inferred) | positional checkpoint paths, `evaluation/split_v1.json` → `evaluation/tta_<ckpt>_split_v1_test.json` |

Environment: the venv under `/projects/$USER/agai/envs/` that a Slurm file in [`../slurm/`](../slurm/) or the
script's docstring names; "(inferred)" means no file names one and the entry is an environment that has the packages
the script imports; stdlib means standard library only (any Python 3). † not in this repository.

## Running

The scripts open every path relative to the working directory, as in the cluster's working copy. Build that folder
with [`../infra/make_workdir.sh`](../infra/make_workdir.sh) (details in
[`../infra/README.md`](../infra/README.md#working-copy)):

```bash
bash v2/infra/make_workdir.sh /projects/$USER/agai-work    # once, from the root of a clone of this repository
source /projects/$USER/agai/agai_env.sh    # cluster variables, then a cd to /projects/$USER/agai/MiniGPT-4
cd /projects/$USER/agai-work
E=/projects/$USER/agai/envs
$E/vision/bin/python cluster/classifier/eval_shortcut.py --models ft_siglip2-so400m-512_s0          # on a GPU node
$E/minigptv/bin/python cluster/classifier/tta_ablation.py /projects/$USER/agai/models/resnet50_split_v1*.pth
python3 cluster/classifier/lowq_table.py ft_siglip2-so400m-512 _s0,_s1,_s2 _labsynth_s0,_labsynth_s1,_labsynth_s2
sbatch cluster/slurm/train_backbone.sbatch --backbone siglip2-so400m-512 --tag s0                  # A100 partition
sbatch cluster/slurm/eval_lowq.sbatch --models ft_siglip2-so400m-512_s0,ft_siglip2-so400m-512_s1
```

A Slurm job runs in `/projects/$USER/agai/MiniGPT-4` whatever folder it was submitted from; see
[`../slurm/README.md`](../slurm/README.md#where-jobs-run).

## Dependencies

- **Sibling module.** `eval_lowq.py`, `eval_probe.py`, `eval_shortcut.py`, `latency_siglip.py`, `probe_backbones.py`
  and `train_backbone.py` import `backbones.py` after putting their own folder on `sys.path`, so these seven files must
  stay together. `backbones.py` reads `USER` from the environment when it is imported.
- **v1 modules.** `resnet_scores.py` and `tta_ablation.py` import `resnet_classifier`
  ([`v1/resnet_classifier.py`](../../v1/resnet_classifier.py)) from the working directory and use its private names
  (`_tfm`, `_CLASSES`, `_TEMPERATURE`, `_DEVICE`, `_normalize_label`).
- **Text copied from v1** (no import): `calibration_metrics.py`'s gate thresholds (`V1_GATE`), and `train_backbone.py`'s
  augmentations, taken from
  [`train_resnet_straw5.py`](../../v1/plant_diagnostic/scripts/resnetscripts/train_resnet_straw5.py).
- **Files from and for other studies.** `cv_probes.py` needs the output of `benchmark/near_dupes.py`, and its outputs
  feed `benchmark/label_audit.py` and `benchmark/mold_tossups.py`. `eval_probe.py` reads the manifests of
  `synthetic/import_synth.py` and the split files of `synthetic/build_train_tree.py`; `train_backbone.py --train-root`
  takes the trees that `synthetic/` builds. The output of `latency_siglip.py` is read by `vlm/latency_student.py` and
  `vlm/latency_summary.py`. `ensemble.py` combines probability files of `resnet_scores.py`, `probe_backbones.py` and
  `vlm/eval_vlm_scores.py`; `calibration_metrics.py` reads those of `resnet_scores.py` and `train_backbone.py`;
  `benchmark/leaderboard.py` picks up the test files of `probe_backbones.py`, `train_backbone.py`, `ensemble.py` and the
  squash files of `resnet_scores.py`.

## Known problems

- **Defaults and the repository.** The scripts write to the top level of `evaluation/`; the files the manuscript uses
  were copied by hand to [`evaluation/paper_data/`](../../evaluation/paper_data/): `calib_*` from
  `calibration_metrics.py`, `latency_siglip2.json` from `latency_siglip.py`, and the test files of the 512-pixel
  ResNet-50 control (`scores_resnet50_512_*`) from `resnet_scores.py`.
- **Inputs never committed.** `evaluation/near_dupes.json` (`cv_probes.py`); the probability files that
  `calibration_metrics.py` paired (the committed summaries name
  `evaluation/scores_resnet50_*_runtime_split_v1_{val,test}.json` and
  `evaluation/ft_siglip2-so400m-512_chatgpt_nof3_lq_s*_split_v1_{val,test}.json`); `evaluation/lowq_*.json`; the
  synthetic manifests; every checkpoint and feature file under `/projects/$USER/agai/`.
- **Photographs.** The scripts open split v1's photographs, 292 of which are not in this repository (see
  [`../benchmark/README.md`](../benchmark/README.md#known-problems)).
- **`calibration_metrics.py`** splits each `--pairs` item at `:`, so Windows paths with a drive letter do not work. Its
  `V1_GATE` (healthy 0.40, drought 0.65, root rot 0.65, frost 0.70, others 0.60) equals `_CLASS_THRESH` in
  [`v1/demo_v5.py`](../../v1/demo_v5.py); the comment beside it names `resnet_classifier.py`, whose `_CLASS_THRESH`
  holds other values (drought 0.70, frost 0.75, gray mold 0.40, white mold 0.45).
- **Flat names.** `latency_siglip.py`'s docstring shows `python cluster/latency_siglip.py`; the flat name and
  `cluster/classifier/latency_siglip.py` both work in the working copy.
