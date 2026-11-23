# `v2/benchmark/`: split v1, its leakage and label audits, and the comparison tables

Part of v2, the lab's 2026 revival of AGAI (October-November 2026). These scripts built split v1, the revival's
leak-free benchmark ([`evaluation/split_v1.json`](../../evaluation/split_v1.json)), audited the photographs for copies
and wrong labels, and re-ran the v1 baselines (ResNet-50 v4, MiniGPT-v2) on fair test sets; the others turn per-image
result files into tables with paired tests.

Results and protocol: [Benchmark](../../docs/BENCHMARK.md); the v1 reproduction and the error analysis are in
[Classifier](../../docs/CLASSIFIER.md). Every run is recorded in the lab notebook,
[`../EXPERIMENT_LOG.md`](../EXPERIMENT_LOG.md), which names these scripts by their flat working-copy paths
(`cluster/make_split.py` for `cluster/benchmark/make_split.py`).

## Files

| File | Purpose | Environment | Inputs → outputs (defaults) |
|---|---|---|---|
| [`clean_eval_sets.py`](clean_eval_sets.py) | which labeled originals are fair test data for which model: every deduplicated original for zero-shot VLMs (`vlm_all_originals`), and those whose content (SHA-256) is not in ResNet v4's paper-era training data (`resnet_v4_clean`) | system `python3` | `plant_diagnostic/data/{holdout,train}/`, git refs `refs/labbox/stash` and `refs/labbox/stash^3`† → `evaluation/clean_eval_sets.json` |
| [`compare_results.py`](compare_results.py) | one accuracy table per evaluation set (`split_v1_test`, `resnet161`, `all1580`) with exact paired McNemar tests; `--drop-leaks` also drops `resnet161` photographs with a near-copy in ResNet v4's training data | stdlib | `evaluation/minigpt_lab_train_overlap.json`†, `evaluation/split_v1.json`, `evaluation/near_dupes.json`† (with `--drop-leaks`), the result files named in `RUNS`† (missing ones are skipped) → stdout |
| [`error_sheet.py`](error_sheet.py) | contact sheet of the photographs a result file gets wrong, with each listed member's prediction | `vision` (inferred) | positional `RESULT_JSON OUT_IMAGE [NAME=RESULT_JSON ...]` and the photographs they name → `OUT_IMAGE` |
| [`eval_resnet_set.py`](eval_resnet_set.py) | a ResNet checkpoint scored with v1's `resnet_classifier.diagnose_or_none` (image and mirror, checkpoint temperature) on `--set resnet161` or `all1580`, or on a split given with `--split-json` and `--split test` | `minigptv` (inferred) | `--ckpt plant_diagnostic/models/resnet_strawberry_v4_release10holdout.pth`†, `evaluation/clean_eval_sets.json`† (without `--split-json`) → `evaluation/resnet_<ckpt>_<set>.json` |
| [`label_audit.py`](label_audit.py) | ranks all 1,576 split v1 labels by how likely they are wrong, from the out-of-fold probabilities of three frozen-backbone probes and the Qwen3-VL-32B annotator's verdict on the true label | `vision` (inferred) | `evaluation/cv5_probe_<backbone>.json`† (from [`../classifier/cv_probes.py`](../classifier/cv_probes.py)), `--teacher /projects/$USER/agai/annotations/v1_Qwen3-VL-32B-Instruct` (from [`../vlm/annotate_vllm.py`](../vlm/annotate_vllm.py)) → `evaluation/label_audit.csv` (committed), `--sheet /projects/$USER/agai/logs/label_audit_top.jpg` |
| [`leaderboard.py`](leaderboard.py) | split v1 test leaderboard (398 photographs) of every recognised result file: accuracy, macro accuracy, per-class counts, exact McNemar against the ResNet-50 lab recipe (seed 42), mean ± sd over seeds | stdlib | `evaluation/split_v1.json`, result files at the top level of `evaluation/` (file-name patterns in `SOURCES`) → stdout; `--md FILE` |
| [`make_manifest.py`](make_manifest.py) | SHA-256 manifest of the evaluation files and of the split v1 photographs present in the repository; a Git LFS pointer contributes its oid, the SHA-256 of the photograph | stdlib | `evaluation/*.json`, `evaluation/*.csv`, `evaluation/paper_data/*`, `evaluation/split_v1.json` and the photographs (relative to `v1/`) → `--out evaluation/MANIFEST.sha256`; run from the repository root |
| [`make_split.py`](make_split.py) | builds split v1 (seed 0): near-duplicate groups stay whole, groups with conflicting labels are excluded, per class at least 25 % test and 10 % validation | stdlib | `evaluation/clean_eval_sets.json`†, `evaluation/near_dupes.json`† → `evaluation/split_v1.json`, symbolic-link trees `/projects/$USER/agai/splits/v1/{train,val,test}/<class>/` |
| [`mcnemar_pair.py`](mcnemar_pair.py) | exact two-sided McNemar test between two result files on the photographs they share | stdlib | positional `A.json B.json [A.json B.json ...]` → stdout |
| [`minigpt_diagnose.py`](minigpt_diagnose.py) | MiniGPT-v2 checkpoints (official stage 2 and stage 3, the lab's fine-tune) scored as 7-class diagnosers on `paper68` and `resnet161`, with the prompt of [`../vlm/zeroshot_qwen3vl.py`](../vlm/zeroshot_qwen3vl.py) | `minigptv` | `checkpoints/checkpoint_stage2.pth`†, `checkpoints/minigptv2_stage3.pth`†, `output/minigpt_strawberry/checkpoint_best.pth`† (missing ones are skipped), `evaluation/holdout/repro_resnet_v4_hamilton.json`†, `evaluation/clean_eval_sets.json`†, `evaluation/minigpt_lab_train_overlap.json`†; optional run names → `evaluation/minigpt_<run>_<set>.json` |
| [`minigpt_train_overlap.py`](minigpt_train_overlap.py) | which test photographs (`paper68`, `resnet161`) the lab's MiniGPT-v2 fine-tune (checkpoint of 18 December 2025) trained on, by LFS content hash | system `python3` | `plant_diagnostic/datasets/stage2_train_7class_fixed.json`, git ref `origin/main`, `evaluation/holdout/repro_resnet_v4_hamilton.json`†, `evaluation/clean_eval_sets.json`† → `evaluation/minigpt_lab_train_overlap.json` |
| [`mirror_dupes.py`](mirror_dupes.py) | mirror-aware near-duplicate audit of split v1 with the manuscript's rule: pHash within 8 and dHash within 10 bits, for the photograph or its mirror, after EXIF orientation | `tools` (inferred) | `--split evaluation/split_v1.json`, `--root .` → `--out evaluation/mirror_dupes_v1.json` |
| [`mold_tossups.py`](mold_tossups.py) | gray-mold and white-mold photographs to which the cross-validated probe ensemble gives the other mold at least `--min-other` 0.15, most contested first, for a label review | `vision` (inferred) | `evaluation/cv5_probe_<backbone>.json`†, `evaluation/label_audit.csv` → `--out DIR` (required): `tossups.csv`, `tossups_NN.jpg` |
| [`near_dupes.py`](near_dupes.py) | perceptual-hash groups (pHash within `--max-dist` 6 bits) over `plant_diagnostic/data/{train,holdout,train_aug}`, groups filed under two labels, and `resnet161` photographs with a near-copy in ResNet v4's paper-era training data | `tools` (inferred) | the photographs, git refs `refs/labbox/stash` and `refs/labbox/stash^3`†, `evaluation/clean_eval_sets.json`† → `evaluation/near_dupes.json` |
| [`repro_resnet_v4.py`](repro_resnet_v4.py) | re-runs the paper's ResNet-50 v4 on the paper's 70-photograph holdout and compares it photograph by photograph with v1's saved results | `minigptv` | `--results evaluation/holdout/resnet_v4_release10holdout_full_holdout.json` (in [`v1/evaluation/holdout/`](../../v1/evaluation/holdout/)), `--ckpt plant_diagnostic/models/resnet_strawberry_v4_release10holdout.pth`†, `--extra-dir` → `--out evaluation/holdout/repro_resnet_v4_hamilton.json` |
| [`sensitivity_mirror.py`](sensitivity_mirror.py) | accuracy and macro accuracy of result files with and without the test photographs that `mirror_dupes.py` flags | stdlib | positional `AUDIT_JSON RESULT_JSON ...` → stdout (the committed run: [`sensitivity_mirror_v1.txt`](../../evaluation/paper_data/sensitivity_mirror_v1.txt)) |
| [`permutation_holm.py`](permutation_holm.py) | seed-averaged paired permutation tests (100,000 sign flips, bootstrap intervals) for the classifier comparisons and the synthetic-data arms, and Holm's correction within each family, also for the P values the manuscript reports for honesty, test-time compute, the report writer and faithfulness | `vision` (numpy) | the per-image result files in `evaluation/` (patterns in `MODELS`; most stayed on the cluster), `evaluation/split_v1.json` → `--out` `.txt` and `.json` (the committed run: [`permutation_holm_v1.txt`](../../evaluation/paper_data/permutation_holm_v1.txt)) |
| [`external_overlap.py`](external_overlap.py) | near-copies between an external, COCO-annotated photograph set and split v1, with the benchmark's rule (pHash within 8 bits, dHash within 10, mirror-aware), per external category | `vision` (imagehash) | `--images` folder, `evaluation/split_v1.json`, the split's photographs → `--out` JSON (the committed run: [`external_mendeley_overlap.json`](../../evaluation/paper_data/external_mendeley_overlap.json)) |
| [`holm_table_tex.py`](holm_table_tex.py) | the LaTeX of Supplementary Table S19 from `permutation_holm_v1.json` | stdlib | positional JSON path → stdout |
| [`sensitivity_subsets.py`](sensitivity_subsets.py) | ResNet-50 (v1 recipe) against SigLIP 2 (recipe) on the split v1 test set without the label-audit flags, the web-sourced file names, the 3 defective photographs, and all of them, with seed-matched McNemar tests | `scipy` | `evaluation/paper_data/calib_*_test_seed*.json`, `evaluation/label_audit.csv` → `--out evaluation/paper_data/sensitivity_subsets_v1.txt`; run from the repository root |
| [`split_coverage.py`](split_coverage.py) | which split v1 photographs are in the repository's copy, per set and condition, and the list of those that are not | stdlib | `--split evaluation/split_v1.json`, `--root v1` → `--out evaluation/split_v1_coverage.json` |

Environment: the venv under `/projects/$USER/agai/envs/` that the script's docstring names; "(inferred)" means no file
names one and the entry is an environment that has the packages the script imports (`imagehash` is in none of the
four frozen environments, see [`../infra/README.md`](../infra/README.md#known-problems)); stdlib means standard
library only (any Python 3). † not in this repository.

## Running

The scripts open every path relative to the working directory, as in the cluster's working copy. Build that folder
with [`../infra/make_workdir.sh`](../infra/make_workdir.sh) (details in
[`../infra/README.md`](../infra/README.md#working-copy)):

```bash
bash v2/infra/make_workdir.sh /projects/$USER/agai-work    # once, from the root of a clone of this repository
source /projects/$USER/agai/agai_env.sh    # cluster variables, then a cd to /projects/$USER/agai/MiniGPT-4
cd /projects/$USER/agai-work
P=evaluation/paper_data
python3 cluster/benchmark/mcnemar_pair.py $P/scores_resnet50_512_s42_runtime_split_v1_test.json \
    $P/scores_resnet50_512_s42_squash_split_v1_test.json
python3 cluster/benchmark/sensitivity_mirror.py $P/mirror_dupes_v1.json \
    $P/scores_resnet50_512_s*_runtime_split_v1_test.json
/projects/$USER/agai/envs/tools/bin/python cluster/benchmark/mirror_dupes.py --out evaluation/mirror_dupes_v1.json
python3 cluster/benchmark/leaderboard.py --md leaderboard_split_v1.md
```

The first two commands need only committed files. Off the cluster, skip the `source` line; the standard-library scripts
run with any Python 3.

## Dependencies

- **v1 modules**, imported from the working directory: `eval_resnet_set.py` and `repro_resnet_v4.py` import
  `resnet_classifier` ([`v1/resnet_classifier.py`](../../v1/resnet_classifier.py)); `minigpt_diagnose.py` imports
  `grounding.inference.load_minigpt`, which loads `minigpt4/` and reads `eval_configs/minigptv2_eval.yaml`.
- **No sibling imports.** Each script stands alone.
- **Files from and for other studies.** `label_audit.py` and `mold_tossups.py` read the output of
  `classifier/cv_probes.py`, and `label_audit.py` the annotations of `vlm/annotate_vllm.py`. The output of
  `near_dupes.py` feeds `make_split.py`, `classifier/cv_probes.py` and `compare_results.py --drop-leaks`; the output of
  `repro_resnet_v4.py` feeds `minigpt_diagnose.py`, `minigpt_train_overlap.py` and `vlm/zeroshot_qwen3vl.py`. The
  result files that `leaderboard.py` and `compare_results.py` read come from `classifier/`, `vlm/` and
  `eval_resnet_set.py`.

## Known problems

- **Inputs never committed.** `evaluation/clean_eval_sets.json`, `near_dupes.json`, `minigpt_lab_train_overlap.json`,
  `holdout/repro_resnet_v4_hamilton.json` and `cv5_probe_*.json`, the per-run result files, the ResNet checkpoints in
  `plant_diagnostic/models/` and the MiniGPT-v2 checkpoints (`checkpoints/`, `output/`). `make_split.py`,
  `minigpt_diagnose.py`, `label_audit.py` and `mold_tossups.py` stop at the first missing file; `eval_resnet_set.py`
  and `repro_resnet_v4.py` need a checkpoint.
- **`compare_results.py`** opens `evaluation/minigpt_lab_train_overlap.json` before anything else, so it fails in a
  working copy built from this repository.
- **`leaderboard.py`** reads only the top level of `evaluation/`. It ignores `evaluation/paper_data/` and prints an
  empty table from the committed files (checked 18 November 2026).
- **`mirror_dupes.py`** writes `evaluation/mirror_dupes_v1.json` by default; the committed copy is
  [`evaluation/paper_data/mirror_dupes_v1.json`](../../evaluation/paper_data/mirror_dupes_v1.json).
- **Git refs.** `clean_eval_sets.py` and `near_dupes.py` read the lab server's stash (`refs/labbox/stash`), which a
  clone of this repository does not have; `minigpt_train_overlap.py` expects `origin/main` to be the GitHub-era tree
  with `plant_diagnostic/data/train_aug` at its root. All three run `git` in the working directory, and the folder
  that `make_workdir.sh` builds is not a git repository. `near_dupes.py` ignores git's exit status, so without the refs
  it runs on and reports no leaks.
- **Photographs.** Of the 1,576 photographs that `evaluation/split_v1.json` lists, 1,284 are at those paths under
  `v1/` in this repository; 292, most of them in `plant_diagnostic/data/holdout/`, are not (count of 18 November 2026).
  Scripts that open the listed photographs fail on a missing one; `mirror_dupes.py` reports it as unreadable and
  continues.
- **Writes into committed files.** In the working copy `evaluation/split_v1.json` and `evaluation/label_audit.csv`
  are links into this repository, so running `make_split.py` or `label_audit.py` there overwrites the committed files;
  `repro_resnet_v4.py` writes into `evaluation/holdout/`, a link to `v1/evaluation/holdout/`.
- **Lab-server paths.** `minigpt_diagnose.py` goes through `eval_configs/minigptv2_eval.yaml`, whose model paths
  start with `/data/AGAI/MiniGPT-4` (see [Paths](../../v1/README.md#paths) in the v1 README).
