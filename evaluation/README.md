# `evaluation/`: benchmark and result files of the 2026 revival

Data shared by the studies of v2, the 2026 revival (October to November 2026): split v1, the leak-free benchmark on
which the v2 studies and the comparison of v1 with v2 are scored; the captioner bake-off set; the label audit; and, in
[`paper_data/`](paper_data/README.md), the result files behind the journal manuscript. The files of v1 (the original
system, 2025 to April 2026) are in [`v1/evaluation/`](../v1/evaluation/): the April 2026 holdout evaluations of the
ResNet-50 and an evaluator from January 2026. "Split v1" names the first version of the revival's split, not the v1
system.

| File | What it is | Written by, read by |
|---|---|---|
| [`split_v1.json`](split_v1.json) | split v1: the 1,576 labeled photographs (originals only, no synthetic or augmented copies) in train 1,017 / val 161 / test 398, listed per class, and 4 photographs excluded because they belong to near-duplicate groups filed under two labels; near-duplicate groups never cross the sets; built on 20 October 2026 with seed 0 | written by [`v2/benchmark/make_split.py`](../v2/benchmark/make_split.py); read by default by most v2 scripts |
| [`bakeoff_set_v1.json`](bakeoff_set_v1.json) | the 38 hard split v1 test photographs of the captioner bake-off (faint mold, black frost centres, wilting stems, look-alikes, rotted crowns), each with its label and the label audit's ensemble probability for that label, and 2 photographs dropped with the reason | no script in the repository writes it; read by [`v2/vlm/caption_bakeoff.py`](../v2/vlm/caption_bakeoff.py) and [`v2/vlm/judge_claims.py`](../v2/vlm/judge_claims.py) (`--set`) |
| [`label_audit.csv`](label_audit.csv) | one row per split v1 photograph (1,576), sorted by how likely the label is wrong: the out-of-fold prediction of an ensemble of three frozen-backbone probes (EVA-02, PE-Core-G, SigLIP 2) and its probability for the label, and the Qwen3-VL-32B annotator's verdict on the label, differential and caption; `both_disagree` marks the 11 rows where both reject the label. Candidates for expert relabeling; no label in split v1 was changed | written by [`v2/benchmark/label_audit.py`](../v2/benchmark/label_audit.py); read by [`v2/benchmark/mold_tossups.py`](../v2/benchmark/mold_tossups.py) |
| [`label_audit_top.jpg`](label_audit_top.jpg) | contact sheet of the first 24 rows of `label_audit.csv`: photograph, label, ensemble prediction, annotator verdict | `v2/benchmark/label_audit.py` (`--sheet`, by default in the cluster's log folder) |
| [`ensemble_errors.jpg`](ensemble_errors.jpg) | contact sheet of the 11 split v1 test photographs that ensemble E2 gets wrong, with the predictions of single models; reviewed in [Classifier, error analysis](../docs/CLASSIFIER.md#error-analysis-and-the-label-noise-ceiling) | [`v2/benchmark/error_sheet.py`](../v2/benchmark/error_sheet.py); the lab notebook calls the sheet `logs/e2_errors.jpg` |
| [`split_v1_coverage.json`](split_v1_coverage.json) | for each set and condition of split v1, how many of its photographs are in this repository, and the 291 paths that are not | [`v2/benchmark/split_coverage.py`](../v2/benchmark/split_coverage.py) |
| [`MANIFEST.sha256`](MANIFEST.sha256) | SHA-256 of the evaluation files and of the 1,285 split v1 photographs in this repository; check with `git lfs pull` then `sha256sum -c evaluation/MANIFEST.sha256` from the repository root | [`v2/benchmark/make_manifest.py`](../v2/benchmark/make_manifest.py) |
| [`paper_data/`](paper_data/README.md) | 54 result files behind the journal manuscript, 31 October to 20 November 2026 | see its README |

## Using the files

The v2 scripts open these files as `evaluation/...` relative to the working directory, and most write their own results
to `evaluation/<name>.json` the same way. On the cluster the working directory was one working copy that held the v1
files, `evaluation/` and the v2 scripts (as `cluster/`) side by side. In this repository
`bash v2/infra/make_workdir.sh DEST` builds such a folder out of symbolic links
([details](../v2/README.md#running-the-scripts)). Its `evaluation/` is a real folder of links to the entries of this
folder and of `v1/evaluation/`, so:

- new result files stay in `DEST/evaluation/`;
- writing to a name that already exists goes through the link into the repository: a rerun of
  `v2/benchmark/make_split.py` or `label_audit.py` rewrites `split_v1.json` or `label_audit.csv` here, and files
  written into `evaluation/paper_data/` land in [`paper_data/`](paper_data/).

The photographs are listed as `plant_diagnostic/data/<train or holdout>/<class>/<file>`. These paths are relative to
`v1/`, and they resolve in `DEST` as well, where `plant_diagnostic/` is a link to `v1/plant_diagnostic/`.

## Photographs not in this repository

292 of the 1,580 paths in `split_v1.json` are not in this repository's copy of the photographs
(`v1/plant_diagnostic/data/`): 74 of the 398 test paths, 36 of the 161 validation paths, 181 of the 1,017 training
paths and 1 of the 4 excluded ones. The same gap affects the files built on the split: 12 of the 40 photographs of
`bakeoff_set_v1.json`, 291 of the 1,576 rows of `label_audit.csv` and the per-image files in `paper_data/`. Scripts
that open the photographs need the lab's copy for these paths. [`split_v1_coverage.json`](split_v1_coverage.json) lists
the 291 paths of the three sets with their set and condition.

The reason: split v1 was built on the cluster (20 October 2026) from the lab server's copy of the photographs, exported
on 19 October 2026 (lab notebook, entries of 19 and 20 October 2026), while the photographs in this repository were
committed on 17 October 2025. The two copies are laid out differently. Of the 292 paths, 228 are under `holdout/` (of
231 `holdout/` paths in the split) and 64 under `train/`. 236 of them have a file of the same class and name in the
other folder here (226 `holdout/` paths under `train/`, 10 `train/` paths under `holdout/`), and 56 have no file of
that name. A matching name does not prove the same picture; for the three photographs compared in the lab notebook
(entry (y), 17 November 2026) the pictures are the same.

## Storage

`.gitattributes` puts every `*.json` file in Git LFS, but four later rules keep the JSON files of this folder in Git
as plain text:

```
evaluation/paper_data/*.json !filter !diff !merge text
evaluation/split_v1.json !filter !diff !merge text
evaluation/bakeoff_set_v1.json !filter !diff !merge text
evaluation/split_v1_coverage.json !filter !diff !merge text
```

A new JSON file elsewhere in `evaluation/` goes to Git LFS unless a rule is added, and moving one of these files needs
its rule changed in the same commit. `label_audit.csv` and the TXT files in `paper_data/` fall under no LFS rule; the two
JPG contact sheets are in Git LFS (`*.jpg`).

## Known problems

- `split_v1.json` cannot be rebuilt from this repository: besides the photographs above, `make_split.py` reads
  `evaluation/clean_eval_sets.json` and `evaluation/near_dupes.json` (written by `v2/benchmark/clean_eval_sets.py` and
  `near_dupes.py`), which stayed on the cluster.
- The script that chose the bake-off set is not in the repository. The file's note gives the rule (ensemble probability
  of the label 0.5 to 0.97, annotator did not reject the label), but three of the 38 photographs fall outside it:
  `overwatered_32.jpg` (probability 0.978), `overwatered_47.jpg` (verdict `does_not_support` in `label_audit.csv`) and
  `overwatered_7.jpg` (0.995 and `does_not_support`).
- Known defects of the data (mislabeled photographs, gray and white mold toss-ups, three non-photographic or degraded
  test images) are listed, not removed: split v1 and every result on it are unchanged
  ([Benchmark, known defects](../docs/BENCHMARK.md#known-defects-of-the-data)).
