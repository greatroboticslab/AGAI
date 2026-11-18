# `v1/plant_diagnostic/data/`: labeled photographs of v1

The labeled strawberry photographs of v1 as committed on 17 October 2025 (v1, 2025 to April 2026): the classifier's
training and holdout folders, and the augmented training folder that the report writer's instruction data points to.
The photographs are in the repository (Git LFS); the v2 revival's benchmark, split v1, lists photographs from these
folders.

## Contents

| Path | Contents |
|---|---|
| [`train/`](train/) | 1,540 original photographs in 7 class folders |
| [`train_aug/`](train_aug/) | 2,159 photographs: the `train/` photographs (all but one, below) and 620 augmented copies (`<name>_augNNN.jpg`) of drought, overwatered and root rot photographs |
| [`holdout/`](holdout/) | 20 photographs in 5 class folders (no `gray_mold/` or `white_mold/`) |
| [`aug_report.json`](aug_report.json) | report of the augmentation run (format of [`../tools/augment_dataset.py`](../tools/augment_dataset.py)): for drought, overwatered and root rot the photographs found, the target (300) and the copies added |
| `README.md` | this page |

| Class | `train/` | `train_aug/` (of which augmented copies) | `holdout/` |
|---|---:|---:|---:|
| drought | 63 | 300 (237) | 4 |
| frost | 111 | 110 (0) | 4 |
| gray_mold | 607 | 607 (0) | 0 |
| healthy | 325 | 325 (0) | 5 |
| overwatered | 80 | 298 (218) | 4 |
| root_rot | 128 | 293 (165) | 3 |
| white_mold | 226 | 226 (0) | 0 |
| **total** | **1,540** | **2,159 (620)** | **20** |

The 3,604 JPEG and PNG files and `aug_report.json` are stored with Git LFS (without Git LFS a clone has small pointer
files in their place); the 114 WebP files and one `.jpg_large` file are regular git files. `.gitignore` ignores new
image files in these class folders, so a new photograph has to be added with `git add -f`.

## What uses it

- [`../datasets/stage2_train_7class_fixed.json`](../datasets/stage2_train_7class_fixed.json), the report writer's
  instruction data, has one record per `train_aug/` photograph (all 2,159 are here), named by lab-server paths
  (`/data/AGAI/MiniGPT-4/plant_diagnostic/data/train_aug/...`). The records in [`../jsondatasets/`](../jsondatasets/)
  name `train/` photographs as `data/train/...`, relative to `v1/plant_diagnostic/`.
- v1's MiniGPT configs name `train_aug/` and `holdout/` by lab-server paths; `v1/hallucination_evaluation/config.py`
  uses `train_aug/` and `holdout/` relative to `v1/`; the scripts in [`../../paper/scripts/`](../../paper/scripts/)
  count the photographs of all three folders and run on `holdout/`.
- v2: [`../../../evaluation/split_v1.json`](../../../evaluation/split_v1.json),
  [`../../../evaluation/label_audit.csv`](../../../evaluation/label_audit.csv),
  [`../../../evaluation/bakeoff_set_v1.json`](../../../evaluation/bakeoff_set_v1.json) and 27 of the 49 files in
  `evaluation/paper_data/` name photographs as `plant_diagnostic/data/<folder>/<class>/<file>`, relative to `v1/` or
  to the folder that [`v2/infra/make_workdir.sh`](../../../v2/infra/make_workdir.sh) builds.
  `v2/benchmark/near_dupes.py`, `clean_eval_sets.py` and `repro_resnet_v4.py` also list these folders directly.

## Split v1 and this copy

[`split_v1.json`](../../../evaluation/split_v1.json) lists 1,580 original photographs (train 1,017, val 161, test 398,
and 4 excluded for label conflicts). It was built on the cluster, whose photographs came from the lab server's data
export of 19 October 2026 (lab notebook [`v2/EXPERIMENT_LOG.md`](../../../v2/EXPERIMENT_LOG.md), entry of that day), and
292 of its paths are not in this copy: 181 train, 36 val, 74 test and 1 excluded.

- 236 of them name a photograph that is here in the other folder, with the same class and file name: 226 listed under
  `holdout/` are in `train/` (and `train_aug/`), among them all 91 `holdout/gray_mold/` and 33 `holdout/white_mold/`
  paths, and 10 listed under `train/` are in `holdout/`. Whether these files are identical to the ones split v1 was
  built from has not been checked.
- 56 name a file that is not here: 54 under `train/` (43 of them gray mold) and 2 under `holdout/`.
- Of the 1,560 originals here (`train/` and `holdout/`), 1,288 are at their split v1 path, 236 are the photographs
  above, and 36 are not in split v1.

Scripts that open split v1's photographs therefore fail or skip photographs on this copy. The April 2026 technical
report's holdout (70 photographs, 10 per class) also lists lab-server paths, and 16 of them exist here
([`../../evaluation/holdout/`](../../evaluation/holdout/)).

## Known problems

- **Not the April 2026 folders.** The technical report counts 1,506 / 2,424 / 70 photographs in `train/`, `train_aug/`
  and `holdout/` ([`../../paper/generated/dataset_counts_table.tex`](../../paper/generated/dataset_counts_table.tex));
  this copy holds 1,540 / 2,159 / 20. The scripts that rebuilt the lab server's folders in April 2026 (holdout release,
  external imports) are in [`../../recovered/labbox_2026-10-19/`](../../recovered/labbox_2026-10-19/), and they move
  files: run them on a copy.
- **Augmentation report.** `aug_report.json` reports 220 overwatered and 172 root rot copies added; `train_aug/` holds
  218 and 165.
- **Odd files.** `train/frost/EWcYwrUWAAAvAOL.jpg_large` is not in `train_aug/` (the augmentation tool copies only image
  extensions it knows) and is named `EWcYwrUWAAAvAOL.jpg` in split v1. 18 file names in these folders contain spaces,
  and one photograph in `train/overwatered/` (with its copies in `train_aug/overwatered/`) has a non-ASCII name.
