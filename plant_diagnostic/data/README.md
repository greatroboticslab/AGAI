# Plant-Diagnostic Image Dataset (not tracked)

This directory contains the strawberry-disease image dataset used to train
and evaluate the deployed ResNet-50 classifier. The image files are
**deliberately excluded from version control** (see [`.gitignore`](../../.gitignore))
because of their size; this README and the placeholder class
subdirectories are tracked so the expected on-disk layout is reproducible.

## Expected layout

```
plant_diagnostic/data/
├── train/                       # compact source-image pool, 7 class subdirs
│   ├── drought/
│   ├── frost/
│   ├── gray_mold/
│   ├── healthy/
│   ├── overwatered/
│   ├── root_rot/
│   └── white_mold/
├── train_aug/                   # augmented training root, same 7 subdirs
│   └── ... (deterministic augmentation descendants of train/)
└── holdout/                     # balanced 70-image evaluation split
    └── ... (10 images per class)
```

## Reproducing the splits

The compact, augmented, and holdout roots are produced by the scripts
under [`../scripts/`](../scripts):

- `release_holdout_to_train_and_rebuild_aug.py` -- relocate holdout
  images and rebuild the augmented root from scratch.
- `augment_train_aug_to_min_count.py` -- raise underrepresented classes
  to the class-floor minimum via deterministic augmentation.
- `import_curated_external_candidates.py` -- import conservatively
  audited external images.

The reproducibility manifest in the appendix of
[`../../paper/main.tex`](../../paper/main.tex) lists checkpoint hashes
and evaluation-artifact paths.

## Class-count snapshot

The image counts in the appendix tables of the manuscript reflect the
state of these directories at the commit hash listed in the appendix.
After regenerating the splits locally, re-run
`paper/scripts/build_paper_assets.py` to refresh the dataset-count
table.
