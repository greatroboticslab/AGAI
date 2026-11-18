# `v1/plant_diagnostic/tools/`: offline augmentation of the training photographs

One script of v1 (2025): `augment_dataset.py` writes augmented copies of the photographs of selected classes until each
class reaches a target count, and copies the original photographs of all classes, so that the output folder is a
complete training tree. Its naming and report format match [`../data/train_aug/`](../data/train_aug/) and
[`../data/aug_report.json`](../data/aug_report.json).

## Contents

| File | What it does |
|---|---|
| [`augment_dataset.py`](augment_dataset.py) | for each class in `--classes` that has fewer than `--target-per-class` photographs in `--src/<class>/`, writes copies `<stem>_augNNN.jpg` into `--dst/<class>/` (at most `--max-per-image` per original, in turns over the originals) with class-specific photometric and mild geometric changes (albumentations if installed, otherwise Pillow); then copies the originals of every class into `--dst` when it differs from `--src`; `--report` writes the per-class counts (found, target, added) as JSON |

## Running

```bash
cd v1/plant_diagnostic
python tools/augment_dataset.py --src data/train --dst data/train_aug --classes drought overwatered root_rot \
    --target-per-class 300 --max-per-image 4 --img-size 256 --report data/aug_report.json
```

These are the docstring's example, with relative instead of lab-server paths; `--classes`, `--target-per-class`,
`--max-per-image` and `--img-size` are also the defaults, and `--src` and `--dst` are required. Augmented copies are
`--img-size` square crops saved as JPEG; the changes are random (no seed is set), so a new run does not reproduce the
committed copies. It writes into `--dst`: run it on a copy if `data/train_aug/` should stay as committed.

## Known problems

- `../data/aug_report.json` reports 220 overwatered and 172 root rot copies added; `../data/train_aug/` holds 218 and
  165. The tool also counts a copy as added when a file of that name already exists. Why the numbers differ is not
  recorded.
- Files with extensions outside its list (`.jpg`, `.jpeg`, `.png`, `.bmp`, `.webp`, `.tif`, `.tiff`) are not copied:
  `data/train/frost/EWcYwrUWAAAvAOL.jpg_large` is missing from `data/train_aug/`.
