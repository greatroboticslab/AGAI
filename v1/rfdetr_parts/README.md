# `v1/rfdetr_parts/`: the v1 plant-part detector (RF-DETR Small)

Part of v1, the original system (2025 to April 2026). Training, threshold tuning and inference code for v1's
plant-part detector, an RF-DETR Small model fine-tuned on six parts (flower, fruit, leaf, root, soil, stem), added on
12 April 2026. The application runs [`detect_images.py`](detect_images.py) through
[`../grounding/detector.py`](../grounding/detector.py); the 2026 revival reused this configuration, dataset and
checkpoint in its detector study ([DETECTOR.md](../../docs/DETECTOR.md)). No file here was changed during the revival.

## Files

| File | What it is |
|---|---|
| [`prepare_dataset.py`](prepare_dataset.py) | builds `dataset/train/` and `dataset/valid/` from a Roboflow COCO-segmentation export (zip): removes the `Strawberries` super-category, numbers the six part classes 1 to 6, and puts 30 % of the images in validation (images with and without roots split separately, seed 42) |
| [`train_rfdetr.py`](train_rfdetr.py) | trains `RFDETRSmall` with the settings in `rfdetr_config.yaml` (`--config` for another file) |
| [`rfdetr_config.yaml`](rfdetr_config.yaml) | training settings: 100 epochs, batch 8 × gradient accumulation 2, resolution 560, learning rate 1e-4 (encoder 1.5e-4), cosine schedule with 5 warm-up epochs, drop path 0.1, EMA, seed 42; dataset and output folders as lab-server paths |
| [`eval_thresholds.py`](eval_thresholds.py) | per-part threshold sweep: up to 20 random validation images per part (`--n-images`, seed 42), precision, recall and F1 at IoU 0.5 for thresholds 0.05 to 0.50 |
| [`detect_images.py`](detect_images.py) | inference command line (`--images ...`): prints JSON with, per image, the parts found and their best score, every box, and the number of boxes; per-part thresholds flower and fruit 0.50, leaf, soil and stem 0.40, root 0.30 (`--threshold` sets one value for all) |
| [`dataset/README.md`](dataset/README.md) | expected layout of the training data (753 training and 322 validation images), which is not in the repository |

`dataset/` and `output/` (the trained checkpoint `checkpoint_best_total.pth`) are local only: they are in
[`.gitignore`](../../.gitignore).

## Running

Run these in an environment with the `rfdetr` package (Python 3.11; see [`../README.md`](../README.md#setup)). The
revival used rfdetr 1.11.2 ([`rfdetr.freeze.2026-10-30.txt`](../../v2/infra/requirements/rfdetr.freeze.2026-10-30.txt));
the version that trained the lab's checkpoint is not recorded.

```bash
cd v1
python rfdetr_parts/prepare_dataset.py --zip <Roboflow COCO export>.zip --out rfdetr_parts/dataset
python rfdetr_parts/train_rfdetr.py      # dataset_dir and output_dir come from rfdetr_config.yaml
python rfdetr_parts/eval_thresholds.py --checkpoint rfdetr_parts/output/checkpoint_best_total.pth \
    --dataset rfdetr_parts/dataset
python rfdetr_parts/detect_images.py --checkpoint rfdetr_parts/output/checkpoint_best_total.pth \
    --images photo1.jpg photo2.jpg
```

- The Roboflow export is not in the repository; `prepare_dataset.py` defaults to
  `/data/AGAI/LatestStrawberry archive.coco-segmentation.zip` on the lab server, and it deletes the output folder
  before writing.
- The COCO files number the parts 1 to 6 (flower, fruit, leaf, root, soil, stem); the model predicts 0 to 5 in the
  same order, which is the mapping in `detect_images.py`, and `eval_thresholds.py` adds 1 before comparing. The
  revival confirmed the mapping on the validation split (lab notebook, 30 October 2026).
- The SHA-256 of the application's checkpoint is listed in the appendix of the April 2026 report
  ([`../paper/sections/08_appendix.tex`](../paper/sections/08_appendix.tex), Reproducibility Manifest).

## Dependencies

- **Used by v1:** [`../grounding/detector.py`](../grounding/detector.py) runs `detect_images.py` once per photograph
  for the application; [`../paper/scripts/run_rfdetr_on_holdout.py`](../paper/scripts/run_rfdetr_on_holdout.py) (the
  April 2026 report's detection rates on the holdout) loads `output/checkpoint_best_total.pth`.
- **Used by v2**, from the working folder that [`../../v2/infra/make_workdir.sh`](../../v2/infra/make_workdir.sh)
  builds: [`../../v2/detector/rfdetr_size.py`](../../v2/detector/rfdetr_size.py) reads `rfdetr_config.yaml` (and its
  `dataset_dir` unless `--dataset-dir` is given);
  [`../../v2/detector/make_detector_split.py`](../../v2/detector/make_detector_split.py) reads `dataset/` and writes
  the split that agrees with split v1 (the notebook's run wrote `rfdetr_parts/dataset_splitv1`, the default of
  [`../../v2/slurm/eval_parts.sbatch`](../../v2/slurm/eval_parts.sbatch));
  [`../../v2/vlm/detect_parts.py`](../../v2/vlm/detect_parts.py) and
  [`../../v2/detector/latency_rfdetr.py`](../../v2/detector/latency_rfdetr.py) load `output/checkpoint_best_total.pth`
  by default, and [`../../v2/slurm/latency_v1_v2.sbatch`](../../v2/slurm/latency_v1_v2.sbatch) checks that it exists.
- **Copies of the thresholds** of `detect_images.py`, as text: `run_rfdetr_on_holdout.py`, `detect_parts.py`,
  `latency_rfdetr.py` and [`../../v2/detector/eval_parts.py`](../../v2/detector/eval_parts.py).

## Known problems

- **Lab-server paths.** `rfdetr_config.yaml` (`dataset_dir`, `output_dir`) and the defaults of `prepare_dataset.py`,
  `eval_thresholds.py` and `detect_images.py` point to `/data/AGAI/MiniGPT-4/rfdetr_parts/...`. Pass the paths, or
  edit a copy of the YAML and give it with `--config` ([Paths](../README.md#paths)).
- **The 560-pixel resolution was never applied.** `train_rfdetr.py` builds `RFDETRSmall()` with the library defaults
  and passes the YAML values only to `train()`; the lab's checkpoint takes 512-pixel input. RF-DETR 1.11.2 rejects 560
  and does not accept `lr_min_factor`, so the revival trained at each model size's default resolution and dropped that
  key ([DETECTOR.md](../../docs/DETECTOR.md)).
- **Optimistic validation score.** The 322-image validation split is random, shares near-copies with the training
  images, and is also where the best checkpoint is chosen, so the lab checkpoint's 0.598 mAP@50:95 on it is
  optimistic. RF-DETR Small retrained on a split that agrees with split v1 scores 0.564 on that split's clean test
  images ([DETECTOR.md](../../docs/DETECTOR.md)).
- **Hand-set thresholds.** The thresholds in `detect_images.py` were set by hand in April 2026 after a run of
  `eval_thresholds.py` whose output was not kept; thresholds chosen on the revival's validation split did no better
  ([MANUSCRIPT_QA.md](../../docs/MANUSCRIPT_QA.md), [DETECTOR.md](../../docs/DETECTOR.md)).
- **Model load per call.** `detect_images.py` loads the checkpoint every time it runs, and the application starts it
  once per photograph (see [`../grounding/README.md`](../grounding/README.md)).
