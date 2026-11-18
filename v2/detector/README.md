# `v2/detector/`: RF-DETR plant-part detector studies

Part of v2, the lab's 2026 revival of AGAI (October-November 2026). Re-evaluation of v1's RF-DETR Small checkpoint,
training of other sizes with the lab's configuration, a detector split that agrees with split v1, per-part metrics on
that split, and the detectors' latency. RF-DETR Large retrained on that split is v2's detector.

Results: [Detector](../../docs/DETECTOR.md). Every run is recorded in the lab notebook,
[`../EXPERIMENT_LOG.md`](../EXPERIMENT_LOG.md), under the scripts' flat names (`cluster/eval_parts.py`).

## Files

| File | Purpose | Environment | Inputs → outputs (defaults) |
|---|---|---|---|
| [`check_class_ids.py`](check_class_ids.py) | which plant part each predicted class id of an RF-DETR checkpoint stands for, from detections matched to ground-truth boxes (IoU ≥ 0.5) | `rfdetr` (inferred) | `--size small` (or medium, large), `--weights` and `--split-dir` (both required; docstring: `rfdetr_parts/output/checkpoint_best_total.pth`†, `rfdetr_parts/dataset/valid`†), `--min-score 0.3` → stdout |
| [`eval_parts.py`](eval_parts.py) | per-part AP50 and AP50:95 (pycocotools), image-level part presence (precision, recall, F1) and a threshold sweep; thresholds chosen on `valid` are applied to `test` with `--thresholds` | `rfdetr` | `--size`, `--weights`, `--dataset-dir`, `--name` (all required; dataset e.g. `rfdetr_parts/dataset_splitv1`†), `--split valid`, `--thresholds` → `evaluation/parts_<name>_<split>.json` |
| [`latency_rfdetr.py`](latency_rfdetr.py) | latency and GPU memory of RF-DETR Small (v1's checkpoint) and Large (retrained on the detector split) on the 49 seeded split v1 test photographs shared by the `latency_*.py` scripts | `rfdetr` | `--weights-small rfdetr_parts/output/checkpoint_best_total.pth`†, `--weights-large /projects/$USER/agai/rfdetr_runs/large_splitv1/checkpoint_best_total.pth`, `--split-json evaluation/split_v1.json` → `--out evaluation/latency_rfdetr.json` |
| [`make_detector_split.py`](make_detector_split.py) | assigns each image of the lab's Roboflow export to train, valid or test by pHash match (≤ 6 bits) to split v1 photographs, keeping near-copies together; writes COCO trees of symbolic links and a report | `tools` (inferred) | `--roboflow-dir rfdetr_parts/dataset`†, `--split evaluation/split_v1.json`, `--out-dir` (required; the notebook's run used `rfdetr_parts/dataset_splitv1`) → `<out-dir>/{train,valid,test}/`, `--report evaluation/detector_split_v1.json` |
| [`rfdetr_size.py`](rfdetr_size.py) | evaluates the lab's checkpoint (`--eval-only --weights`), or trains a size with the lab's hyperparameters and evaluates it on the dataset's validation split; each size trains at its own default resolution, and yaml keys the installed RF-DETR rejects are dropped and printed | `rfdetr` (inferred) | `--size`, `--name` (both required), `--weights`, `--config rfdetr_parts/rfdetr_config.yaml`, `--dataset-dir` (default: the yaml's `dataset_dir`) → run folder `/projects/$USER/agai/rfdetr_runs/<name>/`, `evaluation/rfdetr_<name>_valid.json` |

Environment: the venv under `/projects/$USER/agai/envs/` that a Slurm file in [`../slurm/`](../slurm/) or the
script's docstring names; "(inferred)" means no file names one and the entry is an environment that has the packages
the script imports (`rfdetr` is installed only in `envs/rfdetr`; `imagehash` in none of the four frozen environments,
see [`../infra/README.md`](../infra/README.md#known-problems)). † not in this repository.

## Running

The scripts open every path relative to the working directory, as in the cluster's working copy. Build that folder
with [`../infra/make_workdir.sh`](../infra/make_workdir.sh) (details in
[`../infra/README.md`](../infra/README.md#working-copy)):

```bash
bash v2/infra/make_workdir.sh /projects/$USER/agai-work    # once, from the root of a clone of this repository
source /projects/$USER/agai/agai_env.sh    # cluster variables, then a cd to /projects/$USER/agai/MiniGPT-4
cd /projects/$USER/agai-work
E=/projects/$USER/agai/envs
$E/rfdetr/bin/python cluster/detector/check_class_ids.py --weights rfdetr_parts/output/checkpoint_best_total.pth \
    --split-dir rfdetr_parts/dataset/valid
$E/rfdetr/bin/python cluster/detector/rfdetr_size.py --size large --batch-size 4 --accum 4 --name large \
    --dataset-dir rfdetr_parts/dataset
sbatch cluster/slurm/eval_parts.sbatch large \
    /projects/$USER/agai/rfdetr_runs/large_splitv1/checkpoint_best_total.pth large_splitv1
```

[`../slurm/eval_parts.sbatch`](../slurm/eval_parts.sbatch) runs `eval_parts.py` on `valid` (threshold sweep) and then
on `test` with those thresholds, on `rfdetr_parts/dataset_splitv1` unless a fourth argument names another dataset. A
Slurm job runs in `/projects/$USER/agai/MiniGPT-4` whatever folder it was submitted from; see
[`../slurm/README.md`](../slurm/README.md#where-jobs-run).

## Dependencies

- **No imports** between these scripts and no imports of v1 modules; they need the `rfdetr` package (and
  `pycocotools` for `eval_parts.py`).
- **v1 files on default paths**, from v1's [`rfdetr_parts/`](../../v1/rfdetr_parts/): the configuration
  `rfdetr_config.yaml` (`rfdetr_size.py`), the Roboflow export `dataset/` (`make_detector_split.py`), and v1's
  checkpoint `output/checkpoint_best_total.pth` (`latency_rfdetr.py`).
- **Text copied from v1** (no import): the per-part thresholds of `rfdetr_parts/detect_images.py` in `eval_parts.py`
  and `latency_rfdetr.py`.
- **Files for other studies.** The output of `latency_rfdetr.py` is read by `vlm/latency_summary.py`.
  `vlm/detect_parts.py` runs v1's Small checkpoint for the report-writer study.

## Known problems

- **Not committed.** The lab's Roboflow export (`rfdetr_parts/dataset/{train,valid}`; only its README is here), v1's
  checkpoint `rfdetr_parts/output/checkpoint_best_total.pth`, the detector-split trees (`rfdetr_parts/dataset_splitv1`)
  and the retrained checkpoints under `/projects/$USER/agai/rfdetr_runs/`.
- **Lab-server path.** Without `--dataset-dir`, `rfdetr_size.py` trains and evaluates on the yaml's `dataset_dir`,
  which [`rfdetr_config.yaml`](../../v1/rfdetr_parts/rfdetr_config.yaml) sets to
  `/data/AGAI/MiniGPT-4/rfdetr_parts/dataset` (the lab server).
- **Defaults and the repository.** `eval_parts.py` writes `evaluation/parts_<name>_<split>.json`, `latency_rfdetr.py`
  `evaluation/latency_rfdetr.json` and `make_detector_split.py` `evaluation/detector_split_v1.json`; the committed
  copies are in [`evaluation/paper_data/`](../../evaluation/paper_data/).
- **Writes into v1.** In the working copy `rfdetr_parts/` is a link to `v1/rfdetr_parts/`, so the trees that
  `make_detector_split.py --out-dir rfdetr_parts/dataset_splitv1` writes land in this repository's v1 folder; the
  symbolic links inside them are absolute paths.
- **Photographs.** `make_detector_split.py` hashes split v1's photographs, 292 of which are not in this repository
  (see [`../benchmark/README.md`](../benchmark/README.md#known-problems)).
- **Flat names.** `slurm/eval_parts.sbatch` calls `cluster/eval_parts.py`, and `latency_rfdetr.py`'s docstring shows
  `python cluster/latency_rfdetr.py`; the flat names exist in the working copy.
