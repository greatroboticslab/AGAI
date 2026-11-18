# `v2/synthetic/`: synthetic training images and training trees

Part of v2, the lab's 2026 revival of AGAI (October-November 2026). These scripts import reviewed ChatGPT-generated
photographs as training-only images, build training trees that add synthetic images to split v1's training
photographs, and run one synthetic-data arm end to end; the fine-tunes and evaluations themselves are in
[`../classifier/`](../classifier/).

Results: [Synthetic data](../../docs/SYNTHETIC_DATA.md). Every run is recorded in the lab notebook,
[`../EXPERIMENT_LOG.md`](../EXPERIMENT_LOG.md), under the scripts' flat names (`cluster/build_synth_split.py`).

## Files

| File | Purpose | Environment | Inputs → outputs (defaults) |
|---|---|---|---|
| [`build_synth_split.py`](build_synth_split.py) | split v1's training photographs plus the lab's April 2026 GPT images (files matching `--pattern` in `--pool`), deduplicated, without near-copies of validation, test or training photographs, for the with/without-synthetic ablation | `tools` | `--pool plant_diagnostic/data/train_aug`, `--pattern synth_`, `--name v1_gptsynth`, `--max-dist 6`, `evaluation/split_v1.json` → `/projects/$USER/agai/splits/<name>/train/<class>/` (symbolic links), `evaluation/split_<name>.json` |
| [`build_train_tree.py`](build_train_tree.py) | training tree: split v1's training photographs plus the images of an `import_synth.py` manifest, with at most `--hand-cap` 0.33 of each class's added images showing a hand in the prompt; `--holdout` batches stay out of training as probe files | system `python3` (stdlib) | `--manifest`, `--name` (both required), `--classes`, `--hand-cap 0.33`, `--holdout`, `evaluation/split_v1.json` → `/projects/$USER/agai/splits/<name>/train/<class>/` (symbolic links), `evaluation/split_<name>.json` (with `probe_files` for `classifier/eval_probe.py --only`) |
| [`import_synth.py`](import_synth.py) | imports reviewed ChatGPT PNGs as training-only JPEGs: a seeded random crop and aspect ratio, a phone-like downscale and JPEG quality, pHash deduplication within the pool and against split v1's validation and test photographs, and a manifest with each image's prompt, severity and hand flag; `review.json` in `--raw` can exclude images or override the hand flag | `tools` | `--raw`, `--index` (both required; `/projects/$USER/agai/synth/chatgpt_raw` and prompt-index JSON files), `--name chatgpt_v1`, `--max-dist 6`, `evaluation/split_v1.json` → `/projects/$USER/agai/synth/<name>/<class>/*.jpg`, `evaluation/synth_<name>_manifest.json` |
| [`run_synth_arm.sh`](run_synth_arm.sh) | one arm end to end: `import_synth.py`, `build_train_tree.py`, three SigLIP 2 fine-tunes (seeds 0 to 2, `slurm/train_backbone.sbatch` on `research-gpu`), then `slurm/eval_lowq.sbatch` on the new models and, with a holdout, `slurm/eval_probe.sbatch` on the new models and the two baseline arms | bash (submits Slurm jobs) | positional `NAME TREE [HOLDOUT]`, `/projects/$USER/agai/synth/chatgpt_raw`, `/projects/$USER/agai/synth/prompt_index/batches_{v1,v2,v3,v3b}.json` → the outputs above, three training jobs and their evaluation jobs |

Environment: the venv under `/projects/$USER/agai/envs/` that `run_synth_arm.sh` or the script's docstring names;
stdlib means standard library only (any Python 3).

## Running

The scripts open every path relative to the working directory, as in the cluster's working copy. Build that folder
with [`../infra/make_workdir.sh`](../infra/make_workdir.sh) (details in
[`../infra/README.md`](../infra/README.md#working-copy)):

```bash
bash v2/infra/make_workdir.sh /projects/$USER/agai-work    # once, from the root of a clone of this repository
source /projects/$USER/agai/agai_env.sh    # cluster variables, then a cd to /projects/$USER/agai/MiniGPT-4
cd /projects/$USER/agai-work
/projects/$USER/agai/envs/tools/bin/python cluster/synthetic/build_synth_split.py
python3 cluster/synthetic/build_train_tree.py --manifest evaluation/synth_chatgpt_v1_manifest.json --name v1_chatgpt \
    --hand-cap 0.33 --holdout healthy_5,root_rot_5
cd /projects/$USER/agai/MiniGPT-4    # the folder Slurm jobs run in
bash cluster/synthetic/run_synth_arm.sh chatgpt_v1 v1_chatgpt healthy_5,root_rot_5
```

`run_synth_arm.sh` runs from `/projects/$USER/agai/MiniGPT-4`, as its header says: it writes the manifest and split
file into `evaluation/` of the folder it runs in, and the evaluation job it submits reads them from `evaluation/` of
the folder Slurm jobs run in, which is that one (see [`../slurm/README.md`](../slurm/README.md#where-jobs-run)).

## Dependencies

- **No imports** between these scripts and no imports of v1 code. `build_synth_split.py` reads v1 data by default
  (`plant_diagnostic/data/train_aug`); it and `import_synth.py` hash split v1's photographs, and the two tree builders
  link to them by absolute path.
- **Paths in `run_synth_arm.sh`**, relative to the working directory: `cluster/synthetic/import_synth.py`,
  `cluster/synthetic/build_train_tree.py` and `cluster/slurm/{train_backbone,eval_lowq,eval_probe}.sbatch`, which run
  `cluster/classifier/train_backbone.py`, `eval_lowq.py` and `eval_probe.py`.
- **Files for other studies.** The trees under `/projects/$USER/agai/splits/` are the `--train-root` of
  `classifier/train_backbone.py`; the manifests and split files are the `--manifest` and `--only` of
  `classifier/eval_probe.py`.

## Known problems

- **Inputs outside the repository.** The raw ChatGPT images, `review.json` and the prompt indexes live under
  `/projects/$USER/agai/synth/`.
- **The lab's GPT images are not here.** No file in this repository's `v1/plant_diagnostic/data/train_aug/` matches
  `synth_`, so `build_synth_split.py`'s default pool is empty.
- **`envs/tools`** is created by [`../infra/hamilton_setup_stage1.sh`](../infra/hamilton_setup_stage1.sh) with `gdown`
  only; no script in the repository installs the `imagehash` and Pillow that `import_synth.py` and
  `build_synth_split.py` import.
- **Rebuilt on every run.** `build_synth_split.py` and `build_train_tree.py` delete
  `/projects/$USER/agai/splits/<name>/train` before writing it.
- **Fixed in `run_synth_arm.sh`.** The baseline arms (`ft_siglip2-so400m-512_s0` to `_s2` and `_labsynth_s0` to
  `_labsynth_s2`), the model tag (TREE without its `v1_` prefix), `-p research-gpu`, `--grad-ckpt --batch-size 16`, and
  the four prompt-index files.
- **Photographs.** 292 of split v1's photographs are not in this repository (see
  [`../benchmark/README.md`](../benchmark/README.md#known-problems)): `build_synth_split.py` and `import_synth.py`
  fail on them, and `build_train_tree.py` creates links that point nowhere.
