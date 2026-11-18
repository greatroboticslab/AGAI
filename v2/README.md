# `cluster/`: revival code and lab notebook

The code behind every number in the [documentation](../docs/), grouped by study, and the lab notebook
[`EXPERIMENT_LOG.md`](EXPERIMENT_LOG.md), which records each run (job ID, arguments, result file, conclusion), every
pre-registration and every correction. Hardware, environments and multi-node serving:
[docs/COMPUTE.md](../docs/COMPUTE.md).

Conventions: run from the repository root on the cluster after `source /projects/$USER/agai/agai_env.sh` (a copy of
[`infra/agai_env.sh`](infra/agai_env.sh)); models live under `/projects/$USER/agai/models` and are staged to
node-local NVMe; result files with per-image outputs go to `evaluation/`; Slurm logs to
`/projects/$USER/agai/logs/<job-name>-<jobid>.out`. The notebook's commands use the flat layout of the cluster working
copy (`cluster/<script>`); here the same scripts are grouped by study.

| Folder | Contents | Results |
|---|---|---|
| [`benchmark/`](benchmark/) | leak-free split, leakage and label audits, v1 baselines, leaderboards and paired tests | [Benchmark](../docs/BENCHMARK.md) |
| [`classifier/`](classifier/) | backbone probes and fine-tunes, cross-validation, ensembles, robustness, resolution shortcut | [Classifier](../docs/CLASSIFIER.md) |
| [`synthetic/`](synthetic/) | synthetic-image import, training trees, one arm end to end | [Synthetic data](../docs/SYNTHETIC_DATA.md) |
| [`vlm/`](vlm/) | annotation schema, label-swap honesty, honest student, captioning, claim judges, voting, disagreement resolution | [VLM honesty](../docs/VLM_HONESTY.md), [Captioning](../docs/CAPTIONING.md), [Test-time compute](../docs/TEST_TIME_COMPUTE.md) |
| [`detector/`](detector/) | RF-DETR evaluation and training | [Detector](../docs/DETECTOR.md) |
| [`infra/`](infra/) | cluster setup, model staging and verification, Ray and multi-node helpers | [Compute](../docs/COMPUTE.md) |
| [`slurm/`](slurm/) | launchers: multi-node vLLM, multi-node LoRA, the honest student, synthetic arms | [Compute](../docs/COMPUTE.md) |

Not published here: one-off diagnostics, superseded annotators and the per-experiment Slurm wrappers (each stages a
model and runs one of the scripts below). Their runs are recorded in the notebook.

## `benchmark/`

| Script | Purpose |
|---|---|
| [`repro_resnet_v4.py`](benchmark/repro_resnet_v4.py) | re-run the paper's ResNet-50 v4 on its holdout and compare image by image |
| [`clean_eval_sets.py`](benchmark/clean_eval_sets.py) | which labeled images are fair test data for which model (content hashes) |
| [`near_dupes.py`](benchmark/near_dupes.py) | perceptual-hash near-duplicate audit |
| [`make_split.py`](benchmark/make_split.py) | build split v1: grouped, stratified, leak-free train/val/test |
| [`eval_resnet_set.py`](benchmark/eval_resnet_set.py) | a ResNet checkpoint on an evaluation set (lab recipe: flip TTA + checkpoint temperature) |
| [`minigpt_diagnose.py`](benchmark/minigpt_diagnose.py), [`minigpt_train_overlap.py`](benchmark/minigpt_train_overlap.py) | MiniGPT-v2 checkpoints as diagnosers; which test images the lab's MiniGPT trained on |
| [`label_audit.py`](benchmark/label_audit.py) | rank labels by how likely they are wrong (CV ensemble + VLM annotator) → [`evaluation/label_audit.csv`](../evaluation/label_audit.csv) |
| [`mold_tossups.py`](benchmark/mold_tossups.py) | gray vs white mold toss-ups for expert review |
| [`error_sheet.py`](benchmark/error_sheet.py) | contact sheet of the images a result file gets wrong |
| [`leaderboard.py`](benchmark/leaderboard.py), [`compare_results.py`](benchmark/compare_results.py) | split v1 leaderboard; tables and paired McNemar tests from per-image result files |

## `classifier/`

| Script | Purpose |
|---|---|
| [`backbones.py`](classifier/backbones.py) | backbone registry: local weight paths, input sizes, feature extraction (imported by the scripts below) |
| [`probe_backbones.py`](classifier/probe_backbones.py) | frozen-feature linear probes on split v1 |
| [`cv_probes.py`](classifier/cv_probes.py) | 5-fold grouped stratified CV of the probes over all 1,576 images |
| [`train_backbone.py`](classifier/train_backbone.py) | fine-tune a backbone with the lab's ResNet recipe; `--lowq-aug` phone degradation |
| [`resnet_scores.py`](classifier/resnet_scores.py) | per-class probabilities from ResNet-50 checkpoints on split v1 |
| [`ensemble.py`](classifier/ensemble.py) | ensembles of per-class probability files |
| [`eval_lowq.py`](classifier/eval_lowq.py), [`lowq_table.py`](classifier/lowq_table.py) | low-quality photo robustness and its table over seeds |
| [`eval_shortcut.py`](classifier/eval_shortcut.py) | resolution-shortcut test: does shrinking a photo raise P(healthy)? |
| [`eval_probe.py`](classifier/eval_probe.py) | shortcut probe on held-out synthetic batches |

## `synthetic/`

| Script | Purpose |
|---|---|
| [`import_synth.py`](synthetic/import_synth.py) | import reviewed synthetic images: exclusions, random crop / downscale / JPEG, pHash de-dup, manifest |
| [`build_train_tree.py`](synthetic/build_train_tree.py) | training tree = real train + synthetic manifest, with hand cap and held-out batches |
| [`build_synth_split.py`](synthetic/build_synth_split.py) | split v1 train + the lab's April 2026 GPT images (with/without-synthetic ablation) |
| [`run_synth_arm.sh`](synthetic/run_synth_arm.sh) | one synthetic-data arm end to end (import, tree, 3 seeds, evaluations) |

## `vlm/`

| Script | Purpose |
|---|---|
| [`annotation_spec.py`](vlm/annotation_spec.py) | the structured, label-conditioned annotation schema and prompts (imported by the scripts below) |
| [`annotate_vllm.py`](vlm/annotate_vllm.py) | schema-constrained annotations with vLLM: true / swapped labels, `--pp` for multi-node, `--order` |
| [`score_honesty.py`](vlm/score_honesty.py) | label-swap honesty test |
| [`train_vlm_reports.py`](vlm/train_vlm_reports.py), [`merge_lora.py`](vlm/merge_lora.py) | honest-report student: LoRA distillation filtered to correct teacher verdicts; adapter merging for vLLM |
| [`zeroshot_qwen3vl.py`](vlm/zeroshot_qwen3vl.py) | zero-shot 7-class diagnosis (optional thinking mode) |
| [`train_vlm_lora.py`](vlm/train_vlm_lora.py), [`eval_vlm_scores.py`](vlm/eval_vlm_scores.py) | LoRA / QLoRA classification fine-tunes (single or multi-node); per-class probabilities |
| [`caption_bakeoff.py`](vlm/caption_bakeoff.py) | two-pass captioning (label-blind grounded claims, then text-only reconciliation); `--samples N` for self-consistency |
| [`judge_claims.py`](vlm/judge_claims.py) | zoom-in claim judge with decoys |
| [`consensus_dx.py`](vlm/consensus_dx.py) | zero-shot diagnosis with N sampled answers per photo (voting, differential); `--prompt-style` answer-format ablation |
| [`resolve_dx.py`](vlm/resolve_dx.py) | resolve two-model disagreements: choose, real / synthetic references, multi-round debate |

## `detector/`

| Script | Purpose |
|---|---|
| [`rfdetr_size.py`](detector/rfdetr_size.py) | evaluate the lab's RF-DETR checkpoint or train another size with the lab's configuration |

## `infra/`

| Script | Purpose |
|---|---|
| [`hamilton_setup_stage1.sh`](infra/hamilton_setup_stage1.sh) | stage 1: git-lfs, the repository with its LFS images, public weights (needs no lab data) |
| [`hamilton_setup_envs.sh`](infra/hamilton_setup_envs.sh) | stage 2: Python environments as plain venvs on shared storage |
| [`setup_vision_env.sh`](infra/setup_vision_env.sh) | separate venv for backbone probes / fine-tunes and image generation (timm, open_clip, diffusers) |
| [`agai_env.sh`](infra/agai_env.sh) | environment variables for every job |
| [`hf_fetch.sh`](infra/hf_fetch.sh) | download a Hugging Face model repository with curl, verifying LFS files by SHA-256 |
| [`stage_model.sh`](infra/stage_model.sh) | copy a model from shared storage to node-local NVMe (load time 296 → 5 s for an 8B), verified against its manifest |
| [`make_manifest.py`](infra/make_manifest.py), [`verify_staged.py`](infra/verify_staged.py) | per-model SHA-256 manifest; check staged copies by reading them from disk |
| [`stage_pp_shard.py`](infra/stage_pp_shard.py) | stage only one pipeline-parallel rank's weight files and write a per-node index |
| [`ray_pp_order.py`](infra/ray_pp_order.py) | print the node order vLLM will use for pipeline ranks on a running Ray cluster |
| [`ddp_task.sh`](infra/ddp_task.sh) | one rank of a multi-node LoRA run |

## `slurm/`

| Job script | Runs |
|---|---|
| [`vllm_ray.sbatch`](slurm/vllm_ray.sbatch) | any vLLM script with pipeline parallelism across N nodes on Ray (`VLLM_SCRIPT=` selects the script) |
| [`train_vlm_lora_multinode.sbatch`](slurm/train_vlm_lora_multinode.sbatch) | multi-node LoRA / QLoRA fine-tune, one rank per node |
| [`train_vlm_reports.sbatch`](slurm/train_vlm_reports.sbatch) | honest student end to end: distil, merge, label-swap test |
| [`train_backbone.sbatch`](slurm/train_backbone.sbatch), [`eval_lowq.sbatch`](slurm/eval_lowq.sbatch), [`eval_probe.sbatch`](slurm/eval_probe.sbatch) | single-GPU wrappers submitted by [`synthetic/run_synth_arm.sh`](synthetic/run_synth_arm.sh) |

Example, the 235B claim judge across five nodes:

```bash
sbatch -N 5 --export=ALL,VLLM_SCRIPT=cluster/vlm/judge_claims.py cluster/slurm/vllm_ray.sbatch \
    Qwen3-VL-235B-A22B-Instruct-AWQ 16,20,20,19,19 --modes zoom --max-model-len 8192 --gpu-mem 0.95 --enforce-eager
```
