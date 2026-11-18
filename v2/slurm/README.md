# `v2/slurm/`: Slurm job scripts for the Hamilton cluster

Part of v2, the lab's 2026 revival of AGAI (October-November 2026). The job files that ran the revival's GPU work on
the Hamilton cluster: single-GPU wrappers for the classifier studies, multi-node vLLM and LoRA launchers, the honest
student end to end, the detector metrics, and the manuscript's runs (v1 versus v2 latency, the report writer's split v1
retrain and its reports).

Hardware, partitions and multi-node serving: [Compute](../../docs/COMPUTE.md). Job IDs and results are in the lab
notebook, [`../EXPERIMENT_LOG.md`](../EXPERIMENT_LOG.md).

## Files

| File | Runs (environment) | Partition and resources | Arguments |
|---|---|---|---|
| [`eval_lowq.sbatch`](eval_lowq.sbatch) | `cluster/classifier/eval_lowq.py` (vision) | `research-gpu`, 1 GPU, 10 CPUs, 48 GB, 3 h | passed to the script |
| [`eval_parts.sbatch`](eval_parts.sbatch) ‡ | `cluster/eval_parts.py` (rfdetr) twice: the threshold sweep on `valid`, then `test` with `--thresholds evaluation/parts_<NAME>_valid.json` | `research-gpu` without c4 and c5, no `--gres`, 1 h | `SIZE WEIGHTS NAME [DATASET_DIR]` (default `rfdetr_parts/dataset_splitv1`) |
| [`eval_probe.sbatch`](eval_probe.sbatch) | `cluster/classifier/eval_probe.py` (vision) | `research-gpu`, 1 GPU, 10 CPUs, 48 GB, 1 h | passed to the script |
| [`latency_v1_v2.sbatch`](latency_v1_v2.sbatch) ‡ | v1 versus v2 latency on one GPU; after checking `RFDETR_PYTHON`, `grounding/detector.py` and every weight file it runs `cluster/latency_v1.py` (minigptv), `cluster/latency_rfdetr.py` (rfdetr), `cluster/latency_siglip.py` (vision), `cluster/stage_model.sh`, `cluster/merge_lora.py` (vlm) when `/scratch` has no merged student yet, `cluster/latency_student.py --labels evaluation/latency_siglip2.json` (vllm) and `cluster/latency_summary.py` (vision) | `research-gpu` without c4 and c5, 1 GPU, 8 CPUs, 56 GB, 4 h | `ADAPTER [SIGLIP_CKPT_NAME] [BASE_DIR]`: student adapter (required), classifier checkpoint name (default `ft_siglip2-so400m-512_chatgpt_nof3_lq_s0`), base model (default from the adapter's `adapter_config.json`) |
| [`minigpt_reports.sbatch`](minigpt_reports.sbatch) ‡ | `cluster/minigpt_reports.py` (minigptv) | `research-gpu` without c4 and c5, no `--gres`, 8 h | passed to the script |
| [`reports_after_training.sbatch`](reports_after_training.sbatch) ‡ | `cluster/minigpt_reports.py --name splitv1 --parts evaluation/parts_report_test.json` (minigptv) with the newest `checkpoint_best.pth` under `/projects/$USER/agai/minigpt_runs/minigpt_strawberry_splitv1` | `research-gpu` without c4 and c5, no `--gres`, 8 h | none; submitted with `--dependency=afterok:<training job>` |
| [`select_and_report.sbatch`](select_and_report.sbatch) ‡ | `cluster/minigpt_val_loss.py` (picks the epoch with the lowest validation loss), then `cluster/minigpt_reports.py` with it (minigptv) | `research-gpu` without c4 and c5, no `--gres`, 10 h | none; submitted with `--dependency=afterok:<training job>` |
| [`train_backbone.sbatch`](train_backbone.sbatch) | `cluster/classifier/train_backbone.py` (vision) | `a100`, 1 GPU, 10 CPUs, 60 GB, 12 h | passed to the script |
| [`train_minigpt_splitv1.sbatch`](train_minigpt_splitv1.sbatch) ‡ | v1's `train.py --cfg-path train_configs/minigptv2_strawberry_splitv1.yaml` (minigptv): v1's MiniGPT-v2 report writer retrained on split v1 | `research-gpu` without c4 and c5, no `--gres`, 14 h | optional config path as the first argument |
| [`train_vlm_lora_multinode.sbatch`](train_vlm_lora_multinode.sbatch) | `srun` of `cluster/infra/ddp_task.sh` on every node, which runs `cluster/vlm/train_vlm_lora.py` (vlm, or `VLM_ENV`) | `research-gpu`, 4 nodes with 1 GPU, 8 CPUs, 56 GB each, 24 h | passed to `train_vlm_lora.py` |
| [`train_vlm_reports.sbatch`](train_vlm_reports.sbatch) | the honest student end to end: `cluster/infra/stage_model.sh`, `cluster/vlm/train_vlm_reports.py` and `merge_lora.py` (vlm), `annotate_vllm.py --run student` on validation and test (vllm), `score_honesty.py` against the teacher and the base model (vision) | `research-gpu`, 1 GPU, 8 CPUs, 56 GB, 12 h | `[TEACHER_RUN_DIR] [BASE_MODEL] [TAG]` (defaults `/projects/$USER/agai/annotations/v1_Qwen3-VL-32B-Instruct`, `/projects/$USER/agai/models/Qwen3-VL-8B-Instruct`, `reports_v1`) |
| [`vllm_ray.sbatch`](vllm_ray.sbatch) | one vLLM script with pipeline parallelism over a Ray cluster with token authentication: finds each node's 10.0.0.x address, starts Ray, gets vLLM's rank order (`cluster/infra/ray_pp_order.py`), stages each rank's weight files (`cluster/infra/stage_pp_shard.py`), then runs `$VLLM_SCRIPT` (default `cluster/vlm/caption_bakeoff.py`) with `--model /scratch/$USER/models/<MODEL> --name <MODEL> --pp <nodes>` (vllm) | `research-gpu`, 5 nodes with 1 GPU, 10 CPUs, 56 GB each, 8 h | `MODEL_DIR_NAME LAYER_SPLIT [script arguments]`; the model directory is under `/projects/$USER/agai/models/` |

‡ written for the flat working copy: these six files also `cd /projects/$USER/agai/MiniGPT-4`, write their logs to
`/projects/was4g/agai/logs/`, and, except `train_minigpt_splitv1.sbatch`, call scripts by flat names
(`cluster/<script>`).

Environments are the venvs under `/projects/$USER/agai/envs/`. Every job except `eval_parts.sbatch` sets
`HF_HUB_OFFLINE=1`, so models have to be on disk beforehand ([`../infra/hf_fetch.sh`](../infra/hf_fetch.sh)).
[`../synthetic/run_synth_arm.sh`](../synthetic/run_synth_arm.sh) submits `train_backbone.sbatch`, `eval_lowq.sbatch`
and `eval_probe.sbatch`.

## Where jobs run

Every job sources `/projects/$USER/agai/agai_env.sh`, a copy of [`../infra/agai_env.sh`](../infra/agai_env.sh) whose
last line is `cd "$AGAI_ROOT/MiniGPT-4"`, and the six files marked ‡ then `cd /projects/$USER/agai/MiniGPT-4`
themselves. A job therefore runs in `/projects/$USER/agai/MiniGPT-4` whatever folder `sbatch` was called from, and the
relative paths inside it (`cluster/...`, `evaluation/...`, `train.py`, `rfdetr_parts/...`) resolve there. `sbatch`
itself reads the job file from the submission folder.

For the account that ran the revival, `/projects/$USER/agai/MiniGPT-4` is the original working copy, which the
reorganization left unchanged (lab notebook, entry (ad)). For a job to use a folder built from this repository by
[`../infra/make_workdir.sh`](../infra/make_workdir.sh), that folder has to be at `/projects/$USER/agai/MiniGPT-4`, or
the `cd` lines (the copy of `agai_env.sh` and the six ‡ files) have to name it. In such a folder the flat names that
the ‡ files call exist next to the study paths (see [`../infra/README.md`](../infra/README.md#working-copy)).

## Examples

From the folder the jobs run in, with the arguments the files' own usage lines and the lab notebook give:

```bash
sbatch cluster/slurm/train_backbone.sbatch --backbone siglip2-so400m-512 --tag s0
sbatch -p research-gpu cluster/slurm/train_backbone.sbatch --backbone siglip2-so400m-512 --tag s0 \
    --grad-ckpt --batch-size 8
sbatch cluster/slurm/eval_lowq.sbatch --models ft_siglip2-so400m-512_s0,ft_siglip2-so400m-512_s1
sbatch cluster/slurm/eval_probe.sbatch --models ft_siglip2-so400m-512_s0 \
    --manifest evaluation/synth_chatgpt_v1_manifest.json
sbatch cluster/slurm/eval_parts.sbatch large \
    /projects/$USER/agai/rfdetr_runs/large_splitv1/checkpoint_best_total.pth large_splitv1
sbatch cluster/slurm/latency_v1_v2.sbatch \
    /projects/$USER/agai/adapters/Qwen3-VL-8B-Instruct-clsft_reports_v1    # as job 72394
sbatch cluster/slurm/train_minigpt_splitv1.sbatch
sbatch --dependency=afterok:<training job> cluster/slurm/select_and_report.sbatch
sbatch cluster/slurm/minigpt_reports.sbatch --name april_writer --ckpt output/minigpt_strawberry/checkpoint_best.pth \
    --parts evaluation/parts_report_test.json
sbatch -N 4 cluster/slurm/train_vlm_lora_multinode.sbatch \
    --model /projects/$USER/agai/models/Qwen3-VL-32B-Instruct-bnb4 --grad-accum 4
sbatch --dependency=afterok:<teacher job> cluster/slurm/train_vlm_reports.sbatch
sbatch -N 5 --export=ALL,VLLM_SCRIPT=cluster/vlm/judge_claims.py cluster/slurm/vllm_ray.sbatch \
    Qwen3-VL-235B-A22B-Instruct-AWQ 16,20,20,19,19 --modes zoom --max-model-len 8192 --gpu-mem 0.95 --enforce-eager
```

## Known problems

- **Log path.** The six ‡ files write their logs to `/projects/was4g/agai/logs/%x-%j.out`; for any other account that
  line needs `/projects/%u/agai/logs/`, as in the other six files.
- **No `--gres`.** `eval_parts.sbatch`, `minigpt_reports.sbatch`, `reports_after_training.sbatch`,
  `select_and_report.sbatch` and `train_minigpt_splitv1.sbatch` request no GPU with `--gres`.
- **`train_minigpt_splitv1.sbatch`** takes the config path itself as its first argument; the form in its usage line,
  `[--cfg-path other.yaml]`, would pass `--cfg-path` as the path. The default config,
  [`v1/train_configs/minigptv2_strawberry_splitv1.yaml`](../../v1/train_configs/minigptv2_strawberry_splitv1.yaml),
  names absolute paths under `/projects/was4g/agai/` (Llama weights, stage-2 checkpoint, training records, output
  folder), and the run depends on the revival's changes to v1's `minigpt4/` (commit ee006d9).
- **`reports_after_training.sbatch`** looks for a `checkpoint_best.pth` that the training run does not write (it saves
  one checkpoint per epoch; lab notebook, section 2026-10-30). `select_and_report.sbatch` writes that file first with
  `minigpt_val_loss.py`; it is the job the notebook records.
- **Inputs never committed.** `plant_diagnostic/datasets/split_v1/` (training and validation records, the report test
  set), `evaluation/parts_report_test.json`, `plant_diagnostic/models/resnet_strawberry.pth`,
  `rfdetr_parts/output/checkpoint_best_total.pth`, the MiniGPT-v2 checkpoints, and everything under
  `/projects/$USER/agai/` (models, adapters, annotations, runs).
