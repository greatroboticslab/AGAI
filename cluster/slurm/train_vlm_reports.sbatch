#!/bin/bash
# Honest-report student: LoRA-distil a Qwen3-VL-8B on the teacher's structured annotations (incl. wrong-label
# examples, filtered to the teacher's correct verdicts), merge the adapter, then run the label-swap honesty test on
# val+test with vLLM. One GPU, end to end.
#   sbatch --dependency=afterok:<teacher job> cluster/slurm/train_vlm_reports.sbatch [TEACHER_RUN_DIR] [BASE_MODEL] [TAG]
#SBATCH --job-name=agai-reports
#SBATCH --partition=research-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=56G
#SBATCH --time=12:00:00
#SBATCH --output=/projects/%u/agai/logs/%x-%j.out

set -eo pipefail
source /projects/$USER/agai/agai_env.sh
export HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True VLLM_WORKER_MULTIPROC_METHOD=spawn VLLM_USE_FLASHINFER_SAMPLER=0
T=${1:-/projects/$USER/agai/annotations/v1_Qwen3-VL-32B-Instruct}
BASE=$(bash cluster/infra/stage_model.sh "${2:-/projects/$USER/agai/models/Qwen3-VL-8B-Instruct}")
TAG=${3:-reports_v1}
NAME=$(basename "$BASE")
ADAPTER=/projects/$USER/agai/adapters/${NAME}_${TAG}
MERGED=/scratch/$USER/models/${NAME}-${TAG}
/projects/$USER/agai/envs/vlm/bin/python cluster/vlm/train_vlm_reports.py --model "$BASE" \
    --train "$T/train_true.jsonl,$T/train_swap.jsonl" --val "$T/val_both.jsonl" --tag "$TAG"
/projects/$USER/agai/envs/vlm/bin/python cluster/vlm/merge_lora.py "$BASE" "$ADAPTER" "$MERGED"
/projects/$USER/agai/envs/vllm/bin/python cluster/vlm/annotate_vllm.py --model "$MERGED" --tasks val:both,test:both \
    --run student --gpu-mem 0.90
/projects/$USER/agai/envs/vision/bin/python cluster/vlm/score_honesty.py "$T/test_both.jsonl" \
    /projects/$USER/agai/annotations/student_${NAME}-${TAG}/test_both.jsonl \
    /projects/$USER/agai/annotations/base_Qwen3-VL-8B-Instruct/test_both.jsonl 2>&1 || true
