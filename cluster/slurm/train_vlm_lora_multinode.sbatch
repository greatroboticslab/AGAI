#!/bin/bash
# Multi-node LoRA/QLoRA fine-tune (one GPU per node on Hamilton): srun starts cluster/infra/ddp_task.sh on every node.
# Args pass through to cluster/vlm/train_vlm_lora.py, e.g.
#   sbatch -N 4 cluster/slurm/train_vlm_lora_multinode.sbatch --model /projects/$USER/agai/models/Qwen3-VL-32B-Instruct-bnb4 --grad-accum 4
#SBATCH --job-name=agai-vlm-ddp
#SBATCH --partition=research-gpu
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=56G
#SBATCH --time=1-00:00:00
#SBATCH --output=/projects/%u/agai/logs/%x-%j.out

set -eo pipefail
source /projects/$USER/agai/agai_env.sh
export HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -1)
export MASTER_PORT=$((29500 + SLURM_JOB_ID % 1000))
echo "nodes: $SLURM_JOB_NODELIST"
srun --kill-on-bad-exit=1 bash cluster/infra/ddp_task.sh "$@"
