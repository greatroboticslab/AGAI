#!/bin/bash
# Fine-tune a vision backbone with the lab's ResNet recipe (cluster/classifier/train_backbone.py). Args pass through; on the
# A100 by default (sbatch -p research-gpu ... --grad-ckpt --batch-size 8 for the 32 GB cards).
#   sbatch cluster/slurm/train_backbone.sbatch --backbone siglip2-so400m-512 --tag s0
#SBATCH --job-name=agai-ftbb
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=60G
#SBATCH --time=12:00:00
#SBATCH --output=/projects/%u/agai/logs/%x-%j.out

set -eo pipefail
source /projects/$USER/agai/agai_env.sh
export HF_HUB_OFFLINE=1
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
/projects/$USER/agai/envs/vision/bin/python cluster/classifier/train_backbone.py "$@"
