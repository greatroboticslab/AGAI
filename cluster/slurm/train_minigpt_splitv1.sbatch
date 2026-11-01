#!/bin/bash
# Retrain the v1 MiniGPT-v2 report writer on split v1 (train_configs/minigptv2_strawberry_splitv1.yaml): records of
# training photographs only, validation loss selects the checkpoint, the training instruction is the inference prompt.
#   sbatch cluster/slurm/train_minigpt_splitv1.sbatch [--cfg-path other.yaml]
#SBATCH -J minigpt_splitv1
#SBATCH -p research-gpu
#SBATCH --exclude=c4,c5
#SBATCH -t 14:00:00
#SBATCH -o /projects/was4g/agai/logs/%x-%j.out
set -eo pipefail
source /projects/$USER/agai/agai_env.sh
export HF_HUB_OFFLINE=1
cd /projects/$USER/agai/MiniGPT-4
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
CFG=${1:-train_configs/minigptv2_strawberry_splitv1.yaml}
/projects/$USER/agai/envs/minigptv/bin/python train.py --cfg-path "$CFG"
