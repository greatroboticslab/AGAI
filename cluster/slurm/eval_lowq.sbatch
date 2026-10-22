#!/bin/bash
# Low-quality-photo robustness of saved fine-tuned backbones (cluster/classifier/eval_lowq.py). Args pass through.
#   sbatch cluster/slurm/eval_lowq.sbatch --models ft_siglip2-so400m-512_s0,ft_siglip2-so400m-512_s1
#SBATCH --job-name=agai-lowq
#SBATCH --partition=research-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=48G
#SBATCH --time=3:00:00
#SBATCH --output=/projects/%u/agai/logs/%x-%j.out

set -eo pipefail
source /projects/$USER/agai/agai_env.sh
export HF_HUB_OFFLINE=1
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
/projects/$USER/agai/envs/vision/bin/python cluster/classifier/eval_lowq.py "$@"
