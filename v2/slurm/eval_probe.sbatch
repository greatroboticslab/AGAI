#!/bin/bash
# Shortcut probe of saved fine-tuned backbones on unseen synthetic images (cluster/classifier/eval_probe.py). Args pass through.
#   sbatch cluster/slurm/eval_probe.sbatch --models ft_siglip2-so400m-512_s0 --manifest evaluation/synth_chatgpt_v1_manifest.json
#SBATCH --job-name=agai-probe
#SBATCH --partition=research-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=48G
#SBATCH --time=1:00:00
#SBATCH --output=/projects/%u/agai/logs/%x-%j.out

set -eo pipefail
source /projects/$USER/agai/agai_env.sh
export HF_HUB_OFFLINE=1
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
/projects/$USER/agai/envs/vision/bin/python cluster/classifier/eval_probe.py "$@"
