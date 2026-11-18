#!/bin/bash
# Reports for the detector on/off ablation with one MiniGPT-v2 checkpoint (cluster/vlm/minigpt_reports.py). Args pass through.
#   sbatch cluster/slurm/minigpt_reports.sbatch --name april_writer --ckpt output/minigpt_strawberry/checkpoint_best.pth --parts evaluation/parts_report_test.json
#SBATCH -J reports
#SBATCH -p research-gpu
#SBATCH --exclude=c4,c5
#SBATCH -t 08:00:00
#SBATCH -o /projects/was4g/agai/logs/%x-%j.out
set -eo pipefail
source /projects/$USER/agai/agai_env.sh
export HF_HUB_OFFLINE=1
cd /projects/$USER/agai/MiniGPT-4
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader
/projects/$USER/agai/envs/minigptv/bin/python cluster/minigpt_reports.py "$@"
