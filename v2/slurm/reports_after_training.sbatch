#!/bin/bash
# Reports of the retrained writer for the detector on/off ablation, submitted with --dependency=afterok:<training job>:
# picks the newest checkpoint_best.pth of the split v1 run (the runner names the run folder by its start time).
#   sbatch --dependency=afterok:72341 cluster/slurm/reports_after_training.sbatch
#SBATCH -J reports_splitv1
#SBATCH -p research-gpu
#SBATCH --exclude=c4,c5
#SBATCH -t 08:00:00
#SBATCH -o /projects/was4g/agai/logs/%x-%j.out
set -eo pipefail
source /projects/$USER/agai/agai_env.sh
export HF_HUB_OFFLINE=1
cd /projects/$USER/agai/MiniGPT-4
RUN=/projects/$USER/agai/minigpt_runs/minigpt_strawberry_splitv1   # the runner writes checkpoints here (or in a dated subfolder)
CKPT=$(ls -t $RUN/checkpoint_best.pth $RUN/*/checkpoint_best.pth 2>/dev/null | head -1)
[ -n "$CKPT" ] || { echo "no checkpoint_best.pth under minigpt_runs/minigpt_strawberry_splitv1 (validation never improved?)"; ls -R /projects/$USER/agai/minigpt_runs/minigpt_strawberry_splitv1 | head -20; exit 1; }
echo "using $CKPT"; ls -la "$CKPT"
/projects/$USER/agai/envs/minigptv/bin/python cluster/minigpt_reports.py --name splitv1 --ckpt "$CKPT" --parts evaluation/parts_report_test.json
