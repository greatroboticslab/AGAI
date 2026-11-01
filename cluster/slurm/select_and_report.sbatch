#!/bin/bash
# After the split v1 report-writer training: pick the epoch with the lowest validation loss
# (cluster/vlm/minigpt_val_loss.py), then generate the detector on/off reports with it (cluster/vlm/minigpt_reports.py).
#   sbatch --dependency=afterok:<training job> cluster/slurm/select_and_report.sbatch
#SBATCH -J select_report
#SBATCH -p research-gpu
#SBATCH --exclude=c4,c5
#SBATCH -t 10:00:00
#SBATCH -o /projects/was4g/agai/logs/%x-%j.out
set -eo pipefail
source /projects/$USER/agai/agai_env.sh
export HF_HUB_OFFLINE=1
cd /projects/$USER/agai/MiniGPT-4
RUN=/projects/$USER/agai/minigpt_runs/minigpt_strawberry_splitv1
PY=/projects/$USER/agai/envs/minigptv/bin/python
$PY cluster/minigpt_val_loss.py --name splitv1 --run $RUN --records plant_diagnostic/datasets/split_v1/stage2_val_prompted.json
$PY cluster/minigpt_reports.py --name splitv1 --ckpt $RUN/checkpoint_best.pth --parts evaluation/parts_report_test.json
