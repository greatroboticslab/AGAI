#!/bin/bash
# Per-part detector metrics on the detector split that agrees with split v1 (cluster/detector/eval_parts.py): the
# threshold sweep on the validation split, then the test split with those thresholds.
#   sbatch [--dependency=afterok:<train job>] cluster/slurm/eval_parts.sbatch SIZE WEIGHTS NAME [DATASET_DIR]
#SBATCH -J eval_parts
#SBATCH -p research-gpu
#SBATCH --exclude=c4,c5
#SBATCH -t 01:00:00
#SBATCH -o /projects/was4g/agai/logs/%x-%j.out
set -eo pipefail
source /projects/$USER/agai/agai_env.sh
cd /projects/$USER/agai/MiniGPT-4
SIZE=$1; WEIGHTS=$2; NAME=$3; DS=${4:-rfdetr_parts/dataset_splitv1}
PY=/projects/$USER/agai/envs/rfdetr/bin/python
$PY cluster/eval_parts.py --size "$SIZE" --weights "$WEIGHTS" --dataset-dir "$DS" --split valid --name "$NAME"
$PY cluster/eval_parts.py --size "$SIZE" --weights "$WEIGHTS" --dataset-dir "$DS" --split test --name "$NAME" \
    --thresholds "evaluation/parts_${NAME}_valid.json"
