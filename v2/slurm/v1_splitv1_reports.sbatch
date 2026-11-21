#!/bin/bash
# v1's pipeline as deployed (cluster/latency_v1.py: two-view ResNet-50, acceptance gate, RF-DETR subprocess, MiniGPT-v2
# report with the consistency check) run with the models retrained on split v1 instead of the lab's checkpoints, on
# the 49 photographs of the faithfulness comparison (lab notebook, entry (aj)).
#   sbatch cluster/slurm/v1_splitv1_reports.sbatch
#SBATCH -J v1splitv1
#SBATCH -p research-gpu
#SBATCH --exclude=c4,c5
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=56G
#SBATCH -t 03:00:00
#SBATCH -o /projects/was4g/agai/logs/%x-%j.out
set -eo pipefail
source /projects/$USER/agai/agai_env.sh
export PYTHONUNBUFFERED=1
cd /projects/$USER/agai/MiniGPT-4
R=/projects/$USER/agai
export RFDETR_CHECKPOINT=$R/rfdetr_runs/small_splitv1/checkpoint_best_total.pth
RESNET=$R/models/resnet50_split_v1.pth
WRITER=$R/minigpt_runs/minigpt_strawberry_splitv1/checkpoint_best.pth
[ -x "${RFDETR_PYTHON:-}" ] || { echo "RFDETR_PYTHON is not an executable interpreter: '${RFDETR_PYTHON:-}'" >&2; exit 1; }
grep -q RFDETR_CHECKPOINT grounding/detector.py || { echo "grounding/detector.py ignores RFDETR_CHECKPOINT" >&2; exit 1; }
grep -q writer-ckpt cluster/latency_v1.py || { echo "cluster/latency_v1.py has no --writer-ckpt" >&2; exit 1; }
for f in "$RFDETR_CHECKPOINT" "$RESNET" "$WRITER" evaluation/split_v1.json; do
    [ -f "$f" ] || { echo "missing: $f" >&2; exit 1; }
done
echo "on $(hostname -s), CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader
sha256sum "$RESNET" "$RFDETR_CHECKPOINT" "$WRITER"
$R/envs/minigptv/bin/python cluster/latency_v1.py --resnet "$RESNET" --writer-ckpt "$WRITER" \
    --out evaluation/latency_v1_splitv1.json
