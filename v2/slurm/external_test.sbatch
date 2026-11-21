#!/bin/bash
# External test (lab notebook, entry (al)): v1's ResNet-50 and v2's SigLIP 2 classifier, each with both acceptance
# gates, on the test split of the public Mendeley Data release k2ptxmfjhj (463 photographs, verified by SHA-256).
#   sbatch cluster/slurm/external_test.sbatch
#SBATCH -J external
#SBATCH -p research-gpu
#SBATCH --exclude=c4,c5
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -t 01:00:00
#SBATCH -o /projects/was4g/agai/logs/%x-%j.out
set -eo pipefail
source /projects/$USER/agai/agai_env.sh
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1
cd /projects/$USER/agai/MiniGPT-4
D=/projects/$USER/agai/external/mendeley_k2ptxmfjhj_v2/traditional_test
[ "$(ls "$D"/*.jpg | wc -l)" = 463 ] || { echo "expected 463 photographs in $D" >&2; exit 1; }
for f in "$D/_annotations.coco.json" /projects/$USER/agai/models/resnet50_split_v1.pth \
         /projects/$USER/agai/models/ft_siglip2-so400m-512_chatgpt_nof3_lq_s0.pt \
         evaluation/calib_resnet50_v1_summary.json evaluation/calib_siglip2_recipe_summary.json; do
    [ -f "$f" ] || { echo "missing: $f" >&2; exit 1; }
done
echo "on $(hostname -s), CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
/projects/$USER/agai/envs/vision/bin/python cluster/external_test.py --images "$D" --out evaluation/external_mendeley_test.json
