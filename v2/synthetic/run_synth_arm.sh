#!/bin/bash
# One synthetic-data arm end to end, from /projects/$USER/agai/MiniGPT-4:
#   import the reviewed ChatGPT images -> training tree (hand cap, held-out probe batches) -> SigLIP 2 fine-tunes
#   (3 seeds, same recipe as the real-only and lab-synth arms: batch 16, grad checkpointing on the 32 GB cards) ->
#   low-quality robustness test + shortcut probe on the held-out batches (for this arm and the two baselines).
#   bash cluster/synthetic/run_synth_arm.sh chatgpt_v1 v1_chatgpt healthy_5,root_rot_5
set -eo pipefail
NAME=${1:?synthetic set name}; TREE=${2:?tree name}; HOLDOUT=${3:-}
P=/projects/$USER/agai/synth/prompt_index
/projects/$USER/agai/envs/tools/bin/python cluster/synthetic/import_synth.py --raw /projects/$USER/agai/synth/chatgpt_raw \
    --index $P/batches_v1.json,$P/batches_v2.json,$P/batches_v3.json,$P/batches_v3b.json --name "$NAME"
python3 cluster/synthetic/build_train_tree.py --manifest "evaluation/synth_${NAME}_manifest.json" --name "$TREE" \
    --hand-cap 0.33 ${HOLDOUT:+--holdout "$HOLDOUT"}
TAG=${TREE#v1_}
jobs=()
for s in 0 1 2; do
    jobs+=($(sbatch --parsable -p research-gpu cluster/slurm/train_backbone.sbatch --backbone siglip2-so400m-512 \
        --tag "${TAG}_s$s" --seed $s --train-root "/projects/$USER/agai/splits/$TREE/train" --grad-ckpt --batch-size 16))
done
dep=$(IFS=:; echo "${jobs[*]}")
M=ft_siglip2-so400m-512
new="${M}_${TAG}_s0,${M}_${TAG}_s1,${M}_${TAG}_s2"
base="${M}_s0,${M}_s1,${M}_s2,${M}_labsynth_s0,${M}_labsynth_s1,${M}_labsynth_s2"
sbatch --dependency=afterok:$dep cluster/slurm/eval_lowq.sbatch --models "$new"
[ -n "$HOLDOUT" ] && sbatch --dependency=afterok:$dep cluster/slurm/eval_probe.sbatch --models "$new,$base" \
    --manifest "evaluation/synth_${NAME}_manifest.json" --only "evaluation/split_${TREE}.json"
echo "training jobs: ${jobs[*]}"
