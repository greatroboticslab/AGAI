#!/bin/bash
# v1 vs v2 end-to-end latency and GPU memory on one RTX 5000 Ada (manuscript 3.9 / abstract): the four stage scripts,
# each in its own environment and process, on the same 49 split v1 test photographs, then the summary table.
# The student is merged from its adapter onto the node's /scratch when no merged copy is there yet.
#   sbatch cluster/slurm/latency_v1_v2.sbatch /projects/$USER/agai/adapters/<student adapter> [SIGLIP_CKPT_NAME] [BASE_DIR]
# BASE_DIR defaults to /projects/$USER/agai/models/<basename of the adapter's base_model_name_or_path>: the paper's
# student (Qwen3-VL-8B-Instruct-clsft_reports_v1) sits on the classification-tuned 8B, not on the plain 8B.
#SBATCH -J latency
#SBATCH -p research-gpu
#SBATCH --exclude=c4,c5
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=56G
#SBATCH -t 04:00:00
#SBATCH -o /projects/was4g/agai/logs/%x-%j.out
set -eo pipefail
source /projects/$USER/agai/agai_env.sh
export HF_HUB_OFFLINE=1 VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_WORKER_MULTIPROC_METHOD=spawn PYTHONUNBUFFERED=1
cd /projects/$USER/agai/MiniGPT-4
ADAPTER=${1:?student adapter directory}
SIGLIP=${2:-ft_siglip2-so400m-512_chatgpt_nof3_lq_s0}
E=/projects/$USER/agai/envs
BASE_NAME=$(python3 -c 'import json, os, sys; print(os.path.basename(json.load(open(sys.argv[1]))["base_model_name_or_path"].rstrip("/")))' \
    "$ADAPTER/adapter_config.json")
BASE_SRC=${3:-/projects/$USER/agai/models/$BASE_NAME}
# pre-flight: fail here, not after the GPU slot is spent
[ -x "${RFDETR_PYTHON:-}" ] || { echo "RFDETR_PYTHON is not an executable interpreter: '${RFDETR_PYTHON:-}'" >&2; exit 1; }
grep -q RFDETR_PYTHON grounding/detector.py \
    || { echo "grounding/detector.py ignores RFDETR_PYTHON: the detector subprocess would fail (no conda here)" >&2; exit 1; }
for f in rfdetr_parts/output/checkpoint_best_total.pth /projects/$USER/agai/rfdetr_runs/large_splitv1/checkpoint_best_total.pth \
         /projects/$USER/agai/models/$SIGLIP.pt "$ADAPTER/adapter_config.json" "$BASE_SRC/config.json" \
         plant_diagnostic/models/resnet_strawberry.pth evaluation/split_v1.json; do
    [ -f "$f" ] || { echo "missing: $f" >&2; exit 1; }
done
echo "adapter $ADAPTER on base $BASE_SRC; classifier $SIGLIP"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset} on $(hostname -s)"
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader
echo "== v1 as deployed (minigptv) =="
$E/minigptv/bin/python cluster/latency_v1.py
echo "== RF-DETR Small (lab) and Large (split v1), in-process (rfdetr) =="
$E/rfdetr/bin/python cluster/latency_rfdetr.py \
    --weights-large /projects/$USER/agai/rfdetr_runs/large_splitv1/checkpoint_best_total.pth
echo "== SigLIP 2 classifier (vision) =="
$E/vision/bin/python cluster/latency_siglip.py --model "$SIGLIP"
echo "== student under vLLM (vllm) =="
BASE=$(bash cluster/stage_model.sh "$BASE_SRC")
MERGED=/scratch/$USER/models/$(basename "$BASE")-$(basename "$ADAPTER")
if [ ! -f "$MERGED/config.json" ]; then
    $E/vlm/bin/python cluster/merge_lora.py "$BASE" "$ADAPTER" "$MERGED"
    for f in tokenizer.json tokenizer_config.json vocab.json merges.txt preprocessor_config.json chat_template.json \
             chat_template.jinja video_preprocessor_config.json generation_config.json; do   # the base's files load in vLLM
        [ -f "$BASE/$f" ] && cp "$BASE/$f" "$MERGED/$f"
    done
fi
$E/vllm/bin/python cluster/latency_student.py --model "$MERGED" --labels evaluation/latency_siglip2.json
echo "== summary =="
$E/vision/bin/python cluster/latency_summary.py
