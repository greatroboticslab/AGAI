#!/bin/bash
# One big VLM over the research-gpu nodes: vLLM pipeline parallelism across N nodes (one RTX 5000 Ada each) on a Ray
# cluster, over the 1 GbE network (only activations cross nodes, so 1 GbE is fine for batch inference).
# Order matters: Ray starts first, cluster/infra/ray_pp_order.py reproduces vLLM's rank order (driver node, then Ray node
# IDs, which are random per start), and only then does each node stage the weight files of ITS rank
# (cluster/infra/stage_pp_shard.py; files from earlier runs stay, so later runs copy less).
#   sbatch -N 5 --exclude=c4 cluster/slurm/vllm_ray.sbatch MODEL_DIR_NAME LAYER_SPLIT [caption_bakeoff args]
# Another vLLM script that takes --model/--name/--pp (cluster/vlm/annotate_vllm.py):
#   sbatch -N 5 --exclude=c4 --export=ALL,VLLM_SCRIPT=cluster/vlm/annotate_vllm.py cluster/slurm/vllm_ray.sbatch \
#       MODEL_DIR_NAME LAYER_SPLIT --tasks test:both --dtype auto ...
#SBATCH --job-name=agai-vllm-ray
#SBATCH --partition=research-gpu
#SBATCH --nodes=5
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=56G
#SBATCH --time=8:00:00
#SBATCH --output=/projects/%u/agai/logs/%x-%j.out

set -eo pipefail
source /projects/$USER/agai/agai_env.sh
M=$1; PART=$2; shift 2
SCRIPT=${VLLM_SCRIPT:-cluster/vlm/caption_bakeoff.py}
SRC=/projects/$USER/agai/models/$M
PY=/projects/$USER/agai/envs/vllm/bin/python
RAY=/projects/$USER/agai/envs/vllm/bin/ray
export HF_HUB_OFFLINE=1 VLLM_USE_FLASHINFER_SAMPLER=0 PYTHONUNBUFFERED=1 VLLM_PP_LAYER_PARTITION=$PART NCCL_IB_DISABLE=1
# JIT caches on each node's own disk: five nodes compiling the same Triton kernel into the NFS home cache at once
# fail with "Stale file handle" (GLM-4.6V, job 72117)
export TRITON_CACHE_DIR=/scratch/$USER/cache/triton VLLM_CACHE_ROOT=/scratch/$USER/cache/vllm \
       TORCHINDUCTOR_CACHE_DIR=/scratch/$USER/cache/inductor
nodes=($(scontrol show hostnames "$SLURM_JOB_NODELIST"))
# each node's 10.0.0.x port (the only configured one; its name differs: c3 enp4s0, c5-c8 enp2s0)
declare -A ip ifc host_of
for n in "${nodes[@]}"; do
    read -r "ifc[$n]" "ip[$n]" < <(srun -N1 -n1 -w "$n" --overlap bash -c "ip -4 -o addr | awk '\$4 ~ /^10[.]0[.]0[.]/ {split(\$4, a, \"/\"); print \$2, a[1]; exit}'")
    [ -n "${ip[$n]}" ] || { echo "no 10.0.0.x address on $n"; exit 1; }
    host_of[${ip[$n]}]=$n
done
head=${nodes[0]}
port=$((6379 + SLURM_JOB_ID % 1000))
# Ray token auth: without it anyone on the cluster network could submit code to this Ray cluster as $USER.
# The token lives in ~/.ray/auth_token (shared home, never printed); every ray process below inherits the mode.
$RAY get-auth-token --generate >/dev/null 2>&1
chmod 600 ~/.ray/auth_token
export RAY_AUTH_MODE=token
srun -N1 -n1 -w "$head" --overlap --gres=gpu:1 bash -c "export VLLM_HOST_IP=${ip[$head]} NCCL_SOCKET_IFNAME=${ifc[$head]} GLOO_SOCKET_IFNAME=${ifc[$head]}; exec $RAY start --head --node-ip-address=${ip[$head]} --port=$port --num-gpus=1 --num-cpus=8 --disable-usage-stats --block" &
sleep 20
for n in "${nodes[@]:1}"; do
    srun -N1 -n1 -w "$n" --overlap --gres=gpu:1 bash -c "export VLLM_HOST_IP=${ip[$n]} NCCL_SOCKET_IFNAME=${ifc[$n]} GLOO_SOCKET_IFNAME=${ifc[$n]}; exec $RAY start --address=${ip[$head]}:$port --node-ip-address=${ip[$n]} --num-gpus=1 --num-cpus=8 --block" &
done
export RAY_ADDRESS=${ip[$head]}:$port VLLM_HOST_IP=${ip[$head]} NCCL_SOCKET_IFNAME=${ifc[$head]} GLOO_SOCKET_IFNAME=${ifc[$head]}
for i in $(seq 1 30); do
    g=$($PY -c "import ray; ray.init(address='auto', logging_level='ERROR', log_to_driver=False); print(int(ray.cluster_resources().get('GPU', 0)))" 2>/dev/null || echo 0)
    [ "$g" -ge "${#nodes[@]}" ] && break
    sleep 5
done
echo "Ray cluster: $g GPUs"
order_ips=($($PY cluster/infra/ray_pp_order.py "${ip[$head]}"))
[ "${#order_ips[@]}" -eq "${#nodes[@]}" ] || { echo "rank order incomplete: ${order_ips[*]}"; exit 1; }
echo "PP rank order (vLLM): $(for r in "${!order_ips[@]}"; do echo -n "$r=${host_of[${order_ips[$r]}]}(${order_ips[$r]}) "; done)"
t0=$(date +%s)
stage_pids=()
for r in "${!order_ips[@]}"; do
    srun -N1 -n1 -w "${host_of[${order_ips[$r]}]}" --overlap $PY cluster/infra/stage_pp_shard.py --src "$SRC" --partition "$PART" --rank "$r" &
    stage_pids+=($!)
done
for p in "${stage_pids[@]}"; do wait "$p" || { echo "staging failed"; $RAY stop --force; kill $(jobs -p); exit 1; }; done
echo "staging done in $(( $(date +%s) - t0 ))s"
set +e
$PY "$SCRIPT" --model "/scratch/$USER/models/$M" --name "$M" --pp ${#nodes[@]} "$@"
rc=$?
$RAY stop --force >/dev/null 2>&1
kill $(jobs -p) 2>/dev/null
exit $rc
