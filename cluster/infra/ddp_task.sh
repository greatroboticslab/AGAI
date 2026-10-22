#!/bin/bash
# One rank of a multi-node LoRA run (srun starts one per node; see cluster/slurm/train_vlm_lora_multinode.sbatch).
# Hamilton's nodes only have IPv4 on their 1 GbE port (10.0.0.x; the 10G ports are up but unconfigured) and the
# port name differs per node, so pick it here for NCCL. Then stage the model on local NVMe and train.
set -eo pipefail
export RANK=$SLURM_PROCID WORLD_SIZE=$SLURM_NTASKS LOCAL_RANK=0
iface=$(ip -4 -o addr | awk '$4 ~ /^10\.0\.0\./ {print $2; exit}')
export NCCL_SOCKET_IFNAME=$iface GLOO_SOCKET_IFNAME=$iface NCCL_IB_DISABLE=1
args=("$@")
for i in "${!args[@]}"; do
    if [ "${args[$i]}" = "--model" ]; then args[$((i + 1))]=$(bash cluster/infra/stage_model.sh "${args[$((i + 1))]}"); fi
done
echo "rank $RANK/$WORLD_SIZE on $(hostname -s) via $iface, master $MASTER_ADDR:$MASTER_PORT"
exec /projects/$USER/agai/envs/${VLM_ENV:-vlm}/bin/python cluster/vlm/train_vlm_lora.py "${args[@]}"
