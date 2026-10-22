#!/bin/bash
# AGAI on Hamilton, stage 1 (needs no lab data): git-lfs, the repo with its LFS images, and the public
# weights (MiniGPT-v2 stage-2, EVA-ViT-g, Llama-2-7b-chat-hf). Idempotent; rerun safely.
#   nohup setsid bash hamilton_setup_stage1.sh > /projects/$USER/agai/logs/setup_stage1.log 2>&1 < /dev/null &
# No conda and no huggingface_hub: /projects is NFSv3 with server-side locking (local_lock=none) and
# anything that takes file locks there (conda/mamba, HF cache) blocks forever. venv, pip and curl don't.
# Layout mirrors the old lab box: /data/AGAI/MiniGPT-4  ->  $ROOT/MiniGPT-4
set -eo pipefail
ROOT=/projects/$USER/agai
REPO=$ROOT/MiniGPT-4
mkdir -p "$ROOT"/{bin,envs,cache/pip,cache/torch/hub/checkpoints,incoming,logs}
export PATH=$ROOT/bin:$PATH PIP_CACHE_DIR=$ROOT/cache/pip

echo "== $(date) git-lfs (static binary; the cluster's git has no LFS)"
if [ ! -x "$ROOT/bin/git-lfs" ]; then
    tmp=$(mktemp -d)
    curl -fsSL https://github.com/git-lfs/git-lfs/releases/download/v3.7.1/git-lfs-linux-amd64-v3.7.1.tar.gz | tar xz -C "$tmp"
    find "$tmp" -name git-lfs -type f -exec cp {} "$ROOT/bin/" \;
    rm -rf "$tmp"
fi
git lfs version

echo "== $(date) repo"
if [ ! -d "$REPO/.git" ]; then
    GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/adainstarks/AGAI.git "$REPO"
    git -C "$REPO" remote add lab https://github.com/greatroboticslab/AGAI.git
fi
git -C "$REPO" lfs install --local   # repo-local filter config, so plain git elsewhere is unaffected
git -C "$REPO" lfs pull
git -C "$REPO" lfs ls-files | awk '{print $2}' | sort | uniq -c   # expect only '*' (downloaded)

echo "== $(date) tools venv"
[ -x "$ROOT/envs/tools/bin/python" ] || /usr/bin/python3.9 -m venv "$ROOT/envs/tools"
"$ROOT/envs/tools/bin/pip" install -q --upgrade pip gdown

echo "== $(date) MiniGPT-v2 stage-2 checkpoint + EVA-ViT-g"
mkdir -p "$REPO/checkpoints"
ckpt=$REPO/checkpoints/checkpoint_stage2.pth   # training start point for the strawberry fine-tune
[ -s "$ckpt" ] || { "$ROOT/envs/tools/bin/gdown" --continue "https://drive.google.com/uc?id=1Vi_E7ZtZXRAQcyz4f8E6LtLh2UXABCmu" -O "$ckpt.part" && mv "$ckpt.part" "$ckpt"; }
eva=$ROOT/cache/torch/hub/checkpoints/eva_vit_g.pth   # the code otherwise downloads it at runtime
[ -s "$eva" ] || { curl -fL -sS -C - -o "$eva.part" https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/eva_vit_g.pth && mv "$eva.part" "$eva"; }
ls -lh "$ckpt" "$eva"

echo "== $(date) Llama-2-7b-chat-hf (gated: token from ~/.agai_secrets, given to curl on stdin, not argv)"
source ~/.agai_secrets
llama=$REPO/llama_weights/Llama-2-7b-chat-hf
mkdir -p "$llama"
auth() { printf 'header = "Authorization: Bearer %s"\n' "$HF_TOKEN"; }
auth | curl -fsS -K - "https://huggingface.co/api/models/meta-llama/Llama-2-7b-chat-hf?blobs=true" > "$llama/.manifest.json"
# every file except the *.bin shards (identical weights to the safetensors) and dotfiles
python3 - "$llama/.manifest.json" > "$llama/.files.tsv" <<'EOF'
import json, sys
for s in json.load(open(sys.argv[1]))["siblings"]:
    f = s["rfilename"]
    if not (f.endswith(".bin") or f.startswith(".")):
        print(f, s.get("size", 0), (s.get("lfs") or {}).get("sha256", ""), sep="\t")
EOF
while IFS=$'\t' read -r f size sha; do
    out=$llama/$f
    [ -s "$out" ] && [ "$(stat -c %s "$out")" = "$size" ] && continue
    auth | curl -fL -sS -C - -K - -o "$out.part" "https://huggingface.co/meta-llama/Llama-2-7b-chat-hf/resolve/main/$f"
    [ -n "$sha" ] && echo "$sha  $out.part" | sha256sum -c --quiet
    mv "$out.part" "$out"
    echo "got $f ($size bytes${sha:+, sha256 ok})"
done < "$llama/.files.tsv"

du -sh "$ROOT"
echo "== $(date) STAGE 1 DONE"
