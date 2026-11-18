#!/bin/bash
# AGAI on Hamilton, stage 2: the Python environments, as plain venvs on /projects (conda hangs on
# /projects' NFS locks; see stage 1). Idempotent: finished envs carry a .installed marker and are skipped.
#   envs/minigptv  system python3.9    MiniGPT-v2 demo/training/eval and the ResNet scripts
#   envs/rfdetr    portable python3.11 RF-DETR training/inference (grounding/detector.py calls it via RFDETR_PYTHON)
# Compute nodes only have /usr/bin/python3.9 (the login node also has 3.11), so anything newer comes from a
# relocatable python-build-standalone CPython in $ROOT/python/3.11.
# The lab's env exports never arrived, so minigptv follows the README's install line plus what the code needs:
#   transformers==4.45.2, NOT the README's 4.41.1: grounding/inference.py wraps every forward in a
#     signature-hiding cache_position shim; under 4.41 generate() then builds no attention mask, re-feeds all
#     generated tokens each step (quadratic KV cache -> OOM on 32 GB, repetitive text). 4.42+ slices by
#     cache_position. 4.45 is also the last series exporting the internals minigpt4/models/modeling_llama.py imports.
#   peft with prepare_model_for_kbit_training + 8-bit via BitsAndBytesConfig (minigpt4/models/base_model.py)
#   Gradio 4 (demo_v5.py uses gr.Image(sources=...)); pydantic<2.11 (newer breaks Gradio 4.44)
#   torch cu121 (README); numpy<2 (decord/opencv/timm era)
#   nohup setsid bash hamilton_setup_envs.sh > /projects/$USER/agai/logs/setup_envs.log 2>&1 < /dev/null &
set -eo pipefail
ROOT=/projects/$USER/agai
export PIP_CACHE_DIR=$ROOT/cache/pip PIP_DISABLE_PIP_VERSION_CHECK=1
TORCH_INDEX=https://download.pytorch.org/whl/cu121
PY311=$ROOT/python/3.11/bin/python3.11

echo "== $(date) portable python 3.11"
if [ ! -x "$PY311" ]; then
    url=$(curl -fsS https://api.github.com/repos/astral-sh/python-build-standalone/releases/latest | python3 -c '
import json, sys
for a in json.load(sys.stdin)["assets"]:
    n = a["name"]
    if n.startswith("cpython-3.11.") and n.endswith("-x86_64-unknown-linux-gnu-install_only.tar.gz"):
        print(a["browser_download_url"]); break')
    echo "$url"
    mkdir -p "$ROOT/python/3.11"
    curl -fsSL "$url" | tar xz -C "$ROOT/python/3.11" --strip-components=1
fi
"$PY311" --version

echo "== $(date) envs/minigptv (python3.9)"
if [ ! -f "$ROOT/envs/minigptv/.installed" ]; then
    [ -x "$ROOT/envs/minigptv/bin/python" ] || /usr/bin/python3.9 -m venv "$ROOT/envs/minigptv"
    P=$ROOT/envs/minigptv/bin/pip
    $P install -q --upgrade pip wheel setuptools
    $P install -q --extra-index-url $TORCH_INDEX \
        "torch==2.5.1+cu121" "torchvision==0.20.1+cu121" "numpy<2" \
        "transformers==4.45.2" "tokenizers>=0.20,<0.21" "huggingface-hub>=0.23,<0.26" \
        "accelerate==0.34.2" "peft==0.13.2" "bitsandbytes==0.43.3" sentencepiece \
        "timm==0.6.13" "omegaconf==2.3.0" iopath "webdataset==0.2.48" "decord==0.6.0" \
        opencv-python-headless scikit-image scikit-learn scipy matplotlib psutil pyyaml regex tqdm \
        pandas plotly networkx python-dotenv google-search-results nltk visual-genome wandb pillow \
        "gradio==4.44.1" "pydantic>=2,<2.11"
    $P freeze > "$ROOT/envs/minigptv.freeze.txt"
    touch "$ROOT/envs/minigptv/.installed"
fi
"$ROOT/envs/minigptv/bin/python" -c "import torch, transformers, peft, bitsandbytes, gradio; print('minigptv OK: torch', torch.__version__, '| transformers', transformers.__version__, '| peft', peft.__version__, '| bnb', bitsandbytes.__version__, '| gradio', gradio.__version__)"

echo "== $(date) envs/rfdetr (portable python3.11)"
if [ ! -f "$ROOT/envs/rfdetr/.installed" ] || [ "$(readlink -f "$ROOT/envs/rfdetr/bin/python")" != "$(readlink -f "$PY311")" ]; then
    rm -rf "$ROOT/envs/rfdetr"
    "$PY311" -m venv "$ROOT/envs/rfdetr"
    P=$ROOT/envs/rfdetr/bin/pip
    $P install -q --upgrade pip wheel setuptools
    $P install -q "rfdetr==1.11.2"
    $P freeze > "$ROOT/envs/rfdetr.freeze.txt"
    touch "$ROOT/envs/rfdetr/.installed"
fi
"$ROOT/envs/rfdetr/bin/python" -c "import torch, rfdetr; from rfdetr import RFDETRSmall; print('rfdetr OK: torch', torch.__version__, 'cuda', torch.version.cuda)"

echo "== $(date) envs/vlm (portable python3.11): modern VLMs (Qwen3-VL needs transformers>=4.57)"
if [ ! -f "$ROOT/envs/vlm/.installed" ]; then
    rm -rf "$ROOT/envs/vlm"
    "$PY311" -m venv "$ROOT/envs/vlm"
    P=$ROOT/envs/vlm/bin/pip
    $P install -q --upgrade pip wheel setuptools
    $P install -q --extra-index-url https://download.pytorch.org/whl/cu128 \
        "torch==2.9.*" "torchvision==0.24.*" "transformers==4.57.*" "accelerate>=1.10,<1.11" \
        "peft>=0.17,<0.18" "bitsandbytes>=0.48,<0.49" pillow numpy pandas scikit-learn
    $P freeze > "$ROOT/envs/vlm.freeze.txt"
    touch "$ROOT/envs/vlm/.installed"
fi
"$ROOT/envs/vlm/bin/python" -c "import torch, transformers, peft, bitsandbytes; from transformers import Qwen3VLForConditionalGeneration; print('vlm OK: torch', torch.__version__, '| transformers', transformers.__version__, '| peft', peft.__version__, '| bnb', bitsandbytes.__version__)"

echo "== $(date) envs/vlm5 (portable python3.11): transformers 5.x for Qwen3.8 (qwen3_5 hybrid linear attention)"
if [ ! -f "$ROOT/envs/vlm5/.installed" ]; then
    rm -rf "$ROOT/envs/vlm5"
    "$PY311" -m venv "$ROOT/envs/vlm5"
    P=$ROOT/envs/vlm5/bin/pip
    $P install -q --upgrade pip wheel setuptools
    $P install -q torch torchvision "transformers==5.20.2" "accelerate==1.15.0" "peft==0.21.2" \
        "bitsandbytes==0.50.2" "flash-linear-attention==0.5.2" pillow numpy pandas scikit-learn
    # optional CUDA kernel for the linear-attention layers' short convolution; transformers falls back without it
    $P install -q "causal-conv1d==1.7.0" || echo "causal-conv1d unavailable (torch fallback will be used)"
    $P freeze > "$ROOT/envs/vlm5.freeze.txt"
    touch "$ROOT/envs/vlm5/.installed"
fi
"$ROOT/envs/vlm5/bin/python" -c "import torch, transformers, peft, bitsandbytes; from transformers import AutoModelForImageTextToText; print('vlm5 OK: torch', torch.__version__, 'cuda', torch.version.cuda, '| transformers', transformers.__version__, '| peft', peft.__version__, '| bnb', bitsandbytes.__version__)"

echo "== $(date) ENVS DONE"
