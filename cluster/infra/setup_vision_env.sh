#!/bin/bash
# Separate venv for the vision-backbone probes/fine-tunes and synthetic-image generation (timm, open_clip,
# diffusers, scikit-learn), so installs never touch envs/vlm while fine-tunes run from it. Run on the login node.
set -eo pipefail
source /projects/$USER/agai/agai_env.sh
E=$AGAI_ROOT/envs/vision
[ -f "$E/.installed" ] && { echo "vision env already installed"; exit 0; }
[ -x "$E/bin/python" ] || "$AGAI_ROOT/python/3.11/bin/python3.11" -m venv "$E"
P="$E/bin/python -m pip"
$P install -q --upgrade pip
$P install -q --extra-index-url https://download.pytorch.org/whl/cu128 "torch==2.9.1" "torchvision==0.24.1" \
    "transformers>=4.57" accelerate "diffusers>=0.36" timm open_clip_torch scikit-learn pillow safetensors \
    sentencepiece protobuf
touch "$E/.installed"
"$E/bin/python" -c "import torch, timm, open_clip, diffusers, sklearn, transformers as t; print('vision env OK:', torch.__version__, 'timm', timm.__version__, 'open_clip', open_clip.__version__, 'diffusers', diffusers.__version__, 'transformers', t.__version__)"
