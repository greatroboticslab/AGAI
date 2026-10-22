# Source me on Hamilton:  source /projects/$USER/agai/agai_env.sh
export AGAI_ROOT=/projects/$USER/agai
export PATH=$AGAI_ROOT/bin:$PATH        # standalone git-lfs; the repo's LFS filters need it on PATH
export PIP_CACHE_DIR=$AGAI_ROOT/cache/pip TORCH_HOME=$AGAI_ROOT/cache/torch
export RFDETR_PYTHON=$AGAI_ROOT/envs/rfdetr/bin/python   # grounding/detector.py uses it instead of `conda activate rfdetr`
[ -f ~/.agai_secrets ] && . ~/.agai_secrets
cd "$AGAI_ROOT/MiniGPT-4"
