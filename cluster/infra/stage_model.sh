#!/bin/bash
# Copy a model directory from /projects (NFS, measured ~65 MB/s) to this node's local NVMe (/scratch, GB/s),
# once per node; later jobs on the node find it there and load in seconds. Safe to call from every job:
# files already present with the right size are skipped, and each copy goes through a private temp name.
# If the model has a SHA256SUMS manifest (cluster/infra/make_manifest.py), every staged weight file is verified as read
# from disk (cluster/infra/verify_staged.py; node c5's /scratch silently corrupted files in 2026-10); a mismatching file is
# copied again once, and the job fails if it still differs.
# Prints the local path on stdout.
#   local=$(bash cluster/infra/stage_model.sh /projects/$USER/agai/models/Qwen3-VL-32B-Instruct)
set -eo pipefail
src=${1%/}
dst=/scratch/$USER/models/$(basename "$src")
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
mkdir -p "$dst"
t0=$(date +%s)
copy_missing() {
    (cd "$src" && find . -type f ! -name "*.part" -printf "%P\t%s\n") | while IFS=$'\t' read -r f size; do
        [ -f "$dst/$f" ] && [ "$(stat -c %s "$dst/$f")" = "$size" ] && continue
        mkdir -p "$(dirname "$dst/$f")"
        cp "$src/$f" "$dst/$f.part.$$" && mv "$dst/$f.part.$$" "$dst/$f"
    done
}
copy_missing
if [ -f "$src/SHA256SUMS" ] && ! python3 "$here/verify_staged.py" --fix "$src" "$dst"; then
    copy_missing   # the mismatching files were deleted by --fix
    python3 "$here/verify_staged.py" "$src" "$dst" || {
        echo "staged copy of $(basename "$src") on $(hostname -s) is still corrupt after a re-copy: node storage fault" >&2
        exit 1; }
fi
echo "staged $(basename "$src") on $(hostname -s):/scratch in $(( $(date +%s) - t0 ))s" >&2
echo "$dst"
