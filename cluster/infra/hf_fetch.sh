#!/bin/bash
# Download a Hugging Face model repo into a directory with curl, verifying LFS files by sha256.
# No huggingface_hub: its cache file locks hang forever on /projects (NFSv3, local_lock=none).
# For gated repos the token comes from ~/.agai_secrets and reaches curl on stdin, never argv.
#   bash hf_fetch.sh Qwen/Qwen3-VL-8B-Instruct /projects/$USER/agai/models/Qwen3-VL-8B-Instruct ['\.bin$']
# Third arg: optional regex of files to skip (dotfiles are always skipped). Idempotent; resumes partial files.
set -eo pipefail
repo=$1 dest=$2 skip=${3:-}
[ -n "$repo" ] && [ -n "$dest" ] || { echo "usage: hf_fetch.sh REPO DEST [SKIP_REGEX]"; exit 2; }
[ -f ~/.agai_secrets ] && source ~/.agai_secrets
auth() { if [ -n "$HF_TOKEN" ]; then printf 'header = "Authorization: Bearer %s"\n' "$HF_TOKEN"; fi; }
mkdir -p "$dest"
auth | curl -fsS -K - "https://huggingface.co/api/models/$repo?blobs=true" > "$dest/.manifest.json"
python3 - "$dest/.manifest.json" "$skip" > "$dest/.files.tsv" <<'EOF'
import json, re, sys
skip = re.compile(sys.argv[2]) if sys.argv[2] else None
for s in json.load(open(sys.argv[1]))["siblings"]:
    f = s["rfilename"]
    if f.startswith(".") or (skip and skip.search(f)):
        continue
    print(f, s.get("size", 0), (s.get("lfs") or {}).get("sha256", ""), sep="\t")
EOF
while IFS=$'\t' read -r f size sha; do
    out=$dest/$f
    mkdir -p "$(dirname "$out")"
    [ -s "$out" ] && [ "$(stat -c %s "$out")" = "$size" ] && continue
    auth | curl -fL -sS -C - -K - -o "$out.part" "https://huggingface.co/$repo/resolve/main/$f"
    [ -n "$sha" ] && echo "$sha  $out.part" | sha256sum -c --quiet
    mv "$out.part" "$out"
    echo "got $f ($size bytes${sha:+, sha256 ok})"
done < "$dest/.files.tsv"
du -sh "$dest"
echo "DONE $repo"
