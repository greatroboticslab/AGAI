"""Stage onto this node's NVMe only the parts of a sharded checkpoint that one pipeline-parallel rank needs, so a model
split over N nodes moves ~1/N of its weights per node over our 1 GbE NFS instead of all of it N times.
Copies every non-safetensors file, every safetensors file holding a tensor of this rank's decoder layers, and every
file holding a non-layer tensor (vision tower, embeddings, final norm, lm_head: small, and staged on all ranks so no
rank can silently miss a weight), then writes a per-node index naming only this rank's files (vLLM refuses an index
naming absent files, and for quantized checkpoints it does NOT check that every weight was loaded, so the rank must
be right: compute it with cluster/infra/ray_pp_order.py after Ray is up). Files from earlier ranks stay on the disk.
The layer split must equal VLLM_PP_LAYER_PARTITION of the run.
  python cluster/infra/stage_pp_shard.py --src /projects/$USER/agai/models/Qwen3-VL-235B-A22B-Instruct-AWQ \
      --partition 16,16,16,16,15,15 --rank 2
"""
import argparse
import json
import os
import re
import shutil
import socket
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from verify_staged import verify  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--src", required=True)
ap.add_argument("--partition", required=True)
ap.add_argument("--rank", type=int, required=True)
ap.add_argument("--dst-root", default=f"/scratch/{os.environ['USER']}/models")
args = ap.parse_args()

cfg = json.load(open(os.path.join(args.src, "config.json")))
n_layers = (cfg.get("text_config") or cfg)["num_hidden_layers"]
parts = [int(x) for x in args.partition.split(",")]
assert sum(parts) == n_layers, f"partition sums to {sum(parts)}, model has {n_layers} layers"
lo = sum(parts[:args.rank])
hi = lo + parts[args.rank]
wmap = json.load(open(os.path.join(args.src, "model.safetensors.index.json")))["weight_map"]
need = set()
for name, f in wmap.items():
    m = re.search(r"(?<!visual\.)(?<!vision_tower\.)\blayers\.(\d+)\.", name)
    if m and "visual" not in name and "vision" not in name:
        layer = int(m.group(1))
        if lo <= layer < hi:
            need.add(f)
    else:
        need.add(f)   # non-layer tensor: everyone gets it
dst = os.path.join(args.dst_root, os.path.basename(args.src.rstrip("/")))
os.makedirs(dst, exist_ok=True)
t0, moved = time.time(), 0


def stage_all():
    global moved
    for f in sorted(os.listdir(args.src)):
        src_f = os.path.join(args.src, f)
        if not os.path.isfile(src_f) or f.startswith(".") or f.endswith(".part"):
            continue
        if f.endswith(".safetensors") and f not in need:
            continue   # files staged for another rank stay (ranks move between runs; the index below hides them)
        out = os.path.join(dst, f)
        if os.path.exists(out) and os.path.getsize(out) == os.path.getsize(src_f):
            continue
        shutil.copyfile(src_f, out + f".part.{os.getpid()}")
        os.replace(out + f".part.{os.getpid()}", out)
        moved += os.path.getsize(out)


stage_all()
# verify this rank's weight files as read from disk against the model's SHA256SUMS (see cluster/infra/verify_staged.py);
# copy mismatching files again once, and stop if they still differ (quantized weights are not checked by vLLM)
bad = verify(args.src, dst, sorted(need), fix=True)
if bad:
    print(f"{socket.gethostname()}: re-copying {len(bad)} corrupt file(s): {' '.join(bad)}", flush=True)
    stage_all()
    bad = verify(args.src, dst, bad)
    if bad:
        sys.exit(f"{socket.gethostname()}: staged files still corrupt after a re-copy (node storage fault): {bad}")
# vLLM refuses an index that names files missing on this node, so this node gets an index of its own files only
index = json.load(open(os.path.join(args.src, "model.safetensors.index.json")))
index["weight_map"] = {k: f for k, f in wmap.items() if f in need}
with open(os.path.join(dst, "model.safetensors.index.json.tmp"), "w") as fh:
    json.dump(index, fh)
os.replace(os.path.join(dst, "model.safetensors.index.json.tmp"), os.path.join(dst, "model.safetensors.index.json"))
open(os.path.join(dst, "STAGED_FOR_RANK"), "w").write(f"{args.rank} {args.partition}\n")
print(f"{socket.gethostname()} rank {args.rank} layers {lo}-{hi - 1}: {len(need)}/{len(set(wmap.values()))} weight files "
      f"needed, {moved / 1e9:.1f} GB copied in {time.time() - t0:.0f}s", flush=True)
