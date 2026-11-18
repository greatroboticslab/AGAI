"""Training split v1 plus the lab's ChatGPT-generated images (plant_diagnostic/data/train_aug/<class>/synth_*),
for the with/without-synthetic ablation. Validation and test stay real-only (split v1).

Synthetic images are deduplicated (sha256, then pHash within the pool) and dropped when they are a near-copy
(pHash distance <= --max-dist) of any split v1 val/test image, so nothing generated from a test photo can leak;
near-copies of real train images are dropped too (they are re-encoded real photos, not new data).
Writes /projects/$USER/agai/splits/<name>/train/<class>/ (symlinks: split v1 train + kept synthetic) and
evaluation/split_<name>.json. Run from the repo root with envs/tools (imagehash):
  /projects/$USER/agai/envs/tools/bin/python cluster/synthetic/build_synth_split.py
"""
import argparse
import collections
import hashlib
import json
import os
import re
import shutil

import imagehash
from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument("--pool", default="plant_diagnostic/data/train_aug")
ap.add_argument("--pattern", default=r"synth_", help="filename regex (re.match) selecting synthetic images")
ap.add_argument("--name", default="v1_gptsynth")
ap.add_argument("--max-dist", type=int, default=6)
args = ap.parse_args()

split = json.load(open("evaluation/split_v1.json"))
real = {s: [(c, p) for c, ps in sorted(split[s].items()) for p in ps] for s in ("train", "val", "test")}
pool = [(c, os.path.join(args.pool, c, f)) for c in sorted(os.listdir(args.pool))
        if os.path.isdir(os.path.join(args.pool, c))
        for f in sorted(os.listdir(os.path.join(args.pool, c))) if re.match(args.pattern, f, re.I)]


def phash(p):
    with Image.open(p) as im:
        return int(str(imagehash.phash(im.convert("RGB"))), 16)


def dist(a, b):
    return bin(a ^ b).count("1")


held = {p: phash(p) for _, p in real["val"] + real["test"]}
train_h = [phash(p) for _, p in real["train"]]
kept, kept_h, seen = [], [], set()
dropped = collections.defaultdict(list)
for c, p in pool:
    sha = hashlib.sha256(open(p, "rb").read()).hexdigest()
    if sha in seen:
        dropped["exact_duplicate"].append(p)
        continue
    seen.add(sha)
    h = phash(p)
    near = [q for q, hq in held.items() if dist(h, hq) <= args.max_dist]
    if near:
        dropped["near_copy_of_val_test"].append({"synthetic": p, "real": near})
    elif any(dist(h, k) <= args.max_dist for k in kept_h):
        dropped["near_duplicate_in_pool"].append(p)
    elif any(dist(h, k) <= args.max_dist for k in train_h):
        dropped["near_copy_of_train"].append(p)
    else:
        kept.append((c, p))
        kept_h.append(h)

out = f"/projects/{os.environ['USER']}/agai/splits/{args.name}/train"
if os.path.isdir(out):
    shutil.rmtree(out)   # symlinks only
for c, p in real["train"]:
    os.makedirs(os.path.join(out, c), exist_ok=True)
    src = p.split("plant_diagnostic/data/")[-1].split("/")[0]
    os.symlink(os.path.abspath(p), os.path.join(out, c, f"{src}__{os.path.basename(p)}"))
for c, p in kept:
    os.symlink(os.path.abspath(p), os.path.join(out, c, f"synth__{os.path.basename(p)}"))

counts = {c: {"real": sum(1 for cc, _ in real["train"] if cc == c), "synthetic": sum(1 for cc, _ in kept if cc == c)}
          for c in sorted(split["train"])}
manifest = {"name": args.name, "pool": args.pool, "pattern": args.pattern, "max_dist": args.max_dist,
            "pool_size": len(pool), "kept": len(kept), "dropped_counts": {k: len(v) for k, v in dropped.items()},
            "train_counts": counts, "train_root": out, "val_root": "splits/v1/val (real only)",
            "test": "split_v1 test (real only)", "synthetic_files": [p for _, p in kept], "dropped": dropped}
json.dump(manifest, open(f"evaluation/split_{args.name}.json", "w"), indent=1)
print(json.dumps({k: manifest[k] for k in ("pool_size", "kept", "dropped_counts", "train_counts")}, indent=1))
