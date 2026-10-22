"""Near-duplicate audit of the strawberry images (perceptual hashes; exact-hash matching misses resized,
re-encoded or lightly cropped copies).

Hashes every image under plant_diagnostic/data/{train,holdout,train_aug} and groups images whose 64-bit pHash
differs in <= --max-dist bits. Reports:
  - cross-label groups (the same photo filed under different classes = label noise)
  - for the 161-image ResNet-clean test set: test images with a near-copy in ResNet v4's paper-era training
    data (exact-clean but visually leaked)
Writes evaluation/near_dupes.json (groups, per-image hashes) for building a leak-free split.
Run from the repo root:  python cluster/benchmark/near_dupes.py   (needs pillow, numpy, imagehash)
"""
import argparse
import collections
import hashlib
import json
import os
import subprocess

import imagehash
import numpy as np
from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument("--max-dist", type=int, default=6)
args = ap.parse_args()

CLASSES = ["drought", "frost", "gray_mold", "healthy", "overwatered", "root_rot", "white_mold"]
IMG = (".jpg", ".jpeg", ".png", ".webp", ".jpg_large")
paths = [os.path.join(d, f) for split in ("train", "holdout", "train_aug") for c in CLASSES
         for d in [os.path.join("plant_diagnostic/data", split, c)] if os.path.isdir(d)
         for f in sorted(os.listdir(d)) if f.lower().endswith(IMG)]

hashes, sha = [], {}
for p in paths:
    with Image.open(p) as im:
        hashes.append(int(str(imagehash.phash(im.convert("RGB"))), 16))
    sha[p] = hashlib.sha256(open(p, "rb").read()).hexdigest()
H = np.array(hashes, dtype=np.uint64)
print(f"hashed {len(paths)} images")

# union-find over pairs within max-dist (popcount of XOR)
parent = list(range(len(paths)))


def find(i):
    while parent[i] != i:
        parent[i] = parent[parent[i]]
        i = parent[i]
    return i


for i in range(len(paths)):
    x = np.bitwise_xor(H[i + 1:], H[i])
    d = np.unpackbits(x.view(np.uint8).reshape(-1, 8), axis=1).sum(axis=1)
    for j in np.nonzero(d <= args.max_dist)[0]:
        a, b = find(i), find(i + 1 + int(j))
        if a != b:
            parent[a] = b
groups = collections.defaultdict(list)
for i, p in enumerate(paths):
    groups[find(i)].append(p)
multi = [g for g in groups.values() if len(g) > 1]
label = lambda p: p.split("/")[-2]
cross = [g for g in multi if len({label(p) for p in g}) > 1]
print(f"near-duplicate groups (>1 image): {len(multi)}; groups spanning different labels: {len(cross)}")
for g in cross[:15]:
    print("   LABEL CONFLICT:", sorted({f'{label(p)}:{p.split("/")[-3]}/{os.path.basename(p)}' for p in g})[:4])

# leakage of the 161 ResNet-clean test images into ResNet v4's paper-era training data
paper = subprocess.run("git ls-tree -r refs/labbox/stash -- plant_diagnostic/data/train plant_diagnostic/data/train_aug; "
                       "git ls-tree -r refs/labbox/stash^3 -- plant_diagnostic/data/train plant_diagnostic/data/train_aug",
                       shell=True, capture_output=True, text=True).stdout
blobs = [l.split()[2] for l in paper.splitlines() if l.split()[1] == "blob"]
cat = subprocess.run(["git", "cat-file", "--batch"], input="\n".join(blobs) + "\n",
                     capture_output=True, text=True, errors="replace").stdout
paper_oids = {l[11:].strip() for l in cat.splitlines() if l.startswith("oid sha256:")}
test = [p for ps in json.load(open("evaluation/clean_eval_sets.json"))["resnet_v4_clean"].values() for p in ps]
group_of = {p: find(i) for i, p in enumerate(paths)}
leaked = []
for t in test:
    mates = [p for p in groups[group_of[t]] if p != t and sha[p] in paper_oids]
    if mates:
        leaked.append({"test": t, "near_copies_in_v4_training": mates[:3]})
print(f"ResNet-clean 161 test images with a near-copy in v4's training data: {len(leaked)}")
for l in leaked[:10]:
    print("   LEAK:", l["test"], "~", l["near_copies_in_v4_training"][0])

json.dump({"max_dist": args.max_dist, "phash": {p: f"{h:016x}" for p, h in zip(paths, hashes)},
           "groups": [g for g in groups.values()], "cross_label_groups": cross, "resnet161_leaks": leaked},
          open("evaluation/near_dupes.json", "w"), indent=1)
