"""Which of the lab's current labeled images are fair test data for which model?

- Zero-shot VLMs never trained on any of them, so every original image (data/train + data/holdout) is fair.
- ResNet v4 trained on the paper-era (2026-04-25) train/train_aug, recorded as LFS pointers in the lab's
  stash (refs/labbox/stash: tracked files; ^3: untracked augbal additions). An image is clean for v4 only if
  its content hash is in neither.
Writes evaluation/clean_eval_sets.json with per-class lists. Run from the repo root with system python3.
"""
import collections
import hashlib
import json
import os
import subprocess

STASH = "refs/labbox/stash"
CLASSES = ["drought", "frost", "gray_mold", "healthy", "overwatered", "root_rot", "white_mold"]
IMG = (".jpg", ".jpeg", ".png", ".webp", ".jpg_large")


def lfs_oids(rev):
    """sha256 oids of every LFS image under data/train and data/train_aug in a commit."""
    ls = subprocess.run(["git", "ls-tree", "-r", rev, "--", "plant_diagnostic/data/train",
                         "plant_diagnostic/data/train_aug"], capture_output=True, text=True, check=True).stdout
    blobs = [line.split()[2] for line in ls.splitlines() if line.split()[1] == "blob"]
    out = subprocess.run(["git", "cat-file", "--batch"], input="\n".join(blobs) + "\n",
                         capture_output=True, text=True, errors="replace", check=True).stdout
    return {line[len("oid sha256:"):].strip() for line in out.splitlines() if line.startswith("oid sha256:")}


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


paper_train = lfs_oids(STASH) | lfs_oids(STASH + "^3")
print(f"paper-era train/train_aug images (unique content): {len(paper_train)}")

sets = {"vlm_all_originals": collections.defaultdict(list), "resnet_v4_clean": collections.defaultdict(list)}
seen = set()
for split in ("holdout", "train"):
    for c in CLASSES:
        d = os.path.join("plant_diagnostic/data", split, c)
        for f in sorted(os.listdir(d)) if os.path.isdir(d) else []:
            if not f.lower().endswith(IMG):
                continue
            p = os.path.join(d, f)
            h = sha(p)
            if h in seen:          # exact duplicates count once
                continue
            seen.add(h)
            sets["vlm_all_originals"][c].append(p)
            if h not in paper_train:
                sets["resnet_v4_clean"][c].append(p)

for name, s in sets.items():
    print(f"{name}: {sum(len(v) for v in s.values())} images | " + ", ".join(f"{c} {len(s[c])}" for c in CLASSES))
json.dump({k: dict(v) for k, v in sets.items()}, open("evaluation/clean_eval_sets.json", "w"), indent=1)
