"""Mirror-aware near-duplicate audit of split v1 (the manuscript's audit rule): perceptual hash (pHash) and difference
hash (dHash) of every image and of its mirror, after EXIF orientation; two images are near-duplicates when the pHash
distance is at most --phash bits and the dHash distance at most --dhash bits for the image or for either mirror.
Reports the test images that have a near-duplicate in training or validation (and validation images with one in
training), the mirrored pairs across sets, and writes the groups so that a sensitivity analysis can drop the affected
test images.
  python cluster/benchmark/mirror_dupes.py --split evaluation/split_v1.json --out evaluation/mirror_dupes_v1.json
"""
import argparse
import itertools
import json
import os

import imagehash
from PIL import Image, ImageOps

ap = argparse.ArgumentParser()
ap.add_argument("--split", default="evaluation/split_v1.json")
ap.add_argument("--root", default=".", help="directory that the split's image paths are relative to")
ap.add_argument("--phash", type=int, default=8)
ap.add_argument("--dhash", type=int, default=10)
ap.add_argument("--out", default="evaluation/mirror_dupes_v1.json")
args = ap.parse_args()

split = json.load(open(args.split))
members = {}                                   # path -> (set, label)
for part in ("train", "val", "test"):
    for label, paths in split[part].items():
        for p in paths:
            members[p] = (part, label)
paths = sorted(members)
print(f"{len(paths)} images: " + ", ".join(f"{part} {sum(1 for p in paths if members[p][0] == part)}" for part in ("train", "val", "test")))

hashes = {}
for i, p in enumerate(paths):
    try:
        im = ImageOps.exif_transpose(Image.open(os.path.join(args.root, p))).convert("RGB")
    except Exception as e:
        print("unreadable:", p, e)
        continue
    mir = ImageOps.mirror(im)
    hashes[p] = ((imagehash.phash(im), imagehash.dhash(im)), (imagehash.phash(mir), imagehash.dhash(mir)))
    if (i + 1) % 400 == 0:
        print(f"  hashed {i + 1}")


def near(a, b):
    """(near, mirrored): near if image-image or image-mirror pairs fall within both thresholds"""
    (pa, da), (pam, dam) = a
    (pb, db), _ = b
    if pa - pb <= args.phash and da - db <= args.dhash:
        return True, False
    if pam - pb <= args.phash and dam - db <= args.dhash:
        return True, True
    return False, False


pairs = []
hp = [p for p in paths if p in hashes]
for a, b in itertools.combinations(hp, 2):
    n, m = near(hashes[a], hashes[b])
    if n:
        pairs.append({"a": a, "b": b, "mirrored": m, "sets": sorted({members[a][0], members[b][0]}),
                      "labels": sorted({members[a][1], members[b][1]})})
cross = [q for q in pairs if len(q["sets"]) > 1]
test_hit = sorted({p for q in cross for p in (q["a"], q["b"]) if members[p][0] == "test"
                   and any(members[o][0] in ("train", "val") for o in (q["a"], q["b"]) if o != p)})
val_hit = sorted({p for q in cross for p in (q["a"], q["b"]) if members[p][0] == "val"
                  and any(members[o][0] == "train" for o in (q["a"], q["b"]) if o != p)})
mirror_cross = [q for q in cross if q["mirrored"]]
print(f"near-duplicate pairs: {len(pairs)} ({sum(q['mirrored'] for q in pairs)} mirrored); across sets: {len(cross)} "
      f"({len(mirror_cross)} mirrored)")
print(f"test images with a near-duplicate in training or validation: {len(test_hit)}")
for p in test_hit:
    print("  ", members[p][1], p)
print(f"validation images with a near-duplicate in training: {len(val_hit)}")
print("mirrored pairs across sets:")
for q in mirror_cross:
    print("  ", q["labels"], q["sets"], q["a"], "|", q["b"])
json.dump({"rule": {"phash_max": args.phash, "dhash_max": args.dhash, "mirror": True, "exif_transpose": True},
           "pairs": pairs, "test_images_affected": test_hit, "val_images_affected": val_hit},
          open(args.out, "w"), indent=1)
print("->", args.out)
