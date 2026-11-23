"""Do the photographs of an external test set also occur in split v1? (post hoc check for lab notebook (al), (an))

    python cluster/external_overlap.py --images /projects/$USER/agai/external/mendeley_k2ptxmfjhj_v2/traditional_test \
        --out evaluation/external_mendeley_overlap.json

Same rule as the manuscript's near-duplicate audit (mirror_dupes.py): pHash within 8 bits and dHash within 10 bits, for
the photograph or its mirror, after EXIF orientation. Each external photograph is compared with every split v1
photograph. Reports, per external category, how many photographs have a near-copy in split v1, in which set and
under which label, and the same at a looser pHash limit (12 bits) as a sensitivity check.
"""
import argparse
import collections
import json
import os

import imagehash
import numpy as np
from PIL import Image, ImageOps

ap = argparse.ArgumentParser()
ap.add_argument("--images", required=True, help="folder with the photographs and _annotations.coco.json")
ap.add_argument("--split", default="evaluation/split_v1.json")
ap.add_argument("--root", default=".")
ap.add_argument("--phash", type=int, default=8)
ap.add_argument("--dhash", type=int, default=10)
ap.add_argument("--out", required=True)
a = ap.parse_args()


def hashes(path):
    im = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    mir = ImageOps.mirror(im)
    f = lambda h: h.hash.flatten()
    return f(imagehash.phash(im)), f(imagehash.dhash(im)), f(imagehash.phash(mir)), f(imagehash.dhash(mir))


split = json.load(open(a.split))
ours = [(p, part, label) for part in ("train", "val", "test") for label, ps in split[part].items() for p in ps]
coco = json.load(open(os.path.join(a.images, "_annotations.coco.json")))
cat = {c["id"]: c["name"] for c in coco["categories"]}
cats = collections.defaultdict(set)
for an in coco["annotations"]:
    cats[an["image_id"]].add(cat[an["category_id"]])
ext = [(im["file_name"], ", ".join(sorted(cats.get(im["id"], set())))) for im in coco["images"]]

H_ours, kept = [], []
for i, (p, part, label) in enumerate(ours):
    try:
        H_ours.append(hashes(os.path.join(a.root, p)))
        kept.append((p, part, label))
    except Exception as e:
        print("unreadable:", p, e, flush=True)
    if i % 400 == 0:
        print(f"split v1 {i}/{len(ours)}", flush=True)
H_ext = [hashes(os.path.join(a.images, n)) for n, _ in ext]
print(f"hashed {len(kept)} split v1 and {len(ext)} external photographs", flush=True)
P = np.array([h[0] for h in H_ours]); D = np.array([h[1] for h in H_ours])            # ours, not mirrored
eP = np.array([h[0] for h in H_ext]); eD = np.array([h[1] for h in H_ext])
ePm = np.array([h[2] for h in H_ext]); eDm = np.array([h[3] for h in H_ext])
dist = lambda A, B: (A[:, None, :] != B[None, :, :]).sum(-1)
dp, dd, dpm, ddm = dist(eP, P), dist(eD, D), dist(ePm, P), dist(eDm, D)


def matches(pmax, dmax):
    return ((dp <= pmax) & (dd <= dmax)) | ((dpm <= pmax) & (ddm <= dmax))


out = {"rule": {"phash_max": a.phash, "dhash_max": a.dhash, "mirror": True}, "n_external": len(ext),
       "n_split_v1": len(kept), "by_category": {}, "pairs": []}
M, Mloose = matches(a.phash, a.dhash), matches(12, 64)
for c in sorted({c for _, c in ext}):
    idx = [i for i, (_, cc) in enumerate(ext) if cc == c]
    hit = [i for i in idx if M[i].any()]
    where = collections.Counter()
    for i in hit:
        for j in np.nonzero(M[i])[0][:1]:
            where[f"{kept[j][1]}/{kept[j][2]}"] += 1
    out["by_category"][c] = {"n": len(idx), "with_near_copy": len(hit), "first_match_in": dict(where),
                             "with_near_copy_loose_phash12": int(sum(Mloose[i].any() for i in idx)),
                             "median_min_phash": float(np.median(np.minimum(dp[idx].min(1), dpm[idx].min(1))))}
    print(f"{c:26s} n={len(idx):3d}  near-copy in split v1: {len(hit):3d}  {dict(where)}  loose: "
          f"{out['by_category'][c]['with_near_copy_loose_phash12']}  median min pHash {out['by_category'][c]['median_min_phash']:.0f}", flush=True)
for i in range(len(ext)):
    for j in np.nonzero(M[i])[0]:
        out["pairs"].append({"external": ext[i][0], "category": ext[i][1], "split_v1": kept[j][0], "set": kept[j][1],
                             "label": kept[j][2], "phash": int(min(dp[i, j], dpm[i, j])), "dhash": int(min(dd[i, j], ddm[i, j]))})
tot = int(M.any(1).sum())
out["external_with_near_copy"] = tot
print(f"external photographs with a near-copy in split v1: {tot} of {len(ext)}; pairs: {len(out['pairs'])}")
json.dump(out, open(a.out, "w"), indent=1)
