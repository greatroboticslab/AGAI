"""Detector split that agrees with split v1. The lab's Roboflow export (rfdetr_parts/dataset/{train,valid}, 1075 images)
is matched to the condition photographs by perceptual hash: an annotated image within pHash distance --max-distance of
a split v1 test photograph goes to the detector test split, of a validation photograph to validation, otherwise to
training (annotation-only images included). Annotated images that are near-copies of each other stay in one split
(a group containing any test image is test, else validation if it contains one, else training).
Writes <out-dir>/{train,valid,test}/_annotations.coco.json with images re-indexed and symlinked, and
evaluation/detector_split_v1.json (assignment of every image, match distances, counts).
  python cluster/detector/make_detector_split.py --out-dir rfdetr_parts/dataset_splitv1
"""
import argparse
import collections
import json
import os

import imagehash
from PIL import Image


def phash(path):
    with Image.open(path) as im:
        return int(str(imagehash.phash(im.convert("RGB"))), 16)


def hamming(a, b):
    return bin(a ^ b).count("1")


ap = argparse.ArgumentParser()
ap.add_argument("--roboflow-dir", default="rfdetr_parts/dataset")
ap.add_argument("--split", default="evaluation/split_v1.json")
ap.add_argument("--out-dir", required=True)
ap.add_argument("--max-distance", type=int, default=6)
ap.add_argument("--report", default="evaluation/detector_split_v1.json")
args = ap.parse_args()

split = json.load(open(args.split))
photos = [(p, s, c) for s in ("train", "val", "test") for c, ps in split[s].items() for p in ps]
print(f"hashing {len(photos)} condition photographs", flush=True)
photo_hash = [(phash(p), p, s, c) for p, s, c in photos]

coco = {}
for sub in ("train", "valid"):
    coco[sub] = json.load(open(os.path.join(args.roboflow_dir, sub, "_annotations.coco.json")))
cats = coco["train"]["categories"]
assert [c["id"] for c in cats] == [c["id"] for c in coco["valid"]["categories"]]
items = []  # (sub, image record, annotations)
for sub in ("train", "valid"):
    anns = collections.defaultdict(list)
    for a in coco[sub]["annotations"]:
        anns[a["image_id"]].append(a)
    for im in coco[sub]["images"]:
        items.append((sub, im, anns.get(im["id"], [])))
print(f"hashing {len(items)} annotated images", flush=True)
det_hash = [phash(os.path.join(args.roboflow_dir, sub, im["file_name"])) for sub, im, _ in items]

# nearest condition photograph
assign, match = [], []
for h in det_hash:
    best = min(photo_hash, key=lambda t: hamming(h, t[0]))
    d = hamming(h, best[0])
    if d <= args.max_distance:
        assign.append(best[2])
        match.append({"photo": best[1], "distance": d, "condition": best[3]})
    else:
        assign.append("train")
        match.append({"photo": None, "distance": d, "condition": None})

# near-copies among the annotated images stay together (union-find)
parent = list(range(len(items)))


def find(i):
    while parent[i] != i:
        parent[i] = parent[parent[i]]
        i = parent[i]
    return i


for i in range(len(items)):
    for j in range(i + 1, len(items)):
        if hamming(det_hash[i], det_hash[j]) <= args.max_distance:
            parent[find(i)] = find(j)
groups = collections.defaultdict(list)
for i in range(len(items)):
    groups[find(i)].append(i)
moved = 0
rank = {"test": 2, "val": 1, "train": 0}
for members in groups.values():
    top = max((assign[i] for i in members), key=rank.get)
    for i in members:
        if assign[i] != top:
            moved += 1
            assign[i] = top

out_sub = {"train": "train", "val": "valid", "test": "test"}
for sub in out_sub.values():
    os.makedirs(os.path.join(args.out_dir, sub), exist_ok=True)
new = {sub: {"images": [], "annotations": [], "categories": cats} for sub in out_sub.values()}
next_img, next_ann = 1, 1
report_items = []
for (src_sub, im, anns), s, m, h in zip(items, assign, match, det_hash):
    sub = out_sub[s]
    rec = dict(im, id=next_img)
    new[sub]["images"].append(rec)
    for a in anns:
        new[sub]["annotations"].append(dict(a, id=next_ann, image_id=next_img))
        next_ann += 1
    src = os.path.abspath(os.path.join(args.roboflow_dir, src_sub, im["file_name"]))
    dst = os.path.join(args.out_dir, sub, im["file_name"])
    if not os.path.lexists(dst):
        os.symlink(src, dst)
    report_items.append({"file": im["file_name"], "roboflow_split": src_sub, "split": s, "match": m,
                         "parts": sorted({a["category_id"] for a in anns}), "n_annotations": len(anns)})
    next_img += 1
for sub, d in new.items():
    json.dump(d, open(os.path.join(args.out_dir, sub, "_annotations.coco.json"), "w"))

name_of = {c["id"]: c["name"] for c in cats}
counts = {s: sum(1 for r in report_items if r["split"] == s) for s in ("train", "val", "test")}
matched = {s: sum(1 for r in report_items if r["split"] == s and r["match"]["photo"]) for s in counts}
parts = {s: collections.Counter(name_of[a["category_id"]] for a in new[out_sub[s]]["annotations"]) for s in counts}
report = {"max_distance": args.max_distance, "images": counts, "matched_to_a_condition_photo": matched,
          "unmatched_annotation_only": sum(1 for r in report_items if not r["match"]["photo"]),
          "moved_by_group_rule": moved, "annotations_per_part": {s: dict(parts[s]) for s in parts},
          "conditions_in_test": dict(collections.Counter(r["match"]["condition"] for r in report_items
                                                        if r["split"] == "test" and r["match"]["condition"])),
          "items": report_items}
json.dump(report, open(args.report, "w"), indent=1)
print(json.dumps({k: v for k, v in report.items() if k != "items"}, indent=1))
