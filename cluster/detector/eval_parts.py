"""Per-part metrics of an RF-DETR checkpoint on one COCO split of the detector data: AP50 and AP50:95 per part
(pycocotools), image-level part presence (a part is present when at least one detection of it reaches its
threshold) as precision / recall / F1 per part, and a threshold sweep (0.05 grid) that picks per-part thresholds by
F1. Choose thresholds on the validation split, report on the test split with --thresholds.
Writes evaluation/parts_<name>_<split>.json.
  python cluster/detector/eval_parts.py --size small --weights RUN/checkpoint_best_total.pth \
      --dataset-dir rfdetr_parts/dataset_splitv1 --split valid --name small_splitv1
  python cluster/detector/eval_parts.py ... --split test --thresholds evaluation/parts_small_splitv1_valid.json
"""
import argparse
import collections
import json
import os
import sys

import numpy as np
from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from rfdetr import RFDETRLarge, RFDETRMedium, RFDETRSmall

SIZES = {"small": RFDETRSmall, "medium": RFDETRMedium, "large": RFDETRLarge}
GRID = [round(float(x), 2) for x in np.arange(0.05, 0.96, 0.05)]
LAB_THRESHOLDS = {"flower": 0.50, "fruit": 0.50, "leaf": 0.40, "root": 0.30, "soil": 0.40, "stem": 0.40}

ap = argparse.ArgumentParser()
ap.add_argument("--size", choices=SIZES, required=True)
ap.add_argument("--weights", required=True)
ap.add_argument("--dataset-dir", required=True)
ap.add_argument("--split", default="valid")
ap.add_argument("--name", required=True)
ap.add_argument("--thresholds", default="", help="parts_*.json of the validation run: use its chosen thresholds")
ap.add_argument("--floor", type=float, default=0.01, help="detection threshold for the raw predictions")
args = ap.parse_args()

split_dir = os.path.join(args.dataset_dir, args.split)
gt = COCO(os.path.join(split_dir, "_annotations.coco.json"))
cat_name = {c["id"]: c["name"] for c in gt.loadCats(gt.getCatIds())}
model = SIZES[args.size](pretrain_weights=args.weights)

raw = []  # (image_id, class id as predicted, score, xyxy)
for img_id in gt.getImgIds():
    info = gt.loadImgs(img_id)[0]
    with Image.open(os.path.join(split_dir, info["file_name"])) as im:
        dets = model.predict(im.convert("RGB"), threshold=args.floor)
    if dets is not None and dets.class_id is not None:
        for cid, conf, xyxy in zip(dets.class_id, dets.confidence, dets.xyxy):
            raw.append((img_id, int(cid), float(conf), [float(v) for v in xyxy]))
# RF-DETR emits 0-based ids: 0..5 = flower, fruit, leaf, root, soil, stem (checked against ground-truth boxes with
# cluster/detector/check_class_ids.py); the COCO categories are 1..6, and id 6 is a no-object slot seen only at very
# low scores, so predictions are shifted by one and ids outside the categories dropped.
pred_ids = {r[1] for r in raw}
shift = 0 if pred_ids <= set(cat_name) else 1
dropped = sum(1 for r in raw if r[1] + shift not in cat_name)
raw = [r for r in raw if r[1] + shift in cat_name]
print(f"{len(raw)} raw detections on {len(gt.getImgIds())} images; class-id shift {shift}; dropped {dropped} with ids "
      f"outside the categories", flush=True)
results = [{"image_id": i, "category_id": c + shift, "score": s,
            "bbox": [b[0], b[1], b[2] - b[0], b[3] - b[1]]} for i, c, s, b in raw]

# COCO AP per part
dt = gt.loadRes(results) if results else None
per_part = {}
if dt is not None:
    E = COCOeval(gt, dt, "bbox")
    E.evaluate()
    E.accumulate()
    E.summarize()
    prec = E.eval["precision"]  # [iou, recall, category, area, maxdets]
    for k, cid in enumerate(E.params.catIds):
        p_all = prec[:, :, k, 0, -1]
        p50 = prec[0, :, k, 0, -1]
        per_part[cat_name[cid]] = {"ap50_95": float(p_all[p_all > -1].mean()) if (p_all > -1).any() else None,
                                   "ap50": float(p50[p50 > -1].mean()) if (p50 > -1).any() else None}
    overall = {"map50_95": float(E.stats[0]), "map50": float(E.stats[1]), "map75": float(E.stats[2])}
else:
    overall = {}

# image-level presence
gt_present = collections.defaultdict(set)
for a in gt.loadAnns(gt.getAnnIds()):
    gt_present[a["image_id"]].add(cat_name[a["category_id"]])
best_score = collections.defaultdict(lambda: collections.defaultdict(float))
for i, c, s, _ in raw:
    name = cat_name[c + shift]
    best_score[i][name] = max(best_score[i][name], s)
images = gt.getImgIds()


def presence(name, t):
    tp = fp = fn = 0
    for i in images:
        p = best_score[i][name] >= t
        g = name in gt_present[i]
        tp += p and g
        fp += p and not g
        fn += (not p) and g
    prec_ = tp / (tp + fp) if tp + fp else 0.0
    rec_ = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec_ * rec_ / (prec_ + rec_) if prec_ + rec_ else 0.0
    return {"precision": round(float(prec_), 4), "recall": round(float(rec_), 4), "f1": round(float(f1), 4),
            "tp": int(tp), "fp": int(fp), "fn": int(fn), "images_with_part": int(sum(name in gt_present[i] for i in images))}


sweep = {name: {str(t): presence(name, t) for t in GRID} for name in cat_name.values()}
chosen = {name: float(max(GRID, key=lambda t: (sweep[name][str(t)]["f1"], -t))) for name in cat_name.values()}
if args.thresholds:
    chosen = json.load(open(args.thresholds))["chosen_thresholds"]
out = {"name": args.name, "size": args.size, "weights": args.weights, "split": args.split, "n_images": len(images),
       "coco": overall, "per_part_ap": per_part,
       "presence_at_lab_thresholds": {n: presence(n, LAB_THRESHOLDS[n]) for n in cat_name.values() if n in LAB_THRESHOLDS},
       "chosen_thresholds": chosen, "presence_at_chosen": {n: presence(n, chosen[n]) for n in cat_name.values()},
       "sweep": sweep, "thresholds_from": args.thresholds or f"F1-maximising sweep on this split ({args.split})"}
os.makedirs("evaluation", exist_ok=True)
json.dump(out, open(f"evaluation/parts_{args.name}_{args.split}.json", "w"), indent=1,
          default=lambda o: o.item() if hasattr(o, "item") else str(o))  # numpy scalars
print(json.dumps({k: out[k] for k in ("coco", "per_part_ap", "chosen_thresholds", "presence_at_chosen")}, indent=1))
