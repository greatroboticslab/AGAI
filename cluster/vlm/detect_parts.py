"""Visible plant parts of photographs with the v1 RF-DETR Small checkpoint and the application's per-part thresholds
(rfdetr_parts/detect_images.py), for report-writer prompts that match inference and for the detector on/off ablation.
Input: report records (stage2_*.json, field "image") and/or image paths; output JSON {image: {"visible": [parts],
"scores": {part: best score}}}, resumable.
  python cluster/detect_parts.py --records plant_diagnostic/datasets/split_v1/stage2_train.json \
      plant_diagnostic/datasets/split_v1/stage2_val.json --out evaluation/parts_stage2_trainval.json
"""
import argparse
import json
import os

from PIL import Image
from rfdetr import RFDETRSmall

THRESH = {"flower": 0.50, "fruit": 0.50, "leaf": 0.40, "root": 0.30, "soil": 0.40, "stem": 0.40}
NAMES = ["flower", "fruit", "leaf", "root", "soil", "stem"]

ap = argparse.ArgumentParser()
ap.add_argument("--records", nargs="*", default=[])
ap.add_argument("--images", nargs="*", default=[])
ap.add_argument("--checkpoint", default="rfdetr_parts/output/checkpoint_best_total.pth")
ap.add_argument("--root", default=".")
ap.add_argument("--out", required=True)
args = ap.parse_args()

paths = list(args.images)
for r in args.records:
    paths += [x["image"] for x in json.load(open(r, encoding="utf-8"))]
paths = list(dict.fromkeys(paths))
done = json.load(open(args.out)) if os.path.exists(args.out) else {}
model = RFDETRSmall(pretrain_weights=args.checkpoint)
offset = None
for n, p in enumerate(paths):
    if p in done:
        continue
    with Image.open(os.path.join(args.root, p)) as im:
        dets = model.predict(im.convert("RGB"), threshold=min(THRESH.values()))
    scores = {}
    if dets is not None and dets.class_id is not None:
        ids = [int(c) for c in dets.class_id]
        if offset is None and ids:
            offset = 1 if max(ids) >= len(NAMES) else 0  # COCO ids 1..6 or 0-based
        for cid, conf in zip(ids, dets.confidence):
            name = NAMES[cid - (offset or 0)]
            scores[name] = max(scores.get(name, 0.0), round(float(conf), 4))
    visible = sorted(k for k, v in scores.items() if v >= THRESH[k])
    done[p] = {"visible": visible, "scores": scores}
    if n % 200 == 0:
        json.dump(done, open(args.out, "w"))
        print(f"{n}/{len(paths)} {p}: {visible}", flush=True)
json.dump(done, open(args.out, "w"))
counts = {}
for v in done.values():
    counts[len(v["visible"])] = counts.get(len(v["visible"]), 0) + 1
print(f"done {len(done)} images; images by number of visible parts: {dict(sorted(counts.items()))}; class-id offset {offset}")
