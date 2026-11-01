"""Which plant part does each predicted class id of an RF-DETR checkpoint mean? The lab's detect_images.py maps ids
0..5 to flower..stem, the COCO categories are 1..6, and the model emits ids 0..6. Settle it empirically: on a split with
ground truth, every detection (score >= --min-score) is matched to the ground-truth box of highest IoU (>= 0.5) and the
(predicted id, true part) pairs are tallied.
  python cluster/detector/check_class_ids.py --weights rfdetr_parts/output/checkpoint_best_total.pth --split-dir rfdetr_parts/dataset/valid
"""
import argparse
import collections
import json
import os

from PIL import Image
from rfdetr import RFDETRLarge, RFDETRMedium, RFDETRSmall

SIZES = {"small": RFDETRSmall, "medium": RFDETRMedium, "large": RFDETRLarge}
ap = argparse.ArgumentParser()
ap.add_argument("--size", default="small", choices=SIZES)
ap.add_argument("--weights", required=True)
ap.add_argument("--split-dir", required=True)
ap.add_argument("--min-score", type=float, default=0.3)
ap.add_argument("--limit", type=int, default=0)
args = ap.parse_args()

coco = json.load(open(os.path.join(args.split_dir, "_annotations.coco.json")))
name = {c["id"]: c["name"] for c in coco["categories"]}
gts = collections.defaultdict(list)
for a in coco["annotations"]:
    x, y, w, h = a["bbox"]
    gts[a["image_id"]].append((a["category_id"], (x, y, x + w, y + h)))


def iou(a, b):
    ix1, iy1, ix2, iy2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


model = SIZES[args.size](pretrain_weights=args.weights)
tally = collections.defaultdict(collections.Counter)
unmatched = collections.Counter()
images = coco["images"][: args.limit] if args.limit else coco["images"]
for im in images:
    with Image.open(os.path.join(args.split_dir, im["file_name"])) as img:
        dets = model.predict(img.convert("RGB"), threshold=args.min_score)
    if dets is None or dets.class_id is None:
        continue
    for cid, conf, xyxy in zip(dets.class_id, dets.confidence, dets.xyxy):
        box = tuple(float(v) for v in xyxy)
        best = max(((iou(box, g), c) for c, g in gts[im["id"]]), default=(0.0, None))
        if best[0] >= 0.5:
            tally[int(cid)][name[best[1]]] += 1
        else:
            unmatched[int(cid)] += 1
print(f"{len(images)} images; detections with IoU >= 0.5 to a ground-truth box, by predicted id -> true part:")
for cid in sorted(set(tally) | set(unmatched)):
    row = tally[cid].most_common()
    print(f"  id {cid}: {row}  | unmatched {unmatched[cid]}")
