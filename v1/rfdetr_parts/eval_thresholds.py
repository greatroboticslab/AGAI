#!/usr/bin/env python3
"""
Evaluate RF-DETR Small with per-class confidence threshold sweeps.

Loads the best checkpoint, runs inference on N random images per class from
the validation set, and reports precision/recall at various thresholds.
"""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
from rfdetr import RFDETRSmall

CLASS_NAMES = {1: "flower", 2: "fruit", 3: "leaf", 4: "root", 5: "soil", 6: "stem"}
CLASS_IDS = {v: k for k, v in CLASS_NAMES.items()}


def load_val_index(dataset_dir: str) -> dict:
    """Load val annotations and build per-class image index."""
    ann_path = Path(dataset_dir) / "valid" / "_annotations.coco.json"
    with open(ann_path) as f:
        coco = json.load(f)

    img_map = {img["id"]: img for img in coco["images"]}

    # per-class: list of (image_info, [bboxes])
    class_images = defaultdict(lambda: defaultdict(list))
    for ann in coco["annotations"]:
        cid = ann["category_id"]
        class_images[cid][ann["image_id"]].append(ann["bbox"])

    return img_map, class_images


def iou(box_a, box_b):
    """IoU between two [x, y, w, h] boxes."""
    ax1, ay1 = box_a[0], box_a[1]
    ax2, ay2 = ax1 + box_a[2], ay1 + box_a[3]
    bx1, by1 = box_b[0], box_b[1]
    bx2, by2 = bx1 + box_b[2], by1 + box_b[3]

    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)

    area_a = box_a[2] * box_a[3]
    area_b = box_b[2] * box_b[3]
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def xyxy_to_xywh(box):
    return [box[0], box[1], box[2] - box[0], box[3] - box[1]]


def evaluate_class(model, dataset_dir, img_map, class_images, class_id, n_images, thresholds):
    """Run inference on n_images containing this class, compute P/R at each threshold."""
    image_ids = list(class_images[class_id].keys())
    random.shuffle(image_ids)
    image_ids = image_ids[:n_images]

    all_preds = []  # (confidence, bbox_xywh)
    all_gt = []     # list of lists of bbox_xywh per image
    pred_image_idx = []

    val_dir = Path(dataset_dir) / "valid"

    for img_idx, iid in enumerate(image_ids):
        img_info = img_map[iid]
        img_path = val_dir / img_info["file_name"]
        img = Image.open(img_path).convert("RGB")

        detections = model.predict(img, threshold=0.01)

        gt_boxes = class_images[class_id][iid]
        all_gt.append(gt_boxes)

        if hasattr(detections, "class_id") and detections.class_id is not None:
            # RF-DETR returns 0-indexed class IDs, COCO annotations are 1-indexed
            for det_cls, conf, xyxy in zip(detections.class_id, detections.confidence, detections.xyxy):
                if int(det_cls) + 1 == class_id:
                    all_preds.append((float(conf), xyxy_to_xywh(xyxy.tolist())))
                    pred_image_idx.append(img_idx)

    results = {}
    for thresh in thresholds:
        tp, fp, total_gt = 0, 0, sum(len(g) for g in all_gt)
        matched_gt = defaultdict(set)

        filtered = [(c, b, idx) for (c, b), idx in zip(all_preds, pred_image_idx) if c >= thresh]
        filtered.sort(key=lambda x: -x[0])

        for conf, pred_box, img_idx in filtered:
            best_iou, best_j = 0, -1
            for j, gt_box in enumerate(all_gt[img_idx]):
                if j in matched_gt[img_idx]:
                    continue
                v = iou(pred_box, gt_box)
                if v > best_iou:
                    best_iou = v
                    best_j = j
            if best_iou >= 0.5 and best_j >= 0:
                tp += 1
                matched_gt[img_idx].add(best_j)
            else:
                fp += 1

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / total_gt if total_gt > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        results[thresh] = {"tp": tp, "fp": fp, "gt": total_gt, "prec": prec, "rec": rec, "f1": f1}

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="/data/AGAI/MiniGPT-4/rfdetr_parts/output/checkpoint_best_total.pth")
    parser.add_argument("--dataset", default="/data/AGAI/MiniGPT-4/rfdetr_parts/dataset")
    parser.add_argument("--n-images", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    print(f"Loading model from {args.checkpoint}")
    model = RFDETRSmall(pretrain_weights=args.checkpoint)

    img_map, class_images = load_val_index(args.dataset)

    thresholds = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]

    print(f"\nEvaluating {args.n_images} images per class at IoU=0.50")
    print(f"Thresholds: {thresholds}\n")

    for cid, cname in sorted(CLASS_NAMES.items()):
        n_available = len(class_images[cid])
        n_eval = min(args.n_images, n_available)
        print(f"{'=' * 70}")
        print(f"  {cname.upper()} ({n_eval} images, {sum(len(v) for v in list(class_images[cid].values())[:n_eval])} GT boxes)")
        print(f"{'=' * 70}")
        print(f"  {'thresh':>6}  {'TP':>4} {'FP':>4} {'GT':>4}  {'Prec':>6} {'Recall':>6} {'F1':>6}")
        print(f"  {'-' * 48}")

        results = evaluate_class(model, args.dataset, img_map, class_images, cid, n_eval, thresholds)

        best_f1_thresh = max(results.items(), key=lambda x: x[1]["f1"])
        for thresh in thresholds:
            r = results[thresh]
            marker = " <-- best F1" if thresh == best_f1_thresh[0] else ""
            print(f"  {thresh:>6.2f}  {r['tp']:>4} {r['fp']:>4} {r['gt']:>4}  {r['prec']:>6.3f} {r['rec']:>6.3f} {r['f1']:>6.3f}{marker}")
        print()


if __name__ == "__main__":
    main()
