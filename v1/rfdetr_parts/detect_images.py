#!/usr/bin/env python3
"""
Run RF-DETR Small detection on a list of images, output JSON to stdout.
Called from the rfdetr conda env.

Usage: python detect_images.py --images img1.jpg img2.jpg --threshold 0.35
"""

import argparse
import json
import sys

from PIL import Image
from rfdetr import RFDETRSmall

CLASS_NAMES = {0: "flower", 1: "fruit", 2: "leaf", 3: "root", 4: "soil", 5: "stem"}
CHECKPOINT = "/data/AGAI/MiniGPT-4/rfdetr_parts/output/checkpoint_best_total.pth"

PER_CLASS_THRESHOLDS = {
    "flower": 0.50,
    "fruit": 0.50,
    "leaf": 0.40,
    "root": 0.30,
    "soil": 0.40,
    "stem": 0.40,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", nargs="+", required=True)
    parser.add_argument("--threshold", type=float, default=None,
                        help="Flat threshold (overrides per-class). Default: use per-class best.")
    parser.add_argument("--checkpoint", default=CHECKPOINT)
    args = parser.parse_args()

    # Suppress noisy logs
    import logging
    logging.getLogger("rfdetr").setLevel(logging.WARNING)

    model = RFDETRSmall(pretrain_weights=args.checkpoint)

    # Use lowest per-class threshold as the raw detection floor
    raw_threshold = args.threshold if args.threshold is not None else min(PER_CLASS_THRESHOLDS.values())

    results = {}
    for img_path in args.images:
        try:
            img = Image.open(img_path).convert("RGB")
            dets = model.predict(img, threshold=raw_threshold)

            detections = []
            if hasattr(dets, "class_id") and dets.class_id is not None:
                for cls_id, conf, xyxy in zip(dets.class_id, dets.confidence, dets.xyxy):
                    name = CLASS_NAMES.get(int(cls_id), f"class_{cls_id}")
                    # Apply per-class threshold filtering
                    if args.threshold is None:
                        min_conf = PER_CLASS_THRESHOLDS.get(name, 0.35)
                    else:
                        min_conf = args.threshold
                    if float(conf) < min_conf:
                        continue
                    detections.append({
                        "class": name,
                        "confidence": round(float(conf), 3),
                        "bbox": [round(float(v), 1) for v in xyxy.tolist()],
                    })

            # Aggregate: unique parts with max confidence
            parts = {}
            for d in detections:
                c = d["class"]
                if c not in parts or d["confidence"] > parts[c]:
                    parts[c] = d["confidence"]

            results[img_path] = {
                "detected_parts": parts,
                "all_detections": detections,
                "num_detections": len(detections),
            }
        except Exception as e:
            results[img_path] = {"error": str(e)}

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
