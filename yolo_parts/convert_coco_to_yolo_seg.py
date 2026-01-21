#!/usr/bin/env python3
"""
Convert COCO *instance segmentation* JSON (as exported by MakeSense) to YOLOv8 Seg format.

YOLOv8-seg label format (one object per line):
  class_id x1 y1 x2 y2 ... xn yn
where coordinates are normalized to [0, 1].

Typical usage:
  python yolo_parts/convert_coco_to_yolo_seg.py \
    --coco /path/to/annotations.json \
    --images-dir /path/to/images \
    --out /path/to/output_dataset
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional


# Must match `yolo_parts/strawberry_parts.yaml` and previous training
CLASS_MAP: Dict[str, int] = {
    "leaf": 0,
    "leaves": 0,
    "fruit": 1,
    "flower": 2,
    "crown": 3,
    "stem": 4,
    "root": 5,
    "soil": 6,
}


@dataclass(frozen=True)
class ImageInfo:
    file_name: str
    width: int
    height: int


def _norm_xy(flat_xy: List[float], w: int, h: int) -> List[float]:
    out: List[float] = []
    for i in range(0, len(flat_xy), 2):
        x = float(flat_xy[i])
        y = float(flat_xy[i + 1])
        # clamp
        if x < 0:
            x = 0.0
        if y < 0:
            y = 0.0
        if x > w:
            x = float(w)
        if y > h:
            y = float(h)
        out.extend([x / w, y / h])
    return out


def _ensure_dirs(out_dir: Path) -> None:
    for split in ("train", "val"):
        (out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / split).mkdir(parents=True, exist_ok=True)


def convert(
    coco_json: Path,
    images_dir: Path,
    out_dir: Path,
    train_ratio: float = 0.8,
    seed: int = 0,
) -> Dict[str, int]:
    data = json.loads(coco_json.read_text())

    # categories: coco category_id -> name -> class_id
    coco_cat_to_yolo: Dict[int, int] = {}
    unknown_cats: List[Tuple[int, str]] = []
    for c in data.get("categories", []):
        cid = int(c["id"])
        name = str(c.get("name", "")).lower().strip()
        if name in CLASS_MAP:
            coco_cat_to_yolo[cid] = CLASS_MAP[name]
        else:
            unknown_cats.append((cid, name))

    if unknown_cats:
        print("Warning: unknown categories in COCO JSON (will be skipped):")
        for cid, name in unknown_cats:
            print(f"  - id={cid} name='{name}'")

    # images
    images: Dict[int, ImageInfo] = {}
    for im in data.get("images", []):
        images[int(im["id"])] = ImageInfo(
            file_name=str(im["file_name"]),
            width=int(im["width"]),
            height=int(im["height"]),
        )

    # collect per-image annotations (base_name -> list of yolo label lines)
    per_image_lines: Dict[int, List[str]] = {img_id: [] for img_id in images.keys()}
    skipped = 0
    kept = 0

    for ann in data.get("annotations", []):
        img_id = int(ann["image_id"])
        cat_id = int(ann["category_id"])
        if cat_id not in coco_cat_to_yolo:
            skipped += 1
            continue
        if img_id not in images:
            skipped += 1
            continue

        seg = ann.get("segmentation")
        if not isinstance(seg, list) or not seg:
            skipped += 1
            continue

        # COCO polygon segmentation is typically: [ [x1,y1,...] , [x1,y1,...] ... ]
        # YOLOv8-seg expects one polygon per object. If multiple polygons exist, keep the largest.
        polys: List[List[float]] = []
        for p in seg:
            if isinstance(p, list) and len(p) >= 6 and len(p) % 2 == 0:
                polys.append([float(v) for v in p])
        if not polys:
            skipped += 1
            continue

        def poly_area(poly: List[float]) -> float:
            # shoelace on flat list
            xs = poly[0::2]
            ys = poly[1::2]
            area = 0.0
            for i in range(len(xs)):
                j = (i + 1) % len(xs)
                area += xs[i] * ys[j] - xs[j] * ys[i]
            return abs(area) / 2.0

        poly = max(polys, key=poly_area)
        info = images[img_id]
        norm = _norm_xy(poly, info.width, info.height)

        # filter tiny/degenerate polygons
        xs = norm[0::2]
        ys = norm[1::2]
        if (max(xs) - min(xs)) < 0.001 or (max(ys) - min(ys)) < 0.001:
            skipped += 1
            continue

        cls = coco_cat_to_yolo[cat_id]
        coords_str = " ".join(f"{c:.6f}" for c in norm)
        per_image_lines[img_id].append(f"{cls} {coords_str}")
        kept += 1

    # decide split
    img_ids = sorted(images.keys())
    random.Random(seed).shuffle(img_ids)
    n_train = int(len(img_ids) * train_ratio)
    train_set = set(img_ids[:n_train])

    _ensure_dirs(out_dir)

    written_images = 0
    written_labels = 0
    missing_images = 0

    for img_id in img_ids:
        info = images[img_id]
        src_img = images_dir / info.file_name
        if not src_img.exists():
            missing_images += 1
            continue

        split = "train" if img_id in train_set else "val"
        base = Path(info.file_name).stem
        dst_img = out_dir / "images" / split / (base + src_img.suffix)
        shutil.copy2(src_img, dst_img)
        written_images += 1

        lines = per_image_lines.get(img_id, [])
        if lines:
            (out_dir / "labels" / split / f"{base}.txt").write_text("\n".join(lines) + "\n")
            written_labels += 1
        else:
            # keep empty label file out (YOLO treats missing label file as no objects)
            pass

    return {
        "images_in_json": len(images),
        "annotations_kept": kept,
        "annotations_skipped": skipped,
        "images_written": written_images,
        "labels_written": written_labels,
        "images_missing_on_disk": missing_images,
        "train_images": len(train_set),
        "val_images": len(img_ids) - len(train_set),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--coco", required=True, type=Path, help="Path to COCO JSON export")
    ap.add_argument("--images-dir", required=True, type=Path, help="Directory containing the images referenced by COCO file_name")
    ap.add_argument("--out", required=True, type=Path, help="Output dataset dir (will create images/ and labels/)")
    ap.add_argument("--train-ratio", type=float, default=0.8)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    stats = convert(
        coco_json=args.coco,
        images_dir=args.images_dir,
        out_dir=args.out,
        train_ratio=args.train_ratio,
        seed=args.seed,
    )
    print("\nConversion complete:")
    for k, v in stats.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()

