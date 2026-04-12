#!/usr/bin/env python3
"""
Prepare COCO-format dataset for RF-DETR training from a Roboflow export zip.

Splits the single-folder Roboflow COCO export into train/valid with proper
_annotations.coco.json for each, filtering out the Roboflow-generated
"Strawberries" super-category and doing a root-aware stratified split.
"""

import argparse
import json
import random
import shutil
import zipfile
from collections import defaultdict
from pathlib import Path


SUPERCATEGORY_NAMES = {"Strawberries"}


def load_coco_from_zip(zip_path: str) -> tuple[dict, dict[str, bytes]]:
    """Return (coco_dict, {filename: image_bytes}) from a Roboflow zip."""
    z = zipfile.ZipFile(zip_path)
    coco = json.loads(z.read("train/_annotations.coco.json"))
    image_bytes = {}
    for entry in z.namelist():
        if entry.startswith("train/") and not entry.endswith(".json") and "/" in entry:
            fname = entry.split("/", 1)[1]
            if fname:
                image_bytes[fname] = z.read(entry)
    return coco, image_bytes


def sanitize_annotation_types(coco: dict) -> None:
    """Ensure bbox, area, and segmentation values are numeric (Roboflow sometimes exports strings)."""
    for ann in coco["annotations"]:
        ann["bbox"] = [float(v) for v in ann["bbox"]]
        ann["area"] = float(ann["area"])
        if "segmentation" in ann and isinstance(ann["segmentation"], list):
            ann["segmentation"] = [[float(v) for v in poly] for poly in ann["segmentation"]]
    for img in coco["images"]:
        img["width"] = int(img["width"])
        img["height"] = int(img["height"])


def filter_supercategories(coco: dict) -> tuple[dict, dict[int, int]]:
    """Remove super-categories, remap remaining category IDs to be contiguous from 1."""
    keep_cats = [c for c in coco["categories"] if c["name"] not in SUPERCATEGORY_NAMES]
    keep_cats.sort(key=lambda c: c["id"])
    old_to_new = {}
    new_cats = []
    for i, cat in enumerate(keep_cats):
        new_id = i + 1
        old_to_new[cat["id"]] = new_id
        new_cats.append({**cat, "id": new_id})
    coco["categories"] = new_cats
    coco["annotations"] = [a for a in coco["annotations"] if a["category_id"] in old_to_new]
    for ann in coco["annotations"]:
        ann["category_id"] = old_to_new[ann["category_id"]]
    return coco, old_to_new


def root_aware_split(coco: dict, val_fraction: float, seed: int) -> tuple[set[int], set[int]]:
    """Split image IDs into train/val, ensuring root-positive images are proportionally distributed."""
    rng = random.Random(seed)
    cat_name_to_id = {c["name"]: c["id"] for c in coco["categories"]}
    root_id = cat_name_to_id.get("root")

    anns_by_image = defaultdict(list)
    for ann in coco["annotations"]:
        anns_by_image[ann["image_id"]].append(ann)

    all_ids = [img["id"] for img in coco["images"]]
    root_positive = {iid for iid in all_ids if root_id and any(a["category_id"] == root_id for a in anns_by_image[iid])}
    root_negative = set(all_ids) - root_positive

    def pick_val(ids: set[int]) -> set[int]:
        ordered = sorted(ids)
        rng.shuffle(ordered)
        n_val = max(1, round(len(ordered) * val_fraction))
        return set(ordered[:n_val])

    val_ids = pick_val(root_positive) | pick_val(root_negative)
    train_ids = set(all_ids) - val_ids
    return train_ids, val_ids


def write_split(coco: dict, image_ids: set[int], image_bytes: dict[str, bytes], out_dir: Path):
    """Write a COCO split: images + _annotations.coco.json."""
    out_dir.mkdir(parents=True, exist_ok=True)
    id_to_img = {img["id"]: img for img in coco["images"]}
    split_images = [id_to_img[iid] for iid in sorted(image_ids)]
    split_anns = [a for a in coco["annotations"] if a["image_id"] in image_ids]

    # Renumber annotation IDs sequentially
    for i, ann in enumerate(split_anns, 1):
        ann["id"] = i

    split_coco = {
        "info": coco.get("info", {}),
        "licenses": coco.get("licenses", []),
        "categories": coco["categories"],
        "images": split_images,
        "annotations": split_anns,
    }

    written = 0
    for img in split_images:
        fname = img["file_name"]
        if fname in image_bytes:
            (out_dir / fname).write_bytes(image_bytes[fname])
            written += 1

    (out_dir / "_annotations.coco.json").write_text(json.dumps(split_coco, indent=2))
    return written, len(split_anns)


def main():
    parser = argparse.ArgumentParser(description="Prepare RF-DETR dataset from Roboflow COCO zip")
    parser.add_argument("--zip", default="/data/AGAI/LatestStrawberry archive.coco-segmentation.zip",
                        help="Path to Roboflow COCO segmentation zip")
    parser.add_argument("--out", default="/data/AGAI/MiniGPT-4/rfdetr_parts/dataset",
                        help="Output dataset directory")
    parser.add_argument("--val-fraction", type=float, default=0.3,
                        help="Fraction of images for validation (default: 0.3)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_dir = Path(args.out)
    if out_dir.exists():
        shutil.rmtree(out_dir)

    print(f"Loading zip: {args.zip}")
    coco, image_bytes = load_coco_from_zip(args.zip)
    print(f"  {len(coco['images'])} images, {len(coco['annotations'])} annotations")

    sanitize_annotation_types(coco)
    coco, _ = filter_supercategories(coco)
    print(f"  After filtering super-categories: {len(coco['annotations'])} annotations")
    print(f"  Classes: {', '.join(c['name'] for c in coco['categories'])}")

    train_ids, val_ids = root_aware_split(coco, args.val_fraction, args.seed)

    train_imgs, train_anns = write_split(coco, train_ids, image_bytes, out_dir / "train")
    val_imgs, val_anns = write_split(coco, val_ids, image_bytes, out_dir / "valid")

    print(f"\nDataset written to: {out_dir}")
    print(f"  train: {train_imgs} images, {train_anns} annotations")
    print(f"  valid: {val_imgs} images, {val_anns} annotations")
    print(f"  Classes ({len(coco['categories'])}): {', '.join(c['name'] for c in coco['categories'])}")


if __name__ == "__main__":
    main()
