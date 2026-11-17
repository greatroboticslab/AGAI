#!/usr/bin/env python3
"""
Import human-audited external strawberry images into train and train_aug.

Assumptions
-----------
1. ``data/train`` is the compact canonical root and should receive the approved
   original images as real files.
2. ``data/train_aug`` should also receive each approved original plus one
   deterministic augmented variant per imported image.
3. For classes currently capped at 300 samples in ``train_aug``, old synthetic
   balancing images can be archived out to keep the total count from exceeding
   the cap.
4. Only files listed in the explicit import manifest are eligible for import.
5. All operations must be auditable via a provenance JSON file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

from PIL import Image, ImageEnhance, ImageOps


IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


@dataclass(frozen=True)
class ImportItem:
    label: str
    source_path: Path
    source_name: str


def count_images(root: Path) -> Dict[str, int]:
    return {
        d.name: sum(1 for p in d.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS)
        for d in sorted(root.iterdir())
        if d.is_dir()
    }


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def apply_recipe(img: Image.Image, recipe_name: str) -> Image.Image:
    img = img.convert("RGB")
    if recipe_name == "hflip":
        return ImageOps.mirror(img)
    if recipe_name == "rot_p3":
        return img.rotate(3, resample=Image.Resampling.BICUBIC, fillcolor=(0, 0, 0))
    if recipe_name == "rot_m3":
        return img.rotate(-3, resample=Image.Resampling.BICUBIC, fillcolor=(0, 0, 0))
    if recipe_name == "contrast_p8":
        return ImageEnhance.Contrast(img).enhance(1.08)
    if recipe_name == "bright_p8":
        return ImageEnhance.Brightness(img).enhance(1.08)
    if recipe_name == "hflip_rot_p2":
        return ImageOps.mirror(img.rotate(2, resample=Image.Resampling.BICUBIC, fillcolor=(0, 0, 0)))
    raise ValueError(f"Unknown augmentation recipe: {recipe_name}")


def save_image(img: Image.Image, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    suffix = dst.suffix.lower()
    if suffix in {".jpg", ".jpeg"}:
        img.save(dst, quality=95)
    elif suffix == ".webp":
        img.save(dst, quality=95)
    else:
        img.save(dst)


def load_manifest(path: Path) -> List[ImportItem]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    items: List[ImportItem] = []
    for raw in obj["items"]:
        items.append(
            ImportItem(
                label=raw["label"],
                source_path=Path(raw["source_path"]),
                source_name=raw["source_name"],
            )
        )
    return items


def main() -> None:
    ap = argparse.ArgumentParser(description="Import curated external strawberry candidates.")
    ap.add_argument(
        "--repo-root",
        type=Path,
        default=Path("/data/AGAI/MiniGPT-4"),
    )
    ap.add_argument(
        "--manifest",
        type=Path,
        default=Path("/data/AGAI/MiniGPT-4/evaluation/external_candidate_search/import_manifest_v1.json"),
    )
    ap.add_argument(
        "--train-aug-cap",
        type=int,
        default=300,
        help="Maximum allowed class total in train_aug for capped classes.",
    )
    ap.add_argument(
        "--provenance-out",
        type=Path,
        default=Path("/data/AGAI/MiniGPT-4/plant_diagnostic/data/external_import_provenance_v1.json"),
    )
    args = ap.parse_args()

    repo = args.repo_root.resolve()
    train_root = repo / "plant_diagnostic" / "data" / "train"
    train_aug_root = repo / "plant_diagnostic" / "data" / "train_aug"
    archive_root = repo / "plant_diagnostic" / "data" / "train_aug_external_swap_archive"
    balance_prov_path = repo / "plant_diagnostic" / "data" / "train_aug_balance_provenance.json"

    manifest_items = load_manifest(args.manifest.resolve())
    balance_prov = json.loads(balance_prov_path.read_text(encoding="utf-8"))
    pre_train = count_images(train_root)
    pre_train_aug = count_images(train_aug_root)

    imports_by_label: Dict[str, List[ImportItem]] = {}
    for item in manifest_items:
        imports_by_label.setdefault(item.label, []).append(item)

    # Determine which synthetic balancing images to swap out.
    synthetic_to_archive: Dict[str, List[Path]] = {}
    for label, items in imports_by_label.items():
        required_slots = 2 * len(items)
        current_count = pre_train_aug.get(label, 0)
        if current_count + required_slots <= args.train_aug_cap:
            synthetic_to_archive[label] = []
            continue
        need_remove = current_count + required_slots - args.train_aug_cap
        generated = [Path(rec["generated"]) for rec in balance_prov["actions"].get(label, [])]
        existing = [p for p in generated if p.is_file()]
        if len(existing) < need_remove:
            raise RuntimeError(
                f"Not enough synthetic balancing files available to keep {label} <= {args.train_aug_cap}. "
                f"Need {need_remove}, found {len(existing)}."
            )
        synthetic_to_archive[label] = existing[-need_remove:]

    provenance = {
        "schema_version": "1.0",
        "manifest": str(args.manifest.resolve()),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "train_aug_cap": args.train_aug_cap,
        "pre_train_counts": pre_train,
        "pre_train_aug_counts": pre_train_aug,
        "imports": [],
        "archived_synthetic_train_aug": {},
    }

    # Archive out selected synthetic train_aug files first.
    for label, paths in synthetic_to_archive.items():
        archived: List[str] = []
        for src in paths:
            dst = archive_root / label / src.name
            dst.parent.mkdir(parents=True, exist_ok=True)
            if dst.exists():
                raise FileExistsError(f"Archive destination already exists: {dst}")
            shutil.move(str(src), str(dst))
            archived.append(str(dst.resolve()))
        provenance["archived_synthetic_train_aug"][label] = archived

    recipes = ["hflip", "rot_p3", "rot_m3", "contrast_p8", "bright_p8", "hflip_rot_p2"]

    for idx, item in enumerate(manifest_items):
        if not item.source_path.is_file():
            raise FileNotFoundError(f"Missing import source: {item.source_path}")
        ext = item.source_path.suffix.lower()
        if ext not in IMG_EXTS:
            raise ValueError(f"Unsupported image type for import: {item.source_path}")

        base_name = f"{item.source_name}{ext}"
        train_dst = train_root / item.label / base_name
        train_aug_dst = train_aug_root / item.label / base_name
        aug_name = f"{item.source_name}_aug900{ext}"
        train_aug_aug_dst = train_aug_root / item.label / aug_name

        for dst in [train_dst, train_aug_dst, train_aug_aug_dst]:
            if dst.exists():
                raise FileExistsError(f"Import destination already exists: {dst}")

        train_dst.parent.mkdir(parents=True, exist_ok=True)
        train_aug_dst.parent.mkdir(parents=True, exist_ok=True)

        shutil.copy2(item.source_path, train_dst)
        shutil.copy2(item.source_path, train_aug_dst)

        recipe = recipes[idx % len(recipes)]
        with Image.open(item.source_path) as im:
            aug_img = apply_recipe(im, recipe)
        save_image(aug_img, train_aug_aug_dst)

        provenance["imports"].append(
            {
                "label": item.label,
                "source_path": str(item.source_path.resolve()),
                "source_name": item.source_name,
                "sha256": sha256_file(item.source_path),
                "train_path": str(train_dst.resolve()),
                "train_aug_path": str(train_aug_dst.resolve()),
                "train_aug_augmented_path": str(train_aug_aug_dst.resolve()),
                "augmentation_recipe": recipe,
            }
        )

    post_train = count_images(train_root)
    post_train_aug = count_images(train_aug_root)
    provenance["post_train_counts"] = post_train
    provenance["post_train_aug_counts"] = post_train_aug

    for label, count in post_train_aug.items():
        if label in imports_by_label and count > args.train_aug_cap:
            raise RuntimeError(f"Post-import train_aug count for {label} exceeds cap: {count}")

    args.provenance_out.write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    print("Wrote provenance:", args.provenance_out)
    print("Post train counts:", post_train)
    print("Post train_aug counts:", post_train_aug)


if __name__ == "__main__":
    main()
