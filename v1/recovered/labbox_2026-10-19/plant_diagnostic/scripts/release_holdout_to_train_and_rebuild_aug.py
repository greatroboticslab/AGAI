#!/usr/bin/env python3
"""
Release a fixed number of holdout images back into train, then rebuild train_aug.

Assumptions
-----------
1. ``data/train`` is the canonical compact training root.
2. ``data/holdout`` currently contains more than the desired final support per
   class, and selected images can be moved back into ``data/train``.
3. ``data/train_aug`` should be rebuilt cleanly from ``data/train`` before any
   new balancing augmentation is applied.
4. This script performs real filesystem moves and copies, so it emits a
   provenance record describing the exact transition.
"""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List


CLASS_ORDER = [
    "drought",
    "frost",
    "gray_mold",
    "healthy",
    "overwatered",
    "root_rot",
    "white_mold",
]
IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def count_images(root: Path) -> Dict[str, int]:
    return {
        d.name: sum(1 for p in d.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS)
        for d in sorted(root.iterdir())
        if d.is_dir()
    }


def list_images(root: Path) -> List[Path]:
    return sorted(p for p in root.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS)


def ensure_clean_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def copy_tree_images(src_root: Path, dst_root: Path) -> None:
    for class_dir in sorted(p for p in src_root.iterdir() if p.is_dir()):
        dst_class = dst_root / class_dir.name
        ensure_clean_dir(dst_class)
        for src in list_images(class_dir):
            dst = dst_class / src.name
            if dst.exists():
                raise FileExistsError(f"Destination already exists while rebuilding train_aug: {dst}")
            shutil.copy2(src, dst)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Move holdout images back into train and rebuild train_aug from scratch."
    )
    ap.add_argument("--repo-root", type=Path, default=Path("/data/AGAI/MiniGPT-4"))
    ap.add_argument(
        "--release-per-class",
        type=int,
        default=10,
        help="Number of holdout images per class to move back into train.",
    )
    ap.add_argument(
        "--provenance-out",
        type=Path,
        default=Path(
            "/data/AGAI/MiniGPT-4/plant_diagnostic/data/holdout_release10_train_aug_rebuild_provenance.json"
        ),
    )
    args = ap.parse_args()

    repo = args.repo_root.resolve()
    data_root = repo / "plant_diagnostic" / "data"
    train_root = data_root / "train"
    holdout_root = data_root / "holdout"
    train_aug_root = data_root / "train_aug"

    pre_train = count_images(train_root)
    pre_holdout = count_images(holdout_root)
    pre_train_aug = count_images(train_aug_root)

    moves: Dict[str, List[dict]] = {}
    for label in CLASS_ORDER:
        hold_dir = holdout_root / label
        train_dir = train_root / label
        ensure_clean_dir(train_dir)

        hold_files = list_images(hold_dir)
        if len(hold_files) < args.release_per_class:
            raise RuntimeError(
                f"Holdout for {label} has only {len(hold_files)} images; cannot release {args.release_per_class}."
            )

        # Deterministic and auditable: keep the lexicographically first N images
        # in holdout and release the lexicographically last N images back to train.
        selected = hold_files[-args.release_per_class :]
        label_moves: List[dict] = []
        for src in selected:
            dst = train_dir / src.name
            if dst.exists():
                raise FileExistsError(f"Train destination already exists: {dst}")
            shutil.move(str(src), str(dst))
            label_moves.append(
                {
                    "holdout_path_before": str(src.resolve()),
                    "train_path_after": str(dst.resolve()),
                }
            )
        moves[label] = label_moves

    backup_name = datetime.now(timezone.utc).strftime("train_aug_backup_before_holdout_release10_%Y%m%dT%H%M%SZ")
    backup_root = data_root / backup_name
    if backup_root.exists():
        raise FileExistsError(f"Backup path already exists: {backup_root}")
    shutil.move(str(train_aug_root), str(backup_root))
    ensure_clean_dir(train_aug_root)
    copy_tree_images(train_root, train_aug_root)

    post_train = count_images(train_root)
    post_holdout = count_images(holdout_root)
    post_train_aug = count_images(train_aug_root)

    for label, count in post_holdout.items():
        expected = pre_holdout[label] - args.release_per_class
        if count != expected:
            raise RuntimeError(f"Unexpected post-release holdout count for {label}: got {count}, expected {expected}")
        if post_train_aug[label] != post_train[label]:
            raise RuntimeError(
                f"Fresh train_aug rebuild mismatch for {label}: train={post_train[label]}, train_aug={post_train_aug[label]}"
            )

    provenance = {
        "schema_version": "1.0",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "selection_policy": "sorted_tail",
        "release_per_class": args.release_per_class,
        "train_aug_backup_path": str(backup_root.resolve()),
        "pre_train_counts": pre_train,
        "pre_holdout_counts": pre_holdout,
        "pre_train_aug_counts": pre_train_aug,
        "moves": moves,
        "post_train_counts": post_train,
        "post_holdout_counts": post_holdout,
        "post_train_aug_counts_before_balance": post_train_aug,
    }
    args.provenance_out.write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    print("Wrote provenance:", args.provenance_out)
    print("Post train counts:", post_train)
    print("Post holdout counts:", post_holdout)
    print("Post train_aug counts before balance:", post_train_aug)


if __name__ == "__main__":
    main()
