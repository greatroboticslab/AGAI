#!/usr/bin/env python3
"""Build a mixed synthetic+real holdout and a train snapshot without leakage.

This script copies:
1. Every image from the current real holdout into a new mixed holdout.
2. A deterministic per-class sample of synthetic images from train_aug into that holdout.
3. Every remaining image from train_aug into a new train root, excluding the sampled
   synthetic holdout images so the new train tree does not leak eval examples.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Set


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
SYNTH_MARKERS = ("synth_", "gpt", "chatgpt")


def list_images(root: Path) -> List[Path]:
    if not root.exists():
        return []
    return sorted(
        p for p in root.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS
    )


def is_synthetic(path: Path) -> bool:
    text = f"{path.parent.name}/{path.name}".lower()
    return any(marker in text for marker in SYNTH_MARKERS)


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def count_per_class(root: Path) -> Dict[str, int]:
    return {class_name: len(list_images(root / class_name)) for class_name in CLASS_ORDER}


def copy_image(src: Path, dst_dir: Path, prefix: str = "") -> Path:
    dst = dst_dir / f"{prefix}{src.name}"
    if dst.exists():
        dst = dst_dir / f"{prefix}{src.stem}_{file_sha256(src)[:10]}{src.suffix.lower()}"
    if dst.exists():
        raise FileExistsError(f"Destination already exists: {dst}")
    shutil.copy2(src, dst)
    return dst


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("/data/AGAI/MiniGPT-4/plant_diagnostic/data"),
    )
    parser.add_argument("--holdout-frac", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-out", type=Path, default=None)
    parser.add_argument("--holdout-out", type=Path, default=None)
    parser.add_argument("--provenance-out", type=Path, default=None)
    args = parser.parse_args()

    data_root = args.data_root.resolve()
    real_holdout_root = data_root / "holdout"
    train_aug_root = data_root / "train_aug"
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    train_out = args.train_out or (data_root / f"train_aug_exclusive_synth15_{timestamp}")
    holdout_out = args.holdout_out or (data_root / f"holdout_mixed_exclusive_synth15_{timestamp}")
    provenance_out = args.provenance_out or (
        data_root / f"mixed_split_exclusive_synth15_{timestamp}.json"
    )

    for path in (train_out, holdout_out, provenance_out):
        if path.exists():
            raise FileExistsError(f"Output already exists: {path}")

    rng = random.Random(args.seed)
    train_out.mkdir(parents=True, exist_ok=False)
    holdout_out.mkdir(parents=True, exist_ok=False)
    for class_name in CLASS_ORDER:
        (train_out / class_name).mkdir(parents=True, exist_ok=False)
        (holdout_out / class_name).mkdir(parents=True, exist_ok=False)

    real_holdout_copies: Dict[str, List[dict]] = {class_name: [] for class_name in CLASS_ORDER}
    synth_holdout_copies: Dict[str, List[dict]] = {class_name: [] for class_name in CLASS_ORDER}
    train_copies: Dict[str, List[dict]] = {class_name: [] for class_name in CLASS_ORDER}
    synth_pool_counts: Dict[str, int] = {}
    synth_holdout_counts: Dict[str, int] = {}
    excluded_train_paths: Set[Path] = set()

    for class_name in CLASS_ORDER:
        for src in list_images(real_holdout_root / class_name):
            dst = copy_image(src, holdout_out / class_name)
            real_holdout_copies[class_name].append(
                {"src": str(src), "dst": str(dst), "sha256": file_sha256(src)}
            )

    for class_name in CLASS_ORDER:
        synth_pool = [p for p in list_images(train_aug_root / class_name) if is_synthetic(p)]
        synth_pool_counts[class_name] = len(synth_pool)
        n_eval = max(1, round(len(synth_pool) * args.holdout_frac)) if synth_pool else 0
        synth_holdout_counts[class_name] = n_eval
        pool = synth_pool[:]
        rng.shuffle(pool)
        chosen = sorted(pool[:n_eval])
        excluded_train_paths.update(chosen)
        for src in chosen:
            dst = copy_image(src, holdout_out / class_name, prefix="evalsynth_")
            synth_holdout_copies[class_name].append(
                {"src": str(src), "dst": str(dst), "sha256": file_sha256(src)}
            )

    for class_name in CLASS_ORDER:
        for src in list_images(train_aug_root / class_name):
            if src in excluded_train_paths:
                continue
            dst = copy_image(src, train_out / class_name)
            train_copies[class_name].append(
                {"src": str(src), "dst": str(dst), "sha256": file_sha256(src)}
            )

    provenance = {
        "schema_version": "1.0",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "data_root": str(data_root),
        "train_aug_root": str(train_aug_root),
        "real_holdout_root": str(real_holdout_root),
        "exclusive_train_root": str(train_out),
        "mixed_holdout_root": str(holdout_out),
        "holdout_frac": args.holdout_frac,
        "seed": args.seed,
        "source_train_aug_counts": count_per_class(train_aug_root),
        "real_holdout_counts": count_per_class(real_holdout_root),
        "synthetic_pool_counts": synth_pool_counts,
        "synthetic_holdout_counts": synth_holdout_counts,
        "exclusive_train_counts": count_per_class(train_out),
        "mixed_holdout_counts": count_per_class(holdout_out),
        "real_holdout_copies": real_holdout_copies,
        "synthetic_holdout_copies": synth_holdout_copies,
        "train_copies": train_copies,
    }
    provenance_out.write_text(json.dumps(provenance, indent=2))

    print(f"Created exclusive train root at {train_out}")
    print(f"Created mixed holdout root at {holdout_out}")
    print("Source train_aug counts:", provenance["source_train_aug_counts"])
    print("Real holdout counts:", provenance["real_holdout_counts"])
    print("Synthetic holdout counts:", provenance["synthetic_holdout_counts"])
    print("Exclusive train counts:", provenance["exclusive_train_counts"])
    print("Mixed holdout counts:", provenance["mixed_holdout_counts"])
    print(f"Wrote provenance to {provenance_out}")


if __name__ == "__main__":
    main()
