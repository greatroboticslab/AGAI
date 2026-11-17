#!/usr/bin/env python3
"""Build an evaluation-only holdout with real holdout + synthetic train_aug copy.

This script does not remove anything from train_aug. It copies:
1. Every file from the current real holdout.
2. A deterministic 15% per-class sample of synthetic files from train_aug.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("/data/AGAI/MiniGPT-4/plant_diagnostic/data"),
    )
    parser.add_argument("--holdout-frac", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-root", type=Path, default=None)
    parser.add_argument("--provenance-out", type=Path, default=None)
    args = parser.parse_args()

    data_root = args.data_root.resolve()
    real_holdout_root = data_root / "holdout"
    train_aug_root = data_root / "train_aug"
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    out_root = args.out_root or (data_root / f"holdout_mixed_synth15_{timestamp}")
    provenance_out = args.provenance_out or (
        data_root / f"holdout_mixed_synth15_{timestamp}.json"
    )

    if out_root.exists():
        raise FileExistsError(f"Output holdout already exists: {out_root}")
    if provenance_out.exists():
        raise FileExistsError(f"Provenance path already exists: {provenance_out}")

    rng = random.Random(args.seed)
    out_root.mkdir(parents=True, exist_ok=False)
    for class_name in CLASS_ORDER:
        (out_root / class_name).mkdir(parents=True, exist_ok=False)

    real_copies: Dict[str, List[dict]] = {class_name: [] for class_name in CLASS_ORDER}
    synth_copies: Dict[str, List[dict]] = {class_name: [] for class_name in CLASS_ORDER}
    synth_pool_counts: Dict[str, int] = {}
    synth_eval_counts: Dict[str, int] = {}

    # Copy all real holdout images first.
    for class_name in CLASS_ORDER:
        for src in list_images(real_holdout_root / class_name):
            dst = out_root / class_name / src.name
            shutil.copy2(src, dst)
            real_copies[class_name].append(
                {"src": str(src), "dst": str(dst), "sha256": file_sha256(src)}
            )

    # Copy a deterministic 15% synthetic sample from train_aug.
    for class_name in CLASS_ORDER:
        synth_pool = [p for p in list_images(train_aug_root / class_name) if is_synthetic(p)]
        synth_pool_counts[class_name] = len(synth_pool)
        n_eval = max(1, round(len(synth_pool) * args.holdout_frac)) if synth_pool else 0
        synth_eval_counts[class_name] = n_eval
        pool = synth_pool[:]
        rng.shuffle(pool)
        chosen = sorted(pool[:n_eval])
        for src in chosen:
            dst = out_root / class_name / src.name
            if dst.exists():
                dst = out_root / class_name / f"evalsynth_{src.name}"
                if dst.exists():
                    raise FileExistsError(f"Destination already exists: {dst}")
            shutil.copy2(src, dst)
            synth_copies[class_name].append(
                {"src": str(src), "dst": str(dst), "sha256": file_sha256(src)}
            )

    provenance = {
        "schema_version": "1.0",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "data_root": str(data_root),
        "real_holdout_root": str(real_holdout_root),
        "train_aug_root": str(train_aug_root),
        "output_holdout_root": str(out_root),
        "holdout_frac": args.holdout_frac,
        "seed": args.seed,
        "real_holdout_counts": count_per_class(real_holdout_root),
        "synthetic_pool_counts": synth_pool_counts,
        "synthetic_eval_counts": synth_eval_counts,
        "mixed_holdout_counts": count_per_class(out_root),
        "real_copies": real_copies,
        "synthetic_copies": synth_copies,
    }
    provenance_out.write_text(json.dumps(provenance, indent=2))

    print(f"Created mixed holdout at {out_root}")
    print("Real holdout counts:", provenance["real_holdout_counts"])
    print("Synthetic eval counts:", synth_eval_counts)
    print("Mixed holdout counts:", provenance["mixed_holdout_counts"])
    print(f"Wrote provenance to {provenance_out}")


if __name__ == "__main__":
    main()
