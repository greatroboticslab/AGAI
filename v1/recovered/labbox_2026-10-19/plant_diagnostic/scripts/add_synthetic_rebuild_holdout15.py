#!/usr/bin/env python3
"""
Re-pool real images, take a 15% per-class holdout, then rebuild train_aug as
(train residual) + synthetic_data mapped into matching class folders.

train/   stays real-only (so future holdout splits never sample synthetic).
holdout/ is rebuilt from the real pool (train ∪ existing holdout).
train_aug/ = real residual + synthetic.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
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

SYNTHETIC_MAP = {
    "droughtGPT": "drought",
    "frostGPT": "frost",
    "graymoldGPT": "gray_mold",
    "gray_moldGPT": "gray_mold",
    "healthyGPT": "healthy",
    "overwateredGPT": "overwatered",
    "rootrotGPT": "root_rot",
    "root_rotGPT": "root_rot",
    "whitemoldGPT": "white_mold",
    "white_moldGPT": "white_mold",
}

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
COPY_SUFFIX_RE = re.compile(r"\s*\(\d+\)$")


def list_images(d: Path) -> List[Path]:
    if not d.exists():
        return []
    return sorted(p for p in d.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS)


def count_per_class(root: Path) -> Dict[str, int]:
    return {c: len(list_images(root / c)) for c in CLASS_ORDER}


def canonical_synthetic_dir_name(name: str) -> str:
    """Normalize copied folder names such as ``droughtGPT (1)``."""
    return COPY_SUFFIX_RE.sub("", name)


def file_sha256(path: Path) -> str:
    """Return a content hash for exact duplicate image detection."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def safe_copy(src: Path, dst: Path) -> None:
    if dst.exists():
        raise FileExistsError(f"Destination already exists: {dst}")
    shutil.copy2(src, dst)


def safe_move(src: Path, dst: Path) -> None:
    if dst.exists():
        raise FileExistsError(f"Destination already exists: {dst}")
    shutil.move(str(src), str(dst))


def quarantine_duplicate_real_images(
    train_root: Path,
    duplicate_backup_root: Path,
) -> Dict[str, List[dict]]:
    """Move exact duplicate real images out of the class pool before splitting."""
    duplicate_records: Dict[str, List[dict]] = {c: [] for c in CLASS_ORDER}
    for c in CLASS_ORDER:
        seen: Dict[str, Path] = {}
        for src in list_images(train_root / c):
            digest = file_sha256(src)
            kept = seen.get(digest)
            if kept is None:
                seen[digest] = src
                continue

            dst_dir = duplicate_backup_root / c
            dst_dir.mkdir(parents=True, exist_ok=True)
            dst = dst_dir / src.name
            if dst.exists():
                dst = dst_dir / f"{digest[:12]}_{src.name}"
            safe_move(src, dst)
            duplicate_records[c].append(
                {
                    "kept": str(kept),
                    "quarantined": str(dst),
                    "sha256": digest,
                }
            )
    return duplicate_records


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path,
                    default=Path("/data/AGAI/MiniGPT-4/plant_diagnostic/data"))
    ap.add_argument("--synthetic-root", type=Path,
                    default=Path("/data/AGAI/synthetic_data"))
    ap.add_argument("--holdout-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--provenance-out", type=Path, default=None)
    args = ap.parse_args()

    data = args.data_root.resolve()
    train_root = data / "train"
    holdout_root = data / "holdout"
    train_aug_root = data / "train_aug"

    pre_train = count_per_class(train_root)
    pre_holdout = count_per_class(holdout_root)
    pre_train_aug = count_per_class(train_aug_root)

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    holdout_backup = data / f"holdout_backup_before_synthetic15_{ts}"
    train_aug_backup = data / f"train_aug_backup_before_synthetic15_{ts}"
    real_duplicate_backup = data / f"real_duplicate_backup_before_synthetic15_{ts}"
    if holdout_backup.exists() or train_aug_backup.exists():
        raise FileExistsError("Backup paths already exist; aborting.")

    # --- Step 1: backup current holdout & train_aug, then pool real images
    # back into train/ so we can resample a fresh 15% holdout from the full
    # real pool (train ∪ holdout, both real).
    shutil.move(str(holdout_root), str(holdout_backup))
    shutil.move(str(train_aug_root), str(train_aug_backup))

    holdout_root.mkdir()
    train_aug_root.mkdir()
    holdout_release_renames: Dict[str, List[dict]] = {}
    for c in CLASS_ORDER:
        (holdout_root / c).mkdir()
        (train_aug_root / c).mkdir()
        (train_root / c).mkdir(exist_ok=True)
        # release backed-up holdout images back into train (real pool).
        renames: List[dict] = []
        for src in list_images(holdout_backup / c):
            dst = train_root / c / src.name
            if dst.exists():
                # Same filename already in train (likely from a prior holdout
                # release with a since-reused name). Disambiguate so we don't
                # lose either image.
                dst = train_root / c / f"holdrel_{src.name}"
                if dst.exists():
                    raise FileExistsError(f"Even prefixed dest exists: {dst}")
                renames.append({"original_name": src.name, "stored_as": dst.name})
            safe_move(src, dst)
        holdout_release_renames[c] = renames

    real_duplicates = quarantine_duplicate_real_images(
        train_root,
        real_duplicate_backup,
    )
    pool_after_release = count_per_class(train_root)

    # --- Step 2: stratified 15% holdout per class (deterministic via seed)
    rng = random.Random(args.seed)
    moves_to_holdout: Dict[str, List[dict]] = {}
    for c in CLASS_ORDER:
        files = list_images(train_root / c)
        n = len(files)
        n_hold = max(1, round(n * args.holdout_frac)) if n > 0 else 0
        # shuffle deterministically
        idxs = list(range(n))
        rng.shuffle(idxs)
        chosen = sorted([files[i] for i in idxs[:n_hold]])
        rec: List[dict] = []
        for src in chosen:
            dst = holdout_root / c / src.name
            safe_move(src, dst)
            rec.append({"from_train": str(src), "to_holdout": str(dst)})
        moves_to_holdout[c] = rec

    # --- Step 3: rebuild train_aug as a copy of remaining train (real)
    for c in CLASS_ORDER:
        for src in list_images(train_root / c):
            safe_copy(src, train_aug_root / c / src.name)

    # --- Step 4: copy synthetic into train_aug under mapped class names
    synth_added: Dict[str, int] = {c: 0 for c in CLASS_ORDER}
    synth_skipped_duplicates: Dict[str, List[dict]] = {c: [] for c in CLASS_ORDER}
    synth_records: Dict[str, List[dict]] = {c: [] for c in CLASS_ORDER}
    if not args.synthetic_root.exists():
        raise SystemExit(f"Synthetic root not found: {args.synthetic_root}")

    synthetic_dirs: Dict[str, List[Path]] = {k: [] for k in SYNTHETIC_MAP}
    unknown_synthetic_dirs: List[str] = []
    for p in sorted(args.synthetic_root.iterdir()):
        if not p.is_dir():
            continue
        canonical = canonical_synthetic_dir_name(p.name)
        if canonical not in SYNTHETIC_MAP:
            unknown_synthetic_dirs.append(p.name)
            continue
        synthetic_dirs[canonical].append(p)

    unknown_synthetic_dirs = sorted(unknown_synthetic_dirs)
    if unknown_synthetic_dirs:
        raise SystemExit(
            "Unmapped synthetic class directories found: "
            + ", ".join(unknown_synthetic_dirs)
        )

    class_hashes: Dict[str, Dict[str, Path]] = {c: {} for c in CLASS_ORDER}
    for c in CLASS_ORDER:
        for src in list_images(train_aug_root / c):
            class_hashes[c][file_sha256(src)] = src

    for src_subdir, mapped in sorted(SYNTHETIC_MAP.items()):
        for sdir in synthetic_dirs[src_subdir]:
            for src in list_images(sdir):
                digest = file_sha256(src)
                existing = class_hashes[mapped].get(digest)
                if existing is not None:
                    synth_skipped_duplicates[mapped].append(
                        {
                            "from_synth": str(src),
                            "duplicate_of": str(existing),
                            "sha256": digest,
                        }
                    )
                    continue
                # prefix to avoid name collisions and to make synthetic provenance obvious
                dst_name = f"synth_{sdir.name}_{src.name}"
                dst = train_aug_root / mapped / dst_name
                safe_copy(src, dst)
                class_hashes[mapped][digest] = dst
                synth_added[mapped] += 1
                synth_records[mapped].append(
                    {"from_synth": str(src), "to_train_aug": str(dst), "sha256": digest}
                )

    post_train = count_per_class(train_root)
    post_holdout = count_per_class(holdout_root)
    post_train_aug = count_per_class(train_aug_root)

    # sanity checks
    for c in CLASS_ORDER:
        if post_train[c] + post_holdout[c] != pool_after_release[c]:
            raise RuntimeError(
                f"Real pool mismatch for {c}: train={post_train[c]} + holdout={post_holdout[c]} "
                f"!= pool={pool_after_release[c]}"
            )
        if post_train_aug[c] != post_train[c] + synth_added[c]:
            raise RuntimeError(
                f"train_aug mismatch for {c}: aug={post_train_aug[c]} != train+synth="
                f"{post_train[c] + synth_added[c]}"
            )

    provenance = {
        "schema_version": "1.0",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "holdout_frac": args.holdout_frac,
        "seed": args.seed,
        "synthetic_root": str(args.synthetic_root),
        "synthetic_class_map": SYNTHETIC_MAP,
        "holdout_backup_path": str(holdout_backup),
        "train_aug_backup_path": str(train_aug_backup),
        "real_duplicate_backup_path": str(real_duplicate_backup),
        "pre_train_counts": pre_train,
        "pre_holdout_counts": pre_holdout,
        "pre_train_aug_counts": pre_train_aug,
        "real_pool_counts_after_release": pool_after_release,
        "real_duplicates_quarantined_per_class": {
            c: len(v) for c, v in real_duplicates.items()
        },
        "post_train_counts": post_train,
        "post_holdout_counts": post_holdout,
        "post_train_aug_counts": post_train_aug,
        "synthetic_added_per_class": synth_added,
        "synthetic_skipped_duplicates_per_class": {
            c: len(v) for c, v in synth_skipped_duplicates.items()
        },
        "moves_to_holdout": moves_to_holdout,
        "holdout_release_renames": holdout_release_renames,
        "real_duplicates_quarantined": real_duplicates,
        "synthetic_copies": synth_records,
        "synthetic_skipped_duplicates": synth_skipped_duplicates,
    }

    out = args.provenance_out or (
        data / f"synthetic15_holdout_rebuild_provenance_{ts}.json"
    )
    out.write_text(json.dumps(provenance, indent=2), encoding="utf-8")

    print("Wrote provenance:", out)
    print("Real pool per class (after release):", pool_after_release)
    print("Post train (real, 85%):           ", post_train)
    print("Post holdout (real, 15%):         ", post_holdout)
    print("Synthetic added to train_aug:     ", synth_added)
    print("Post train_aug (real+synth):      ", post_train_aug)


if __name__ == "__main__":
    main()
