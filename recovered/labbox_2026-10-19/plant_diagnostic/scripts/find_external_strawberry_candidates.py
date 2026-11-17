#!/usr/bin/env python3
"""
Download external strawberry-image candidates and compare them against the local corpus.

Assumptions
-----------
1. The local strawberry corpus under ``plant_diagnostic/data`` is the reference set for
   exact-duplicate screening.
2. External images are only *candidate* additions until manually reviewed for class
   correctness and license suitability.
3. Exact SHA-256 matches indicate byte-for-byte duplication.
4. Perceptual dHash distance is used as a coarse near-duplicate screen, not a proof of
   semantic equivalence.

Notes
-----
This script is intentionally conservative. It records the closest local image and hash
distance for each downloaded candidate so a human can decide whether the image is worth
keeping. It does not auto-import anything into training data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Dict, Iterable, List, Optional
from urllib.parse import urlparse

import numpy as np
import requests
from PIL import Image, UnidentifiedImageError


IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)


@dataclass(frozen=True)
class LocalImageRecord:
    path: str
    sha256: str
    dhash: int


def iter_local_images(root: Path) -> Iterable[Path]:
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMG_EXTS:
            yield path


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def dhash_from_image(image: Image.Image, size: int = 8) -> int:
    """Return a simple difference hash for near-duplicate screening."""
    gray = image.convert("L").resize((size + 1, size), Image.Resampling.LANCZOS)
    arr = np.asarray(gray, dtype=np.int16)
    diff = arr[:, 1:] > arr[:, :-1]
    bits = 0
    for bit in diff.flatten():
        bits = (bits << 1) | int(bit)
    return bits


def hamming_distance(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def load_local_corpus(corpus_root: Path) -> List[LocalImageRecord]:
    records: List[LocalImageRecord] = []
    for path in iter_local_images(corpus_root):
        data = path.read_bytes()
        try:
            with Image.open(BytesIO(data)) as im:
                dhash = dhash_from_image(im)
        except UnidentifiedImageError:
            continue
        records.append(
            LocalImageRecord(
                path=str(path.resolve()),
                sha256=sha256_bytes(data),
                dhash=dhash,
            )
        )
    return records


def closest_local_match(candidate_hash: int, locals_: List[LocalImageRecord]) -> Optional[Dict[str, object]]:
    if not locals_:
        return None
    best = min(locals_, key=lambda rec: hamming_distance(candidate_hash, rec.dhash))
    return {
        "path": best.path,
        "sha256": best.sha256,
        "dhash_distance": hamming_distance(candidate_hash, best.dhash),
    }


def infer_extension(content_type: str, url: str) -> str:
    parsed_path = Path(urlparse(url).path)
    if parsed_path.suffix.lower() in IMG_EXTS:
        return parsed_path.suffix.lower()
    content_type = (content_type or "").lower()
    if "jpeg" in content_type or "jpg" in content_type:
        return ".jpg"
    if "png" in content_type:
        return ".png"
    if "webp" in content_type:
        return ".webp"
    if "bmp" in content_type:
        return ".bmp"
    if "tiff" in content_type:
        return ".tif"
    raise ValueError(f"Could not infer image extension from URL/content-type: {url} / {content_type}")


def download_image(url: str, timeout_s: int) -> tuple[bytes, str]:
    resp = requests.get(url, timeout=timeout_s, headers={"User-Agent": USER_AGENT})
    resp.raise_for_status()
    content_type = resp.headers.get("content-type", "")
    if "image" not in content_type.lower():
        raise ValueError(f"URL did not return an image content type: {content_type}")
    return resp.content, content_type


def main() -> int:
    ap = argparse.ArgumentParser(description="Download and compare external strawberry-image candidates.")
    ap.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="JSON manifest listing candidate image URLs and proposed labels.",
    )
    ap.add_argument(
        "--corpus-root",
        type=Path,
        default=Path("/data/AGAI/MiniGPT-4/plant_diagnostic/data"),
        help="Local corpus root used for duplicate screening.",
    )
    ap.add_argument(
        "--download-root",
        type=Path,
        required=True,
        help="Directory where candidate images will be written.",
    )
    ap.add_argument(
        "--report-out",
        type=Path,
        required=True,
        help="JSON report path for duplicate-screening results.",
    )
    ap.add_argument("--timeout", type=int, default=20, help="HTTP timeout in seconds.")
    args = ap.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    candidates = manifest.get("candidates", [])
    if not isinstance(candidates, list) or not candidates:
        raise RuntimeError("Manifest must contain a non-empty 'candidates' list.")

    args.download_root.mkdir(parents=True, exist_ok=True)
    local_records = load_local_corpus(args.corpus_root)
    local_sha = {rec.sha256: rec for rec in local_records}

    report: Dict[str, object] = {
        "schema_version": "1.0",
        "manifest": str(args.manifest.resolve()),
        "corpus_root": str(args.corpus_root.resolve()),
        "download_root": str(args.download_root.resolve()),
        "local_image_count": len(local_records),
        "results": [],
    }

    for idx, cand in enumerate(candidates):
        url = cand["url"]
        proposed_label = cand.get("proposed_label")
        note = cand.get("note")
        source_page = cand.get("source_page")
        result: Dict[str, object] = {
            "index": idx,
            "url": url,
            "source_page": source_page,
            "proposed_label": proposed_label,
            "note": note,
            "status": "pending",
        }
        try:
            data, content_type = download_image(url, args.timeout)
            sha = sha256_bytes(data)
            ext = infer_extension(content_type, url)
            filename = f"{idx:03d}_{proposed_label or 'unlabeled'}{ext}"
            out_path = args.download_root / filename
            out_path.write_bytes(data)
            with Image.open(BytesIO(data)) as im:
                dhash = dhash_from_image(im)
                width, height = im.size
            result.update(
                {
                    "status": "downloaded",
                    "download_path": str(out_path.resolve()),
                    "sha256": sha,
                    "width": width,
                    "height": height,
                    "content_type": content_type,
                }
            )
            if sha in local_sha:
                result["duplicate_status"] = "exact_duplicate"
                result["closest_local"] = {
                    "path": local_sha[sha].path,
                    "sha256": local_sha[sha].sha256,
                    "dhash_distance": 0,
                }
            else:
                result["duplicate_status"] = "non_exact"
                result["closest_local"] = closest_local_match(dhash, local_records)
        except Exception as exc:  # noqa: BLE001
            result.update({"status": "error", "error": str(exc)})
        report["results"].append(result)

    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote report to {args.report_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
