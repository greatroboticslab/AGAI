#!/usr/bin/env python3
"""
End-to-end grounding test: RF-DETR part detection + MiniGPT diagnosis.

Picks 1 random image per disease class, detects visible plant parts
with RF-DETR (subprocess, rfdetr conda env), injects detected/undetected
parts into the MiniGPT prompt, and generates a diagnostic report.

Usage:
    # All 7 classes, grounding ON (default)
    python test_grounding.py

    # Specific classes only
    python test_grounding.py --classes overwatered root_rot white_mold

    # Disable grounding (let MiniGPT describe freely)
    python test_grounding.py --no-ground
"""

import argparse
import random
from pathlib import Path

from grounding.config import (
    DISEASE_CLASSES, GROUND_PARTS, SEED, TRAIN_DIR,
)
from grounding.detector import run_rfdetr
from grounding.inference import load_minigpt, run_minigpt
from grounding.analyzer import analyze
from grounding import report


def pick_images(classes, seed=SEED):
    """Select one random image per disease class."""
    rng = random.Random(seed)
    selected = {}
    for cls in classes:
        cls_dir = TRAIN_DIR / cls
        imgs = [f for f in cls_dir.iterdir()
                if f.suffix.lower() in (".jpg", ".jpeg", ".png")]
        selected[cls] = str(rng.choice(imgs))
    return selected


def main():
    parser = argparse.ArgumentParser(
        description="RF-DETR + MiniGPT grounding test",
    )
    parser.add_argument(
        "--classes", nargs="*", default=None,
        help="Disease classes to test (default: all 7)",
    )
    parser.add_argument(
        "--no-ground", action="store_true",
        help="Disable grounding — let MiniGPT describe any parts freely",
    )
    parser.add_argument(
        "--seed", type=int, default=SEED,
        help=f"Random seed for image selection (default: {SEED})",
    )
    args = parser.parse_args()

    grounding_on = GROUND_PARTS and not args.no_ground
    classes = args.classes if args.classes else DISEASE_CLASSES

    images = pick_images(classes, seed=args.seed)
    report.print_header(grounding_on, classes)
    print(f"\nSelected images:")
    for cls in classes:
        print(f"  {cls}: {Path(images[cls]).name}")

    print(f"\n[Phase 1] Running RF-DETR detection (per-class best thresholds)...")
    rfdetr_results = run_rfdetr([images[c] for c in classes])
    if not rfdetr_results:
        print("RF-DETR failed. Aborting.")
        return

    print(f"\n[Phase 2] Loading MiniGPT...")
    chat, conv_template = load_minigpt()
    print("MiniGPT loaded.")

    all_results = {}

    for cls in classes:
        img_path = images[cls]
        det = rfdetr_results.get(img_path, {})
        detected_parts = det.get("detected_parts", {})
        num_dets = det.get("num_detections", 0)

        report.print_detection(
            cls, img_path, detected_parts, num_dets, grounding_on,
        )

        print(f"\n  [MiniGPT generating...]")
        response = run_minigpt(
            chat, conv_template, img_path, cls,
            detected_parts=detected_parts if grounding_on else None,
        )
        report.print_response(response)

        result = analyze(detected_parts, response, disease=cls)
        report.print_analysis(result)
        all_results[cls] = result

    report.print_summary(all_results)
    report.print_footer()


if __name__ == "__main__":
    main()
