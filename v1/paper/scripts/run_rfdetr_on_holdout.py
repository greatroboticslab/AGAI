#!/usr/bin/env python3
"""Run RF-DETR Small on the 70 disease-holdout images and report per-class
visible-part detection rates and average per-image detection counts.

This is a deployment-distribution measurement (not COCO AP, since the
holdout has no part annotations). It reports the rate at which each plant
part is detected (i.e., 'visible part list is non-empty for part p') for
each disease class, and inference latency per image.

Run inside the rfdetr conda env. Writes:
  - paper/generated/rfdetr_holdout_table.tex
  - paper/generated/rfdetr_holdout_summary.json
  - paper/figures/generated/rfdetr_holdout_heatmap.{pdf,png}
"""

from __future__ import annotations
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from rfdetr import RFDETRSmall

REPO_ROOT = Path(__file__).resolve().parents[2]
HOLDOUT_ROOT = REPO_ROOT / "plant_diagnostic" / "data" / "holdout"
CHECKPOINT = REPO_ROOT / "rfdetr_parts" / "output" / "checkpoint_best_total.pth"
FIG_ROOT = REPO_ROOT / "paper" / "figures" / "generated"
TEX_ROOT = REPO_ROOT / "paper" / "generated"

CLASS_NAMES = {0: "flower", 1: "fruit", 2: "leaf", 3: "root", 4: "soil", 5: "stem"}
PART_LIST = ["flower", "fruit", "leaf", "root", "soil", "stem"]
PER_CLASS_THRESHOLDS = {
    "flower": 0.50, "fruit": 0.50, "leaf": 0.40,
    "root": 0.30, "soil": 0.40, "stem": 0.40,
}
DISEASE_CLASSES = [
    "drought", "frost", "gray_mold", "healthy",
    "overwatered", "root_rot", "white_mold",
]
DISPLAY = {
    "drought": "Drought", "frost": "Frost", "gray_mold": "Gray mold",
    "healthy": "Healthy", "overwatered": "Overwatered",
    "root_rot": "Root rot", "white_mold": "White mold",
}
IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def main() -> None:
    FIG_ROOT.mkdir(parents=True, exist_ok=True)
    TEX_ROOT.mkdir(parents=True, exist_ok=True)

    import logging
    logging.getLogger("rfdetr").setLevel(logging.WARNING)

    print("Loading RF-DETR Small...")
    model = RFDETRSmall(pretrain_weights=str(CHECKPOINT))
    raw_threshold = min(PER_CLASS_THRESHOLDS.values())

    # detection_rate[disease][part] = fraction of disease images with at least
    # one detection of part above threshold
    counts = {d: {p: 0 for p in PART_LIST} for d in DISEASE_CLASSES}
    image_counts = {d: 0 for d in DISEASE_CLASSES}
    avg_detections = {d: 0.0 for d in DISEASE_CLASSES}
    parts_per_image = {d: 0.0 for d in DISEASE_CLASSES}
    latencies_ms: list[float] = []
    per_image_records = []

    print("Running inference on holdout...")
    for disease in DISEASE_CLASSES:
        ddir = HOLDOUT_ROOT / disease
        for img_path in sorted(ddir.iterdir()):
            if not img_path.is_file() or img_path.suffix.lower() not in IMG_EXTS:
                continue
            img = Image.open(img_path).convert("RGB")

            t0 = time.perf_counter()
            dets = model.predict(img, threshold=raw_threshold)
            t_ms = (time.perf_counter() - t0) * 1000.0
            latencies_ms.append(t_ms)

            visible_parts: dict[str, float] = {}
            n_dets_total = 0
            if hasattr(dets, "class_id") and dets.class_id is not None:
                for cid, conf, _xy in zip(dets.class_id, dets.confidence, dets.xyxy):
                    name = CLASS_NAMES.get(int(cid), f"class_{cid}")
                    if float(conf) < PER_CLASS_THRESHOLDS.get(name, 0.35):
                        continue
                    n_dets_total += 1
                    if name not in visible_parts or float(conf) > visible_parts[name]:
                        visible_parts[name] = float(conf)

            image_counts[disease] += 1
            avg_detections[disease] += n_dets_total
            parts_per_image[disease] += len(visible_parts)
            for p in visible_parts:
                counts[disease][p] += 1
            per_image_records.append({
                "image": str(img_path),
                "disease": disease,
                "visible_parts": list(visible_parts.keys()),
                "num_total_detections": n_dets_total,
                "latency_ms": t_ms,
            })

    # Aggregate
    rate = {
        d: {p: (counts[d][p] / image_counts[d] if image_counts[d] else 0.0) for p in PART_LIST}
        for d in DISEASE_CLASSES
    }
    for d in DISEASE_CLASSES:
        if image_counts[d]:
            avg_detections[d] /= image_counts[d]
            parts_per_image[d] /= image_counts[d]
    overall_rate = {p: float(np.mean([rate[d][p] for d in DISEASE_CLASSES])) for p in PART_LIST}
    median_latency = float(np.median(latencies_ms))
    p95_latency = float(np.percentile(latencies_ms, 95))

    # Save raw summary
    summary = {
        "image_counts": image_counts,
        "detection_rate": rate,
        "overall_part_detection_rate": overall_rate,
        "avg_total_detections_per_image": avg_detections,
        "avg_distinct_parts_per_image": parts_per_image,
        "median_latency_ms": median_latency,
        "p95_latency_ms": p95_latency,
        "per_image": per_image_records,
        "thresholds": PER_CLASS_THRESHOLDS,
        "checkpoint": str(CHECKPOINT),
    }
    (TEX_ROOT / "rfdetr_holdout_summary.json").write_text(json.dumps(summary, indent=2))

    # ── LaTeX heatmap-style table ─────────────────────────────────────────────
    rows = []
    for d in DISEASE_CLASSES:
        cells = " & ".join(f"{100*rate[d][p]:.0f}\\%" for p in PART_LIST)
        rows.append(
            f"{DISPLAY[d]} & {image_counts[d]} & {cells} & {parts_per_image[d]:.2f} \\\\"
        )
    overall_row = (
        "\\textbf{All classes} & \\textbf{" + str(sum(image_counts.values())) + "} & "
        + " & ".join(f"\\textbf{{{100*overall_rate[p]:.0f}\\%}}" for p in PART_LIST)
        + f" & \\textbf{{{np.mean(list(parts_per_image.values())):.2f}}} \\\\"
    )
    table = (
        "\\begin{tabular}{lr" + "r" * (len(PART_LIST) + 1) + "}\n"
        "\\toprule\n"
        "Disease & $n$ & " + " & ".join(p.title() for p in PART_LIST) + " & Mean parts \\\\\n"
        "\\midrule\n"
        + "\n".join(rows) + "\n"
        + overall_row + "\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
    )
    (TEX_ROOT / "rfdetr_holdout_table.tex").write_text(table)

    # ── Heatmap figure ────────────────────────────────────────────────────────
    matrix = np.array([[rate[d][p] for p in PART_LIST] for d in DISEASE_CLASSES])
    fig, ax = plt.subplots(figsize=(8.4, 5.0))
    im = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(np.arange(len(PART_LIST)))
    ax.set_yticks(np.arange(len(DISEASE_CLASSES)))
    ax.set_xticklabels([p.title() for p in PART_LIST])
    ax.set_yticklabels([DISPLAY[d] for d in DISEASE_CLASSES])
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            color = "white" if matrix[i, j] > 0.5 else "#111111"
            ax.text(j, i, f"{100*matrix[i, j]:.0f}", ha="center", va="center",
                    color=color, fontsize=10)
    ax.set_xlabel("Plant part")
    ax.set_ylabel("Disease class")
    ax.set_title("RF-DETR per-class part-detection rate on the disease holdout (\\%)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIG_ROOT / f"rfdetr_holdout_heatmap.{ext}", bbox_inches="tight")
    plt.close(fig)

    print(f"\nMedian latency: {median_latency:.1f} ms; P95: {p95_latency:.1f} ms")
    print("Overall per-part detection rate on holdout:")
    for p in PART_LIST:
        print(f"  {p}: {100*overall_rate[p]:.1f}%")


if __name__ == "__main__":
    main()
