#!/usr/bin/env python3
"""Render the 8 misclassified holdout images with truth/predicted/p1 captions."""

from __future__ import annotations
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
HOLDOUT_V4 = REPO_ROOT / "evaluation" / "holdout" / "resnet_v4_release10holdout_full_holdout.json"
FIG_ROOT = REPO_ROOT / "paper" / "figures" / "generated"

DISPLAY = {
    "drought": "Drought",
    "frost": "Frost",
    "gray_mold": "Gray mold",
    "healthy": "Healthy",
    "overwatered": "Overwatered",
    "root_rot": "Root rot",
    "white_mold": "White mold",
}

CLASS_THRESH = {
    "healthy": 0.40, "overwatered": 0.60, "root_rot": 0.65, "drought": 0.65,
    "frost": 0.70, "gray_mold": 0.60, "white_mold": 0.60,
}


def main() -> None:
    FIG_ROOT.mkdir(parents=True, exist_ok=True)
    d = json.loads(HOLDOUT_V4.read_text())
    miscls = [p for p in d["predictions"] if p["pred_top1"] != p["truth"]]
    miscls = sorted(miscls, key=lambda p: (p["truth"], Path(p["path"]).name))

    n = len(miscls)
    cols = 4
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(13, 3.6 * rows))
    axes = axes.ravel()
    for ax in axes:
        ax.axis("off")

    for i, p in enumerate(miscls):
        path = Path(p["path"])
        with Image.open(path) as img:
            axes[i].imshow(np.asarray(img.convert("RGB")))
        prob = float(p["p1"])
        tau = CLASS_THRESH.get(p["pred_top1"], 0.55)
        gated = "demoted" if prob < max(tau, 0.50) else "accepted"
        title = (
            f"{DISPLAY[p['truth']]} $\\rightarrow$ {DISPLAY[p['pred_top1']]}\n"
            f"$p_1$ = {prob:.2f}, gate: {gated}"
        )
        axes[i].set_title(title, fontsize=10)

    fig.suptitle(
        "Misclassified holdout images for the deployed v4 checkpoint",
        y=1.0, fontsize=13,
    )
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIG_ROOT / f"misclassification_grid.{ext}", bbox_inches="tight")
    plt.close(fig)
    print(f"Rendered {n} misclassified images.")


if __name__ == "__main__":
    main()
