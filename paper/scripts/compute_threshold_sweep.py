#!/usr/bin/env python3
"""Threshold sensitivity sweep and risk-coverage curve from the v4 holdout predictions.

Outputs:
  - paper/figures/generated/threshold_sweep.{pdf,png}
  - paper/figures/generated/risk_coverage.{pdf,png}
  - paper/generated/threshold_sweep_summary.tex
"""

from __future__ import annotations
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
HOLDOUT_V4 = REPO_ROOT / "evaluation" / "holdout" / "resnet_v4_release10holdout_full_holdout.json"
FIG_ROOT = REPO_ROOT / "paper" / "figures" / "generated"
TEX_ROOT = REPO_ROOT / "paper" / "generated"

CLASS_THRESH = {
    "healthy": 0.40,
    "overwatered": 0.60,
    "root_rot": 0.65,
    "drought": 0.65,
    "frost": 0.70,
    "gray_mold": 0.60,
    "white_mold": 0.60,
}
LOW_CONF_DEMOTE = 0.50


def main() -> None:
    FIG_ROOT.mkdir(parents=True, exist_ok=True)
    TEX_ROOT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {"figure.dpi": 180, "savefig.dpi": 300, "axes.spines.top": False,
         "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.22,
         "grid.linestyle": ":", "font.size": 11}
    )

    d = json.loads(HOLDOUT_V4.read_text())
    preds = d["predictions"]
    p1 = np.array([float(p["p1"]) for p in preds])
    correct = np.array([1 if p["pred_top1"] == p["truth"] else 0 for p in preds])

    # ── Threshold sweep: uniform tau over [0, 1) ──────────────────────────────
    taus = np.linspace(0.0, 0.99, 100)
    accept_rates = []
    cond_accs = []
    for t in taus:
        accept = p1 >= t
        if accept.sum() == 0:
            accept_rates.append(0.0)
            cond_accs.append(np.nan)
        else:
            accept_rates.append(float(accept.mean()))
            cond_accs.append(float(correct[accept].mean()))

    # Deployed gate operating point
    deployed_accept = []
    for p in preds:
        lab = p["pred_top1"]
        prob = float(p["p1"])
        tau = CLASS_THRESH.get(lab, 0.55)
        deployed_accept.append(prob >= tau and prob >= LOW_CONF_DEMOTE)
    deployed_accept = np.array(deployed_accept)
    dep_accept_rate = float(deployed_accept.mean())
    dep_cond_acc = float(correct[deployed_accept].mean()) if deployed_accept.any() else float("nan")

    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    ax.plot(taus, accept_rates, color="#4c78a8", linewidth=2.0, label="Acceptance rate")
    ax.set_xlabel("Uniform threshold $\\tau$ on top-1 probability")
    ax.set_ylabel("Acceptance rate", color="#4c78a8")
    ax.tick_params(axis="y", colors="#4c78a8")
    ax.set_ylim(0, 1.05)
    ax2 = ax.twinx()
    ax2.plot(taus, cond_accs, color="#e15759", linewidth=2.0, label="Conditional accuracy")
    ax2.set_ylabel("Conditional accuracy on accepted set", color="#e15759")
    ax2.tick_params(axis="y", colors="#e15759")
    ax2.set_ylim(0, 1.05)
    ax2.grid(False)

    # Mark deployed operating point
    ax.scatter([np.nan], [dep_accept_rate])  # for legend alignment
    ax.axhline(dep_accept_rate, color="#4c78a8", linestyle="--", alpha=0.5)
    ax2.axhline(dep_cond_acc, color="#e15759", linestyle="--", alpha=0.5)
    ax.text(
        0.02, dep_accept_rate + 0.02,
        f"Deployed gate: accept={100*dep_accept_rate:.1f}%, acc|accept={100*dep_cond_acc:.1f}%",
        fontsize=9, color="#22313f",
    )
    ax.set_title("Threshold sensitivity on the balanced holdout")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIG_ROOT / f"threshold_sweep.{ext}", bbox_inches="tight")
    plt.close(fig)

    # ── Risk-coverage curve ───────────────────────────────────────────────────
    order = np.argsort(-p1)  # high confidence first
    p1_sorted = p1[order]
    correct_sorted = correct[order]
    n = len(p1_sorted)
    coverage = np.arange(1, n + 1) / n
    cum_correct = np.cumsum(correct_sorted)
    selective_acc = cum_correct / np.arange(1, n + 1)
    risk = 1.0 - selective_acc

    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    ax.plot(coverage, selective_acc, color="#3182bd", linewidth=2.0)
    ax.set_xlabel("Coverage")
    ax.set_ylabel("Selective accuracy")
    ax.set_xlim(0, 1)
    ax.set_ylim(0.5, 1.02)
    ax.set_title("Risk-coverage curve, ordered by descending top-1 probability")
    ax.scatter([dep_accept_rate], [dep_cond_acc], color="#e15759", s=70, zorder=5,
               label=f"Deployed gate ({100*dep_accept_rate:.0f}%, {100*dep_cond_acc:.1f}%)")
    ax.legend(frameon=False, loc="lower left")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIG_ROOT / f"risk_coverage.{ext}", bbox_inches="tight")
    plt.close(fig)

    # ── LaTeX summary fragment ────────────────────────────────────────────────
    # Selective accuracy at fixed coverage levels
    levels = [0.50, 0.70, 0.85, 1.00]
    rows = []
    for cov in levels:
        k = max(1, int(np.ceil(cov * n)))
        rows.append((cov, k, float(selective_acc[k - 1])))
    body = "\n".join(
        f"{int(100*cov)}\\% & {k} & {100*acc:.2f}\\% \\\\"
        for cov, k, acc in rows
    )
    table = (
        "\\begin{tabular}{lrr}\n"
        "\\toprule\n"
        "Coverage & Accepted images & Selective top-1 accuracy \\\\\n"
        "\\midrule\n"
        + body + "\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
    )
    (TEX_ROOT / "selective_accuracy_table.tex").write_text(table)

    print(f"Deployed gate operating point: accept={100*dep_accept_rate:.1f}%, acc|accept={100*dep_cond_acc:.2f}%")
    for cov, k, acc in rows:
        print(f"  Coverage {int(100*cov)}% (top {k} by p1): selective acc = {100*acc:.2f}%")


if __name__ == "__main__":
    main()
