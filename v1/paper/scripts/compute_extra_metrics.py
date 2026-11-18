#!/usr/bin/env python3
"""Compute additional metrics from the v4 holdout predictions:
- Bootstrap 95% CIs for top-1, top-2, ECE
- Threshold-gate behavior on the holdout (acceptance rate, conditional accuracy)
- Misclassification list

Outputs LaTeX fragments under paper/generated/.
"""

from __future__ import annotations
import json
import math
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
HOLDOUT_V4 = REPO_ROOT / "evaluation" / "holdout" / "resnet_v4_release10holdout_full_holdout.json"
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
DEFAULT_THRESH = 0.55
LOW_CONF_DEMOTE = 0.50

DISPLAY = {
    "drought": "Drought",
    "frost": "Frost",
    "gray_mold": "Gray mold",
    "healthy": "Healthy",
    "overwatered": "Overwatered",
    "root_rot": "Root rot",
    "white_mold": "White mold",
}


def bootstrap_ci(values: np.ndarray, n_boot: int = 10000, seed: int = 42) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(values)
    means = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        means[i] = values[idx].mean()
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


def main() -> None:
    d = json.loads(HOLDOUT_V4.read_text())
    preds = d["predictions"]
    n = len(preds)

    correct1 = np.array([1 if p["pred_top1"] == p["truth"] else 0 for p in preds])
    correct2 = np.array(
        [1 if p["truth"] in {t["label"] for t in p["top2"]} else 0 for p in preds]
    )
    p1 = np.array([float(p["p1"]) for p in preds])
    # ECE per-sample contribution: |p1 - correct| computed by bin externally is hard;
    # instead we bootstrap top-1 and top-2 directly.

    top1_lo, top1_hi = bootstrap_ci(correct1)
    top2_lo, top2_hi = bootstrap_ci(correct2)

    # Threshold-gate behavior: for each sample, compute accept under per-class thresholds + 0.50 demote.
    accepts = []
    correct_given_accept = []
    for p in preds:
        lab = p["pred_top1"]
        prob = float(p["p1"])
        tau = CLASS_THRESH.get(lab, DEFAULT_THRESH)
        accept = prob >= tau and prob >= LOW_CONF_DEMOTE
        accepts.append(int(accept))
        if accept:
            correct_given_accept.append(int(p["pred_top1"] == p["truth"]))

    accept_rate = float(np.mean(accepts))
    n_accepted = int(np.sum(accepts))
    cond_acc = float(np.mean(correct_given_accept)) if correct_given_accept else float("nan")
    cond_lo, cond_hi = bootstrap_ci(np.array(correct_given_accept)) if correct_given_accept else (float("nan"), float("nan"))

    # Per-class accept rate
    by_class = {}
    for c in DISPLAY:
        rows = [p for p in preds if p["truth"] == c]
        acc_count = 0
        cor_in_acc = 0
        for p in rows:
            lab = p["pred_top1"]
            prob = float(p["p1"])
            tau = CLASS_THRESH.get(lab, DEFAULT_THRESH)
            accept = prob >= tau and prob >= LOW_CONF_DEMOTE
            if accept:
                acc_count += 1
                if p["pred_top1"] == p["truth"]:
                    cor_in_acc += 1
        by_class[c] = (
            len(rows),
            acc_count,
            cor_in_acc,
        )

    # Misclassification list
    miscls = [
        (Path(p["path"]).name, p["truth"], p["pred_top1"], float(p["p1"]))
        for p in preds
        if p["pred_top1"] != p["truth"]
    ]

    # ── Write bootstrap CI snippet ────────────────────────────────────────────
    boot_table = (
        "\\begin{tabular}{lcc}\n"
        "\\toprule\n"
        "Aggregate metric & Point estimate & 95\\% bootstrap CI \\\\\n"
        "\\midrule\n"
        f"Top-1 accuracy & {100*correct1.mean():.2f}\\% & [{100*top1_lo:.2f}, {100*top1_hi:.2f}] \\\\\n"
        f"Top-2 accuracy & {100*correct2.mean():.2f}\\% & [{100*top2_lo:.2f}, {100*top2_hi:.2f}] \\\\\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
    )
    (TEX_ROOT / "bootstrap_ci_table.tex").write_text(boot_table)

    # ── Write acceptance-gate snippet ─────────────────────────────────────────
    gate_rows = []
    for c, (support, accepted, cor) in by_class.items():
        rate = accepted / support if support else 0.0
        ca = (cor / accepted) if accepted else float("nan")
        ca_str = f"{100*ca:.1f}\\%" if accepted else "--"
        gate_rows.append(
            f"{DISPLAY[c]} & {support} & {accepted} & {100*rate:.1f}\\% & {ca_str} \\\\"
        )
    gate_rows.append(
        f"\\textbf{{Total}} & \\textbf{{{n}}} & \\textbf{{{n_accepted}}} & "
        f"\\textbf{{{100*accept_rate:.1f}\\%}} & "
        f"\\textbf{{{100*cond_acc:.1f}\\% [{100*cond_lo:.1f}, {100*cond_hi:.1f}]}} \\\\"
    )
    gate_table = (
        "\\begin{tabular}{lrrrr}\n"
        "\\toprule\n"
        "Class & $n$ & Accepted & Acceptance rate & Acc. $\\mid$ accepted \\\\\n"
        "\\midrule\n"
        + "\n".join(gate_rows)
        + "\n\\bottomrule\n"
        "\\end{tabular}\n"
    )
    (TEX_ROOT / "acceptance_gate_table.tex").write_text(gate_table)

    # ── Write misclassification list ──────────────────────────────────────────
    if miscls:
        rows = []
        for fname, truth, pred, prob in sorted(miscls, key=lambda x: (x[1], x[0])):
            fname_tex = fname.replace("_", "\\_")
            rows.append(
                f"\\texttt{{{fname_tex}}} & {DISPLAY.get(truth, truth)} & "
                f"{DISPLAY.get(pred, pred)} & {prob:.3f} \\\\"
            )
        mis_table = (
            "\\begin{tabular}{lllr}\n"
            "\\toprule\n"
            "Image & Ground truth & Predicted top-1 & $p_1$ \\\\\n"
            "\\midrule\n"
            + "\n".join(rows)
            + "\n\\bottomrule\n"
            "\\end{tabular}\n"
        )
        (TEX_ROOT / "misclassification_table.tex").write_text(mis_table)

    print(f"Bootstrap top-1: {100*correct1.mean():.2f}% [{100*top1_lo:.2f}, {100*top1_hi:.2f}]")
    print(f"Bootstrap top-2: {100*correct2.mean():.2f}% [{100*top2_lo:.2f}, {100*top2_hi:.2f}]")
    print(f"Accept rate: {100*accept_rate:.1f}% ({n_accepted}/{n})")
    print(f"Acc | accepted: {100*cond_acc:.1f}% [{100*cond_lo:.1f}, {100*cond_hi:.1f}]")
    print(f"Misclassified: {len(miscls)}")


if __name__ == "__main__":
    main()
