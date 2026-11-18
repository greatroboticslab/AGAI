#!/usr/bin/env python3
"""McNemar paired test comparing v3 and v4 ResNet checkpoints on the
shared 70-image holdout. Both checkpoints were evaluated on the same
holdout root after the release-10-to-train operation.

Outputs paper/generated/mcnemar_table.tex.
"""

from __future__ import annotations
import json
import math
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
HOLDOUT_V3 = REPO_ROOT / "evaluation" / "holdout" / "resnet_v3_release10holdout_full_holdout.json"
HOLDOUT_V4 = REPO_ROOT / "evaluation" / "holdout" / "resnet_v4_release10holdout_full_holdout.json"
TEX_ROOT = REPO_ROOT / "paper" / "generated"


def chi2_sf_1df(x: float) -> float:
    """Survival function of chi-squared with 1 df. For 1 df,
    P(X >= x) = erfc(sqrt(x/2)).
    """
    if x <= 0:
        return 1.0
    return math.erfc(math.sqrt(x / 2.0))


def main() -> None:
    v3 = json.loads(HOLDOUT_V3.read_text())["predictions"]
    v4 = json.loads(HOLDOUT_V4.read_text())["predictions"]
    assert len(v3) == len(v4), f"holdout size mismatch: {len(v3)} vs {len(v4)}"

    by_path_v3 = {p["path"]: p for p in v3}
    n11 = n10 = n01 = n00 = 0  # (v3 correct, v4 correct), etc.
    for p4 in v4:
        p3 = by_path_v3.get(p4["path"])
        assert p3 is not None and p3["truth"] == p4["truth"], "path/truth mismatch"
        c3 = int(p3["pred_top1"] == p3["truth"])
        c4 = int(p4["pred_top1"] == p4["truth"])
        if c3 == 1 and c4 == 1: n11 += 1
        elif c3 == 1 and c4 == 0: n10 += 1
        elif c3 == 0 and c4 == 1: n01 += 1
        else: n00 += 1

    # Discordant cells
    b = n10  # v3 correct, v4 wrong
    c = n01  # v3 wrong, v4 correct
    n = n11 + n10 + n01 + n00

    # Continuity-corrected McNemar (recommended for small samples)
    if b + c == 0:
        chi2 = 0.0
        p_value = 1.0
    else:
        chi2 = ((abs(b - c) - 1) ** 2) / (b + c)
        p_value = chi2_sf_1df(chi2)

    # Exact McNemar binomial p-value (two-sided)
    # P(X >= max(b,c) | X ~ Binom(b+c, 0.5)) doubled
    if b + c == 0:
        exact_p = 1.0
    else:
        k = max(b, c)
        m = b + c
        # tail probability
        tail = sum(math.comb(m, i) for i in range(k, m + 1)) / (2 ** m)
        exact_p = min(1.0, 2 * tail)

    table = (
        "\\begin{tabular}{lr}\n"
        "\\toprule\n"
        "Quantity & Value \\\\\n"
        "\\midrule\n"
        f"Holdout size $n$ & {n} \\\\\n"
        f"Both correct ($n_{{11}}$) & {n11} \\\\\n"
        f"Only v3 correct ($n_{{10}} \\equiv b$) & {b} \\\\\n"
        f"Only v4 correct ($n_{{01}} \\equiv c$) & {c} \\\\\n"
        f"Both incorrect ($n_{{00}}$) & {n00} \\\\\n"
        f"v3 top-1 accuracy & {100*(n11+n10)/n:.2f}\\% \\\\\n"
        f"v4 top-1 accuracy & {100*(n11+n01)/n:.2f}\\% \\\\\n"
        f"McNemar $\\chi^2$ (continuity-corrected, 1 df) & {chi2:.3f} \\\\\n"
        f"$\\chi^2$ asymptotic two-sided $p$ & {p_value:.3f} \\\\\n"
        f"Exact binomial two-sided $p$ & {exact_p:.3f} \\\\\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
    )
    (TEX_ROOT / "mcnemar_table.tex").write_text(table)
    print(table)


if __name__ == "__main__":
    main()
