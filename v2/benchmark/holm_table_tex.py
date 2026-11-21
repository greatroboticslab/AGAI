"""LaTeX for Supplementary Table S19 from evaluation/paper_data/permutation_holm_v1.json (no number typed by hand).

    python -P v2/benchmark/holm_table_tex.py evaluation/paper_data/permutation_holm_v1.json > table_s19.tex
"""
import json
import sys


def fmt_p(p, n_perm_floor=False):
    if n_perm_floor and p < 1e-4:
        return "$<10^{-4}$"
    if p >= 0.995:
        return "1"
    if p >= 0.1:
        return f"{p:.2f}"
    if p >= 0.001:
        return f"{p:.2g}" if p >= 0.01 else f"{p:.1g}" if round(p, 3) == p else f"{p:.2g}"
    m, e = f"{p:.1e}".split("e")
    return f"${m}\\times10^{{{int(e)}}}$"


def num(x):
    """signed number in math mode, with a true minus; -0.0 shown as 0.0"""
    v = round(x, 1)
    return "$0.0$" if v == 0 else f"${v:+.1f}$"


def tex(s):
    return s.replace("%", "\\%").replace("_", "\\_").replace(" (scores)", "").replace("3-5", "3--5")


d = json.load(open(sys.argv[1], encoding="utf-8"))
fam = d["families"]
rows = []
for name, label in (("classifier comparisons (3.3, Tables S4, S5)", "Classifiers"),
                    ("synthetic-data arms against real photographs only (3.4, Table S8)", "Synthetic data (vs real photographs only)")):
    rows.append(f"\\multicolumn{{4}}{{@{{}}l}}{{\\textit{{{label}: seed-averaged paired permutation test}}}} \\\\")
    for r in fam[name]:
        a, b = r["a"], r["b"]
        if label.startswith("Synthetic"):
            a = a.replace("SigLIP 2 ", "")
            comp = f"{tex(a)} ({r['seeds'][0]})"
        else:
            comp = f"{tex(a)} ({r['seeds'][0]}) vs {tex(b)} ({r['seeds'][1]})"
        diff = f"{num(r['diff_points'])} ({num(r['ci95'][0])} to {num(r['ci95'][1])})"
        rows.append(f"{comp} & {diff} & {fmt_p(r['p_perm'], True)} & {fmt_p(r['p_holm'])} \\\\")
    rows.append("\\midrule")
for name, label in (("honesty (3.6)", "Honesty"), ("test-time compute (Tables S13-S15)", "Test-time compute"),
                    ("report writer (Table S16)", "Report writer (Table S16)"),
                    ("faithfulness, pre-registered (3.9)", "Claim-level faithfulness, pre-registered")):
    rows.append(f"\\multicolumn{{4}}{{@{{}}l}}{{\\textit{{{label}: P values of the main text}}}} \\\\")
    for r in fam[name]:
        rows.append(f"{tex(r['test'])} & & {fmt_p(r['p_reported'])} & {fmt_p(r['p_holm'])} \\\\")
    rows.append("\\midrule")
rows[-1] = "\\bottomrule"
body = "\n".join(rows)
print(r"""\begin{table}[pos=!htbp]
\centering
\caption{Seed-averaged paired permutation tests and Holm-adjusted $P$ values (post hoc). For the classifiers and the synthetic-data arms, each photograph's correctness was averaged over all seeds of a model (number of seeds in parentheses), and the two models' per-photograph accuracies were exchanged at random (100,000 sign flips) for a two-sided $P$; differences are in percentage points, with 95\% bootstrap intervals over the 398 split v1 test photographs. Unlike the matched-seed McNemar tests of the main text, this test does not treat the repeated outcomes of one photograph as independent. For the other families the $P$ values are those of the main text. Holm's step-down correction is applied within each family. Code: v2/benchmark/permutation\_holm.py; results: evaluation/paper\_data/permutation\_holm\_v1.json.}
\label{stab:holm}
\footnotesize
\setlength{\tabcolsep}{4pt}
\begin{tabularx}{\linewidth}{@{}>{\raggedright\arraybackslash}X r r r@{}}
\toprule
Comparison & Difference (points) & $P$ & Holm $P$ \\
\midrule
""" + body + r"""
\end{tabularx}
\end{table}""")
