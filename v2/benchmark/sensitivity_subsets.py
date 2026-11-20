"""Does the classifier comparison survive removing the questionable test images?

    python -P v2/benchmark/sensitivity_subsets.py --out evaluation/paper_data/sensitivity_subsets_v1.txt

Uses the per-image test results in evaluation/paper_data (calib_<model>_test_seed<k>.json) and drops, in turn:
(a) the images the label audit flags (evaluation/label_audit.csv, both_disagree = True: the out-of-fold ensemble and
the 32B annotator both disagree with the label); (b) images whose file name shows a web source (stock-photo sites,
Reddit-style names, web-page titles); (c) the three non-photographic or degraded test images of docs/BENCHMARK.md;
(d) all of these. Reports accuracy per model (mean and SD over seeds) and the paired gap between the models, with the
exact McNemar test pooled over seed-matched pairs as in the classifier comparison.
"""
import argparse
import csv
import glob
import json
import os
import re
from statistics import mean, stdev

from scipy.stats import binomtest

WEB = re.compile(r"stock-photo|shutterstock|alamy|istock|dreamstime|depositphotos|adobestock|123rf|getty|-v0-[a-z0-9]{6,}|"
                 r"([a-z]+-){5,}", re.I)
DEFECTS = {"plant_diagnostic/data/holdout/root_rot/rootrot_41.jpg", "plant_diagnostic/data/train/root_rot/rootrot_147.jpg",
           "plant_diagnostic/data/holdout/gray_mold/GrayMold0398.jpg"}
MODELS = {"ResNet-50 (v1 recipe)": "calib_resnet50_v1_test_seed*.json", "SigLIP 2 (recipe)": "calib_siglip2_recipe_test_seed*.json"}


def load(pattern):
    runs = []
    for p in sorted(glob.glob(os.path.join("evaluation/paper_data", pattern))):
        d = json.load(open(p, encoding="utf-8"))
        runs.append({r["image"]: r["pred"] == r["truth"] for r in d["results"]})
    return runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    runs = {m: load(p) for m, p in MODELS.items()}
    images = sorted(set(runs[next(iter(runs))][0]))
    flagged = {r["image"] for r in csv.DictReader(open("evaluation/label_audit.csv", encoding="utf-8"))
               if r["both_disagree"].strip().lower() == "true"}
    subsets = {"all test images": set(),
               "(a) without label-audit flags": flagged,
               "(b) without web-sourced names": {i for i in images if WEB.search(os.path.basename(i))},
               "(c) without the 3 defective images": DEFECTS}
    subsets["(d) without (a), (b) and (c)"] = set().union(*list(subsets.values())[1:])
    lines = [f"Test images: {len(images)}; seeds: " + ", ".join(f"{m} {len(r)}" for m, r in runs.items()), ""]
    m1, m2 = list(runs)
    for name, drop in subsets.items():
        keep = [i for i in images if i not in drop]
        accs = {m: [100 * mean(r[i] for i in keep) for r in rs] for m, rs in runs.items()}
        k = min(len(runs[m1]), len(runs[m2]))
        b = c = 0  # seed-matched pairs: m2 right and m1 wrong, and the reverse
        for s in range(k):
            for i in keep:
                x, y = runs[m1][s][i], runs[m2][s][i]
                b += (y and not x)
                c += (x and not y)
        p = binomtest(b, b + c, 0.5).pvalue if b + c else 1.0
        lines.append(f"{name}: n = {len(keep)} (dropped {len(images) - len(keep)})")
        for m, v in accs.items():
            lines.append(f"    {m:24s} {mean(v):5.1f} +- {stdev(v) if len(v) > 1 else 0:.1f} %")
        lines.append(f"    gap {m2} - {m1}: {mean(accs[m2]) - mean(accs[m1]):+.1f} points; seed-matched pairs "
                     f"(first {k} seeds) {b} vs {c}, exact McNemar P = {p:.2g}")
    txt = "\n".join(lines)
    print(txt)
    with open(a.out, "w", encoding="utf-8", newline="\n") as f:
        f.write(txt + "\n")


if __name__ == "__main__":
    main()
