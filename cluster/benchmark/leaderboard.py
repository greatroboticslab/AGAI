"""One leaderboard for split v1 test (398 real images): every result file found in evaluation/, sorted by accuracy,
with macro accuracy, per-class counts, an exact McNemar test against the reference (the lab's ResNet-50 recipe,
seed 42), and mean +- sd for any method run with several seeds. Run from the repo root:
  python3 cluster/benchmark/leaderboard.py [--md evaluation/leaderboard_split_v1.md]
"""
import argparse
import glob
import json
import math
import os
import re
import statistics

CLASSES = ["drought", "frost", "gray_mold", "healthy", "overwatered", "root_rot", "white_mold"]
E = "evaluation"
ap = argparse.ArgumentParser()
ap.add_argument("--md", default="")
args = ap.parse_args()
test = {p for ps in json.load(open(f"{E}/split_v1.json"))["test"].values() for p in ps}

# (glob, label function, group for seed statistics)
SOURCES = [
    ("resnet_resnet50_split_v1_split_v1_test.json", lambda m: "ResNet-50, lab recipe (seed 42)", "ResNet-50 lab recipe"),
    ("resnet_resnet50_v1_s*_split_v1_test.json", lambda m: f"ResNet-50, lab recipe (seed {m.group(1)})", "ResNet-50 lab recipe"),
    ("resnet_resnet50_v1gptsynth_s*_split_v1_test.json", lambda m: f"ResNet-50 + lab GPT synthetic (seed {m.group(1)})",
     "ResNet-50 + GPT synthetic"),
    ("scores_resnet50_*_squash_split_v1_test.json", lambda m: f"ResNet-50 {m.group(1)}, squash preprocessing",
     "ResNet-50 squash preprocessing"),
    ("finetuned_Qwen3-VL-8B-Instruct_lora*_test.json", lambda m: f"Qwen3-VL-8B LoRA{(' ' + m.group(1).strip('_')) if m.group(1) else ''}",
     "Qwen3-VL-8B LoRA"),
    ("finetuned_Qwen3-VL-32B-Instruct-bnb4_lora*_test.json", lambda m: "Qwen3-VL-32B QLoRA", "Qwen3-VL-32B QLoRA"),
    ("zeroshot_Qwen3-VL-8B-Instruct_none_all1580.json", lambda m: "Qwen3-VL-8B zero-shot", None),
    ("zeroshot_Qwen3-VL-32B-Instruct_4bit_all1580.json", lambda m: "Qwen3-VL-32B zero-shot (4-bit)", None),
    ("zeroshot_Qwen3.8-27B_4bit_all1580.json", lambda m: "Qwen3.8-27B zero-shot (4-bit)", None),
    ("zeroshot_*_api_split_v1_test.json", lambda m: f"{m.group(1)} zero-shot (API)", None),
    ("probe_*_split_v1_test.json", lambda m: f"{m.group(1)} frozen + linear probe", None),
    ("ft_*_split_v1_test.json", lambda m: f"{m.group(1)} fine-tuned (lab recipe)",
     lambda m: re.sub(r"_s\d+$", "", m.group(1)) + " fine-tuned"),
    ("ensemble_*_split_v1_test.json", lambda m: f"ensemble {m.group(1)}", None),
]


def pattern_regex(g):
    return re.compile("^" + re.escape(g).replace(r"\*", "(.*?)") + "$")


rows, seen = [], set()
for g, label, group in SOURCES:
    rx = pattern_regex(g)
    for f in sorted(glob.glob(os.path.join(E, g))):
        b = os.path.basename(f)
        if b in seen:
            continue
        m = rx.match(b)
        if not m:
            continue
        if g.startswith("resnet_resnet50_v1_s") and "gptsynth" in b:
            continue
        seen.add(b)
        res = [r for r in json.load(open(f))["results"] if r["image"] in test]
        if len(res) != len(test):
            continue
        ok = {r["image"]: r["pred"] == r["truth"] for r in res}
        truth = {r["image"]: r["truth"] for r in res}
        per = {c: (sum(ok[i] for i in ok if truth[i] == c), sum(truth[i] == c for i in ok)) for c in CLASSES}
        rows.append({"label": label(m), "file": b, "group": group(m) if callable(group) else group, "ok": ok,
                     "acc": sum(ok.values()) / len(ok), "macro": sum(a / n for a, n in per.values()) / len(per), "per": per})


def mcnemar(a, b):
    oa = sum(a[i] and not b[i] for i in a)
    ob = sum(b[i] and not a[i] for i in a)
    n, k = oa + ob, min(oa, ob)
    return oa, ob, (min(1.0, 2 * sum(math.comb(n, j) for j in range(k + 1)) / 2 ** n) if n else 1.0)


ref = next((r for r in rows if r["label"] == "ResNet-50, lab recipe (seed 42)"), None)
rows.sort(key=lambda r: (-r["acc"], -r["macro"]))
out = [f"## split v1 test leaderboard ({len(test)} real images)", "",
       "| # | method | accuracy | per-class avg | " + " | ".join(CLASSES) + " | vs ResNet s42 (only this / only ResNet, p) |",
       "|---|---|---|---|" + "---|" * len(CLASSES) + "---|"]
for i, r in enumerate(rows, 1):
    cmp = ""
    if ref is not None and r is not ref:
        oa, ob, p = mcnemar(r["ok"], ref["ok"])
        cmp = f"{oa} / {ob}, p = {p:.2g}"
    out.append(f"| {i} | {r['label']} | {sum(r['ok'].values())}/{len(r['ok'])} = {r['acc']:.1%} | {r['macro']:.1%} | "
               + " | ".join(f"{a}/{n}" for a, n in r["per"].values()) + f" | {cmp} |")
groups = {}
for r in rows:
    if r["group"]:
        groups.setdefault(r["group"], []).append(r)
out += ["", "### Runs with several seeds (mean +- sd over seeds)", "", "| method | runs | accuracy | per-class avg |", "|---|---|---|---|"]
for gname, rs in sorted(groups.items(), key=lambda kv: -statistics.mean(x["acc"] for x in kv[1])):
    accs, macs = [x["acc"] * 100 for x in rs], [x["macro"] * 100 for x in rs]
    sd = lambda v: statistics.stdev(v) if len(v) > 1 else 0.0  # noqa: E731
    out.append(f"| {gname} | {len(rs)} | {statistics.mean(accs):.1f} +- {sd(accs):.1f} | {statistics.mean(macs):.1f} +- {sd(macs):.1f} |")
text = "\n".join(out)
print(text)
if args.md:
    open(args.md, "w").write(text + "\n")
