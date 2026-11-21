"""Seed-averaged paired permutation tests for the classifier comparisons, and Holm's correction within each family of
comparisons in the manuscript (post hoc; lab notebook, entry (ak)).

    python3 cluster/benchmark/permutation_holm.py --out evaluation/paper_data/permutation_holm_v1   # writes .txt, .json

Several classifier comparisons in the manuscript pit the best seed of one model against the best seed of another
(exact McNemar on that pair). Choosing best seeds inflates the evidence. Here each photograph's correctness is averaged
over all seeds of a model, the difference between two models is averaged over the 398 split v1 test photographs, and
the two models' per-photograph accuracies are exchanged at random (sign flips, 100,000 draws, seed 0) for a two-sided
P; a 95 % bootstrap interval over photographs goes with each difference. Holm's step-down correction is then applied
within each family: the classifier comparisons, the synthetic-data arms, and, from the P values the manuscript
reports, honesty, test-time compute, the report writer (Table S16) and claim-level faithfulness.
"""
import argparse
import glob
import json
import os
import re

import numpy as np

E = "evaluation"
RESNET = [r"^resnet_resnet50_split_v1_split_v1_test\.json$", r"^resnet_resnet50_v1_s\d+_split_v1_test\.json$"]
MODELS = {
    "ResNet-50 (lab recipe)": RESNET,
    "SigLIP 2 fine-tuned": [r"^ft_siglip2-so400m-512_s\d+_split_v1_test\.json$"],
    "EVA-02-L fine-tuned": [r"^ft_eva02-large-448_s\d+_split_v1_test\.json$"],
    "EVA-02-L probe": [r"^probe_eva02-large-448_split_v1_test\.json$"],
    "PE-Core-G probe": [r"^probe_pe-core-g14-448_split_v1_test\.json$"],
    "SigLIP 2 probe": [r"^probe_siglip2-so400m-512_split_v1_test\.json$"],
    "DINOv2-g probe": [r"^probe_dinov2-giant_split_v1_test\.json$"],
    "ConvNeXt V2-L probe": [r"^probe_convnextv2-large-384_split_v1_test\.json$"],
    "BioCLIP 2 probe": [r"^probe_bioclip-2_split_v1_test\.json$"],
    "Qwen3-VL-8B LoRA": [r"^finetuned_Qwen3-VL-8B-Instruct_lora(_s\d+)?_test\.json$"],
    "Qwen3-VL-32B QLoRA": [r"^finetuned_Qwen3-VL-32B-Instruct-bnb4_lora_test\.json$"],
    "E1 ensemble": [r"^ensemble_E1_eva_siglip_equal_split_v1_test\.json$"],
    "E2 ensemble": [r"^ensemble_E2_8b_eva_siglip_equal_split_v1_test\.json$"],
    "E3 ensemble": [r"^ensemble_E3_resnet_8b_equal_split_v1_test\.json$"],
    "ResNet-50 at 512 px": [r"^scores_resnet50_512_s\d+_runtime_split_v1_test\.json$"],
    "ResNet-50 at 256 px (scores)": [r"^scores_resnet50_split_v1_runtime_split_v1_test\.json$",
                                     r"^scores_resnet50_v1_s\d+_runtime_split_v1_test\.json$"],
}
SYN = {"lab GPT synthetic (601)": "labsynth", "ChatGPT set, first 165": "chatgpt_a", "ChatGPT set, all 235": "chatgpt",
       "ChatGPT set without frost 3-5": "chatgpt_nof3", "real + low-quality augmentation": "lq",
       "ChatGPT set + augmentation": "chatgpt_lq", "ChatGPT set without frost 3-5 + augmentation": "chatgpt_nof3_lq"}
for name, tag in SYN.items():
    MODELS["SigLIP 2 " + name] = [rf"^ft_siglip2-so400m-512_{tag}_s\d+_split_v1_test\.json$"]
CLASSIFIER = [("SigLIP 2 fine-tuned", "ResNet-50 (lab recipe)"), ("EVA-02-L fine-tuned", "ResNet-50 (lab recipe)"),
              ("EVA-02-L probe", "ResNet-50 (lab recipe)"), ("PE-Core-G probe", "ResNet-50 (lab recipe)"),
              ("SigLIP 2 probe", "ResNet-50 (lab recipe)"), ("DINOv2-g probe", "ResNet-50 (lab recipe)"),
              ("ConvNeXt V2-L probe", "ResNet-50 (lab recipe)"), ("BioCLIP 2 probe", "ResNet-50 (lab recipe)"),
              ("Qwen3-VL-8B LoRA", "ResNet-50 (lab recipe)"), ("Qwen3-VL-32B QLoRA", "Qwen3-VL-8B LoRA"),
              ("SigLIP 2 fine-tuned", "EVA-02-L fine-tuned"), ("E2 ensemble", "Qwen3-VL-8B LoRA"),
              ("E2 ensemble", "EVA-02-L probe"), ("E1 ensemble", "EVA-02-L probe"), ("E3 ensemble", "Qwen3-VL-8B LoRA"),
              ("ResNet-50 at 512 px", "ResNet-50 at 256 px (scores)")]
SYNTH = [("SigLIP 2 " + n, "SigLIP 2 fine-tuned") for n in SYN]
# P values as the manuscript reports them (20 November draft), families without a seed problem
REPORTED = {
    "honesty (3.6)": [("student vs 32B teacher", 1.3e-5), ("student vs Qwen3.8-27B", 0.011),
                      ("student vs Qwen3-VL-235B", 0.064), ("student vs base-8B student", 0.019)],
    "test-time compute (Tables S13-S15)": [("vote vs greedy, 8B", 0.08), ("vote vs greedy, 32B", 0.46),
                                           ("vote vs greedy, 27B", 0.23), ("vote vs greedy, 235B", 0.50),
                                           ("vote vs greedy, GLM-4.6V", 0.86), ("choose between two", 0.03),
                                           ("choose + real references", 0.69), ("choose + synthetic references", 1.0),
                                           ("debate, round 1", 0.80), ("debate, round 3", 1.0),
                                           ("evidence first, 8B", 1.4e-5), ("evidence first, 32B", 0.0003),
                                           ("evidence first, 27B", 0.32), ("evidence first, 235B", 2.7e-5)],
    "report writer (Table S16)": [("H_part, with vs without part lists", 0.064),
                                  ("condition consistency", 0.031), ("April writer H_part", 0.45)],
    "faithfulness, pre-registered (3.9)": [("Wilcoxon on per-photograph rates", 0.086),
                                           ("McNemar on reports with an unsupported claim", 0.093)],
}


def load(patterns, test):
    files = sorted(f for f in os.listdir(E) if any(re.match(p, f) for p in patterns))
    runs = []
    for f in files:
        res = {r["image"]: r["pred"] == r["truth"] for r in json.load(open(os.path.join(E, f)))["results"]}
        if all(i in res for i in test):
            runs.append([res[i] for i in test])
    return files, np.array(runs, dtype=float)


def holm(ps):
    m, order = len(ps), np.argsort(ps)
    adj, run = np.empty(m), 0.0
    for k, i in enumerate(order):
        run = max(run, min(1.0, (m - k) * ps[i]))
        adj[i] = run
    return adj


def compare(a, b, rng, n_perm=100_000, n_boot=10_000):
    d = a.mean(0) - b.mean(0)
    t = d.mean()
    hits = 0
    for _ in range(n_perm // 10_000):  # in chunks, so the sign matrix stays small
        signs = rng.integers(0, 2, size=(10_000, d.size), dtype=np.int8) * 2 - 1
        hits += int(np.sum(np.abs(signs @ d / d.size) >= abs(t) - 1e-12))
    p = (1 + hits) / (1 + n_perm)
    idx = rng.integers(0, d.size, size=(n_boot, d.size))
    boots = d[idx].mean(1)
    return {"diff_points": 100 * t, "ci95": [100 * np.percentile(boots, 2.5), 100 * np.percentile(boots, 97.5)],
            "p_perm": float(p), "photos_a_better": int((d > 0).sum()), "photos_b_better": int((d < 0).sum())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    split = json.load(open(os.path.join(E, "split_v1.json")))
    test = sorted(p for ps in split["test"].values() for p in ps)
    rng = np.random.default_rng(0)
    data, lines, out = {}, [], {"models": {}, "families": {}}
    for name, pats in MODELS.items():
        files, runs = load(pats, test)
        data[name] = runs
        acc = 100 * runs.mean(1) if len(runs) else np.array([])
        out["models"][name] = {"files": files, "seeds": int(len(runs)),
                               "accuracy_mean": float(acc.mean()) if len(acc) else None,
                               "accuracy_sd": float(acc.std(ddof=1)) if len(acc) > 1 else None}
        lines.append(f"{name:48s} {len(runs)} run(s)  " + (f"{acc.mean():.1f} +- {acc.std(ddof=1) if len(acc) > 1 else 0:.1f} %"
                                                          if len(acc) else "NO FILES"))
    lines.append("")
    for fam, pairs in (("classifier comparisons (3.3, Tables S4, S5)", CLASSIFIER),
                       ("synthetic-data arms against real photographs only (3.4, Table S8)", SYNTH)):
        rows = []
        for x, y in pairs:
            if len(data[x]) == 0 or len(data[y]) == 0:
                rows.append({"a": x, "b": y, "missing": True})
                continue
            r = compare(data[x], data[y], rng)
            r.update(a=x, b=y, seeds=[int(len(data[x])), int(len(data[y]))])
            rows.append(r)
        ok = [r for r in rows if "p_perm" in r]
        for r, q in zip(ok, holm(np.array([r["p_perm"] for r in ok]))):
            r["p_holm"] = float(q)
        out["families"][fam] = rows
        lines.append(f"== {fam}: seed-averaged paired permutation test, Holm within the family ({len(ok)} tests)")
        for r in rows:
            if r.get("missing"):
                lines.append(f"  {r['a']} vs {r['b']}: files missing")
                continue
            lines.append(f"  {r['a']} ({r['seeds'][0]}) vs {r['b']} ({r['seeds'][1]}): {r['diff_points']:+.1f} points "
                         f"[{r['ci95'][0]:+.1f}, {r['ci95'][1]:+.1f}], photos {r['photos_a_better']}/{r['photos_b_better']}, "
                         f"P = {r['p_perm']:.2g}, Holm P = {r['p_holm']:.2g}")
        lines.append("")
    for fam, items in REPORTED.items():
        adj = holm(np.array([p for _, p in items]))
        out["families"][fam] = [{"test": n, "p_reported": p, "p_holm": float(q)} for (n, p), q in zip(items, adj)]
        lines.append(f"== {fam}: P values as reported, Holm within the family ({len(items)} tests)")
        for (n, p), q in zip(items, adj):
            lines.append(f"  {n}: P = {p:.2g}, Holm P = {q:.2g}")
        lines.append("")
    txt = "\n".join(lines)
    print(txt)
    open(a.out + ".txt", "w").write(txt + "\n")
    json.dump(out, open(a.out + ".json", "w"), indent=1)


if __name__ == "__main__":
    main()
