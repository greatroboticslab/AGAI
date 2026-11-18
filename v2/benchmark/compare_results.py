"""One table for every diagnosis run, plus exact paired McNemar tests. Reads the per-image result JSONs in
evaluation/ (written by eval_resnet_set.py, minigpt_diagnose.py, zeroshot_qwen3vl.py). Run from the repo root.
"""
import json
import math
import os

CLASSES = ["drought", "frost", "gray_mold", "healthy", "overwatered", "root_rot", "white_mold"]
E = "evaluation"
RUNS = {  # label -> {set: file}; split_v1_test rows are filtered to the evaluation/split_v1.json test paths
    "ResNet-50 retrained on split v1 train": {"split_v1_test": "resnet_resnet50_split_v1_split_v1_test.json"},
    "Qwen3-VL-8B, fine-tuned (LoRA, split v1)": {"split_v1_test": "finetuned_Qwen3-VL-8B-Instruct_lora_test.json"},
    "Qwen3-VL-32B, fine-tuned (QLoRA, split v1)": {"split_v1_test": "finetuned_Qwen3-VL-32B-Instruct-bnb4_lora_test.json"},
    "ResNet-50 v4 (trained on lab data)": {"resnet161": "resnet_resnet_strawberry_v4_release10holdout_resnet161.json"},
    "MiniGPT-v2 lab fine-tuned (all)": {"resnet161": "minigpt_lab_finetuned_resnet161.json"},
    "MiniGPT-v2 lab fine-tuned (unseen only)": {"resnet161": "minigpt_lab_finetuned_resnet161.json"},
    "MiniGPT-v2 official stage-3 [vqa], zero-shot": {"resnet161": "minigpt_official_stage3_vqa_resnet161.json"},
    "Qwen3-VL-8B, zero-shot": {"resnet161": "zeroshot_Qwen3-VL-8B-Instruct_none_resnet161.json",
                               "all1580": "zeroshot_Qwen3-VL-8B-Instruct_none_all1580.json",
                               "split_v1_test": "zeroshot_Qwen3-VL-8B-Instruct_none_all1580.json"},
    "Qwen3-VL-32B (4-bit), zero-shot": {"resnet161": "zeroshot_Qwen3-VL-32B-Instruct_4bit_resnet161.json",
                                        "all1580": "zeroshot_Qwen3-VL-32B-Instruct_4bit_all1580.json",
                                        "split_v1_test": "zeroshot_Qwen3-VL-32B-Instruct_4bit_all1580.json"},
    "Qwen3.8-27B (4-bit), zero-shot": {"resnet161": "zeroshot_Qwen3.8-27B_4bit_resnet161.json",
                                       "all1580": "zeroshot_Qwen3.8-27B_4bit_all1580.json",
                                       "split_v1_test": "zeroshot_Qwen3.8-27B_4bit_all1580.json"},
}
overlap = set(json.load(open(f"{E}/minigpt_lab_train_overlap.json"))["resnet161"])
# --drop-leaks: also drop test images with a visual near-copy in ResNet v4's training data (cluster/benchmark/near_dupes.py)
import sys
leaks = ({l["test"] for l in json.load(open(f"{E}/near_dupes.json"))["resnet161_leaks"]}
         if "--drop-leaks" in sys.argv else set())


def load(label, set_name):
    rows = json.load(open(os.path.join(E, RUNS[label][set_name])))["results"]
    if "unseen only" in label:
        rows = [r for r in rows if r["image"] not in overlap]
    if set_name == "resnet161":
        rows = [r for r in rows if r["image"] not in leaks]
    if set_name == "split_v1_test":
        test = {p for ps in json.load(open(f"{E}/split_v1.json"))["test"].values() for p in ps}
        rows = [r for r in rows if r["image"] in test]
    return {r["image"]: r["pred"] == r["truth"] for r in rows}, {r["image"]: r["truth"] for r in rows}


def mcnemar(a, b):
    """exact two-sided McNemar on paired correctness dicts (same images)"""
    common = a.keys() & b.keys()
    only_a = sum(a[i] and not b[i] for i in common)
    only_b = sum(b[i] and not a[i] for i in common)
    n, k = only_a + only_b, min(only_a, only_b)
    p = min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n) if n else 1.0
    return only_a, only_b, p


for set_name in ("split_v1_test", "resnet161", "all1580"):
    labels = [l for l in RUNS if set_name in RUNS[l] and os.path.exists(os.path.join(E, RUNS[l][set_name]))]
    print(f"\n## {set_name}\n")
    print("| model | n | accuracy | per-class avg | " + " | ".join(CLASSES) + " |")
    print("|---|---|---|---|" + "---|" * len(CLASSES))
    data = {}
    for l in labels:
        ok, truth = load(l, set_name)
        data[l] = ok
        per = {c: [sum(ok[i] for i in ok if truth[i] == c), sum(truth[i] == c for i in ok)] for c in CLASSES}
        macro = sum(v[0] / v[1] for v in per.values() if v[1]) / sum(1 for v in per.values() if v[1])
        print(f"| {l} | {len(ok)} | {sum(ok.values())}/{len(ok)} = {sum(ok.values()) / len(ok):.1%} | {macro:.1%} | "
              + " | ".join(f"{v[0]}/{v[1]}" for v in per.values()) + " |")
    print("\npaired McNemar (images only model A got right / only model B got right, exact p):")
    ref = labels[0]
    pairs = [(ref, l) for l in labels[1:]] if set_name in ("resnet161", "split_v1_test") else \
        [(a, b) for i, a in enumerate(labels) for b in labels[i + 1:]]
    for a, b in pairs:
        oa, ob, p = mcnemar(data[a], data[b])
        print(f"  {a}  vs  {b}: {oa} / {ob}, p = {p:.2g}")
