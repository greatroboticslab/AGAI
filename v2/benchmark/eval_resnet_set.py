"""Evaluate a ResNet checkpoint (the lab's resnet_classifier: TTA hflip + checkpoint temperature) on an image set
from evaluation/clean_eval_sets.json, reporting overall and macro accuracy in the zero-shot script's format.
  python cluster/benchmark/eval_resnet_set.py --set resnet161 [--ckpt plant_diagnostic/models/resnet_strawberry_v4_release10holdout.pth]
  python cluster/benchmark/eval_resnet_set.py --split-json evaluation/split_v1.json --split test --ckpt /projects/$USER/agai/models/resnet50_split_v1.pth
"""
import argparse
import collections
import json
import os
import sys

sys.path.insert(0, os.getcwd())
from resnet_classifier import load_resnet, diagnose_or_none  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--set", choices=["resnet161", "all1580"], default="resnet161")
ap.add_argument("--ckpt", default="plant_diagnostic/models/resnet_strawberry_v4_release10holdout.pth")
ap.add_argument("--split-json", default=None, help="score a split from cluster/benchmark/make_split.py instead of --set")
ap.add_argument("--split", default="test")
args = ap.parse_args()

if args.split_json:
    sets = json.load(open(args.split_json))[args.split]
    args.set = f"{os.path.basename(args.split_json).replace('.json', '')}_{args.split}"
else:
    key = "resnet_v4_clean" if args.set == "resnet161" else "vlm_all_originals"
    sets = json.load(open("evaluation/clean_eval_sets.json"))[key]
model = load_resnet(args.ckpt)
results, per_class = [], collections.defaultdict(lambda: [0, 0])
for c, paths in sorted(sets.items()):
    for p in paths:
        r = diagnose_or_none(model, p)
        results.append({"image": p, "truth": c, "pred": r["label"], "p1": r["p1"]})
        per_class[c][1] += 1
        per_class[c][0] += r["label"] == c
n = len(results)
correct = sum(r["pred"] == r["truth"] for r in results)
summary = {"model": os.path.basename(args.ckpt), "set": args.set, "n": n,
           "accuracy": f"{correct}/{n} = {correct / n:.3f}",
           "per_class": {k: f"{v[0]}/{v[1]}" for k, v in sorted(per_class.items())},
           "macro_accuracy": round(sum(v[0] / v[1] for v in per_class.values()) / len(per_class), 3)}
out = f"evaluation/resnet_{os.path.basename(args.ckpt).replace('.pth', '')}_{args.set}.json"
json.dump({"summary": summary, "results": results}, open(out, "w"), indent=2)
print(json.dumps(summary, indent=2))
