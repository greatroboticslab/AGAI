"""Re-run the paper's ResNet-50 v4 on the paper's 70-image holdout and compare, image by image, with the
lab's saved results (evaluation/holdout/resnet_v4_release10holdout_full_holdout.json; TTA hflip + T=0.7513).

The holdout was re-carved after the paper, so images that moved are found by file name, looking first in
--extra-dir (paper-era copies recovered from git LFS: some names now belong to different images in train/),
then data/train, data/holdout, data/train_aug.
Run from the repo root with the minigptv env:  python cluster/benchmark/repro_resnet_v4.py --extra-dir DIR
"""
import argparse
import collections
import json
import os
import sys

sys.path.insert(0, os.getcwd())
from resnet_classifier import load_resnet, diagnose_or_none  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--results", default="evaluation/holdout/resnet_v4_release10holdout_full_holdout.json")
ap.add_argument("--ckpt", default="plant_diagnostic/models/resnet_strawberry_v4_release10holdout.pth")
ap.add_argument("--extra-dir", default=None)
ap.add_argument("--out", default="evaluation/holdout/repro_resnet_v4_hamilton.json")
args = ap.parse_args()

paper = json.load(open(args.results, encoding="utf-8"))
index = collections.defaultdict(list)
roots = ["plant_diagnostic/data/train", "plant_diagnostic/data/holdout", "plant_diagnostic/data/train_aug"]
if args.extra_dir:
    roots.insert(0, args.extra_dir)
for root in roots:
    for dirpath, _, files in os.walk(root):
        for f in files:
            index[f].append(os.path.join(dirpath, f))

model = load_resnet(args.ckpt)
rows, missing = [], []
for p in paper["predictions"]:
    name = os.path.basename(p["path"])
    cands = [p["path"]] if os.path.isfile(p["path"]) else index.get(name, [])
    if not cands:
        missing.append(name)
        continue
    r = diagnose_or_none(model, cands[0])
    rows.append({
        "image": name, "found_at": cands[0], "truth": p["truth"],
        "paper_pred": p["pred_top1"], "paper_p1": p["p1"],
        "repro_pred": r["label"], "repro_p1": r["p1"], "abs_dp1": abs(r["p1"] - p["p1"]),
    })

agree = sum(r["paper_pred"] == r["repro_pred"] for r in rows)
dps = sorted(r["abs_dp1"] for r in rows)
acc_paper = sum(r["paper_pred"] == r["truth"] for r in rows)
acc_repro = sum(r["repro_pred"] == r["truth"] for r in rows)
summary = {
    "paper_samples": len(paper["predictions"]), "found": len(rows), "missing": missing,
    "top1_agreement": f"{agree}/{len(rows)}",
    "abs_dp1_max": dps[-1] if dps else None, "abs_dp1_median": dps[len(dps) // 2] if dps else None,
    "accuracy_on_found_paper": f"{acc_paper}/{len(rows)}", "accuracy_on_found_repro": f"{acc_repro}/{len(rows)}",
    "paper_top1_accuracy_all70": paper["summary"]["top1_accuracy"],
}
json.dump({"summary": summary, "rows": rows}, open(args.out, "w"), indent=2)
print(json.dumps(summary, indent=2))
for r in rows:
    if r["paper_pred"] != r["repro_pred"] or r["abs_dp1"] > 1e-3:
        print(f"DIFF {r['image']}: paper {r['paper_pred']} {r['paper_p1']:.4f} vs repro {r['repro_pred']} {r['repro_p1']:.4f} ({r['found_at']})")
