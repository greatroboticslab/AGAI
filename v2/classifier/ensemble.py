"""Ensembles of per-class probability files on split v1 (evaluation/scores_*.json, evaluation/probe_*.json).

Two rules, both decided without looking at test:
  equal:  plain average of the members' class probabilities (no tuning at all)
  tuned:  weights on a 0.1 grid over the simplex, chosen on val (macro accuracy, then accuracy), applied unchanged
          to test
Prints test accuracy / macro / per-class for every member and both ensembles, plus exact McNemar p against each
member. Usage (NAME=val_file,test_file):
  python cluster/classifier/ensemble.py resnet=evaluation/scores_resnet50_split_v1_squash_split_v1_val.json,evaluation/...test.json \
      vlm8b=evaluation/scores_Qwen3-VL-8B-Instruct_lora_split_v1_val.json,evaluation/...test.json [--out NAME]
"""
import argparse
import itertools
import json
import math

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("members", nargs="+")
ap.add_argument("--out", default="")
args = ap.parse_args()

CLASSES = ["drought", "frost", "gray_mold", "healthy", "overwatered", "root_rot", "white_mold"]


def load(path):
    rows = json.load(open(path))["results"]
    return {r["image"]: (r["truth"], np.array([r["probs"][c] for c in CLASSES], dtype=float)) for r in rows}


members = {}
for m in args.members:
    name, files = m.split("=", 1)
    val_f, test_f = files.split(",")
    members[name] = {"val": load(val_f), "test": load(test_f)}
names = list(members)
data = {}
for s in ("val", "test"):
    imgs = sorted(set.intersection(*[set(members[n][s]) for n in names]))
    assert all(len(members[n][s]) == len(imgs) for n in names), f"{s}: members cover different images"
    y = np.array([CLASSES.index(members[names[0]][s][i][0]) for i in imgs])
    P = np.stack([np.stack([members[n][s][i][1] for i in imgs]) for n in names])   # [model, image, class]
    data[s] = (imgs, y, P / P.sum(-1, keepdims=True))


def scores(pred, y):
    per = [float(np.mean(pred[y == k] == k)) for k in range(len(CLASSES))]
    return float(np.mean(pred == y)), float(np.mean(per)), [f"{int(np.sum((pred == k) & (y == k)))}/{int(np.sum(y == k))}" for k in range(len(CLASSES))]


def mcnemar(a_ok, b_ok):
    oa, ob = int(np.sum(a_ok & ~b_ok)), int(np.sum(b_ok & ~a_ok))
    n, k = oa + ob, min(oa, ob)
    p = min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n) if n else 1.0
    return oa, ob, p


grid = [w for w in itertools.product(*[np.arange(0, 1.01, 0.1)] * len(names)) if abs(sum(w) - 1) < 1e-6]
_, yv, Pv = data["val"]
best = max(grid, key=lambda w: scores(np.tensordot(w, Pv, 1).argmax(-1), yv)[:2])
imgs, yt, Pt = data["test"]
rows = {n: Pt[i].argmax(-1) for i, n in enumerate(names)}
rows["ensemble equal"] = Pt.mean(0).argmax(-1)
rows["ensemble tuned " + "/".join(f"{w:.1f}" for w in best)] = np.tensordot(best, Pt, 1).argmax(-1)
print(f"test n={len(yt)}; tuned weights chosen on val: {dict(zip(names, [round(float(w), 1) for w in best]))}")
print("| model | accuracy | per-class avg | " + " | ".join(CLASSES) + " |")
print("|---|---|---|" + "---|" * len(CLASSES))
for n, pred in rows.items():
    acc, mac, per = scores(pred, yt)
    print(f"| {n} | {int(np.sum(pred == yt))}/{len(yt)} = {acc:.1%} | {mac:.1%} | " + " | ".join(per) + " |")
print("\nexact McNemar, ensemble vs each member (only ensemble right / only member right, p):")
for e in [k for k in rows if k.startswith("ensemble")]:
    for n in names:
        oa, ob, p = mcnemar(rows[e] == yt, rows[n] == yt)
        print(f"  {e} vs {n}: {oa} / {ob}, p = {p:.2g}")
if args.out:
    for e in [k for k in rows if k.startswith("ensemble")]:
        tag = "equal" if e.endswith("equal") else "tuned"
        res = [{"image": i, "truth": CLASSES[t], "pred": CLASSES[int(q)]} for i, t, q in zip(imgs, yt, rows[e])]
        json.dump({"summary": {"model": f"{args.out}_{tag}", "members": names, "rule": e, "set": "split_v1_test"},
                   "results": res}, open(f"evaluation/ensemble_{args.out}_{tag}_split_v1_test.json", "w"), indent=1)
