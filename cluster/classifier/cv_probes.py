"""5-fold grouped, stratified cross-validation of frozen-feature linear probes over all 1,576 split v1 images
(features saved by cluster/classifier/probe_backbones.py in features/<name>.npz).

Near-duplicate groups (evaluation/near_dupes.json) never straddle folds. C is chosen by an inner 3-fold grouped CV
on each outer training part (macro accuracy), so nothing is tuned on the image being predicted. Every image is
predicted exactly once, by a model that never saw it or its near-copies: about 4x more test predictions than the
398-image split v1 test, so much tighter comparisons (paired exact McNemar on all 1,576). Equal-weight probability
ensembles of backbones are scored from the same out-of-fold predictions.
Writes evaluation/cv5_probe_<name>.json (per image: truth, pred, probs, fold).
  python cluster/classifier/cv_probes.py --backbones eva02-large-448,pe-core-g14-448,siglip2-so400m-512 --ensembles eva02-large-448+pe-core-g14-448
"""
import argparse
import itertools
import json
import math
import os

import numpy as np
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

ap = argparse.ArgumentParser()
ap.add_argument("--backbones", required=True)
ap.add_argument("--ensembles", default="", help="comma-separated, members joined by +")
ap.add_argument("--folds", type=int, default=5)
ap.add_argument("--jobs", type=int, default=8)
args = ap.parse_args()

split = json.load(open("evaluation/split_v1.json"))
CLASSES = sorted(split["train"])
paths = sorted(p for s in ("train", "val", "test") for c in CLASSES for p in split[s][c])
label = {p: CLASSES.index(c) for s in ("train", "val", "test") for c in CLASSES for p in split[s][c]}
y = np.array([label[p] for p in paths])
nd = json.load(open("evaluation/near_dupes.json"))
group_of = {p: gi for gi, g in enumerate(nd["groups"]) for p in g}
groups = np.array([group_of.get(p, 10**6 + i) for i, p in enumerate(paths)])
feat_dir = f"/projects/{os.environ['USER']}/agai/features"
outer = list(StratifiedGroupKFold(n_splits=args.folds, shuffle=True, random_state=0).split(paths, y, groups))
CS = (1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 0.1, 0.3)


def macro(pred, yy):
    return float(np.mean([np.mean(pred[yy == k] == k) for k in np.unique(yy)]))


def fit_predict(Xtr, ytr, Xte, C):
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(C=C, max_iter=5000, class_weight="balanced").fit(sc.transform(Xtr), ytr)
    return clf.predict_proba(sc.transform(Xte))


def inner_score(X, yy, gg, C):
    inner = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=1).split(X, yy, gg)
    return float(np.mean([macro(fit_predict(X[a], yy[a], X[b], C).argmax(1), yy[b]) for a, b in inner]))


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (c - h) / d, (c + h) / d


def mcnemar(a, b):
    oa, ob = int(np.sum(a & ~b)), int(np.sum(b & ~a))
    n, k = oa + ob, min(oa, ob)
    return oa, ob, (min(1.0, 2 * sum(math.comb(n, j) for j in range(k + 1)) / 2 ** n) if n else 1.0)


probs, chosen = {}, {}
for name in args.backbones.split(","):
    d = np.load(f"{feat_dir}/{name}.npz")
    row = {p: x for s in ("train", "val", "test") for p, x in zip(d[f"paths_{s}"], d[f"X_{s}"])}
    X = np.stack([row[p] for p in paths])
    P = np.zeros((len(paths), len(CLASSES)))
    chosen[name] = []
    for tr, te in outer:
        scores = Parallel(n_jobs=args.jobs)(delayed(inner_score)(X[tr], y[tr], groups[tr], C) for C in CS)
        C = CS[int(np.argmax(scores))]
        chosen[name].append(C)
        P[te] = fit_predict(X[tr], y[tr], X[te], C)
    probs[name] = P
    print(f"{name}: C per fold {chosen[name]}", flush=True)

for e in [x for x in args.ensembles.split(",") if x]:
    members = e.split("+")
    probs["ens:" + e] = np.mean([probs[m] for m in members], axis=0)

fold_of = np.zeros(len(paths), dtype=int)
for k, (_, te) in enumerate(outer):
    fold_of[te] = k
print(f"\n### 5-fold grouped CV over {len(paths)} images (out-of-fold predictions)\n")
print("| model | accuracy (95% CI) | per-class avg | " + " | ".join(CLASSES) + " |")
print("|---|---|---|" + "---|" * len(CLASSES))
ok = {}
for name, P in sorted(probs.items(), key=lambda kv: -np.mean(kv[1].argmax(1) == y)):
    pred = P.argmax(1)
    ok[name] = pred == y
    k = int(ok[name].sum())
    lo, hi = wilson(k, len(y))
    per = " | ".join(f"{int(np.sum((pred == i) & (y == i)))}/{int(np.sum(y == i))}" for i in range(len(CLASSES)))
    print(f"| {name} | {k}/{len(y)} = {k / len(y):.1%} ({lo:.1%}-{hi:.1%}) | {macro(pred, y):.1%} | {per} |")
    if not name.startswith("ens:"):
        json.dump({"summary": {"model": f"cv5_probe_{name}", "set": "split_v1_all_cv5", "n": len(y),
                               "accuracy": f"{k}/{len(y)} = {k / len(y):.3f}", "macro_accuracy": round(macro(pred, y), 3),
                               "C_per_fold": chosen[name]},
                   "results": [{"image": p, "truth": CLASSES[t], "pred": CLASSES[int(q)], "fold": int(f),
                                "probs": {c: round(float(v), 5) for c, v in zip(CLASSES, pr)}}
                               for p, t, q, f, pr in zip(paths, y, pred, fold_of, P)]},
                  open(f"evaluation/cv5_probe_{name}.json", "w"), indent=1)
print("\npaired exact McNemar (only A right / only B right, p):")
for a, b in itertools.combinations(sorted(ok, key=lambda n: -ok[n].mean()), 2):
    oa, ob, p = mcnemar(ok[a], ok[b])
    print(f"  {a} vs {b}: {oa} / {ob}, p = {p:.2g}")
