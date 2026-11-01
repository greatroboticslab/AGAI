"""Calibration, acceptance gate and per-class metrics of a classifier on split v1, from its saved per-image
probability files (cluster/classifier/resnet_scores.py, train_backbone.py: zero-shot format plus "probs", with the
fitted temperature in "summary"). The stored probabilities are temperature-scaled; the unscaled ones follow exactly
from p_raw ∝ p_cal^T, so both columns of the paper's calibration table come from the same files.

Per seed (one val/test pair) and pooled over seeds (mean ± sd), with and without temperature scaling:
  accuracy, macro accuracy (mean per-class recall), macro-F1, ECE (10 equal-width bins), AURC and the risk-coverage
  curve, the deployed v1 gate (class thresholds, p1 >= max(tau_c, 0.5)) and a gate whose thresholds are chosen on the
  validation images (per predicted class, the smallest tau >= 0.5 at which the accepted validation images of that
  class are >= 95 % correct), per-class precision / recall / F1, the confusion matrix, 95 % bootstrap intervals.
Writes evaluation/calib_<name>_<split>_seed<k>.json (per-image records) and evaluation/calib_<name>_summary.json
plus a Markdown table on stdout.
  python cluster/classifier/calibration_metrics.py --name resnet50 \
      --pairs evaluation/scores_A_runtime_split_v1_val.json:evaluation/scores_A_runtime_split_v1_test.json ...
"""
import argparse
import json
import random

import numpy as np

V1_GATE = {"healthy": 0.40, "drought": 0.65, "root_rot": 0.65, "frost": 0.70}  # resnet_classifier.py; others 0.60
V1_DEFAULT = 0.60
TARGET = 0.95
GRID = np.round(np.arange(0.50, 0.995, 0.01), 2)


def load(path):
    d = json.load(open(path))
    classes = list(d["results"][0]["probs"].keys())
    P = np.array([[r["probs"][c] for c in classes] for r in d["results"]], dtype=np.float64)
    P = np.clip(P, 1e-6, 1.0)
    P /= P.sum(1, keepdims=True)
    y = np.array([classes.index(r["truth"]) for r in d["results"]])
    T = float(d["summary"].get("temperature", 1.0))
    images = [r["image"] for r in d["results"]]
    return classes, P, y, T, images


def unscale(P, T):
    """probabilities before temperature scaling: softmax(logits) = normalize(softmax(logits/T)^T)"""
    Q = P ** T
    return Q / Q.sum(1, keepdims=True)


def basic(P, y, K):
    pred = P.argmax(1)
    cm = np.zeros((K, K), dtype=int)
    for t, p in zip(y, pred):
        cm[t, p] += 1
    recall = np.array([cm[k, k] / max(cm[k].sum(), 1) for k in range(K)])
    precision = np.array([cm[k, k] / max(cm[:, k].sum(), 1) for k in range(K)])
    f1 = np.array([2 * p * r / (p + r) if p + r > 0 else 0.0 for p, r in zip(precision, recall)])
    return {"accuracy": float((pred == y).mean()), "macro_accuracy": float(recall.mean()), "macro_f1": float(f1.mean()),
            "precision": precision.tolist(), "recall": recall.tolist(), "f1": f1.tolist(), "confusion": cm.tolist()}


def ece(P, y, bins=10):
    p1 = P.max(1)
    hit = (P.argmax(1) == y).astype(float)
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p1 > lo) & (p1 <= hi) if lo > 0 else (p1 >= lo) & (p1 <= hi)
        if m.any():
            e += m.mean() * abs(hit[m].mean() - p1[m].mean())
    return float(e)


def risk_coverage(P, y):
    """selective risk at every coverage when images are accepted in order of decreasing top-1 probability;
    AURC = mean selective risk over coverages 1/N ... 1 (Geifman & El-Yaniv)"""
    p1 = P.max(1)
    wrong = (P.argmax(1) != y).astype(float)
    order = np.argsort(-p1, kind="stable")
    risk = np.cumsum(wrong[order]) / np.arange(1, len(y) + 1)
    cov = np.arange(1, len(y) + 1) / len(y)
    return float(risk.mean()), cov, risk


def gate(P, y, classes, thresholds, default):
    p1 = P.max(1)
    pred = P.argmax(1)
    tau = np.array([max(thresholds.get(classes[k], default), 0.5) for k in pred])
    acc = p1 >= tau
    return {"coverage": float(acc.mean()), "accuracy_accepted": float((pred[acc] == y[acc]).mean()) if acc.any() else None,
            "n_accepted": int(acc.sum())}, acc


def choose_thresholds(Pv, yv, classes):
    """per predicted class, the smallest tau in the grid at which accepted validation images of that class are
    >= TARGET correct; if the target is never met, the tau with the highest such accuracy (ties: smallest)"""
    pred, p1 = Pv.argmax(1), Pv.max(1)
    chosen, note = {}, {}
    for k, c in enumerate(classes):
        m = pred == k
        best, best_acc = None, -1.0
        for t in GRID:
            a = m & (p1 >= t)
            if not a.any():
                break
            acc = float((yv[a] == k).mean())
            if acc >= TARGET:
                best, best_acc, how = float(t), acc, "target met"
                break
            if acc > best_acc:
                best, best_acc, how = float(t), acc, "target not met: best available"
        chosen[c] = best if best is not None else 0.5
        note[c] = f"{how} ({best_acc:.3f} on {int((m & (p1 >= chosen[c])).sum())} accepted of {int(m.sum())} predicted)"
    return chosen, note


def boot(fn, n, reps, seed=0):
    rng = random.Random(seed)
    vals = []
    for _ in range(reps):
        idx = [rng.randrange(n) for _ in range(n)]
        vals.append(fn(np.array(idx)))
    vals = np.array(vals, dtype=float)
    return [float(np.nanpercentile(vals, 2.5)), float(np.nanpercentile(vals, 97.5))]


def evaluate(P, y, classes, Pv, yv, reps):
    K = len(classes)
    out = basic(P, y, K)
    out["ece"] = ece(P, y)
    out["aurc"], cov, risk = risk_coverage(P, y)
    out["risk_coverage"] = {"coverage": cov[::max(1, len(cov) // 40)].round(4).tolist(),
                            "risk": risk[::max(1, len(risk) // 40)].round(4).tolist()}
    out["gate_v1"], _ = gate(P, y, classes, V1_GATE, V1_DEFAULT)
    th, note = choose_thresholds(Pv, yv, classes)
    out["gate_val"], acc_mask = gate(P, y, classes, th, 0.5)
    out["gate_val"]["thresholds"] = th
    out["gate_val"]["how_chosen"] = note
    n = len(y)
    ci = {}
    ci["accuracy"] = boot(lambda i: basic(P[i], y[i], K)["accuracy"], n, reps)
    ci["macro_accuracy"] = boot(lambda i: basic(P[i], y[i], K)["macro_accuracy"], n, reps, 1)
    ci["macro_f1"] = boot(lambda i: basic(P[i], y[i], K)["macro_f1"], n, reps, 2)
    ci["ece"] = boot(lambda i: ece(P[i], y[i]), n, reps, 3)
    ci["aurc"] = boot(lambda i: risk_coverage(P[i], y[i])[0], n, reps, 4)
    ci["gate_val_coverage"] = boot(lambda i: gate(P[i], y[i], classes, th, 0.5)[0]["coverage"], n, reps, 5)
    ci["gate_val_accuracy"] = boot(lambda i: (gate(P[i], y[i], classes, th, 0.5)[0]["accuracy_accepted"] or np.nan), n, reps, 6)
    out["ci95"] = ci
    return out, acc_mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--pairs", nargs="+", required=True, help="VAL_JSON:TEST_JSON per seed")
    ap.add_argument("--reps", type=int, default=10000)
    args = ap.parse_args()
    summary = {"name": args.name, "seeds": []}
    for k, pair in enumerate(args.pairs):
        vpath, tpath = pair.split(":")
        classes, Pv, yv, T, _ = load(vpath)
        classes_t, Pt, yt, Tt, images = load(tpath)
        assert classes == classes_t and abs(T - Tt) < 1e-6, (vpath, tpath)
        seed = {"val": vpath, "test": tpath, "temperature": T, "n_val": len(yv), "n_test": len(yt)}
        records = None
        for label, (Ptest, Pval) in {"with_temperature": (Pt, Pv), "without_temperature": (unscale(Pt, T), unscale(Pv, T))}.items():
            seed[label], acc_mask = evaluate(Ptest, yt, classes, Pval, yv, args.reps)
            if label == "with_temperature":
                _, acc_v1 = gate(Ptest, yt, classes, V1_GATE, V1_DEFAULT)
                raw = unscale(Pt, T)
                records = [{"image": im, "truth": classes[t], "pred": classes[int(p.argmax())],
                            "p_cal": [round(float(v), 5) for v in p], "p_raw": [round(float(v), 5) for v in r],
                            "p1": round(float(p.max()), 5), "accepted_v1_gate": bool(a1), "accepted_val_gate": bool(av)}
                           for im, t, p, r, a1, av in zip(images, yt, Pt, raw, acc_v1, acc_mask)]
        json.dump({"classes": classes, "temperature": T, "source": tpath, "results": records},
                  open(f"evaluation/calib_{args.name}_test_seed{k}.json", "w"), indent=1)
        summary["seeds"].append(seed)
        print(f"seed {k}: T={T:.3f} acc {seed['with_temperature']['accuracy']*100:.1f} "
              f"ECE {seed['without_temperature']['ece']*100:.2f} -> {seed['with_temperature']['ece']*100:.2f} "
              f"AURC {seed['without_temperature']['aurc']:.4f} -> {seed['with_temperature']['aurc']:.4f} "
              f"gate_v1 cov {seed['with_temperature']['gate_v1']['coverage']*100:.1f} acc {seed['with_temperature']['gate_v1']['accuracy_accepted']*100:.1f} "
              f"gate_val cov {seed['with_temperature']['gate_val']['coverage']*100:.1f} acc {seed['with_temperature']['gate_val']['accuracy_accepted']*100:.1f}")
    # pooled table: mean ± sd over seeds
    rows = []
    for metric, scale, fmt in (("accuracy", 100, "{:.1f}"), ("macro_accuracy", 100, "{:.1f}"), ("macro_f1", 1, "{:.3f}"),
                               ("ece", 100, "{:.2f}"), ("aurc", 1, "{:.4f}")):
        vals = {lab: np.array([s[lab][metric] * scale for s in summary["seeds"]]) for lab in ("without_temperature", "with_temperature")}
        rows.append((metric, *[f"{fmt.format(v.mean())} ± {fmt.format(v.std(ddof=1)) if len(v) > 1 else '-'}" for v in vals.values()]))
    for g in ("gate_v1", "gate_val"):
        for field in ("coverage", "accuracy_accepted"):
            vals = np.array([s["with_temperature"][g][field] * 100 for s in summary["seeds"]])
            rows.append((f"{g} {field}", "", f"{vals.mean():.1f} ± {vals.std(ddof=1) if len(vals) > 1 else 0:.1f}"))
    summary["pooled"] = {r[0]: {"without": r[1], "with": r[2]} for r in rows}
    json.dump(summary, open(f"evaluation/calib_{args.name}_summary.json", "w"), indent=1)
    print(f"\n| metric | without temperature | with temperature |\n|---|---|---|")
    for r in rows:
        print(f"| {r[0]} | {r[1]} | {r[2]} |")


if __name__ == "__main__":
    main()
