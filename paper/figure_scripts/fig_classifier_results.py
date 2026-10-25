"""Fig. Condition-classifier results from an evaluation JSON (schema of eval_resnet_straw5.py output:
summary, per_class, class_order, confusion_matrix, reliability, predictions[{truth, pred_top1, p1, top2}]).

usage: python fig_classifier_results.py <eval.json> <out_stem> [--outdir DIR]
The paper version must be built from the clean (leak-free) test evaluation only.
"""
import argparse
import json
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, to_rgb
from matplotlib.lines import Line2D

sys.path.insert(0, "D:/agai-paper/draft/figsrc")
from agstyle import *  # noqa

ap = argparse.ArgumentParser()
ap.add_argument("json")
ap.add_argument("stem")
ap.add_argument("--outdir", default="D:/agai-paper/draft/figures")
ap.add_argument("--note", default="")
args = ap.parse_args()

setup(7.0)
d = json.load(open(args.json, encoding="utf-8"))
src = d["class_order"]
idx = [src.index(c) for c in CLASS_ORDER]
cm = np.array(d["confusion_matrix"], dtype=float)[np.ix_(idx, idx)]
preds = d["predictions"]
truth = np.array([CLASS_ORDER.index(p["truth"]) for p in preds])
pred = np.array([CLASS_ORDER.index(p["pred_top1"]) for p in preds])
conf = np.array([float(p["p1"]) for p in preds])
correct = (truth == pred).astype(float)
K = len(CLASS_ORDER)

BLUE = ["#f4f8fd", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
cmap = LinearSegmentedColormap.from_list("seqblue", BLUE)

fig = plt.figure(figsize=(W2, 128 * MM))
gs = fig.add_gridspec(2, 2, left=0.115, right=0.985, bottom=0.075, top=0.955, wspace=0.42, hspace=0.42,
                      width_ratios=[1.05, 1.0])

# (a) confusion matrix, row-normalised
ax = fig.add_subplot(gs[0, 0])
rown = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
ax.imshow(rown, cmap=cmap, vmin=0, vmax=1)
for i in range(K):
    for j in range(K):
        if cm[i, j] == 0:
            continue
        v = rown[i, j]
        col = "white" if v > 0.55 else INK
        ax.text(j, i, f"{v * 100:.0f}" if v >= 0.005 else "<1", ha="center", va="center", fontsize=6.2,
                color=col, fontweight="bold" if i == j else "normal")
short = {"healthy": "Healthy", "drought": "Drought", "frost": "Frost", "overwatered": "Overwater",
         "root_rot": "Root rot", "gray_mold": "Gray mold", "white_mold": "White mold"}
ax.set_xticks(range(K), [short[c] for c in CLASS_ORDER], rotation=40, ha="right")
ax.set_yticks(range(K), [short[c] for c in CLASS_ORDER])
ax.set_xlabel("Predicted condition")
ax.set_ylabel("True condition")
for s in ax.spines.values():
    s.set_visible(False)
ax.tick_params(length=0)
ax.set_title("(a) Confusion matrix (row %)", loc="left", fontweight="bold", pad=5)

# (b) per-class precision, recall, F1
ax = fig.add_subplot(gs[0, 1])
tp = np.diag(cm)
prec = tp / np.maximum(cm.sum(axis=0), 1)
rec = tp / np.maximum(cm.sum(axis=1), 1)
f1 = np.where(prec + rec > 0, 2 * prec * rec / np.maximum(prec + rec, 1e-9), 0)
ys = np.arange(K)[::-1]
for y, k in zip(ys, range(K)):
    c = CLASS_ORDER[k]
    ax.barh(y, f1[k], height=0.5, color=CLASS_COLOR[c], lw=0)
    ax.plot(prec[k], y, marker="o", ms=4.2, mfc="white", mec=INK, mew=0.8, ls="none")
    ax.plot(rec[k], y, marker="D", ms=3.6, mfc=INK, mec="white", mew=0.6, ls="none")
    ax.text(1.03, y, f"{f1[k]:.2f}", va="center", fontsize=6.2, color=INK2)
ax.set_yticks(ys, [CLASS_LABEL[c] for c in CLASS_ORDER])
ax.set_xlim(0, 1.12)
ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
ax.xaxis.grid(True)
ax.set_axisbelow(True)
for s in ("top", "right", "left"):
    ax.spines[s].set_visible(False)
ax.tick_params(axis="y", length=0)
ax.set_xlabel("Score", labelpad=2)
ax.legend(handles=[Line2D([], [], color="#9a9893", lw=5, label="F1 (bar)"),
                   Line2D([], [], marker="o", ms=4.2, mfc="white", mec=INK, ls="none", label="Precision"),
                   Line2D([], [], marker="D", ms=3.6, mfc=INK, mec="white", ls="none", label="Recall")],
          loc="upper center", bbox_to_anchor=(0.45, -0.17), ncol=3, frameon=False, handlelength=1.4,
          columnspacing=1.2, borderaxespad=0.1)
ax.set_title("(b) Per-class scores", loc="left", fontweight="bold", pad=5)

# (c) reliability diagram
ax = fig.add_subplot(gs[1, 0])
edges = np.linspace(0, 1, 11)
binid = np.clip(np.digitize(conf, edges) - 1, 0, 9)
accs, confs, cnts = [], [], []
for b in range(10):
    m = binid == b
    cnts.append(m.sum())
    accs.append(correct[m].mean() if m.any() else np.nan)
    confs.append(conf[m].mean() if m.any() else np.nan)
cnts = np.array(cnts)
ece = np.nansum([cnts[b] / len(conf) * abs(accs[b] - confs[b]) for b in range(10) if cnts[b]])
centers = (edges[:-1] + edges[1:]) / 2
ax.plot([0, 1], [0, 1], color=AXIS, lw=0.8)
for b in range(10):
    if not cnts[b]:
        continue
    ax.bar(centers[b], accs[b], width=0.085, color="#2a78d6", lw=0)
    lo, hi = sorted([accs[b], confs[b]])
    ax.bar(centers[b], hi - lo, bottom=lo, width=0.085, color="#e34948", alpha=0.25, lw=0)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1.02)
ax.set_xlabel("Confidence (top-1 probability)")
ax.set_ylabel("Accuracy")
ax.yaxis.grid(True)
ax.set_axisbelow(True)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.text(0.03, 0.95, f"ECE = {ece * 100:.1f}%\nn = {len(conf)}", transform=ax.transAxes, va="top", fontsize=6.5,
        color=INK)
ax.legend(handles=[Line2D([], [], color="#2a78d6", lw=5, label="Observed accuracy"),
                   Line2D([], [], color="#e34948", alpha=0.35, lw=5, label="Calibration gap")],
          loc="upper left", bbox_to_anchor=(0.0, 0.83), frameon=False, handlelength=1.4)
ax.set_title("(c) Reliability diagram", loc="left", fontweight="bold", pad=5)

# (d) risk-coverage
ax = fig.add_subplot(gs[1, 1])
order = np.argsort(-conf)
cov = np.arange(1, len(conf) + 1) / len(conf)
risk = 1 - np.cumsum(correct[order]) / np.arange(1, len(conf) + 1)
aurc = float(np.sum((risk[1:] + risk[:-1]) / 2 * np.diff(cov)))
ax.plot(cov, risk, color="#2a78d6", lw=1.6)
ax.fill_between(cov, risk, color="#2a78d6", alpha=0.10, lw=0)
for thr in (0.5,):
    m = conf >= thr
    if m.any():
        c0, r0 = m.mean(), 1 - correct[m].mean()
        ax.plot(c0, r0, marker="o", ms=5, mfc="#e34948", mec="white", mew=1.0, ls="none", zorder=5)
        ax.annotate(f"p1 >= {thr:.2f}: coverage {c0 * 100:.0f}%, risk {r0 * 100:.1f}%", (c0, r0),
                    xytext=(-6, 14), textcoords="offset points", ha="right", fontsize=6.2, color=INK2,
                    arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.6))
ax.set_xlim(0, 1)
ax.set_ylim(0, max(0.05, risk.max() * 1.15))
ax.set_xlabel("Coverage (fraction of images accepted)")
ax.set_ylabel("Selective risk (error rate)")
ax.yaxis.grid(True)
ax.set_axisbelow(True)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.text(0.03, 0.95, f"AURC = {aurc:.3f}", transform=ax.transAxes, va="top", fontsize=6.5, color=INK)
ax.set_title("(d) Risk-coverage", loc="left", fontweight="bold", pad=5)
if args.note:
    fig.text(0.5, 0.5, args.note, ha="center", va="center", fontsize=28, color="#e34948", alpha=0.25,
             rotation=25, fontweight="bold")
save(fig, args.stem, args.outdir)
print(json.dumps(dict(acc=float(correct.mean()), macro_f1=float(f1.mean()), ece=float(ece), aurc=float(aurc))))
