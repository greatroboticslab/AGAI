"""Fig. Evaluation-data audit: (a) leakage paths, (b) near-duplicate pairs, (c) rate per class."""
import csv
import json
import sys
import numpy as np
from PIL import Image, ImageOps
from matplotlib.patches import Rectangle

sys.path.insert(0, "D:/agai-paper/draft/figsrc")
from agstyle import *  # noqa

setup(7.0)
rows = list(csv.DictReader(open("D:/agai-paper/audit/holdout_vs_train_pairs.csv", encoding="utf-8")))
summ = json.load(open("D:/agai-paper/audit/dup_audit_summary.json"))

W, H = 190.0, 100.0
fig, ax = canvas(W * MM, H * MM)

# ---------------- (a) leakage paths (schematic)
panel_label(ax, 1.0, H - 1.0, "(a) How test images reach training")
y0 = H - 9.0
pool = box(ax, 2.0, y0 - 16.0, 30.0, 14.0, "data", "Image pool", ["web-sourced photographs,", "resized and re-saved copies"],
           title_size=6.6, body_size=5.7)
box(ax, 44.0, y0 - 7.5, 26.0, 9.0, "classify", "Training set", [], title_size=6.6)
box(ax, 44.0, y0 - 19.5, 26.0, 9.0, "gate", "Test set", [], title_size=6.6)
arrow(ax, 32.0, y0 - 6.0, 44.0, y0 - 3.0, color=INK2, lw=0.8)
arrow(ax, 32.0, y0 - 12.0, 44.0, y0 - 15.0, color=INK2, lw=0.8)
paths = [
    ("1", "Near-duplicate copies of one photograph", "fall on both sides of a random split"),
    ("2", "Test images used for checkpoint selection", "or to fit the calibration temperature"),
    ("3", "Warm start from an earlier checkpoint", "that was trained on today's test images"),
]
px = 76.0
for i, (n_, t1, t2) in enumerate(paths):
    yy = y0 - 2.5 - i * 7.2
    ax.add_patch(Rectangle((px, yy - 2.7), 5.0, 5.4, fc="#e34948", ec="none"))
    ax.text(px + 2.5, yy, n_, ha="center", va="center", fontsize=6.6, color="white", fontweight="bold")
    ax.text(px + 6.8, yy + 1.2, t1, fontsize=6.0, color=INK, va="center", fontweight="bold")
    ax.text(px + 6.8, yy - 1.4, t2, fontsize=5.8, color=INK2, va="center")
arrow(ax, 57.0, y0 - 10.5, 57.0, y0 - 7.5, color="#e34948", lw=1.0, style="<|-|>", ms=6)
ax.text(59.0, y0 - 9.0, "leak", fontsize=5.6, color="#b23a3a", va="center")
ax.text(138.0, y0 - 1.3, "Controls required for a clean evaluation", fontsize=6.4, fontweight="bold", color=INK)
ctrls = ["pHash near-duplicate groups kept within one split",
         "validation images kept apart from the test set",
         "no warm start from earlier checkpoints",
         "test set scored once per trained model"]
for i, c in enumerate(ctrls):
    yy = y0 - 5.6 - i * 3.9
    ax.text(138.6, yy, "\u2713", fontsize=6.4, color="#008300", va="center", family="DejaVu Sans")
    ax.text(141.8, yy, c, fontsize=5.7, color=INK2, va="center")

# ---------------- (b) near-duplicate pairs
yb_top = 63.0
panel_label(ax, 1.0, yb_top + 1.5, "(b) Test images with a near-duplicate in the training pool")
near = [r for r in rows if r["near_dup"] == "True"]
near.sort(key=lambda r: (r["cls"] != "gray_mold", int(r["phash_d"]) + int(r["dhash_d"])))
pick = []
seen_cls = {}
for r in near:
    if seen_cls.get(r["cls"], 0) >= (2 if r["cls"] == "gray_mold" else 1):
        continue
    seen_cls[r["cls"]] = seen_cls.get(r["cls"], 0) + 1
    pick.append(r)
    if len(pick) == 5:
        break
t = 15.5
for i, r in enumerate(pick):
    x = 2.0 + i * 2 * (t + 0.6) + i * 2.6
    for j, pth in enumerate((r["holdout"], r["best_train"])):
        im = ImageOps.exif_transpose(Image.open(pth)).convert("RGB")
        im = square_crop(im).resize((240, 240), Image.LANCZOS)
        image_tile(ax, im, x + j * (t + 0.6), yb_top - 3.5 - t, t, t, radius=0.6, border=AXIS, lw=0.4)
    ax.text(x + t / 2, yb_top - 2.3, "test", ha="center", fontsize=5.6, color=INK2)
    ax.text(x + t + 0.6 + t / 2, yb_top - 2.3, "training", ha="center", fontsize=5.6, color=INK2)
    ax.text(x + t + 0.3, yb_top - 4.2 - t - 1.2, CLASS_LABEL[r['cls']], ha="center", fontsize=5.8, color=INK,
            va="top", fontweight="bold")
    ax.text(x + t + 0.3, yb_top - 4.2 - t - 4.2, f"pHash {r['phash_d']}, dHash {r['dhash_d']} (of 64 bits)",
            ha="center", fontsize=5.3, color=INK2, va="top")

# ---------------- (c) rate per class
ax2 = fig.add_axes([0.118, 0.085, 0.37, 0.205])
pc = summ["per_class"]
ys = np.arange(len(CLASS_ORDER))[::-1]
for y, c in zip(ys, CLASS_ORDER):
    n, k = pc[c]["n"], pc[c]["near"]
    ax2.barh(y, 100 * k / n, height=0.55, color=CLASS_COLOR[c], lw=0)
    ax2.text(100 * k / n + 0.6, y, f"{k}/{n}", va="center", fontsize=5.9, color=INK2)
ax2.set_yticks(ys, [CLASS_LABEL[c] for c in CLASS_ORDER])
ax2.set_xlim(0, 20)
ax2.set_xlabel("Test images with a near-duplicate in training (%)")
ax2.xaxis.grid(True)
ax2.set_axisbelow(True)
for s_ in ("top", "right", "left"):
    ax2.spines[s_].set_visible(False)
ax2.tick_params(axis="y", length=0)
tot = summ["total"]
ax.text(1.0, 33.2, f"(c) Near-duplicate rate per condition (overall {tot['near']}/{tot['n']} = {100 * tot['near'] / tot['n']:.1f}%)",
        fontsize=7.5, fontweight="bold", va="top", color=INK)

# ---------------- (d) leak-free split v1 composition
# Totals and test per class: D:/agai-paper/github/docs/BENCHMARK.md:31 and :50-51 (notebook cluster/EXPERIMENT_LOG.md:21-23).
# Train and val per class are not printed in the docs; they follow from the split rule (BENCHMARK.md:42, per class
# test >= 25%, val >= 10%, both at their rounded-up minimum because the minimums add up to the reported 398 and 161)
# and the class totals of evaluation/label_audit.csv (1576 images). Root rot train = 84 matches EXPERIMENT_LOG.md:406.
SPLIT = {"healthy": (209, 33, 81), "drought": (41, 7, 17), "frost": (74, 12, 29), "overwatered": (46, 8, 19),
         "root_rot": (84, 13, 33), "gray_mold": (421, 65, 163), "white_mold": (142, 23, 56)}
TOTALS = (1017, 161, 398)
assert tuple(sum(v[i] for v in SPLIT.values()) for i in range(3)) == TOTALS
SHADE = (1.0, 0.58, 0.30)   # train, val, test: class colour mixed with white


def tint(c, f):
    from matplotlib.colors import to_rgb
    return tuple(1 - f * (1 - v) for v in to_rgb(c))


ax.text(104.0, 33.2, "(d) Leak-free split v1: near-duplicate groups never cross splits", fontsize=7.5,
        fontweight="bold", va="top", color=INK)
DX0, DY0, DW, DH = 122.0, 8.5, 35.0, 18.4
ax3 = fig.add_axes([DX0 / W, DY0 / H, DW / W, DH / H])
ax3.set_facecolor("none")
ys = np.arange(len(CLASS_ORDER))[::-1]
for y, c in zip(ys, CLASS_ORDER):
    left = 0
    for v, f in zip(SPLIT[c], SHADE):
        ax3.barh(y, v, left=left, height=0.64, color=tint(CLASS_COLOR[c], f), edgecolor="white", lw=0.4)
        left += v
ax3.set_yticks(ys, [CLASS_LABEL[c] for c in CLASS_ORDER])
ax3.set_ylim(-0.5, len(CLASS_ORDER) - 0.5)
ax3.set_xlim(0, 660)
ax3.set_xticks([0, 200, 400, 600])
ax3.set_xlabel("Images")
ax3.xaxis.grid(True)
ax3.set_axisbelow(True)
for s_ in ("top", "right", "left"):
    ax3.spines[s_].set_visible(False)
ax3.tick_params(axis="y", length=0)
# numbers as a small table to the right of the bars: one column per split, totals in the bottom row
COLX = (171.0, 180.0, 189.0)
yrow = lambda y: DY0 + (y + 0.5) / len(CLASS_ORDER) * DH  # noqa: E731
for y, c in zip(ys, CLASS_ORDER):
    for xc, v in zip(COLX, SPLIT[c]):
        ax.text(xc, yrow(y), str(v), fontsize=5.9, color=INK2, ha="right", va="center")
yh = DY0 + DH + 2.0
yt = DY0 - 2.5
ax.plot([COLX[0] - 8.0, COLX[2]], [DY0 - 0.9, DY0 - 0.9], color=AXIS, lw=0.5)
ax.text(COLX[0] - 9.0, yt, "total", fontsize=5.9, color=INK2, ha="right", va="center")
for xc, name, t in zip(COLX, ("train", "val", "test"), TOTALS):
    ax.text(xc, yh, name, fontsize=5.9, color=INK, ha="right", va="center", fontweight="bold")
    ax.text(xc, yt, str(t), fontsize=5.9, color=INK, ha="right", va="center", fontweight="bold")
# key to the shading, above the bars
for kx, name, f in zip((DX0, DX0 + 11.0, DX0 + 20.5), ("train", "val", "test"), SHADE):
    ax.add_patch(Rectangle((kx, yh - 0.9), 2.6, 1.8, fc=tint(INK2, f), ec="none"))
    ax.text(kx + 3.4, yh, name, fontsize=5.9, color=INK2, va="center")

save(fig, "fig_leakage_audit")
print("ok", [r["cls"] for r in pick])
