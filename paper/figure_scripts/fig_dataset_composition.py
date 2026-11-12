"""Fig. Dataset composition: (a) images per condition, (b) plant-part instances, (c) object size."""
import json
import sys
import zipfile
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb

sys.path.insert(0, "D:/agai-paper/draft/figsrc")
from agstyle import *  # noqa

setup(7.0)
counts = json.load(open("D:/agai-paper/draft/notes/dataset_counts.json"))["per_class"]

z = zipfile.ZipFile("D:/agai-paper/labdata/LatestStrawberry archive.coco-segmentation.zip")
coco = json.loads(z.read("train/_annotations.coco.json"))
cats = {c["id"]: c["name"] for c in coco["categories"]}
imsz = {i["id"]: (float(i["width"]), float(i["height"])) for i in coco["images"]}
PARTS = ["leaf", "fruit", "stem", "flower", "soil", "root"]
inst = {p: 0 for p in PARTS}
imgs_with = {p: set() for p in PARTS}
rel = {p: [] for p in PARTS}
for a in coco["annotations"]:
    p = cats[a["category_id"]]
    if p not in inst:
        continue
    inst[p] += 1
    imgs_with[p].add(a["image_id"])
    w, h = imsz[a["image_id"]]
    rel[p].append(max(float(a["bbox"][2]) * float(a["bbox"][3]) / (w * h), 1e-6))
n_img = len(coco["images"])
n_ann = sum(inst.values())


def tint(c, k):
    r, g, b = to_rgb(c)
    return (1 - k * (1 - r), 1 - k * (1 - g), 1 - k * (1 - b))


fig = plt.figure(figsize=(W2, 64 * MM))
gs = fig.add_gridspec(1, 3, width_ratios=[1.25, 1.0, 1.0], left=0.105, right=0.985, bottom=0.19,
                      top=0.86, wspace=0.55)

# (a) images per condition: real + synthetic, stacked with a white gap
ax = fig.add_subplot(gs[0])
ys = np.arange(len(CLASS_ORDER))[::-1]
for y, c in zip(ys, CLASS_ORDER):
    real, syn = counts[c]["real_total"], counts[c]["synthetic"]
    ax.barh(y, real, height=0.56, color=CLASS_COLOR[c], lw=0)
    ax.barh(y, syn, left=real + 6, height=0.56, color=tint(CLASS_COLOR[c], 0.38), lw=0)
    ax.text(real + syn + 18, y, f"{real}" + (f" + {syn}" if syn else ""), va="center", fontsize=6.3,
            color=INK2)
ax.set_yticks(ys, [CLASS_LABEL[c] for c in CLASS_ORDER])
ax.set_xlim(0, 980)
ax.set_xlabel("Number of images")
ax.xaxis.grid(True)
ax.set_axisbelow(True)
for s in ("top", "right", "left"):
    ax.spines[s].set_visible(False)
ax.tick_params(axis="y", length=0)
tr = sum(v["real_total"] for v in counts.values())
ts = sum(v["synthetic"] for v in counts.values())
from matplotlib.patches import Patch
ax.legend(handles=[Patch(fc="#7d7c78", lw=0, label=f"Real (n = {tr:,})"),
                   Patch(fc=tint("#7d7c78", 0.38), lw=0, label=f"Synthetic (n = {ts})")],
          loc="center right", frameon=False, handlelength=1.0, borderaxespad=0.0, bbox_to_anchor=(1.0, 0.64))
ax.set_title("(a) Images per condition", loc="left", fontweight="bold", color=INK, pad=6)

# (b) plant-part instances
ax = fig.add_subplot(gs[1])
yp = np.arange(len(PARTS))[::-1]
part_color = "#52514e"
ax.barh(yp, [inst[p] for p in PARTS], height=0.56, color=part_color, lw=0)
for y, p in zip(yp, PARTS):
    ax.text(inst[p] + 250, y, f"{inst[p]:,}", va="center", fontsize=6.3, color=INK2)
    ax.text(24400, y, f"{len(imgs_with[p])}", va="center", ha="right", fontsize=6.3, color=MUTED)
ax.text(24400, len(PARTS) - 0.45, "images", va="bottom", ha="right", fontsize=6.0, color=MUTED)
ax.set_ylim(-0.6, len(PARTS) - 0.1)
ax.set_yticks(yp, [p.capitalize() for p in PARTS])
ax.set_xlim(0, 24500)
ax.set_xticks([0, 5000, 10000, 15000], ["0", "5K", "10K", "15K"])
ax.set_xlabel("Annotated instances")
ax.xaxis.grid(True)
ax.set_axisbelow(True)
for s in ("top", "right", "left"):
    ax.spines[s].set_visible(False)
ax.tick_params(axis="y", length=0)
ax.set_title("(b) Plant-part labels", loc="left", fontweight="bold", color=INK, pad=6)

# (c) relative object size per part (log scale box plot)
ax = fig.add_subplot(gs[2])
data = [np.array(rel[p]) * 100 for p in PARTS]
bp = ax.boxplot(data, vert=False, positions=yp, widths=0.5, whis=(5, 95), showfliers=False,
                patch_artist=True, medianprops=dict(color=INK, lw=1.0),
                boxprops=dict(fc=tint(part_color, 0.25), ec=part_color, lw=0.6),
                whiskerprops=dict(color=part_color, lw=0.6), capprops=dict(color=part_color, lw=0.6))
ax.set_xscale("log")
ax.set_xlim(0.01, 100)
ax.set_xticks([0.01, 0.1, 1, 10, 100], ["0.01", "0.1", "1", "10", "100"])
med_w, med_h = np.median([v[0] for v in imsz.values()]), np.median([v[1] for v in imsz.values()])
thr = 32 * 32 / (med_w * med_h) * 100
ax.axvline(thr, color="#e34948", lw=0.7)
ax.set_yticks(yp, [p.capitalize() for p in PARTS])
ax.set_xlabel("Box area (% of image, log scale);\nred line: 32 x 32 px at the median image size")
ax.xaxis.grid(True, which="major")
ax.set_axisbelow(True)
for s in ("top", "right", "left"):
    ax.spines[s].set_visible(False)
ax.tick_params(axis="y", length=0)
ax.set_title("(c) Object size", loc="left", fontweight="bold", color=INK, pad=6)

save(fig, "fig_dataset_composition")
print("ok", tr, ts, n_img, n_ann, round(thr, 3))
