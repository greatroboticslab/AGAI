"""Fig. Plant-part annotation workflow in Roboflow: (a) model-assisted labeling loop, (b) images per batch,
(c) quality of the label-assist models. Numbers read from the Roboflow project (mtsu-2h73y/strawberry-archive)."""
import sys
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

sys.path.insert(0, "D:/agai-paper/draft/figsrc")
from agstyle import *  # noqa

setup(7.0)
W, H = 190.0, 80.0
fig = plt.figure(figsize=(W * MM, H * MM))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W)
ax.set_ylim(0, H)
ax.axis("off")

# ---------------- (a) loop diagram
panel_label(ax, 1.0, H - 1.0, "(a) Model-assisted labeling loop")
A = dict(color=INK2, lw=0.8, ms=6)
box(ax, 2.0, 50.0, 34.0, 20.0, "data", "1  Hand annotation", ["about 100 images,", "polygons for leaf, fruit,", "stem, flower, root, soil"],
    title_size=6.6, body_size=5.6)
box(ax, 44.0, 50.0, 40.0, 20.0, "detect", "2  Train in Roboflow", ["instance segmentation:", "RF-DETR-Seg Small,", "YOLOv11-seg, YOLO26-seg"],
    title_size=6.6, body_size=5.6)
box(ax, 44.0, 22.0, 40.0, 20.0, "classify", "3  Pre-label a batch", ["the trained model draws", "masks on new images", "(auto-label job)"],
    title_size=6.6, body_size=5.6)
box(ax, 2.0, 22.0, 34.0, 20.0, "gate", "4  Annotator review", ["fix or redraw each mask;", "accept the image or", "send it back"],
    title_size=6.6, body_size=5.6)
arrow(ax, 36.0, 60.0, 44.0, 60.0, **A)
arrow(ax, 64.0, 50.0, 64.0, 42.0, **A)
arrow(ax, 44.0, 32.0, 36.0, 32.0, **A)
arrow(ax, 19.0, 42.0, 19.0, 50.0, color="#e34948", lw=0.8, ms=6)
ax.text(20.5, 46.0, "accepted images join the training set;", fontsize=5.4, color="#b23a3a", va="center")
ax.text(20.5, 43.6, "the model is retrained", fontsize=5.4, color="#b23a3a", va="center")
box(ax, 2.0, 3.0, 82.0, 13.0, "output", "5  Final export (April 2026)", ["1075 images, 28,069 instance masks, six part classes;",
    "1046 images came through reviewed pre-labeling, 29 were relabeled by hand"], title_size=6.6, body_size=5.6)
arrow(ax, 19.0, 22.0, 19.0, 16.0, **A)
ax.text(20.5, 19.0, "done", fontsize=5.4, color=INK2, va="center")

# ---------------- (b) images per batch
batches = [("healthy folder", 305, "auto"), ("black soil folder", 211, "auto"), ("needs-annotation folder", 143, "auto"),
           ("April 8 upload", 114, "auto"), ("root rot folder", 103, "auto"), ("frost folder", 86, "auto"),
           ("January 20 job", 47, "auto"), ("added through the API", 31, "auto"), ("relabeled by hand (Apr 7)", 29, "hand"),
           ("smaller January batches", 6, "auto")]
axb = fig.add_axes([0.62, 0.53, 0.355, 0.36])
ys = list(range(len(batches)))[::-1]
for y, (name, n, kind) in zip(ys, batches):
    col = "#2a78d6" if kind == "auto" else "#898781"
    axb.barh(y, n, height=0.62, color=col, lw=0)
    axb.text(n + 5, y, str(n), va="center", fontsize=5.6, color=INK2)
axb.set_yticks(ys, [b[0] for b in batches], fontsize=5.6)
axb.set_xlim(0, 360)
axb.set_xticks([0, 100, 200, 300])
axb.tick_params(axis="x", labelsize=5.6)
axb.xaxis.grid(True)
axb.set_axisbelow(True)
for s in ("top", "right", "left"):
    axb.spines[s].set_visible(False)
axb.tick_params(axis="y", length=0)
axb.set_xlabel("Images", fontsize=6.0, labelpad=1)
ax.text(104.0, H - 1.0, "(b) Images per annotation batch", fontsize=7.5, fontweight="bold", va="top")
from matplotlib.patches import Patch
axb.legend(handles=[Patch(fc="#2a78d6", lw=0, label="pre-labeled, reviewed"), Patch(fc="#898781", lw=0, label="labeled by hand")],
           loc="lower right", frameon=False, fontsize=5.6, handlelength=1.0, borderaxespad=0.2)

# ---------------- (c) label-assist model quality
axc = fig.add_axes([0.62, 0.10, 0.355, 0.27])
models = [("RF-DETR-Seg S", -0.09, 19.2), ("YOLOv11-seg", -0.03, 19.5), ("YOLO26-seg", 0.03, 21.2), ("RF-DETR-Seg S", 0.09, 22.6),
          ("RF-DETR-Seg S", 1.0, 64.0)]
for name, x, m in models:
    axc.plot(x, m, marker="o", ms=5, mfc="#1baf7a" if "RF" in name else "#eda100", mec="white", mew=0.8, ls="none")
axc.annotate("RF-DETR-Seg S, 64.0%", (1.0, 64.0), xytext=(-8, -2), textcoords="offset points", ha="right",
             fontsize=5.6, color=INK2, va="center")
axc.text(0.16, 21.0, "four models, 19.2 to 22.6%", fontsize=5.6, color=INK2, va="center")
from matplotlib.lines import Line2D
axc.legend(handles=[Line2D([], [], marker="o", ls="none", ms=5, mfc="#1baf7a", mec="white", label="RF-DETR-Seg"),
                    Line2D([], [], marker="o", ls="none", ms=5, mfc="#eda100", mec="white", label="YOLO segmentation")],
           loc="upper left", frameon=False, fontsize=5.6, handletextpad=0.3, borderaxespad=0.2)
axc.set_xlim(-0.25, 1.25)
axc.set_ylim(0, 75)
axc.set_xticks([0, 1], ["Jan 20, 2026", "Apr 7, 2026"], fontsize=5.8)
axc.set_yticks([0, 20, 40, 60], ["0", "20", "40", "60"], fontsize=5.6)
axc.set_ylabel("Roboflow mAP (%)", fontsize=6.0)
axc.yaxis.grid(True)
axc.set_axisbelow(True)
for s in ("top", "right"):
    axc.spines[s].set_visible(False)
ax.text(104.0, 34.5, "(c) Label-assist models trained in Roboflow", fontsize=7.5, fontweight="bold", va="top")

save(fig, "fig_annotation_workflow")
print("ok")
