"""Fig. Explanations that can reject a wrong label: the label-swap test on the split v1 test (398 photos x 2 labels).
(a) Test design. (b) Share of true labels accepted against share of seeded wrong labels rejected; dashed lines are
constant balanced honesty. (c) Share of wrong-label cases whose explanation names the true condition in its
differential.
Numbers: D:/agai-paper/github/docs/VLM_HONESTY.md (results table :58-66, McNemar :72, design :36-48) and
cluster/EXPERIMENT_LOG.md. The re-verified student run (83.2% balanced, VLM_HONESTY.md:109-112) is not plotted; the
figure uses the headline run, as the docs' tables do."""
import math
import sys

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle

sys.path.insert(0, "D:/agai-paper/draft/figsrc")
from agstyle import *  # noqa

setup(7.0)
W, H = 190.0, 92.0
fig = plt.figure(figsize=(W * MM, H * MM))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W)
ax.set_ylim(0, H)
ax.axis("off")


def axmm(x, y, w, h):
    a = fig.add_axes([x / W, y / H, w / W, h / H])
    a.set_facecolor("none")
    return a


BLUE = ROLE["classify"][1]
ZS = MUTED
S_LAB, S_VAL = 6.0, 5.6
HALO = [pe.withStroke(linewidth=2.2, foreground="white")]

# label, accepts true, rejects wrong, balanced, differential, student?
MODELS = [
    ("8B student, from the\nclassification-tuned 8B", 80.9, 85.4, 83.2, 97.7, True),  # verified node (job 72215)
    ("Qwen3-VL-235B", 85.9, 75.5, 80.7, 88.1, False),
    ("8B student, from\nthe base 8B", 79.3, 81.9, 80.6, 93.7, True),
    ("Qwen3.8-27B", 90.5, 67.8, 79.1, 75.1, False),
    ("Qwen3-VL-32B, teacher", 78.6, 76.6, 77.6, 95.2, False),
    ("Qwen3-VL-8B", 72.4, 74.1, 73.2, 89.2, False),
    ("GLM-4.6V", 50.0, 78.1, 64.1, 78.4, False),
]
for _, a_, r_, b_, _, _ in MODELS:
    assert abs((a_ + r_) / 2 - b_) <= 0.051, (a_, r_, b_)

# ------------------------------------------------------------------ (a) schematic
panel_label(ax, 1.0, H - 1.0, "(a) The label-swap test")
A = dict(color=INK2, lw=0.8, ms=6)
ytop, ybot, bh = 80.0, 67.5, 8.6
box(ax, 1.0, 64.5, 13.0, 18.5, "data", "Test photo", ["398 photos"], title_size=6.2, body_size=5.6)
box(ax, 19.0, ytop - bh / 2, 22.0, bh, "output", "True label", ["annotated once"], title_size=6.2, body_size=5.6)
box(ax, 19.0, ybot - bh / 2, 22.0, bh, "gate", "Wrong label", ["seeded, annotated once"], title_size=6.2,
    body_size=5.6)
box(ax, 46.0, ytop - bh / 2, 28.0, bh, "neutral", "Honest: accepts", ["clearly or partially supports"],
    title_size=6.2, body_size=5.6)
box(ax, 46.0, ybot - bh / 2, 28.0, bh, "neutral", "Honest: rejects", ["does not support or cannot tell"],
    title_size=6.2, body_size=5.6)
arrow(ax, 14.0, 76.5, 19.0, ytop, **A)
arrow(ax, 14.0, 71.0, 19.0, ybot, **A)
arrow(ax, 41.0, ytop, 46.0, ytop, **A)
arrow(ax, 41.0, ybot, 46.0, ybot, **A)
ax.text(46.6, ybot - bh / 2 - 2.3, "and is the true condition in the", fontsize=S_VAL, color=INK2, va="center")
ax.text(46.6, ybot - bh / 2 - 4.6, "differential? (panel c)", fontsize=S_VAL, color=INK2, va="center")
box(ax, 1.0, 45.0, 73.0, 8.4, "output", None, [], fill="#f7f7f5", edge=AXIS)
ax.text(37.5, 50.9, "Balanced honesty = (accept-true + reject-wrong) / 2", fontsize=6.2, color=INK,
        ha="center", va="center", fontweight="bold")
ax.text(37.5, 47.5, "chance = 50%; a sycophantic model accepts both labels", fontsize=S_VAL, color=INK2,
        ha="center", va="center")

# ------------------------------------------------------------------ (b) scatter
BX0, BY0, BW, BH = 93.0, 12.0, 92.0, 68.0
XMIN, XMAX, YMIN, YMAX = 46.0, 96.0, 61.0, 91.0
pb = axmm(BX0, BY0, BW, BH)
pb.set_xlim(XMIN, XMAX)
pb.set_ylim(YMIN, YMAX)
ax.text(BX0 - 13.0, H - 1.0, "(b) Accepting true labels against rejecting wrong ones (398 photos \u00d7 2 labels)",
        fontsize=7.5, fontweight="bold", ha="left", va="top", color=INK)
pb.set_xticks(range(50, 96, 10))
pb.set_yticks(range(65, 91, 5))
pb.grid(color=GRID, lw=0.5)
pb.set_axisbelow(True)
for s in ("top", "right"):
    pb.spines[s].set_visible(False)
pb.tick_params(labelsize=S_LAB)
pb.set_xlabel("Accepts the true label (%)", fontsize=6.2, labelpad=1.5)
pb.set_ylabel("Rejects the wrong label (%)", fontsize=6.2, labelpad=1.5)
# iso-lines of balanced honesty, labelled where they enter the top edge
p0 = pb.transData.transform((60, 80))
p1 = pb.transData.transform((61, 79))
ang = math.degrees(math.atan2(p1[1] - p0[1], p1[0] - p0[0]))
for b in (70, 75, 80, 85):
    xs = [max(XMIN, 2 * b - YMAX), min(XMAX, 2 * b - YMIN)]
    pb.plot(xs, [2 * b - x for x in xs], color="#a9a79f", lw=0.6, ls=(0, (3.0, 2.0)), zorder=1)
    xt = 2 * b - YMAX + 0.9
    pb.text(xt, 2 * b - xt - 0.25, f"balanced {b}%" if b == 70 else f"{b}%", fontsize=S_VAL, color=MUTED,
            rotation=ang, rotation_mode="anchor", ha="left", va="top", path_effects=HALO)

OFF = {  # label placement: dx, dy (data units), ha, va
    "8B student, from the\nclassification-tuned 8B": (1.0, 0.0, "left", "center"),
    "Qwen3-VL-235B": (0.0, -1.3, "center", "top"),
    "8B student, from\nthe base 8B": (1.0, 0.0, "left", "center"),
    "Qwen3.8-27B": (0.0, -1.3, "center", "top"),
    "Qwen3-VL-32B, teacher": (-1.0, 0.0, "right", "center"),
    "Qwen3-VL-8B": (-1.0, 0.0, "right", "center"),
    "GLM-4.6V": (1.0, 0.0, "left", "center"),
}
for name, acc, rej, bal, _, st in MODELS:
    if st:
        pb.plot(acc, rej, marker="s", ms=4.6, mfc=BLUE, mec="white", mew=0.5, ls="none", zorder=5)
    else:
        pb.plot(acc, rej, marker="o", ms=4.8, mfc=ZS, mec="white", mew=0.5, ls="none", zorder=5)
    dx, dy, ha, va = OFF[name]
    pb.text(acc + dx, rej + dy, f"{name} ({bal:.1f})", fontsize=5.8, color=INK, ha=ha, va=va,
            linespacing=1.1, path_effects=HALO, zorder=6)
# student vs teacher
t, s = MODELS[4], MODELS[0]
pb.add_patch(FancyArrowPatch((t[1], t[2]), (s[1], s[2]), arrowstyle="-|>", mutation_scale=6, color=BLUE, lw=0.8,
                             connectionstyle="arc3,rad=-0.45", shrinkA=4, shrinkB=4, zorder=4))
pb.text(74.4, 82.6, "student vs teacher:", fontsize=S_VAL, color=INK2, ha="right", va="center", path_effects=HALO)
pb.text(74.4, 81.1, "McNemar p = 0.00013", fontsize=S_VAL, color=INK2, ha="right", va="center", path_effects=HALO)
# key (lower left of the plot, left of the 70% line), drawn in the plot with a halo so the grid stays behind it
kx, ky = 47.6, 63.2
pb.plot(kx, ky + 2.7, marker="s", ms=4.6, mfc=BLUE, mec="white", mew=0.5, ls="none", zorder=5)
pb.text(kx + 1.1, ky + 2.7, "distilled 8B student", fontsize=S_VAL, color=INK2, va="center", path_effects=HALO,
        zorder=6)
pb.plot(kx, ky + 1.35, marker="o", ms=4.8, mfc=ZS, mec="white", mew=0.5, ls="none", zorder=5)
pb.text(kx + 1.1, ky + 1.35, "zero-shot VLM", fontsize=S_VAL, color=INK2, va="center", path_effects=HALO,
        zorder=6)
pb.text(kx - 0.65, ky, "label: model (balanced honesty, %)", fontsize=S_VAL, color=INK2, va="center",
        path_effects=HALO, zorder=6)
ax.text(BX0, 2.0, "Qwen3-VL-235B scored on 396 photos and the student from the base 8B on 397.", fontsize=S_VAL,
        color=MUTED, va="center")

# ------------------------------------------------------------------ (c) differential bars
panel_label(ax, 1.0, 41.5, "(c) True condition named in the differential")
ax.text(1.0, 37.0, "share of the wrong-label cases (%)", fontsize=S_VAL, color=INK2, va="center")
CX0, CY0, CW, CH = 33.0, 7.5, 37.0, 26.0
pc = axmm(CX0, CY0, CW, CH)
pc.set_xlim(0, 100)
pc.set_ylim(0, CH)
pc.set_xticks([0, 25, 50, 75, 100])
pc.grid(axis="x", color=GRID, lw=0.5)
pc.set_axisbelow(True)
pc.set_yticks([])
for s_ in ("top", "right", "left"):
    pc.spines[s_].set_visible(False)
pc.tick_params(axis="x", labelsize=S_LAB)
CNAME = {"8B student, from the\nclassification-tuned 8B": "8B student (from tuned 8B)",
         "8B student, from\nthe base 8B": "8B student (from base 8B)"}
ORDERED = sorted(MODELS, key=lambda m: -m[4])
heights = [3.6] * len(ORDERED)
scale = CH / sum(heights)
y = CH
for (name, _, _, _, diff, st), h in zip(ORDERED, heights):
    name = CNAME.get(name, name)
    h *= scale
    yc = y - h / 2
    y -= h
    pc.add_patch(Rectangle((0, yc - 1.15), diff, 2.3, fc=BLUE if st else ZS, ec="none", zorder=2))
    pc.text(diff - 1.5, yc, f"{diff:.1f}", fontsize=S_VAL, color="white", ha="right", va="center",
            fontweight="bold", zorder=3)
    ax.text(1.0, CY0 + yc, name.replace("from\nthe", "from the"), fontsize=S_VAL, color=INK, va="center",
            linespacing=1.1)

save(fig, "fig_honesty")
print("ok")
