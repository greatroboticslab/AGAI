"""Fig. Synthetic training data and robustness to low-detail photos (SigLIP 2 so400m fine-tunes, lab recipe, 3 seeds
per arm, real split v1 test).
(a) Clean test accuracy against accuracy on the same photos shrunk to 128 px thumbnails, per training arm; arms with
train-time low-quality augmentation (p = 0.3) on the shaded band.
(b) Per-class change in test accuracy when the lab's 601 GPT images are added (SigLIP 2, 3 seeds; ResNet-50, 5 seeds).
(c) The roots shortcut on a held-out synthetic probe (first designed arm only): healthy dug-up roots called healthy,
and accuracy on the whole 70-image probe; bars span the 3 seeds.
Numbers: D:/agai-paper/github/docs/SYNTHETIC_DATA.md (arms :89-96, per class :31-37, mechanism :39-41, roots probe
:111-115, frost batch :119-123) and cluster/EXPERIMENT_LOG.md (:445-448 probe, :475-482 and :501-503 arms)."""
import sys

import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.patches import Rectangle

sys.path.insert(0, "D:/agai-paper/draft/figsrc")
from agstyle import *  # noqa

setup(7.0)
W, H = 190.0, 106.0
fig = plt.figure(figsize=(W * MM, H * MM))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W)
ax.set_ylim(0, H)
ax.axis("off")


def axmm(x, y, w, h):
    a = fig.add_axes([x / W, y / H, w / W, h / H])
    a.set_facecolor("none")
    return a


def tint(c, f):
    r, g, b = to_rgb(c)
    return (1 - f * (1 - r), 1 - f * (1 - g), 1 - f * (1 - b))


def style_x(a, label, ticks, minor=None):
    a.set_xticks(ticks)
    if minor:
        a.set_xticks(minor, minor=True)
        a.grid(axis="x", which="minor", color=GRID, lw=0.35)
    a.grid(axis="x", which="major", color=GRID, lw=0.5)
    a.set_axisbelow(True)
    a.set_yticks([])
    for s in ("top", "right", "left"):
        a.spines[s].set_visible(False)
    a.tick_params(axis="x", labelsize=S_LAB, which="both")
    a.tick_params(axis="x", which="minor", length=1.5)
    a.set_xlabel(label, fontsize=6.2, labelpad=1.5)


def minus(v, signed=True):
    s = f"{abs(v):.1f}"
    if v < 0:
        return "\u2212" + s
    return ("+" + s) if (signed and v > 0) else s


BLUE = ROLE["classify"][1]
DARK = INK2
LIGHT = "#b9b7ae"
SHADE = "#f3f2ee"
GUIDE = "#efeee9"
S_LAB, S_VAL, S_HEAD = 6.0, 5.6, 5.9

# ------------------------------------------------------------------ (a) dumbbells
ARMS = [
    ("No augmentation", [
        ("real only", 95.6, 83.7),
        ("+ lab GPT set (601)", 95.7, 85.8),
        ("+ designed set, first 165", 96.0, 77.9),
        ("+ designed set, all 235", 94.2, 83.0),
        ("+ designed set without frost 3-5", 96.1, 81.9)]),
    ("With low-quality augmentation (p = 0.3)", [
        ("real only", 95.1, 90.0),
        ("+ designed set, all 235", 95.7, 91.6),
        ("+ designed set without frost 3-5\n(recommended)", 96.3, 92.1)]),
]
AX0, AY0, AW, AH = 47.0, 52.0, 55.0, 44.0
XMIN, XMAX = 74.0, 99.0
pa = axmm(AX0, AY0, AW, AH)
pa.set_xlim(XMIN, XMAX)
pa.set_ylim(0, AH)
MMPT = AW / (XMAX - XMIN)
panel_label(ax, 1.0, H - 1.0, "(a) Clean and 128 px thumbnail accuracy per training arm")
style_x(pa, "Test accuracy (%), mean of 3 seeds per arm", [75, 80, 85, 90, 95], [77.5, 82.5, 87.5, 92.5, 97.5])

HEAD, ROW, ROW2, GAP = 3.4, 4.1, 5.8, 1.6
y = AH - 0.4
rows, heads = [], []
for gi, (gname, items) in enumerate(ARMS):
    if gi:
        y -= GAP
    g_top = y
    heads.append((y - HEAD / 2, gname))
    y -= HEAD
    for it in items:
        h = ROW2 if "\n" in it[0] else ROW
        rows.append((y - h / 2, gi, it))
        y -= h
    if gi == 1:
        ax.add_patch(Rectangle((1.0, AY0 + y - 0.2), AX0 + AW + 1.0, g_top - y + 0.4, fc=SHADE, ec="none",
                               zorder=0))
assert y > 0.0, y
for yh, name in heads:
    ax.text(2.0, AY0 + yh, name, fontsize=S_HEAD, color=MUTED, fontweight="bold", va="center")
# direct series labels in the first header row
yk = heads[0][0]
pa.plot(80.6, yk, marker="o", ms=4.0, mfc=BLUE, mec="white", mew=0.5, ls="none")
pa.text(80.6 + 1.0 / MMPT + 0.3, yk, "128 px thumbnails", fontsize=S_VAL, color=INK2, va="center")
pa.plot(92.3, yk, marker="o", ms=4.0, mfc=DARK, mec="white", mew=0.5, ls="none")
pa.text(92.3 + 1.0 / MMPT + 0.3, yk, "clean photos", fontsize=S_VAL, color=INK2, va="center")
for yr, gi, (label, clean, thumb) in rows:
    rec = "recommended" in label
    ax.text(4.0, AY0 + yr, label, fontsize=S_LAB, color=INK, va="center", linespacing=1.1,
            fontweight="bold" if rec else "normal")
    pa.plot([XMIN, XMAX], [yr, yr], color=GUIDE, lw=0.5, zorder=0.5)
    pa.plot([thumb, clean], [yr, yr], color=AXIS, lw=1.2, zorder=2, solid_capstyle="butt")
    pa.plot(thumb, yr, marker="o", ms=4.2, mfc=BLUE, mec="white", mew=0.5, ls="none", zorder=4)
    pa.plot(clean, yr, marker="o", ms=4.2, mfc=DARK, mec="white", mew=0.5, ls="none", zorder=4)
    pa.text(thumb - 1.2 / MMPT, yr, f"{thumb:.1f}", fontsize=S_VAL, color=INK2, va="center", ha="right",
            fontweight="bold" if rec else "normal")
    pa.text(clean + 1.2 / MMPT, yr, f"{clean:.1f}", fontsize=S_VAL, color=INK2, va="center", ha="left",
            fontweight="bold" if rec else "normal")

# ------------------------------------------------------------------ (b) per-class change, lab GPT set
PC = [  # key, test n, delta SigLIP 2 (3 seeds), delta ResNet-50 (5 seeds)
    ("drought", 17, -15.7, -16.5),
    ("overwatered", 19, 8.8, 8.4),
    ("frost", 29, 2.3, 0.7),
    ("root_rot", 33, 3.0, -1.8),
    ("gray_mold", 163, 1.4, 2.1),
    ("healthy", 81, -2.9, 1.0),
    ("white_mold", 56, 0.0, -3.2),
]
BL, BX0, BW = 109.0, 135.0, 52.0
BXMIN, BXMAX = -21.0, 12.0
pb = axmm(BX0, AY0, BW, AH)
pb.set_xlim(BXMIN, BXMAX)
pb.set_ylim(0, AH)
BMMPT = BW / (BXMAX - BXMIN)
ax.text(BL, H - 1.0, "(b) Change per class from adding the lab GPT set", fontsize=7.5, fontweight="bold",
        ha="left", va="top", color=INK)
style_x(pb, "Change in class accuracy (percentage points)", [-20, -15, -10, -5, 0, 5, 10])
pb.axvline(0, color=INK2, lw=0.6, zorder=3)
pitch = AH / len(PC)
BH = 2.0
for i, (k, n, ds, dr) in enumerate(PC):
    yc = AH - pitch * (i + 0.5)
    ax.text(BL, AY0 + yc, f"{CLASS_LABEL[k]} ({n})", fontsize=S_LAB, color=INK, va="center")
    for j, (v, fc, ec) in enumerate(((ds, CLASS_COLOR[k], "none"), (dr, tint(CLASS_COLOR[k], 0.42), CLASS_COLOR[k]))):
        yb = yc + (BH / 2 + 0.15) * (1 if j == 0 else -1)
        if v != 0:
            pb.add_patch(Rectangle((min(0, v), yb - BH / 2), abs(v), BH, fc=fc, ec=ec, lw=0.5, zorder=2))
        if v < 0:
            pb.text(v - 0.8 / BMMPT, yb, minus(v), fontsize=S_VAL, color=INK2, va="center", ha="right")
        else:
            pb.text(v + 0.8 / BMMPT, yb, minus(v), fontsize=S_VAL, color=INK2, va="center", ha="left")
    if i == 0:
        pb.text(-0.8 / BMMPT, yc + BH / 2 + 0.15, "SigLIP 2, 3 seeds", fontsize=S_VAL, color=INK, va="center",
                ha="right", zorder=4)
        pb.text(-0.8 / BMMPT, yc - BH / 2 - 0.15, "ResNet-50, 5 seeds", fontsize=S_VAL, color=INK, va="center",
                ha="right", zorder=4)
yn = AH - pitch * 1.5
pb.text(-1.2 / BMMPT, yn + 1.2, "the overwatered gain", fontsize=S_VAL, color=INK2, va="center", ha="right")
pb.text(-1.2 / BMMPT, yn - 1.2, "is paid for by drought", fontsize=S_VAL, color=INK2, va="center", ha="right")

# ------------------------------------------------------------------ (c) roots shortcut
PROBE = [  # arm, healthy roots called healthy (of 6), 70-image probe accuracy (%)
    ("real only", (0, 1), (57, 66), LIGHT),
    ("+ lab GPT set (601)", (0, 1), (74, 83), LIGHT),
    ("+ designed set, first 165", (5, 6), (89, 96), DARK),
]
CY0, CH = 9.0, 16.5
panel_label(ax, 1.0, 40.5, "(c) Roots shortcut on a held-out synthetic probe")
fig.canvas.draw()
_r = fig.canvas.get_renderer()
_x1 = ax.transData.inverted().transform(ax.texts[-1].get_window_extent(_r))[1][0]
ax.text(_x1 + 1.5, 40.5, "first designed arm only; bars span the 3 seeds; a diagnostic, not a benchmark",
        fontsize=S_VAL, color=INK2, ha="left", va="top")
C1X, C1W = 47.0, 50.0
C2X, C2W = 118.0, 62.0
c1 = axmm(C1X, CY0, C1W, CH)
c2 = axmm(C2X, CY0, C2W, CH)
c1.set_xlim(-0.4, 6.4)
c2.set_xlim(50, 100)
for a in (c1, c2):
    a.set_ylim(0, CH)
style_x(c1, "Called healthy, of 6", [0, 1, 2, 3, 4, 5, 6])
style_x(c2, "Probe accuracy (%)", [50, 60, 70, 80, 90, 100])
ax.text(C1X, CY0 + CH + 6.0, "Healthy dug-up roots (6 photos)", fontsize=6.2, color=INK, fontweight="bold",
        va="center")
ax.text(C1X, CY0 + CH + 3.2, "rotted roots stay 8 of 8 with the designed set", fontsize=S_VAL, color=INK2,
        va="center")
ax.text(C2X, CY0 + CH + 6.0, "Whole probe (70 photos)", fontsize=6.2, color=INK, fontweight="bold", va="center")
cp = CH / len(PROBE)
for i, (arm, r6, r70, col) in enumerate(PROBE):
    yc = CH - cp * (i + 0.5)
    ax.text(4.0, CY0 + yc, arm, fontsize=S_LAB, color=INK, va="center")
    for a, (lo, hi), mmpt, unit, xmax in ((c1, r6, C1W / 6.8, "", 6.4), (c2, r70, C2W / 50.0, "%", 100)):
        a.plot([a.get_xlim()[0], a.get_xlim()[1]], [yc, yc], color=GUIDE, lw=0.5, zorder=0.5)
        a.plot([lo, hi], [yc, yc], color=col, lw=3.6, solid_capstyle="butt", zorder=3)
        txt = f"{lo} to {hi}{unit}"
        if hi + 1.0 / mmpt + len(txt) * 1.05 / mmpt < xmax:
            a.text(hi + 1.0 / mmpt, yc, txt, fontsize=S_VAL, color=INK2, va="center", ha="left")
        else:
            a.text(lo - 1.0 / mmpt, yc, txt, fontsize=S_VAL, color=INK2, va="center", ha="right")
save(fig, "fig_synthetic_robustness")
print("ok")
