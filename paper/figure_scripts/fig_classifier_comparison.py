"""Fig. Diagnosis accuracy of the revival models.
(a) Leak-free split v1 test (398 photos): accuracy as mean +- SD over seeds (dot and bar), macro accuracy (mean of
per-class accuracies, small hollow diamond); single-seed and exploratory results hollow. Zero-shot VLMs for context.
(b) 5-fold grouped cross-validation over all 1576 images, frozen backbones with a logistic-regression head, Wilson 95%
CI where reported.
Numbers: D:/agai-paper/github/docs/CLASSIFIER.md (zero-shot table :36-38, CV table :66-76, fine-tuned table :92-99,
ensembles :119 and :123, species note :81-82, error review :162-172) and cluster/EXPERIMENT_LOG.md (:166-178 E2 error
review, :196 and :351 ceiling, :197-230 CV counts). The ceiling band follows the notebook: of E2's 11 test errors,
1 clear label error + 3 probable label problems + 4 ambiguous photos."""
import sys

import matplotlib.pyplot as plt

sys.path.insert(0, "D:/agai-paper/draft/figsrc")
from agstyle import *  # noqa

setup(7.0)
W, H = 190.0, 88.0
fig = plt.figure(figsize=(W * MM, H * MM))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W)
ax.set_ylim(0, H)
ax.axis("off")


def axmm(x, y, w, h):
    return fig.add_axes([x / W, y / H, w / W, h / H])


BLUE = ROLE["classify"][1]     # revival models
AMBER = ROLE["knowledge"][1]   # species-identification pretraining
BASE = MUTED                   # the lab's v1 classifier
ZS = "#aeaca4"                 # zero-shot context
BAND = "#f4ecd9"               # label-noise ceiling
GUIDE = "#efeee9"              # row guides
S_LAB, S_VAL, S_HEAD = 6.0, 5.6, 5.9

# ------------------------------------------------------------------ (a) split v1 test
# label, accuracy, SD (None = single run), macro, value tag, kind, colour
GROUPS = [
    ("v1 classifier", [
        ("ResNet-50, lab recipe", 90.2, 1.5, 85.5, "5 seeds", "multi", BASE),
        ("ResNet-50 + lab GPT synthetic", 90.4, 0.9, 84.2, "5 seeds", "multi", BASE)]),
    ("Fine-tuned VLMs (classification)", [
        ("Qwen3-VL-32B, QLoRA", 93.7, None, 88.0, "1 seed", "single", BLUE),
        ("Qwen3-VL-8B, LoRA", 94.2, 0.3, 90.2, "3 seeds", "multi", BLUE)]),
    ("Fine-tuned vision backbones (native resolution)", [
        ("SigLIP 2 so400m", 95.6, 1.3, 94.0, "3 seeds", "multi", BLUE),
        ("EVA-02-L", 95.7, 1.1, 93.3, "3 seeds", "multi", BLUE),
        ("SigLIP 2 + designed synthetic\n+ low-quality aug. (recommended)", 96.3, 0.8, 94.3, "3 seeds", "multi", BLUE),
        ("PE-Core-G/14", 96.7, None, 95.7, "1 seed", "single", BLUE)]),
    ("Ensembles", [
        ("E2: 8B LoRA + EVA-02 + SigLIP 2 probes", 97.2, None, 96.1, "pre-declared", "ens", BLUE),
        ("X3: six fine-tuned backbones + 8B LoRA", 97.7, None, 97.7, "exploratory", "expl", BLUE)]),
    ("Zero-shot VLMs (greedy, same 7-class prompt)", [
        ("Qwen3-VL-8B", 77.1, None, None, None, "zs", ZS),
        ("Qwen3-VL-32B (4-bit)", 72.4, None, None, None, "zs", ZS),
        ("Qwen3.8-27B (4-bit, thinking off)", 65.6, None, None, None, "zs", ZS)]),
]

AX0, AY0, AW, AH = 50.0, 12.0, 70.0, 65.4   # plot box in mm
XMIN, XMAX = 60.0, 100.0
pa = axmm(AX0, AY0, AW, AH)
pa.set_xlim(XMIN, XMAX)
pa.set_ylim(0, AH)
MMPT = AW / (XMAX - XMIN)                      # mm per accuracy point

panel_label(ax, 1.0, H - 1.0, "(a) Diagnosis accuracy on the split v1 test (398 photos)")

# key
ky = H - 7.4
kx = 3.0
ax.plot(kx, ky, marker="o", ms=4.2, mfc=BLUE, mec="white", mew=0.5, ls="none")
ax.plot([kx - 1.6, kx + 1.6], [ky, ky], color=BLUE, lw=1.0, zorder=1)
ax.text(kx + 2.6, ky, "accuracy, mean \u00b1 SD over seeds", fontsize=S_VAL, color=INK2, va="center")
kx = 47.0
ax.plot(kx, ky, marker="o", ms=4.2, mfc="white", mec=BLUE, mew=0.9, ls="none")
ax.text(kx + 2.2, ky, "single seed or exploratory", fontsize=S_VAL, color=INK2, va="center")
kx = 79.0
ax.plot(kx, ky, marker="D", ms=3.0, mfc="white", mec=INK2, mew=0.7, ls="none")
ax.text(kx + 2.0, ky, "macro accuracy (mean of per-class accuracies)", fontsize=S_VAL, color=INK2, va="center")

# rows
GAP, HEAD, ROW, ROW2 = 0.9, 3.0, 3.3, 5.2
y = AH - 0.6
rows, heads = [], []
for gi, (gname, items) in enumerate(GROUPS):
    if gi:
        y -= GAP
    heads.append((y - HEAD / 2, gname))
    y -= HEAD
    for it in items:
        h = ROW2 if "\n" in it[0] else ROW
        rows.append((y - h / 2, it))
        y -= h
assert y > 0.3, y

pa.axvspan(97.0, 98.0, color=BAND, lw=0, zorder=0)
pa.set_xticks([60, 70, 80, 90, 100])
pa.set_xticks([65, 75, 85, 95], minor=True)
pa.grid(axis="x", which="major", color=GRID, lw=0.5)
pa.grid(axis="x", which="minor", color=GRID, lw=0.35)
pa.set_axisbelow(True)
pa.set_yticks([])
for s in ("top", "right", "left"):
    pa.spines[s].set_visible(False)
pa.tick_params(axis="x", labelsize=S_LAB, which="both")
pa.tick_params(axis="x", which="minor", length=1.5)
pa.set_xlabel("Test accuracy (%)", fontsize=6.2, labelpad=1.5)

for yh, name in heads:
    ax.text(1.0, AY0 + yh, name, fontsize=S_HEAD, color=MUTED, fontweight="bold", va="center")

for yr, (label, m, sd, mac, tag, kind, col) in rows:
    pa.plot([XMIN, XMAX], [yr, yr], color=GUIDE, lw=0.5, zorder=0.5)
    ax.text(3.0, AY0 + yr, label, fontsize=S_LAB, color=INK, va="center", linespacing=1.1,
            fontweight="bold" if "recommended" in label else "normal")
    if sd:
        pa.plot([m - sd, m + sd], [yr, yr], color=col, lw=1.0, solid_capstyle="butt", zorder=3)
    hollow = kind in ("single", "expl")
    pa.plot(m, yr, marker="o", ms=4.2 if kind != "zs" else 3.8, ls="none", zorder=4,
            mfc="white" if hollow else col, mec=col if hollow else "white", mew=0.9 if hollow else 0.5)
    if mac is not None:
        pa.plot(mac, yr, marker="D", ms=3.0, mfc="white", mec=col, mew=0.7, ls="none", zorder=5)
    # value text
    if kind == "zs":
        pa.text(m + 0.55 + 1.0 / MMPT, yr, f"{m:.1f}", fontsize=S_VAL, color=INK2, va="center", ha="left")
        continue
    if kind == "multi":
        txt = f"{m:.1f} \u00b1 {sd:.1f} ({tag})"
    elif kind == "single":
        txt = f"{m:.1f} ({tag})"
    elif kind == "ens":
        txt = f"{m:.1f}, {tag}"
    else:
        txt = f"{m:.1f} (macro {mac:.1f}), {tag}"
    left = min(m - (sd or 0.0), mac if mac is not None else m) - 0.75 / MMPT
    pa.text(left - 1.0 / MMPT, yr, txt, fontsize=S_VAL, color=INK2, va="center", ha="right")

# ceiling annotations: band label in the first (empty) header row, explanation in the zero-shot rows
pa.text(96.6, heads[0][0], "label-noise ceiling, about 97 to 98%", fontsize=S_VAL, color=INK2, va="center",
        ha="right", fontweight="bold")
yz = [r[0] for r in rows[-2:]]
pa.text(96.6, yz[0], "Of E2's 11 test errors, 4 look like", fontsize=S_VAL, color=INK2, va="center", ha="right")
pa.text(96.6, yz[1], "label errors and 4 are ambiguous photos", fontsize=S_VAL, color=INK2, va="center", ha="right")

# ------------------------------------------------------------------ (b) 5-fold grouped CV
CV = [  # label, accuracy, CI low, CI high, kind
    ("Ensemble: EVA-02 +\nPE-Core-G + SigLIP 2", 96.8, 95.8, 97.5, "ens"),
    ("PE-Core-G/14", 96.3, 95.2, 97.1, "gen"),
    ("SigLIP 2 so400m", 95.7, 94.6, 96.6, "gen"),
    ("EVA-02-L", 95.6, 94.5, 96.5, "gen"),
    ("DINOv3 ViT-L/16", 95.1, 93.9, 96.1, "gen"),
    ("ConvNeXt V2-L", 94.3, 93.0, 95.3, "gen"),
    ("DINOv3 ViT-7B", 94.2, None, None, "gen"),
    ("PlantNet DINOv2-B", 93.5, None, None, "species"),
    ("DINOv2-g", 93.4, None, None, "gen"),
    ("BioCLIP 2.5 ViT-H", 92.7, None, None, "species"),
    ("BioCLIP 2", 92.3, None, None, "species"),
]
BL, BX0, BW = 127.0, 150.5, 38.0
BXMIN, BXMAX = 91.5, 98.8
pb = axmm(BX0, AY0, BW, AH)
pb.set_xlim(BXMIN, BXMAX)
pb.set_ylim(0, AH)
BMMPT = BW / (BXMAX - BXMIN)
ax.text(BL, H - 1.0, "(b) Frozen backbones + linear head,\n5-fold grouped CV over 1576 images", fontsize=7.5,
        fontweight="bold", ha="left", va="top", color=INK, linespacing=1.15)
pitch = AH / len(CV)
pb.set_xticks([92, 94, 96, 98])
pb.set_xticks([93, 95, 97], minor=True)
pb.grid(axis="x", which="major", color=GRID, lw=0.5)
pb.grid(axis="x", which="minor", color=GRID, lw=0.35)
pb.set_axisbelow(True)
pb.set_yticks([])
for s in ("top", "right", "left"):
    pb.spines[s].set_visible(False)
pb.tick_params(axis="x", labelsize=S_LAB, which="both")
pb.tick_params(axis="x", which="minor", length=1.5)
pb.set_xlabel("CV accuracy (%)", fontsize=6.2, labelpad=1.5)
ax.text(BX0 + BW / 2, 2.2, "bars: Wilson 95% CI, where reported", fontsize=S_VAL, color=MUTED, ha="center",
        va="center")
for i, (label, m, lo, hi, kind) in enumerate(CV):
    yr = AH - pitch * (i + 0.5)
    col = AMBER if kind == "species" else BLUE
    pb.plot([BXMIN, BXMAX], [yr, yr], color=GUIDE, lw=0.5, zorder=0.5)
    ax.text(BL, AY0 + yr, label, fontsize=S_LAB, color=INK, va="center", linespacing=1.1)
    if lo is not None:
        pb.plot([lo, hi], [yr, yr], color=col, lw=1.0, solid_capstyle="butt", zorder=3)
    pb.plot(m, yr, marker="o", ms=4.2, mfc=col, mec="white", mew=0.5, ls="none", zorder=4)
    xr = (hi if hi is not None else m + 0.7 / BMMPT) + 0.9 / BMMPT
    pb.text(xr, yr, f"{m:.1f}", fontsize=S_VAL, color=INK2, va="center", ha="left")
ysp = AH - pitch * 9.5
for k, line in enumerate(["Species-ID models", "(PlantNet, BioCLIP)", "transfer no better", "than web-scale ones"]):
    pb.text(BXMAX - 0.05, ysp + 3.6 - 2.4 * k, line, fontsize=S_VAL, color=INK2, va="center", ha="right")

save(fig, "fig_classifier_comparison")
print("ok")
