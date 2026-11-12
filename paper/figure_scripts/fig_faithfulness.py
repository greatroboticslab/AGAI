"""Fig. Claim-level faithfulness of local captioners and automatic hallucination filters (38 hard split v1 test
photos; blind human ratings, interim: one rater, 615 of 845 claims).
(a) Two-pass captioning protocol and blind rating. (b) Share of claims rated "not there" with 95% Wilson intervals.
(c) Filters on one common set of 562 rated claims (515 real, 47 not there): false claims removed against real claims
removed; the diagonal is a filter that drops claims at random.
Numbers: D:/agai-paper/github/docs/CAPTIONING.md (status :13, protocol :17-37, ratings :64-69, filters :99-114,
ratio range :129-130) and cluster/EXPERIMENT_LOG.md (:570-579 ratings, :633-649 filters)."""
import sys

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt

sys.path.insert(0, "D:/agai-paper/draft/figsrc")
from agstyle import *  # noqa

setup(7.0)
W, H = 190.0, 100.0
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
GRAYM = MUTED
S_LAB, S_VAL = 6.0, 5.6
HALO = [pe.withStroke(linewidth=2.2, foreground="white")]

# ------------------------------------------------------------------ (a) protocol
panel_label(ax, 1.0, H - 1.0, "(a) Two-pass captioning and blind rating")
A = dict(color=INK2, lw=0.8, ms=6)
TS, BS = 6.2, 5.6
y1 = H - 20.0   # centre of the top row
box(ax, 1.0, y1 - 7.0, 13.0, 14.0, "data", "Photo", ["38 hard", "test photos"], title_size=TS, body_size=BS)
box(ax, 19.0, y1 - 8.0, 30.0, 16.0, "language", "Pass 1: captioner",
    ["sees the image only,", "no label; lists at most 10", "grounded observations"], title_size=TS, body_size=BS)
box(ax, 57.0, y1 - 8.0, 31.0, 16.0, "knowledge", "Pass 2: reconciler",
    ["text only; gets the label", "and its definition; writes", "the verdict and caption"], title_size=TS,
    body_size=BS)
arrow(ax, 14.0, y1, 19.0, y1, **A)
arrow(ax, 49.0, y1, 57.0, y1, **A)
ax.text(53.0, y1 + 2.0, "as text", fontsize=BS, color=INK2, ha="center", va="center")
yr = H - 40.0   # centre of the rater box
box(ax, 19.0, yr - 6.5, 69.0, 13.0, "gate", "Blind human rater",
    ["marks each pass-1 claim real, not there or unsure;", "captioners anonymized as A to D, shuffled per photo"],
    title_size=TS, body_size=BS)
arrow(ax, 34.0, y1 - 8.0, 34.0, yr + 6.5, **A)
ax.text(35.2, (y1 - 8.0 + yr + 6.5) / 2, "claims", fontsize=BS, color=INK2, va="center")

# ------------------------------------------------------------------ (b) "not there" rates
EX = [  # captioner, rate %, CI low, CI high, claims, photos rated, mold seen (of 13)
    ("Qwen3-VL-32B", 4.0, 2.0, 8.1, 174, 34, "9 of 13"),
    ("GLM-4.6V", 5.0, 1.4, 16.5, 40, 9, "not rated"),
    ("Qwen3.8-27B", 10.2, 6.7, 15.2, 197, 34, "11 of 13"),
    ("Qwen3-VL-235B", 10.3, 6.8, 15.2, 204, 34, "11 of 13"),
]
panel_label(ax, 1.0, 50.5, "(b) Claims rated \"not there\" (95% Wilson CI)")
ax.text(1.0, 46.3, "one blind rater; 615 of 845 claims rated so far", fontsize=S_VAL, color=INK2,
        va="center")
PX0, PY0, PW, PH = 21.0, 11.0, 37.0, 26.0
pp = axmm(PX0, PY0, PW, PH)
pp.set_xlim(0, 20)
pp.set_ylim(0, PH)
pp.set_xticks([0, 5, 10, 15, 20])
pp.grid(axis="x", color=GRID, lw=0.5)
pp.set_axisbelow(True)
pp.set_yticks([])
for s_ in ("top", "right", "left"):
    pp.spines[s_].set_visible(False)
pp.tick_params(axis="x", labelsize=S_LAB)
pp.set_xlabel("Claims rated \"not there\" (%)", fontsize=6.2, labelpad=1.5)
MMB = PW / 20.0
cp = PH / len(EX)
CX1, CX2 = 66.0, 76.5   # table columns (mm): claims (photos), mold seen
ax.text(CX1, PY0 + PH + 2.6, "claims (photos)", fontsize=S_VAL, color=MUTED, ha="center", va="center")
ax.text(CX2 + 6.0, PY0 + PH + 2.6, "mold seen", fontsize=S_VAL, color=MUTED, ha="center", va="center")
for i, (name, r, lo, hi, n, ph, mold) in enumerate(EX):
    yc = PH - cp * (i + 0.5)
    pp.plot([0, 20], [yc, yc], color="#efeee9", lw=0.5, zorder=0.5)
    pp.plot([lo, hi], [yc, yc], color=INK2, lw=1.0, solid_capstyle="butt", zorder=3)
    pp.plot(r, yc, marker="o", ms=4.4, mfc="white", mec=INK2, mew=0.9, ls="none", zorder=4)
    pp.text(hi + 0.8 / MMB, yc, f"{r:.1f}", fontsize=S_VAL, color=INK2, va="center", ha="left")
    ax.text(1.0, PY0 + yc, name, fontsize=S_LAB, color=INK, va="center")
    ax.text(CX1, PY0 + yc, f"{n} ({ph})", fontsize=S_VAL, color=INK2, ha="center", va="center")
    ax.text(CX2 + 6.0, PY0 + yc, mold, fontsize=S_VAL, color=INK2, ha="center", va="center")

# ------------------------------------------------------------------ (c) filter trade-off
CONS = [("agreement", 41.7, 59.6), ("self-consistency", 52.0, 66.0),
        ("agreement, else\nself-consistency", 34.6, 51.1)]
JUDGE = [("8B", 21.4, 59.6), ("8B + zoom", 16.3, 48.9), ("27B", 8.5, 31.9), ("27B + zoom", 9.9, 36.2),
         ("32B", 10.5, 38.3), ("32B + zoom", 12.8, 44.7), ("235B + zoom", 8.2, 29.8)]
MAJ = ("majority of 3 judges", 7.2, 36.2)
for _, xr, yf in CONS + JUDGE + [MAJ]:
    assert 0 < xr < yf < 70
ax.text(97.0, H - 1.0, "(c) Hallucination filters: false against real claims removed", fontsize=7.5,
        fontweight="bold", ha="left", va="top", color=INK)
ax.text(97.0, H - 5.6, "common set of 562 rated claims (515 real, 47 not there); interim ratings", fontsize=S_VAL,
        color=INK2, ha="left", va="top")
QX0, QY0, QW, QH = 106.0, 12.0, 80.0, 74.0
XMAX, YMIN, YMAX = 56.0, 20.0, 70.0
pq = axmm(QX0, QY0, QW, QH)
pq.set_xlim(0, XMAX)
pq.set_ylim(YMIN, YMAX)
pq.set_xticks(range(0, 56, 10))
pq.set_yticks(range(20, 71, 10))
pq.grid(color=GRID, lw=0.5)
pq.set_axisbelow(True)
for s_ in ("top", "right"):
    pq.spines[s_].set_visible(False)
pq.tick_params(labelsize=S_LAB)
pq.set_xlabel("Real claims removed (%, of 515)", fontsize=6.2, labelpad=1.5)
pq.set_ylabel("False claims removed (%, of 47)", fontsize=6.2, labelpad=1.5)
pq.plot([YMIN, XMAX], [YMIN, XMAX], color=MUTED, lw=0.7, zorder=1)
_p0, _p1 = pq.transData.transform((30, 30)), pq.transData.transform((31, 31))
_ang = __import__("math").degrees(__import__("math").atan2(_p1[1] - _p0[1], _p1[0] - _p0[0]))
pq.text(41.0, 39.6, "random filter (y = x)", fontsize=S_VAL, color=MUTED, ha="left", va="top", rotation=_ang,
        rotation_mode="anchor", path_effects=HALO)
for name, xr, yf in CONS:
    pq.plot(xr, yf, marker="s", ms=4.4, mfc=GRAYM, mec="white", mew=0.5, ls="none", zorder=5)
for name, xr, yf in JUDGE:
    pq.plot(xr, yf, marker="o", ms=4.6, mfc=BLUE, mec="white", mew=0.5, ls="none", zorder=5)
pq.plot(MAJ[1], MAJ[2], marker="o", ms=4.6, mfc="white", mec=BLUE, mew=0.9, ls="none", zorder=5)

LAB = {  # name: (x, y, ha, va) label anchor in data units
    "agreement": (43.0, 59.6, "left", "center"),
    "self-consistency": (50.6, 66.0, "right", "center"),
    "agreement, else\nself-consistency": (36.0, 51.1, "left", "center"),
    "8B": (22.8, 59.6, "left", "center"),
    "8B + zoom": (17.7, 48.9, "left", "center"),
    "27B": (9.9, 31.9, "left", "center"),
    "27B + zoom": (11.3, 36.2, "left", "center"),
    "32B": (11.9, 38.3, "left", "center"),
    "32B + zoom": (14.2, 44.7, "left", "center"),
    "235B + zoom": (9.6, 29.8, "left", "center"),
}
for name, xr, yf in CONS + JUDGE:
    lx, ly, ha, va = LAB[name]
    pq.text(lx, ly, name, fontsize=S_VAL + 0.2, color=INK, ha=ha, va=va, linespacing=1.1, path_effects=HALO,
            zorder=6)
pq.text(1.0, 52.6, "majority of 3 judges", fontsize=S_VAL + 0.2, color=INK, ha="left", va="center",
        path_effects=HALO, zorder=6)
pq.text(1.0, 50.6, "(exploratory)", fontsize=S_VAL, color=INK2, ha="left", va="center", path_effects=HALO,
        zorder=6)
pq.plot([3.6, MAJ[1] - 0.25], [49.6, MAJ[2] + 0.9], color=MUTED, lw=0.5, zorder=4)
# key (top left)
for k, (mk, fc, ec, mew, txt) in enumerate([
        ("s", GRAYM, "white", 0.5, "consistency filter"),
        ("o", BLUE, "white", 0.5, "judge, whole photo; \"+ zoom\": plus an enlarged crop"),
        ("o", "white", BLUE, 0.9, "exploratory")]):
    yk = 68.4 - 1.9 * k
    pq.plot(1.2, yk, marker=mk, ms=4.4, mfc=fc, mec=ec, mew=mew, ls="none", zorder=5)
    pq.text(2.3, yk, txt, fontsize=S_VAL, color=INK2, ha="left", va="center", path_effects=HALO, zorder=6)
for k, line in enumerate(["False removed per real removed:", "judges 2.8 to 3.7, consistency 1.3 to 1.5,",
                          "exploratory majority 5.0"]):
    pq.text(55.0, 30.0 - 2.6 * k, line, fontsize=S_VAL, color=INK2, ha="right", va="center", path_effects=HALO)

save(fig, "fig_faithfulness")
print("ok")
