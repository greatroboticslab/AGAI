"""Fig. Visible signs that the knowledge base lists for the six non-healthy conditions.
(a) Each symptom and visual-indicator statement (101 in total) is coded into one or more sign groups; the coding is
saved to notes/symptom_coding.csv. Dot area = number of statements; ring = the cell holds a hallmark phrase of the
report consistency check (knowledge_graph/rag_retriever.py, unique_indicators). (b) Visible sign groups shared by
each pair of conditions (signs that need touch, smell or cutting are left out)."""
import csv
import json
import sys

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Circle, Rectangle, FancyBboxPatch

sys.path.insert(0, "D:/agai-paper/draft/figsrc")
from agstyle import *  # noqa

KB = json.load(open("D:/agai-paper/worktree/knowledge_graph/disease_knowledge_base.json", encoding="utf-8"))["diseases"]
COND = [("drought", "drought"), ("frost", "frost_injury"), ("overwatered", "overwatering"),
        ("root_rot", "root_rot"), ("gray_mold", "gray_mold"), ("white_mold", "white_mold")]

BLOCKS = [
    ("Whole plant", [("WILT", "Wilting, limp or collapsing"), ("STUNT", "Stunted or small new growth")]),
    ("Leaves", [("YEL", "Yellowing or paling"), ("BRN", "Browning or dry, papery margins"),
                ("CURL", "Curling, cupping or folding"), ("GLASS", "Glassy or blistered tissue"),
                ("DULL", "Dull, matte surface")]),
    ("Flowers\nand fruit", [("FLOWER", "Damaged or aborted flowers"), ("FRUIT", "Misshapen or shriveled fruit"),
                            ("CALYX", "Brown, papery calyx")]),
    ("Rot and\nfungal growth", [("ROT", "Soft rot or water-soaked lesions"), ("GRAY", "Gray fuzzy spores"),
                                ("WHITE", "White cottony mycelium"), ("SCLER", "Black sclerotia")]),
    ("Stem, crown\nand root", [("STEM", "Stem or runner damage"), ("CROWN", "Crown rot, softening or collapse"),
                               ("ROOT", "Brown, mushy roots")]),
    ("Soil", [("SOILDRY", "Dry, cracked soil"), ("SOILWET", "Wet soil, standing water, algae")]),
    ("Touch, smell\nor cutting", [("ODOR", "Root-zone odor"), ("CUT", "Discolored inside when cut"),
                                  ("PULL", "Plant pulls out easily")]),
]
HIDDEN = {"ODOR", "CUT", "PULL"}

# Sign groups for every statement: S = symptoms[i], V = visual_indicators[i] of the knowledge-base entry.
CODE = {
    "drought": {"S0": "CURL BRN", "S1": "DULL", "S2": "STUNT CURL", "S3": "SOILDRY", "S4": "FRUIT", "S5": "WILT",
                "S6": "WILT", "S7": "YEL BRN", "S8": "FLOWER", "S9": "STUNT",
                "V0": "CURL", "V1": "BRN", "V2": "SOILDRY", "V3": "WILT", "V4": "DULL", "V5": "FRUIT", "V6": "FLOWER"},
    "frost": {"S0": "GLASS", "S1": "FLOWER", "S2": "GLASS BRN", "S3": "FRUIT", "S4": "FLOWER", "S5": "BRN",
              "S6": "FLOWER", "S7": "FRUIT", "S8": "STEM", "S9": "FLOWER",
              "V0": "FLOWER", "V1": "GLASS", "V2": "BRN", "V3": "FRUIT", "V4": "FLOWER", "V5": "GLASS"},
    "overwatered": {"S0": "SOILWET", "S1": "WILT", "S2": "YEL", "S3": "CROWN", "S4": "CROWN", "S5": "SOILWET",
                    "S6": "YEL", "S7": "ROOT", "S8": "ODOR", "S9": "GLASS",
                    "V0": "SOILWET", "V1": "YEL", "V2": "WILT", "V3": "CROWN", "V4": "SOILWET", "V5": "ROOT",
                    "V6": "GLASS"},
    "root_rot": {"S0": "WILT", "S1": "CUT", "S2": "ROOT", "S3": "ODOR", "S4": "STUNT", "S5": "BRN", "S6": "CUT",
                 "S7": "PULL", "S8": "YEL CURL STUNT", "S9": "WILT",
                 "V0": "ROOT", "V1": "CROWN", "V2": "WILT", "V3": "ODOR", "V4": "CUT", "V5": "PULL", "V6": "ROOT"},
    "gray_mold": {"S0": "GRAY", "S1": "ROT", "S2": "ROT", "S3": "GRAY", "S4": "FRUIT", "S5": "CALYX", "S6": "STEM",
                  "S7": "FLOWER", "S8": "ROT", "S9": "WHITE",
                  "V0": "GRAY", "V1": "ROT", "V2": "GRAY", "V3": "FRUIT", "V4": "ROT", "V5": "FLOWER", "V6": "STEM"},
    "white_mold": {"S0": "WHITE", "S1": "ROT", "S2": "WHITE", "S3": "ROT", "S4": "CROWN", "S5": "SCLER",
                   "S6": "WILT", "S7": "STEM WHITE WILT", "S8": "CUT", "S9": "WHITE SCLER",
                   "V0": "WHITE", "V1": "STEM", "V2": "SCLER", "V3": "CROWN", "V4": "ROT", "V5": "WHITE",
                   "V6": "STEM"},
}
# The 18 hallmark phrases of the consistency check (rag_retriever.py, unique_indicators).
HALLMARKS = {"drought": ["cracked soil", "limp petioles"],
             "overwatered": ["mushy crown", "algae growth", "waterlogged"],
             "root_rot": ["foul odor", "red stele", "brown roots"],
             "frost": ["black eye", "translucent leaves", "glassy tissue"],
             "gray_mold": ["gray fuzz", "dusty spores", "tan mummies", "botrytis"],
             "white_mold": ["white cotton", "black sclerotia", "cottony mycelium"]}

# ------------------------------------------------------------------ counts
ROWS = [c for _, rs in BLOCKS for c, _ in rs]
LABEL = {c: l for _, rs in BLOCKS for c, l in rs}
cnt = {s: {r: 0 for r in ROWS} for s, _ in COND}
stmts = {}
with open("D:/agai-paper/draft/notes/symptom_coding.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["condition", "statement_id", "statement", "sign_groups"])
    for s, key in COND:
        e = KB[key]
        items = [("S%d" % i, t) for i, t in enumerate(e["symptoms"])] + \
                [("V%d" % i, t) for i, t in enumerate(e["visual_indicators"])]
        assert {i for i, _ in items} == set(CODE[s]), (s, {i for i, _ in items} ^ set(CODE[s]))
        stmts[s] = items
        for i, t in items:
            codes = CODE[s][i].split()
            for c in codes:
                assert c in ROWS, c
                cnt[s][c] += 1
            w.writerow([s, i, t, "; ".join(LABEL[c] for c in codes)])
print("statements coded:", sum(len(v) for v in stmts.values()))

ring = set()
for s, phrases in HALLMARKS.items():
    for p in phrases:
        hit = [i for i, t in stmts[s] if p.lower() in t.lower()]
        assert hit, (s, p)
        for i in hit:
            for c in CODE[s][i].split():
                ring.add((s, c))


def flat(o):
    if isinstance(o, str):
        return [o]
    if isinstance(o, list):
        return [x for v in o for x in flat(v)]
    if isinstance(o, dict):
        return [x for v in o.values() for x in flat(v)]
    return []


print("hallmark phrases that also appear in another condition's knowledge-base entry:")
for s, phrases in HALLMARKS.items():
    for p in phrases:
        others = [o for o, k in COND + [("healthy", "healthy")]
                  if o != s and any(p.lower() in t.lower() for t in flat(KB[k]))]
        if others:
            print("   %-18r of %-12s also in %s" % (p, s, others))

vis = {s: {c for c in ROWS if c not in HIDDEN and cnt[s][c] > 0} for s, _ in COND}
ncond = {c: sum(1 for s, _ in COND if cnt[s][c] > 0) for c in ROWS}
print("rows listed for >= 3 conditions:", [c for c in ROWS if ncond[c] >= 3])
print("visible groups per condition:", {s: len(vis[s]) for s, _ in COND})
pairs = []
for a in range(6):
    for b in range(a):
        sa, sb = COND[a][0], COND[b][0]
        pairs.append((len(vis[sa] & vis[sb]), sb, sa, sorted(vis[sa] & vis[sb])))
for p in sorted(pairs, reverse=True):
    print("   shared %d  %s / %s  %s" % p)

# ------------------------------------------------------------------ figure
setup(7.0)
W, H = 190.0, 122.0
fig = plt.figure(figsize=(W * MM, H * MM))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W)
ax.set_ylim(0, H)
ax.axis("off")

# ---------------- (a) matrix
panel_label(ax, 1.0, H - 1.0, "(a) Signs the knowledge base lists for each condition")
RH, BG = 3.9, 1.9
CX = [58.0 + 10.5 * i for i in range(6)]
XS = 120.0
top = H - 21.5
for (s, _), cx in zip(COND, CX):
    ax.add_patch(FancyBboxPatch((cx - 3.6, top + 0.8), 7.2, 1.3, boxstyle="round,pad=0,rounding_size=0.5",
                                fc=CLASS_COLOR[s], ec="none"))
    ax.text(cx - 1.4, top + 3.0, CLASS_LABEL[s], rotation=40, rotation_mode="anchor", ha="left", va="bottom",
            fontsize=6.0, color=INK)
ax.text(XS - 0.2, top + 3.0, "Conditions", rotation=40, rotation_mode="anchor", ha="left", va="bottom",
        fontsize=5.8, color=INK2, style="italic")

y = top
row_y = {}
for bi, (bname, rs) in enumerate(BLOCKS):
    if bi:
        y -= BG
    if bname.startswith("Touch"):
        ax.plot([2.0, XS + 8.0], [y + BG / 2, y + BG / 2], color=AXIS, lw=0.6, ls=(0, (2.5, 1.5)))
    y0 = y
    for c, _ in rs:
        row_y[c] = y - RH / 2
        y -= RH
    ax.text(2.0, (y0 + y) / 2, bname, ha="left", va="center", fontsize=5.8, fontweight="bold", color=INK2,
            linespacing=1.15, style="italic" if bname.startswith("Touch") else "normal")
    ax.plot([17.4, 17.4], [y0 - 0.6, y + 0.6], color=AXIS, lw=0.7, solid_capstyle="round")

for c in ROWS:
    yc = row_y[c]
    n_c = ncond[c]
    if n_c >= 3:
        ax.add_patch(Rectangle((18.5, yc - RH / 2 + 0.2), XS + 7.6 - 18.5, RH - 0.4, fc="#f6efdc", ec="none",
                               zorder=0))
    ax.text(19.2, yc, LABEL[c], ha="left", va="center", fontsize=5.9, color=INK2 if c in HIDDEN else INK)
    for (s, _), cx in zip(COND, CX):
        n = cnt[s][c]
        if n == 0:
            ax.add_patch(Circle((cx, yc), 0.22, fc="#d4d2ca", ec="none", zorder=1))
            continue
        r = 0.55 * n ** 0.5
        if c in HIDDEN:
            ax.add_patch(Circle((cx, yc), r, fc="white", ec=CLASS_COLOR[s], lw=0.9, zorder=2))
        else:
            ax.add_patch(Circle((cx, yc), r, fc=CLASS_COLOR[s], ec="white", lw=0.4, zorder=2))
        if (s, c) in ring:
            ax.add_patch(Circle((cx, yc), r + 0.45, fc="none", ec=INK, lw=0.55, zorder=3))
    ax.add_patch(Rectangle((XS, yc - 0.55), 1.3 * n_c, 1.1, fc=INK2 if n_c >= 3 else "#b9b7ae", ec="none",
                           zorder=2))
    ax.text(XS + 1.3 * n_c + 0.8, yc, str(n_c), ha="left", va="center", fontsize=5.6, color=INK2)

# ---------------- (b) pairwise shared visible signs
XB = 131.0
ax.text(XB, H - 1.0, "(b) Visible sign groups shared\nby each pair of conditions", fontsize=7.5, fontweight="bold",
        ha="left", va="top", color=INK, linespacing=1.15)
C = 5.9
x0 = 153.5
ytb = H - 14.5
cmap = LinearSegmentedColormap.from_list("shared", ["#f4f3ef", "#b9b7ae", "#4a4945"])
VMAX = 5.0
for i, (si, _) in enumerate(COND):
    yc = ytb - (i + 0.5) * C
    ax.add_patch(FancyBboxPatch((x0 - 2.9, yc - 0.8), 1.6, 1.6, boxstyle="round,pad=0,rounding_size=0.4",
                                fc=CLASS_COLOR[si], ec="none"))
    ax.text(x0 - 3.6, yc, CLASS_LABEL[si], ha="right", va="center", fontsize=5.9, color=INK)
    for j, (sj, _) in enumerate(COND[: i + 1]):
        xc = x0 + (j + 0.5) * C
        if i == j:
            v = len(vis[si])
            ax.add_patch(Rectangle((xc - C / 2 + 0.25, yc - C / 2 + 0.25), C - 0.5, C - 0.5, fc="white", ec=AXIS,
                                   lw=0.7))
            ax.text(xc, yc, str(v), ha="center", va="center", fontsize=6.2, color=INK, fontweight="bold")
        else:
            v = len(vis[si] & vis[sj])
            ax.add_patch(Rectangle((xc - C / 2, yc - C / 2), C, C, fc=cmap(min(v, VMAX) / VMAX), ec="white", lw=0.9))
            ax.text(xc, yc, str(v), ha="center", va="center", fontsize=6.2, color="white" if v >= 3 else INK)
ybot = ytb - 6 * C
for j, (sj, _) in enumerate(COND):
    ax.text(x0 + (j + 0.5) * C + 0.8, ybot - 1.2, CLASS_LABEL[sj], rotation=40, rotation_mode="anchor", ha="right",
            va="top", fontsize=5.9, color=INK)
ax.text(XB, ybot - 16.0, "Diagonal (outlined): visible sign groups listed\nfor the condition.", fontsize=5.6,
        color=INK2, ha="left", va="top", linespacing=1.2)

# ---------------- key to (a), bottom right
ky = 38.0
ax.text(XB, ky, "Key to (a)", fontsize=6.4, fontweight="bold", color=INK, ha="left", va="center")
ky -= 5.0
ax.text(XB, ky, "Statements per cell:", fontsize=5.8, color=INK2, ha="left", va="center")
for k, n in enumerate([1, 2, 4, 6]):
    cx = XB + 23.0 + 6.0 * k
    ax.add_patch(Circle((cx, ky), 0.55 * n ** 0.5, fc=MUTED, ec="none"))
    ax.text(cx + 0.55 * n ** 0.5 + 0.7, ky, str(n), fontsize=5.6, color=INK2, ha="left", va="center")
ky -= 5.0
ax.add_patch(Circle((XB + 1.6, ky), 0.95, fc=MUTED, ec="none"))
ax.add_patch(Circle((XB + 1.6, ky), 1.4, fc="none", ec=INK, lw=0.55))
ax.text(XB + 4.6, ky, "Holds a hallmark phrase of the report\nconsistency check (18 phrases)", fontsize=5.8, color=INK2,
        ha="left", va="center", linespacing=1.15)
ky -= 6.2
ax.add_patch(Circle((XB + 1.6, ky), 0.95, fc="white", ec=MUTED, lw=0.9))
ax.text(XB + 4.6, ky, "Needs touch, smell or cutting; not\nvisible on an intact plant", fontsize=5.8, color=INK2,
        ha="left", va="center", linespacing=1.15)
ky -= 6.2
ax.add_patch(Rectangle((XB, ky - 1.2), 3.4, 2.4, fc="#f6efdc", ec="none"))
ax.text(XB + 4.6, ky, "Listed for three or more conditions", fontsize=5.8, color=INK2, ha="left", va="center")
ky -= 4.6
ax.add_patch(Rectangle((XB, ky - 0.55), 3.4, 1.1, fc=INK2, ec="none"))
ax.text(XB + 4.6, ky, "Number of conditions listing the sign", fontsize=5.8, color=INK2, ha="left", va="center")

save(fig, "fig_symptom_overlap")
print("ok")
