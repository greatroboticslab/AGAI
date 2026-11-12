"""Fig. Knowledge base: (a) schema, (b) gray mold graph built from disease_knowledge_base.json."""
import json
import math
import re
import sys
from matplotlib.patches import Circle, FancyBboxPatch

sys.path.insert(0, "D:/agai-paper/draft/figsrc")
from agstyle import *  # noqa

setup(7.0)
KB = json.load(open("D:/agai-paper/worktree/knowledge_graph/disease_knowledge_base.json", encoding="utf-8"))["diseases"]
TYPE_COL = {"condition": "#6250d6", "symptom": "#2a78d6", "treatment": "#008300", "cause": "#c98500",
            "recovery": "#898781"}

W, H = 190.0, 92.0
fig, ax = canvas(W * MM, H * MM)

# ---------------- (a) schema
panel_label(ax, 1.0, H - 1.0, "(a) Schema per condition")
cx, cy = 34.0, 42.0
box(ax, cx - 11, cy - 5, 22, 10, "language", "Condition", [], title_size=7.0, fill="#ecebfa", edge="#6250d6")
nodes = [("Symptom", "10 per condition", "symptom", (cx, cy + 25), "shows symptom"),
         ("Treatment", "7 to 9", "treatment", (cx + 21, cy - 18), "treated by"),
         ("Cause", "7 or 8", "cause", (cx - 21, cy - 18), "causes"),
         ("Recovery", "timeline", "recovery", (cx, cy - 31), "recovery time")]
for name, sub, t, (x, y), rel in nodes:
    f = {"symptom": "#e7f0fb", "treatment": "#e3f1e3", "cause": "#fdf2d9", "recovery": "#f3f2ee"}[t]
    box(ax, x - 10.5, y - 4.6, 21, 9.2, "neutral", name, [sub], fill=f, edge=TYPE_COL[t], title_size=6.6, body_size=5.6)
    if t == "cause":
        arrow(ax, x + 6, y + 4.6, cx - 6, cy - 5, color=TYPE_COL[t], lw=0.8)
        ax.text((x + cx) / 2 - 7.5, (y + cy) / 2 + 0.5, rel, fontsize=5.6, color=INK2, rotation=0)
    elif t == "symptom":
        arrow(ax, cx, cy + 5, x, y - 4.6, color=TYPE_COL[t], lw=0.8)
        ax.text(cx + 1.2, (cy + y) / 2, rel, fontsize=5.6, color=INK2)
    elif t == "treatment":
        arrow(ax, cx + 6, cy - 5, x - 6, y + 4.6, color=TYPE_COL[t], lw=0.8)
        ax.text((x + cx) / 2 + 1.5, (y + cy) / 2 + 0.5, rel, fontsize=5.6, color=INK2)
    else:
        arrow(ax, cx, cy - 5, x, y + 4.6, color=TYPE_COL[t], lw=0.8)
        ax.text(cx + 1.4, (cy - 5 + y + 4.6) / 2, rel, fontsize=5.6, color=INK2, ha="center", va="center",
                rotation=90)
attrs = ["severity level", "visual indicators (6 or 7)", "part-specific treatments", "extension sources (3 or 4)",
         "prevention summary"]
ax.text(4.0, 85.0, "Attributes per condition", fontsize=6.4, fontweight="bold", color=INK)
for i, a in enumerate(attrs):
    ax.text(4.0 + (i % 2) * 31.0, 81.0 - (i // 2) * 3.2, "\u2022 " + a, fontsize=5.6, color=INK2,
            family="DejaVu Sans")
ax.plot([1.5, 64.5], [3.0, 3.0], alpha=0)

# ---------------- (b) gray mold graph (radial)
panel_label(ax, 70.0, H - 1.0, "(b) Graph for gray mold built from the knowledge base")
gm = KB["gray_mold"]
SHORT = {
 "symptom": ["Gray, fuzzy sporulation", "Soft, water-soaked patches", "Collapsed, sunken tissue",
             "Dusty gray spore clouds", "Mummified, tan berries", "Brown, papery sepals",
             "Stem and petiole cankers", "Brown spots on petals", "Spread between touching fruit",
             "White mycelium at dawn"],
 "treatment": ["Remove infected tissue", "Thin plants for airflow", "Drip irrigation only",
               "Water early in the morning", "Straw mulch under trusses", "Remove spent petals daily",
               "Fungicide rotation from bloom", "Biofungicides (organic)", "Potassium fertilization"],
 "cause": ["Botrytis cinerea", "15 to 25 C, RH above 80%", "Poor air circulation", "Wetness over 12 h",
           "Fruit-to-fruit contact", "Overripe, damaged fruit", "Senescent petals on fruit", "Rain splash and wind"],
 "recovery": ["Spread halted in 3 to 5 days"],
}


def short(s, n=30):
    s = re.split(r" \u2014 | - |\(|;|:", s)[0].strip().rstrip(",.")
    if len(s) <= n:
        return s
    out = ""
    for w in s.split():
        if len(out) + len(w) + 1 > n:
            break
        out = (out + " " + w).strip()
    return out.rstrip(",") + "..."


FS = 5.4
cxb, cyb, cwb, chb = 117.0, 40.5, 22.0, 9.0          # center node (box)
ax.add_patch(FancyBboxPatch((cxb, cyb), cwb, chb, boxstyle="round,pad=0,rounding_size=1.2",
                            fc="#ecebfa", ec=TYPE_COL["condition"], lw=0.8, zorder=4))
ax.text(cxb + cwb / 2, cyb + chb / 2, "Gray mold", ha="center", va="center", fontsize=7.0, fontweight="bold",
        color=INK, zorder=5)
YMID = cyb + chb / 2


def column(items, t, header, x_spine, x_dot, y_top, step, side):
    """List items on one side of a vertical spine; returns the spine's mid y."""
    ys = [y_top - i * step for i in range(len(items))]
    col = TYPE_COL[t]
    ax.plot([x_spine, x_spine], [ys[0], ys[-1]], color=col, lw=0.7, zorder=2)
    for it, y in zip(items, ys):
        ax.plot([x_spine, x_dot], [y, y], color=col, lw=0.6, zorder=2)
        ax.add_patch(Circle((x_dot, y), 0.85, fc=col, ec="white", lw=0.4, zorder=3))
        if side == "right":
            ax.text(x_dot + 1.8, y, it, ha="left", va="center", fontsize=FS, color=INK2)
        else:
            ax.text(x_dot - 1.8, y, it, ha="right", va="center", fontsize=FS, color=INK2)
    hx = x_dot + 1.8 if side == "right" else x_dot - 1.8
    ax.text(hx, ys[0] + 3.6, header, ha="left" if side == "right" else "right", va="center", fontsize=6.0,
            fontweight="bold", color=col)
    return (ys[0] + ys[-1]) / 2


# causes on the left
ym_c = column(SHORT["cause"], "cause", "Causes (8)", 110.0, 106.0, YMID + 3.5 * 4.4, 4.4, "left")
ax.plot([110.0, 113.5, 113.5, cxb], [ym_c, ym_c, YMID, YMID], color=TYPE_COL["cause"], lw=0.7, zorder=2)
# symptoms (upper right) and treatments (lower right)
ym_s = column(SHORT["symptom"], "symptom", "Symptoms (10)", 148.0, 152.0, 80.0, 3.65, "right")
ym_t = column(SHORT["treatment"], "treatment", "Treatments (9)", 148.0, 152.0, 35.0, 3.6, "right")
ax.plot([cxb + cwb, 143.5, 143.5, 148.0], [YMID + 1.5, YMID + 1.5, ym_s, ym_s], color=TYPE_COL["symptom"],
        lw=0.7, zorder=2)
ax.plot([cxb + cwb, 143.5, 143.5, 148.0], [YMID - 1.5, YMID - 1.5, ym_t, ym_t], color=TYPE_COL["treatment"],
        lw=0.7, zorder=2)
# recovery below the node
ry = 27.0
ax.plot([cxb + cwb / 2, cxb + cwb / 2], [cyb, ry + 1.0], color=TYPE_COL["recovery"], lw=0.7, zorder=2)
ax.add_patch(Circle((cxb + cwb / 2, ry), 0.85, fc=TYPE_COL["recovery"], ec="white", lw=0.4, zorder=3))
ax.text(cxb + cwb / 2, ry - 2.6, "Recovery time", ha="center", va="center", fontsize=6.0, fontweight="bold",
        color=INK2)
ax.text(cxb + cwb / 2, ry - 5.6, SHORT["recovery"][0], ha="center", va="center", fontsize=FS, color=INK2)
save(fig, "fig_knowledge_base")
print("ok")
