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
        ax.text(cx - 1.2, (cy + y) / 2 - 1.0, rel, fontsize=5.6, color=INK2, ha="right")
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


gx, gy = 125.0, 42.0
ax.add_patch(Circle((gx, gy), 3.2, fc=TYPE_COL["condition"], ec="white", lw=0.8, zorder=5))
ax.text(gx, gy - 5.2, "Gray mold", ha="center", fontsize=6.6, fontweight="bold", color=INK, zorder=6)
groups = [("symptom", SHORT["symptom"], 28, 152, 32.0),       # top arc
          ("treatment", SHORT["treatment"], -64, 18, 31.0),    # right arc
          ("cause", SHORT["cause"], 162, 244, 31.0),           # left arc
          ("recovery", SHORT["recovery"], 270, 270, 25.0)]
for t, items, a0, a1, R in groups:
    n = len(items)
    for i, it in enumerate(items):
        ang = math.radians(a0 + (a1 - a0) * (i / (n - 1) if n > 1 else 0.5))
        x, y = gx + R * math.cos(ang) * 1.18, gy + R * math.sin(ang)
        ax.plot([gx, x], [gy, y], color=TYPE_COL[t], lw=0.5, alpha=0.55, zorder=2)
        ax.add_patch(Circle((x, y), 1.25, fc=TYPE_COL[t], ec="white", lw=0.5, zorder=4))
        c = math.cos(ang)
        ha = "left" if c > 0.02 else ("right" if c < -0.02 else "center")
        dx = 1.9 if ha == "left" else (-1.9 if ha == "right" else 0)
        dy = 0 if ha != "center" else (2.4 if math.sin(ang) > 0 else -2.6)
        ax.text(x + dx, y + dy, it, ha=ha, va="center", fontsize=5.1,
                color=INK2, zorder=6)
# legend
lx = 76.0
for i, (t, lab) in enumerate([("condition", "condition"), ("symptom", "symptom (shows symptom)"),
                              ("treatment", "treatment (treated by)"), ("cause", "cause (causes)"),
                              ("recovery", "recovery time")]):
    ax.add_patch(Circle((lx + 1.2, 6.0 - 0.0 + 0 * i), 1.0, fc=TYPE_COL[t], ec="white", lw=0.4))
    ax.text(lx + 3.0, 6.0, lab, fontsize=5.6, color=INK2, va="center")
    lx += 3.5 + 1.05 * len(lab) + 3.0
save(fig, "fig_knowledge_base")
print("ok")
