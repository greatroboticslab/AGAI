"""Fig. 1, AGAI system architecture.

One left-to-right row for the image path (classifier, acceptance gate, part detector, report writer,
hallmark check, report), the abstain branch under the gate, and one row for follow-up questions answered
from the knowledge base. All arrows are straight and none cross. The three models replaced in version 2 carry
a 'v2' line. The only photograph is a real test photograph of the dataset (the input thumbnail).
"""
import json
import sys
from PIL import Image, ImageOps

sys.path.insert(0, "D:/agai-paper/draft/figsrc")
from agstyle import *  # noqa

setup(7.0)

W, H = 190.0, 99.0
fig, ax = canvas(W * MM, H * MM)

T_SIZE, B_SIZE, V_SIZE = 6.5, 5.7, 5.5
LINE = B_SIZE * 0.43          # mm per body line
V2_COLOR = "#6250d6"


def stage(x, y, w, h, role, title, lines, v2=None, num=None):
    """A box with a bold title, body lines and an optional bottom strip naming the v2 replacement."""
    f, e = ROLE[role]
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.2", fc=f, ec=e, lw=0.75,
                                zorder=2))
    strip = 5.0 if v2 else 0.0
    if v2:
        ax.plot([x + 1.2, x + w - 1.2], [y + strip, y + strip], color=e, lw=0.45, zorder=3)
        ax.text(x + w / 2, y + strip / 2, v2, ha="center", va="center", fontsize=V_SIZE, color=V2_COLOR,
                fontweight="bold", zorder=3)
    block = T_SIZE * 0.42 + len(lines) * LINE
    top = y + strip + (h - strip) / 2 + block / 2
    label = f"{num}  {title}" if num else title
    ax.text(x + w / 2, top - T_SIZE * 0.21, label, ha="center", va="center", fontsize=T_SIZE,
            fontweight="bold", color=INK, zorder=3)
    for i, ln in enumerate(lines):
        ax.text(x + w / 2, top - T_SIZE * 0.42 - LINE * (i + 0.5), ln, ha="center", va="center",
                fontsize=B_SIZE, color=INK2, zorder=3)


def tag(x, y, text, ha="center", color=INK2, italic=False):
    ax.text(x, y, text, ha=ha, va="center", fontsize=5.6, color=color, style="italic" if italic else "normal",
            zorder=4, bbox=dict(boxstyle="square,pad=0.12", fc="white", ec="none"))


A = dict(color=INK2, lw=0.8, ms=6)
D = dict(color=MUTED, lw=0.7, ms=5, dashed=True)

# ---------------------------------------------------------------- main row: the image path
Y0, BH = 52.0, 27.0
YC = Y0 + BH / 2 + 2.5           # arrows run through the text part, above the v2 strips
cols = {"photo": (2.0, 14.0), "cls": (21.0, 25.5), "gate": (52.0, 17.5), "det": (75.0, 25.0),
        "wri": (105.5, 28.5), "chk": (139.5, 22.5), "rep": (167.5, 21.0)}

# input photograph (real gray-mold test photograph)
cand = json.load(open("D:/agai-paper/draft/notes/example_candidates.json"))
img = ImageOps.exif_transpose(Image.open(cand["gray_mold"][6])).convert("RGB")
img = square_crop(img).resize((300, 300), Image.LANCZOS)
px, pw = cols["photo"]
image_tile(ax, img, px, YC - pw / 2, pw, pw, radius=1.0, border=AXIS, lw=0.5, zorder=3)
ax.text(px + pw / 2, YC - pw / 2 - 2.6, "photograph x", ha="center", va="center", fontsize=5.7, color=INK2)

x, w = cols["cls"]
stage(x, Y0, w, BH, "classify", "Classifier", ["ResNet-50, 256 px", "image + mirror view,", "temperature-scaled",
      "p(c | x), 7 conditions"], v2="v2: SigLIP 2 so400m", num="1")
x, w = cols["gate"]
stage(x, Y0, w, BH, "gate", "Gate", ["accept the top", "condition if", r"$p_1 \geq \max(\tau_c,\,0.5)$"], num="2")
x, w = cols["det"]
stage(x, Y0, w, BH, "detect", "Part detector", ["RF-DETR Small, 512 px", "leaf, fruit, stem,", "flower, soil, root",
      "visible parts V"], v2="v2: RF-DETR Large", num="3")
x, w = cols["wri"]
stage(x, Y0, w, BH, "language", "Report writer", ["MiniGPT-v2: EVA ViT-g", "+ LLaMA-2-7B, 8-bit, LoRA",
      "prompt fixes d, lists", "the parts it may name"], v2="v2: Qwen3-VL-8B explainer", num="4")
x, w = cols["chk"]
stage(x, Y0, w, BH, "gate", "Hallmark check", ["names a sign of", "another condition?", "regenerate once,",
      "then warn"])
x, w = cols["rep"]
stage(x, Y0, w, BH, "output", "Report", ["diagnosis with a", "confidence badge,", "visible signs,", "management,", "next questions"])

# arrows along the row
def right(a, b):
    return cols[a][0] + cols[a][1], cols[b][0]

for (a, b, lab) in [("photo", "cls", None), ("cls", "gate", None), ("gate", "det", "yes"),
                    ("det", "wri", None), ("wri", "chk", None), ("chk", "rep", "pass")]:
    x0, x1 = right(a, b)
    arrow(ax, x0 + 0.4, YC, x1 - 0.2, YC, **A)
    if lab:
        ax.text((x0 + x1) / 2, YC + 1.9, lab, ha="center", va="center", fontsize=5.6, color=INK2)
# what each arrow carries, under the arrow
for (a, b, lab) in [("cls", "gate", "p"), ("gate", "det", "d"), ("det", "wri", "d, V")]:
    x0, x1 = right(a, b)
    ax.text((x0 + x1) / 2, YC - 2.0, lab, ha="center", va="center", fontsize=5.9, color=INK2, style="italic")

# regenerate loop above writer and check
xw = cols["wri"][0] + cols["wri"][1] * 0.72
xc = cols["chk"][0] + cols["chk"][1] * 0.5
arrow(ax, xc, Y0 + BH, xw, Y0 + BH, conn="arc3,rad=0.42", color="#c43c3b", lw=0.8, ms=6)
ax.text((xw + xc) / 2, Y0 + BH + 7.4, "hallmark found: regenerate", ha="center", va="center", fontsize=5.6,
        color="#b23a3a")

# group headers
def header(x0, x1, text):
    y = 94.0
    ax.plot([x0, x0, x1, x1], [y - 2.0, y - 1.0, y - 1.0, y - 2.0], color=AXIS, lw=0.6)
    ax.text((x0 + x1) / 2, y + 1.3, text, ha="center", va="center", fontsize=6.3, fontweight="bold", color=INK2)

header(cols["cls"][0], cols["gate"][0] + cols["gate"][1], "Decide the diagnosis d")
header(cols["det"][0], cols["chk"][0] + cols["chk"][1], "Explain d on the visible parts V")
header(cols["rep"][0], cols["rep"][0] + cols["rep"][1], "Output")

# ---------------------------------------------------------------- abstain branch
gx, gw = cols["gate"]
ab_w, ab_h, ab_y = 30.0, 13.5, 30.0
stage(gx + gw / 2 - ab_w / 2, ab_y, ab_w, ab_h, "gate", "Abstain", ["reply asks for close-ups of", "leaves, stems, roots, soil"])
arrow(ax, gx + gw / 2, Y0, gx + gw / 2, ab_y + ab_h + 0.2, **A)
ax.text(gx + gw / 2 + 1.4, (Y0 + ab_y + ab_h) / 2, "no", ha="left", va="center", fontsize=5.6, color=INK2)

# ---------------------------------------------------------------- knowledge row
KY, KH = 4.0, 19.0
x, w = cols["det"]
stage(x, KY, w, KH, "data", "Follow-up", ["question q, typed", "in the chat panel"])
x, w = cols["wri"]
stage(x, KY, w, KH, "knowledge", "Question router", ["keywords pick one of", "seven answer types"])
x, w = cols["chk"]
stage(x, KY, w, KH, "knowledge", "Knowledge base", ["symptoms, causes,", "treatments and", "extension sources", "for 7 conditions"])
x, w = cols["rep"]
stage(x, KY, w, KH, "output", "Answer", ["the matching text", "and its sources;", "no model is rerun"])
KC = KY + KH / 2
for (a, b) in [("det", "wri"), ("wri", "chk"), ("chk", "rep")]:
    x0, x1 = right(a, b)
    arrow(ax, x0 + 0.4, KC, x1 - 0.2, KC, **A)

# knowledge base supplies the hallmark phrases to the check
xk = cols["chk"][0] + cols["chk"][1] / 2
arrow(ax, xk, KY + KH, xk, Y0 - 0.2, color="#c98500", lw=0.8, ms=6)
tag(xk, (KY + KH + Y0) / 2, "hallmark\nphrases", color="#9a6b00")
# a question sent with a photograph goes into the writer's prompt
xr = cols["wri"][0] + cols["wri"][1] / 2
arrow(ax, xr, KY + KH, xr, Y0 - 0.2, **D)
tag(xr, (KY + KH + Y0) / 2, "question sent\nwith a photo", color=MUTED)

# ---------------------------------------------------------------- legend
lx, ly = 3.0, 18.0
ax.plot([lx, lx + 7], [ly, ly], color=INK2, lw=0.8)
ax.text(lx + 9, ly, "every request", va="center", fontsize=5.6, color=INK2)
ax.plot([lx, lx + 7], [ly - 4.2, ly - 4.2], color=MUTED, lw=0.7, ls=(0, (2.5, 1.5)))
ax.text(lx + 9, ly - 4.2, "only some requests", va="center", fontsize=5.6, color=INK2)
ax.text(lx, ly - 8.6, "v2:", va="center", fontsize=V_SIZE, color=V2_COLOR, fontweight="bold")
ax.text(lx + 9, ly - 8.6, "model replaced in version 2", va="center", fontsize=5.6, color=INK2)

save(fig, "fig_system_architecture")
print("ok")
