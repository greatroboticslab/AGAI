"""Fig. AGAI system architecture as layered lanes (input, perception, decision and grounding,
report generation and knowledge, output)."""
import io
import json
import sys
import zipfile
import numpy as np
from PIL import Image, ImageOps
from matplotlib.patches import FancyBboxPatch, Polygon, Rectangle, Circle

sys.path.insert(0, "D:/agai-paper/draft/figsrc")
from agstyle import *  # noqa

setup(7.0)
P = json.load(open("D:/agai-paper/draft/figsrc/arch_params.json", encoding="utf-8"))

W, H = 190.0, 128.0
fig, ax = canvas(W * MM, H * MM)
LX, LW = 1.0, 188.0
lanes = [("Input", 107.5, 19.5), ("Perception", 83.5, 22.0), ("Decision and\ngrounding", 59.5, 22.0),
         ("Generation and\nknowledge", 31.5, 26.0), ("Output", 1.0, 28.5)]
for name, y, h in lanes:
    lane(ax, LX, y, LW, h, name, label_w=6.5, size=6.3)
X0 = 9.5  # content start


def thumb(path_or_img, x, y, s, crop=(0.5, 0.5, 1.0)):
    img = path_or_img if isinstance(path_or_img, Image.Image) else Image.open(path_or_img)
    img = ImageOps.exif_transpose(img).convert("RGB")
    img = square_crop(img, *crop).resize((260, 260), Image.LANCZOS)
    image_tile(ax, img, x, y, s, s, radius=0.8, border=AXIS, lw=0.4, zorder=3)


# ---------------- Input lane
cand = json.load(open("D:/agai-paper/draft/notes/example_candidates.json"))
thumb(cand["gray_mold"][6], X0 + 0.5, 109.3, 16.0)
box(ax, X0 + 18.5, 110.0, 52, 14.5, "data", "Plant photograph x",
    ["uploaded or captured in the web interface", "any framing: whole plant or single organ"], align="left")
box(ax, 139.5, 110.0, 48.5, 14.5, "data", "Web interface",
    ["upload or camera capture,", "box overlay and chat panel"], align="left")

# ---------------- Perception lane
bx, by, bw, bh = X0, 85.5, 86, 18.0
box(ax, bx, by, bw, bh, "classify", "Condition classifier: ResNet-50", [
    P["cls_line1"], P["cls_line2"], "output: calibrated probabilities p(c | x)"], align="left")
# tiny probability glyph
gx, gy = bx + bw - 17.5, by + 3.0
vals = [0.03, 0.02, 0.01, 0.02, 0.04, 0.83, 0.05]
for i, (c, v) in enumerate(zip(CLASS_ORDER, vals)):
    ax.add_patch(Rectangle((gx + i * 2.3, gy), 1.7, 0.4 + v * 11.0, fc=CLASS_COLOR[c], lw=0, zorder=4))
ax.plot([gx - 0.6, gx + 7 * 2.3], [gy, gy], color=INK2, lw=0.5, zorder=4)
ax.text(gx + 8.0, gy + 12.6, "p(c | x)", ha="center", va="center", fontsize=5.6, color=INK2, zorder=4)

dx, dy, dw, dh = 100.5, 85.5, 87.5, 18.0
box(ax, dx, dy, dw, dh, "detect", "Plant-part detector: RF-DETR Small", [
    "six parts: leaf, fruit, stem, flower, soil, root", P["det_line2"], "output: visible parts V with boxes"],
    align="left")
# detection thumbnail with boxes from the annotation set
z = zipfile.ZipFile("D:/agai-paper/labdata/LatestStrawberry archive.coco-segmentation.zip")
coco = json.loads(z.read("train/_annotations.coco.json"))
ann_c = json.load(open("D:/agai-paper/draft/notes/ann_cands.json"))
iid = ann_c[4]["iid"]
meta = [i for i in coco["images"] if i["id"] == iid][0]
im = Image.open(io.BytesIO(z.read("train/" + meta["file_name"]))).convert("RGB")
w0, h0 = im.size
side = min(w0, h0)
ox, oy = (w0 - side) / 2, (h0 - side) / 2
tsz, tx, ty = 15.0, dx + dw - 16.5, dy + 1.5
thumb(im, tx, ty, tsz)
PART_COLOR = {"leaf": "#2a78d6", "fruit": "#eda100", "stem": "#e87ba4", "flower": "#6250d6",
              "soil": "#eb6834", "root": "#1baf7a"}
cats = {c["id"]: c["name"] for c in coco["categories"]}
for a in coco["annotations"]:
    if a["image_id"] != iid:
        continue
    p = cats[a["category_id"]]
    if p not in ("fruit", "flower") and not (p == "leaf" and float(a["area"]) > 0.04 * side * side):
        continue
    x, y, bw_, bh_ = [float(v) for v in a["bbox"]]
    x0 = (x - ox) / side
    y0 = (y - oy) / side
    if x0 < 0 or y0 < 0 or x0 + bw_ / side > 1 or y0 + bh_ / side > 1:
        continue
    ax.add_patch(Rectangle((tx + x0 * tsz, ty + (1 - y0 - bh_ / side) * tsz), bw_ / side * tsz, bh_ / side * tsz,
                           fc="none", ec=PART_COLOR[p], lw=0.7, zorder=5))

# ---------------- Decision and grounding lane
gcx, gcy = X0 + 12.0, 70.5
dia = Polygon([[gcx - 12.0, gcy], [gcx, gcy + 8.6], [gcx + 12.0, gcy], [gcx, gcy - 8.6]], closed=True,
              fc=ROLE["gate"][0], ec=ROLE["gate"][1], lw=0.7, zorder=2)
ax.add_patch(dia)
ax.text(gcx, gcy + 1.7, "Acceptance", ha="center", va="center", fontsize=6.2, fontweight="bold", zorder=3)
ax.text(gcx, gcy - 1.6, r"$p_1 \geq \max(\tau_{c}, 0.5)$ ?", ha="center", va="center", fontsize=5.9, zorder=3)
box(ax, X0 + 28.0, 72.0, 34.5, 7.8, "output", "Diagnosis d accepted", ["label and calibrated confidence"],
    align="left", title_size=6.6, body_size=5.9)
box(ax, X0 + 28.0, 61.5, 34.5, 8.4, "gate", "Abstain", ["fixed reply asking for close-ups", "of leaves, stems, roots and soil"],
    align="left", title_size=6.6, body_size=5.9)
box(ax, 100.5, 61.5, 87.5, 18.3, "neutral", "Prompt builder", [
    "system prompt fixes the diagnosis d (never re-predicted),",
    "lists VISIBLE parts V and NOT VISIBLE parts (all others),",
    "and sets the format: diagnosis, visible cues, recommendations"], align="left", fill="#ffffff")

# ---------------- Generation and knowledge lane
kx, ky, kw, kh = X0, 33.5, 44.0, 22.0
box(ax, kx, ky, kw, kh, "knowledge", None, [])
ax.text(kx + 1.6, ky + kh - 2.6, "Disease knowledge base", fontsize=6.8, fontweight="bold", va="center")
ax.text(kx + 1.6, ky + kh - 5.6, P["kg_line"], fontsize=5.5, color=INK2, va="center")
nodes = {"gray mold": (kx + 7.0, ky + 8.6), "fruit": (kx + 19.5, ky + 12.2), "sporulation": (kx + 19.5, ky + 4.6),
         "remove fruit": (kx + 34.0, ky + 12.2), "source": (kx + 34.0, ky + 4.6)}
edges = [("gray mold", "fruit"), ("gray mold", "sporulation"), ("fruit", "remove fruit"),
         ("gray mold", "remove fruit"), ("remove fruit", "source"), ("sporulation", "fruit")]
for a_, b_ in edges:
    ax.plot([nodes[a_][0], nodes[b_][0]], [nodes[a_][1], nodes[b_][1]], color="#c98500", lw=0.6, zorder=3)
kinds = {"gray mold": "#6250d6", "fruit": "#eda100", "sporulation": "#52514e", "remove fruit": "#008300",
         "source": "#898781"}
lab_off = {"gray mold": (0.0, -2.6, "center"), "fruit": (0.0, 2.4, "center"), "sporulation": (0.0, -2.5, "center"),
           "remove fruit": (0.0, 2.4, "center"), "source": (0.0, -2.5, "center")}
for n_, (x_, y_) in nodes.items():
    ax.add_patch(Circle((x_, y_), 1.25, fc=kinds[n_], ec="white", lw=0.6, zorder=4))
    dx_, dy_, ha_ = lab_off[n_]
    ax.text(x_ + dx_, y_ + dy_, n_, ha=ha_, va="center", fontsize=5.0, color=INK2, zorder=4)

box(ax, kx + kw + 3.5, 33.5, 32.0, 22.0, "knowledge", "Question router", [P["ret_line1"], P["ret_line2"],
    "and returns the matching", "knowledge-base section"], align="left", title_size=6.8, body_size=5.8)

mx, my, mw, mh = 100.5, 33.5, 53.0, 22.0
box(ax, mx, my, mw, mh, "language", "Report writer: MiniGPT-v2", [
    P["vlm_line1"], P["vlm_line2"], P["vlm_line3"], "writes the report from the prompt"], align="left",
    title_size=6.8, body_size=5.8)
cx_, cy_, cw_, ch_ = 156.5, 33.5, 31.5, 22.0
box(ax, cx_, cy_, cw_, ch_, "gate", "Consistency check", ["mentions a hallmark of", "another condition?",
    "if so: regenerate", P["regen_line"]], align="left", title_size=6.8, body_size=5.8)

# ---------------- Output lane
fx, fw = X0, 42.0
box(ax, fx, 3.0, fw, 24.5, "output", "Follow-up Q&A", ["the user asks a question q;", "it is answered from the", "knowledge base with sources,", "without rerunning the models"],
    align="left", title_size=6.8, body_size=5.8)
rx, ry, rw, rh = 54.5, 3.0, 108.0, 24.5
box(ax, rx, ry, rw, rh, "output", None, [], fill="#ffffff", edge="#008300")
ax.text(rx + 2.0, ry + rh - 2.6, "Diagnostic report (illustrative layout)", fontsize=6.8, fontweight="bold",
        va="center", color=INK)
rows = [("Diagnosis", "Gray mold (confidence badge: green >= 0.90, yellow >= 0.70)"),
        ("Visible cues", "symptoms described only on the detected parts"),
        ("Recommendation", "management steps written by the fine-tuned report writer"),
        ("Follow-ups", "suggested questions; optional web resources")]
for i, (k_, v_) in enumerate(rows):
    yy = ry + rh - 7.6 - i * 4.6
    ax.text(rx + 2.0, yy, k_, fontsize=5.9, fontweight="bold", color=INK, va="center")
    ax.text(rx + 23.0, yy, v_, fontsize=5.9, color=INK2, va="center")
thumb(cand["gray_mold"][6], 167.5, 7.2, 18.0)
ax.add_patch(Rectangle((167.5 + 0.10 * 18, 7.2 + 0.06 * 18), 0.80 * 18, 0.66 * 18, fc="none", ec="#eda100", lw=0.9, zorder=5))
ax.text(167.5 + 0.12 * 18, 7.2 + 0.73 * 18, "fruit", fontsize=5.0, color="white", zorder=6, va="bottom",
        bbox=dict(boxstyle="square,pad=0.15", fc="#eda100", ec="none"))
ax.text(176.5, 4.6, "annotated image", ha="center", fontsize=5.6, color=INK2)

# ---------------- arrows
A = dict(color=INK2, lw=0.8, ms=6)
arrow(ax, X0 + 44, 110.0, X0 + 44, 103.6, **A)                         # image -> classifier
arrow(ax, X0 + 60, 110.0, 128, 103.6, conn="arc3,rad=-0.12", **A)       # image -> detector
arrow(ax, gcx, 85.5, gcx, gcy + 8.2, **A)                              # classifier -> gate
arrow(ax, gcx + 10.5, gcy + 0.6, X0 + 28.0, 75.9, **A)                 # gate -> accept
ax.text(gcx + 11.5, gcy + 3.6, "yes", fontsize=5.6, color=INK2)
arrow(ax, gcx + 5.0, gcy - 4.2, X0 + 28.0, 65.6, **A)                  # gate -> abstain
ax.text(gcx + 6.8, gcy - 6.5, "no", fontsize=5.6, color=INK2)
arrow(ax, X0 + 62.5, 75.9, 100.5, 75.9, **A)                           # diagnosis -> prompt
ax.text(X0 + 64.0, 77.2, "d", fontsize=6.0, color=INK2, style="italic")
arrow(ax, 144.0, 85.5, 144.0, 79.8, **A)                               # detector -> prompt
ax.text(145.3, 82.6, "V", fontsize=6.0, color=INK2, style="italic")
arrow(ax, kx + kw, 44.5, kx + kw + 3.5, 44.5, **A)                     # KG -> retriever
arrow(ax, kx + kw + 35.5, 50.0, 100.5, 66.0, conn="arc3,rad=0.15", dashed=True, color=MUTED, lw=0.7, ms=5)  # router -> prompt (typed questions)
ax.text(78.5, 59.6, "typed question only", fontsize=5.3, color=MUTED, ha="left")
ax.plot([kx + 30.0, kx + 30.0, 171.0, 171.0], [ky, 30.6, 30.6, 33.5], color="#c98500", lw=0.6, ls=(0, (2.0, 1.4)), zorder=1)
ax.text(120.0, 31.4, "hallmark phrases of each condition", fontsize=5.2, color="#9a6b00", ha="center")
arrow(ax, 127.0, 61.5, 127.0, 55.5, **A)                               # prompt -> MiniGPT
arrow(ax, mx + mw, 44.5, cx_, 44.5, **A)                               # MiniGPT -> check
arrow(ax, cx_ + 6.0, 55.5, mx + mw - 6.0, 55.5, conn="arc3,rad=0.9", color="#e34948", lw=0.8, ms=6)
ax.text(cx_ + 9.0, 58.2, "regenerate", fontsize=5.4, color="#b23a3a", ha="left", va="center")
arrow(ax, cx_ + 6.0, 33.5, rx + rw - 6.0, 27.5, **A)                   # check -> report
ax.text(cx_ + 7.2, 29.6, "pass", fontsize=5.6, color=INK2)
arrow(ax, X0 + 45.0, 61.5, rx + 1.5, 27.5, dashed=True, color=MUTED, lw=0.7, ms=5)  # abstain -> report
arrow(ax, kx + kw + 8.0, 33.5, fx + fw - 8.0, 27.5, dashed=True, color=MUTED, lw=0.7, ms=5)  # retriever -> Q&A

save(fig, "fig_system_architecture")
print("ok")
