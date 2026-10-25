"""Fig. Report generation: (a) MiniGPT-v2 architecture, (b) instruction-data construction, (c) inference prompt + check."""
import json
import sys
from PIL import Image, ImageOps
from matplotlib.patches import Rectangle, FancyBboxPatch

sys.path.insert(0, "D:/agai-paper/draft/figsrc")
from agstyle import *  # noqa

setup(7.0)
cand = json.load(open("D:/agai-paper/draft/notes/example_candidates.json"))
W, H = 190.0, 104.0
fig, ax = canvas(W * MM, H * MM)
FROZEN, TRAIN = "#898781", "#e34948"


def tag(x, y, text, color):
    ax.text(x, y, text, fontsize=5.4, color="white", ha="center", va="center", zorder=6,
            bbox=dict(boxstyle="round,pad=0.25,rounding_size=0.3", fc=color, ec="none"))


def tokens(x, y, n, color, w=1.5, gap=0.35, h=3.2):
    for i in range(n):
        ax.add_patch(Rectangle((x + i * (w + gap), y), w, h, fc=color, ec="none", zorder=4))
    return x + n * (w + gap)


# ---------------- (a) architecture
panel_label(ax, 1.0, H - 1.0, "(a) MiniGPT-v2 report writer")
ya = H - 27.0
img = ImageOps.exif_transpose(Image.open(cand["gray_mold"][6])).convert("RGB")
img = square_crop(img).resize((220, 220), Image.LANCZOS)
image_tile(ax, img, 2.0, ya - 1.0, 18.0, 18.0, radius=0.8, border=AXIS, lw=0.4)
ax.text(11.0, ya - 3.2, "448 x 448 input", ha="center", fontsize=5.5, color=INK2)
box(ax, 24.0, ya, 30.0, 16.0, "classify", "EVA ViT-g/14", ["image encoder", "32 x 32 patch grid"], title_size=6.6,
    body_size=5.7)
tag(39.0, ya + 18.2, "frozen", FROZEN)
arrow(ax, 20.0, ya + 8.0, 24.0, ya + 8.0)
box(ax, 58.0, ya, 27.0, 16.0, "data", "Merge 4 neighbors", ["1024 to 256 tokens", "dim 1408 to 5632"], title_size=6.4,
    body_size=5.7)
arrow(ax, 54.0, ya + 8.0, 58.0, ya + 8.0)
box(ax, 89.0, ya, 22.0, 16.0, "gate", "Linear", ["projection", "5632 to 4096"], title_size=6.6, body_size=5.7)
tag(100.0, ya + 18.2, "trained", TRAIN)
arrow(ax, 85.0, ya + 8.0, 89.0, ya + 8.0)
# token row into the LLM
tx0 = 115.0
ax.text(tx0, ya + 17.6, "input sequence", fontsize=5.6, color=INK2, fontweight="bold")
x1 = tokens(tx0, ya + 9.0, 6, "#2a78d6")
ax.text(tx0 + 5.5, ya + 13.6, "visual (256)", fontsize=5.2, color=INK2, ha="center")
x2 = tokens(x1 + 1.2, ya + 9.0, 9, "#c3c2b7")
ax.text((x1 + x2) / 2 + 0.6, ya + 13.6, "prompt text", fontsize=5.2, color=INK2, ha="center")
arrow(ax, 111.0, ya + 8.0, tx0 - 0.6, ya + 10.6)
box(ax, 115.0, ya - 10.0, 52.0, 14.0, "language", "LLaMA-2-7B-chat (8-bit)", [r"LoRA on q, v of 32 layers: $W = W_0 + \frac{\alpha}{r}BA$",
    r"$r = 64$, $\alpha = 128$; 33.6 M trainable"], title_size=6.6, body_size=5.7)
tag(158.0, ya + 4.2, "LoRA trained", TRAIN)
arrow(ax, 141.0, ya + 9.0, 141.0, ya + 4.0)
ax.text(171.0, ya - 1.6, "report", fontsize=5.8, color=INK2, ha="left")
tokens(171.0, ya - 7.0, 5, "#008300")
arrow(ax, 167.0, ya - 3.0, 171.0, ya - 5.4)

# ---------------- (b) instruction data
yb = 6.0
panel_label(ax, 1.0, 64.5, "(b) Instruction data (2159 image-report pairs)")
img2 = ImageOps.exif_transpose(Image.open(cand["drought"][15])).convert("RGB")
img2 = square_crop(img2, 0.5, 0.35).resize((200, 200), Image.LANCZOS)
image_tile(ax, img2, 2.0, 41.0, 16.0, 16.0, radius=0.8, border=AXIS, lw=0.4)
box(ax, 2.0, 32.5, 16.0, 6.5, "output", "label", ["Drought"], title_size=5.8, body_size=5.6)
box(ax, 22.0, 32.5, 40.0, 25.0, "neutral", None, [], fill="#fbfbf9", edge="#c3c2b7")
ax.text(23.5, 55.0, "Vision model (OpenAI o3) rules", fontsize=6.2, fontweight="bold", va="center")
rules = ["use the given label exactly", "describe only what is visible", "3 to 6 visual clues, likely cause",
         "2 or 3 steps with measurements", "2 or 3 checks for the next 7 days", "prefer cultural measures"]
for i, r in enumerate(rules):
    ax.text(23.5, 51.2 - i * 3.0, "\u2022 " + r, fontsize=5.4, color=INK2, va="center", family="DejaVu Sans")
arrow(ax, 18.0, 49.0, 22.0, 49.0)
arrow(ax, 18.0, 35.7, 22.0, 38.0)
box(ax, 66.0, 32.5, 30.0, 25.0, "output", None, [], fill="#f4faf4", edge="#008300")
ax.text(67.5, 55.0, "Report (excerpt)", fontsize=6.2, fontweight="bold", va="center")
excerpt = ["Diagnosis: Drought. ... several", "stems droop toward the straw,", "and surrounding soil and mulch",
           "appear dry and powdery. ...", "irrigate to wet the root zone", "5-7 cm (2-3 in) deep ...", "151 to 296 words (mean 207)"]
for i, e in enumerate(excerpt):
    ax.text(67.5, 51.2 - i * 2.9, e, fontsize=5.3, color=INK2 if i < 6 else MUTED, va="center",
            style="italic" if i < 6 else "normal")
arrow(ax, 62.0, 45.0, 66.0, 45.0)
# class distribution of the 2,159 records
counts = {"healthy": 325, "drought": 300, "frost": 110, "overwatered": 298, "root_rot": 293, "gray_mold": 607,
          "white_mold": 226}
bx0, by0 = 4.0, 9.0
ax.text(bx0 - 2.0, 27.5, "Reports per condition", fontsize=6.0, fontweight="bold", va="center")
for i, c in enumerate(CLASS_ORDER):
    y = 23.5 - i * 2.75
    ax.add_patch(Rectangle((bx0 + 18.0, y - 0.9), counts[c] / 607 * 50.0, 1.8, fc=CLASS_COLOR[c], ec="none"))
    ax.text(bx0 + 17.0, y, CLASS_LABEL[c], fontsize=5.3, color=INK2, ha="right", va="center")
    ax.text(bx0 + 19.0 + counts[c] / 607 * 50.0, y, str(counts[c]), fontsize=5.3, color=INK2, va="center")
ax.text(bx0 + 18.0, 2.6, "counts include 620 augmented copies of real images", fontsize=5.2, color=MUTED, ha="left")

# ---------------- (c) inference prompt and check
panel_label(ax, 100.0, 64.5, "(c) Prompt at inference and consistency check")
box(ax, 100.0, 22.0, 52.0, 37.5, "neutral", None, [], fill="#fbfbf9", edge="#c3c2b7")
ax.text(101.5, 57.0, "System prompt (abridged)", fontsize=6.2, fontweight="bold", va="center")
prompt = [("The diagnosis has already been determined:", INK2), ("  Gray Mold", "#6250d6"),
          ("VISIBLE (describe these): fruit, leaf, stem", INK2),
          ("NOT VISIBLE (do NOT mention these at all):", INK2), ("  flower, root, soil", "#e34948"),
          ("You must ONLY describe symptoms on the", INK2), ("VISIBLE parts listed above.", INK2),
          ("The green sepals on top of a strawberry", INK2), ("are the calyx, part of the fruit.", INK2),
          ("1) Diagnosis 2) Cues 3) Recommendation", INK2)]
for i, (t, col) in enumerate(prompt):
    ax.text(101.5, 53.2 - i * 3.05, t, fontsize=5.25, color=col, va="center", family="DejaVu Sans Mono")
ax.text(126.0, 18.6, "d and V from the classifier and the detector", fontsize=5.2, color=MUTED, ha="center")
arrow(ax, 152.0, 52.5, 156.0, 52.5)
box(ax, 156.0, 46.0, 32.0, 13.5, "language", "MiniGPT-v2", ["temperature 0.2,", "top-p 0.9"], title_size=6.4, body_size=5.6)
arrow(ax, 172.0, 46.0, 172.0, 42.6)
box(ax, 156.0, 22.0, 32.0, 20.6, "gate", "Hallmark check", ["18 phrases of the other", "conditions, e.g. \"gray fuzz\",",
    "\"black eye\", \"red stele\"", "flagged: regenerate once;", "still flagged: warning"], title_size=6.4, body_size=5.4)
arrow(ax, 160.0, 46.0, 160.0, 42.6, color="#e34948", style="<|-", lw=0.7)
ax.text(161.0, 44.3, "retry", fontsize=5.0, color="#b23a3a")
arrow(ax, 172.0, 22.0, 172.0, 17.0)
box(ax, 156.0, 6.0, 32.0, 11.0, "output", "Report shown", ["with confidence badge", "and part boxes"], title_size=6.2,
    body_size=5.4)

save(fig, "fig_report_generation")
print("ok")
