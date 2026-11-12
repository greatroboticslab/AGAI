"""Fig. Training pipelines of the three models (lanes)."""
import io
import json
import sys
import zipfile
from PIL import Image, ImageOps

sys.path.insert(0, "D:/agai-paper/draft/figsrc")
from agstyle import *  # noqa

setup(7.0)
cand = json.load(open("D:/agai-paper/draft/notes/example_candidates.json"))
W, H = 190.0, 112.0
fig, ax = canvas(W * MM, H * MM)
LX, LW = 1.0, 188.0
lanes = [("Condition\nclassifier", 75.0, 35.5), ("Plant-part\ndetector", 38.0, 35.0), ("Report\nwriter", 1.0, 35.0)]
for name, y, h in lanes:
    lane(ax, LX, y, LW, h, name, label_w=7.0, size=6.3)
X0 = 10.0


def thumb(img, x, y, s):
    img = ImageOps.exif_transpose(img).convert("RGB")
    img = square_crop(img).resize((200, 200), Image.LANCZOS)
    image_tile(ax, img, x, y, s, s, radius=0.8, border=AXIS, lw=0.4, zorder=3)


def tag(x, y, text, color):
    ax.text(x, y, text, fontsize=5.2, color="white", ha="center", va="center", zorder=6,
            bbox=dict(boxstyle="round,pad=0.22,rounding_size=0.3", fc=color, ec="none"))


A = dict(color=INK2, lw=0.8, ms=6)
FROZEN, TRAIN = "#898781", "#e34948"

# ---------------- classifier lane (y 75 to 110.5)
yc = 79.0
thumb(Image.open(cand["frost"][8]), X0, yc + 13.0, 13.0)
thumb(Image.open(cand["gray_mold"][6]), X0 + 3.0, yc + 9.0, 13.0)
ax.text(X0 + 8.0, yc + 5.5, "split v1 training", ha="center", fontsize=5.4, color=INK2)
ax.text(X0 + 8.0, yc + 2.8, "(1017 photographs)", ha="center", fontsize=5.0, color=MUTED)
box(ax, 30.0, yc, 31.0, 27.0, "data", "Online augmentation", ["90-100% crop to 256 px", "rotate +/-3 deg, flip",
    "perspective, color jitter", "blur, autocontrast", "random erasing (p 0.15)"], title_size=6.3, body_size=5.4)
arrow(ax, 26.5, yc + 13.5, 30.0, yc + 13.5, **A)
box(ax, 65.0, yc, 30.0, 27.0, "classify", "Two-phase fine-tuning", ["head only: 5 epochs, lr 1e-3",
    "all stages: 15 epochs,", "lr 1e-4 x 0.25 / 0.5 / 0.75 / 1", "weighted CE, smoothing 0.05",
    "AdamW, AMP, clip 1.0"], title_size=6.3, body_size=5.4)
arrow(ax, 61.0, yc + 13.5, 65.0, yc + 13.5, **A)
box(ax, 99.0, yc, 27.0, 27.0, "gate", "Select epoch", ["best accuracy on", "the 161 validation", "photographs"],
    title_size=6.3, body_size=5.4)
arrow(ax, 95.0, yc + 13.5, 99.0, yc + 13.5, **A)
box(ax, 130.0, yc, 27.0, 27.0, "gate", "Fit temperature T", ["L-BFGS on the", "validation negative", "log-likelihood"],
    title_size=6.1, body_size=5.4)
arrow(ax, 126.0, yc + 13.5, 130.0, yc + 13.5, **A)
box(ax, 161.0, yc, 27.0, 27.0, "output", "Deployed", ["ResNet-50 weights", "temperature T", r"gate thresholds $\tau_c$",
    "(set by hand, 2025)"], title_size=6.3, body_size=5.4)
arrow(ax, 157.0, yc + 13.5, 161.0, yc + 13.5, **A)
tag(80.0, yc + 29.6, "ImageNet init.", "#2a78d6")

# ---------------- detector lane (y 38 to 73)
yd = 41.5
z = zipfile.ZipFile("D:/agai-paper/labdata/LatestStrawberry archive.coco-segmentation.zip")
coco = json.loads(z.read("train/_annotations.coco.json"))
ann_c = json.load(open("D:/agai-paper/draft/notes/ann_cands.json"))
meta = [i for i in coco["images"] if i["id"] == ann_c[12]["iid"]][0]
thumb(Image.open(io.BytesIO(z.read("train/" + meta["file_name"]))), X0 + 1.5, yd + 11.0, 14.0)
ax.text(X0 + 8.5, yd + 7.8, "845 train images", ha="center", fontsize=5.4, color=INK2)
ax.text(X0 + 8.5, yd + 5.1, "(split agrees w/ v1)", ha="center", fontsize=5.0, color=MUTED)
box(ax, 30.0, yd, 31.0, 27.0, "detect", "RF-DETR Small", ["DINOv2 backbone,", "windowed attention", "multi-scale projector",
    "deformable decoder", "13 x 300 queries"], title_size=6.3, body_size=5.4)
arrow(ax, 26.5, yd + 13.5, 30.0, yd + 13.5, **A)
box(ax, 65.0, yd, 30.0, 27.0, "detect", "Set prediction", ["Hungarian matching", "of queries to parts", "class loss (IA-BCE)",
    "L1 + GIoU box loss", "no NMS at inference"], title_size=6.3, body_size=5.4)
arrow(ax, 61.0, yd + 13.5, 65.0, yd + 13.5, **A)
box(ax, 99.0, yd, 27.0, 27.0, "data", "Optimization", ["100 epochs, batch 8", "x 2 accumulation", "AdamW, lr 1e-4", "warm-up + cosine",
    "EMA 0.993"], title_size=6.3, body_size=5.4)
arrow(ax, 95.0, yd + 13.5, 99.0, yd + 13.5, **A)
box(ax, 130.0, yd, 27.0, 27.0, "gate", "Select checkpoint", ["best AP50 of the", "EMA weights on the", "67 validation", "images; scored on", "163 test images"],
    title_size=6.1, body_size=5.4)
arrow(ax, 126.0, yd + 13.5, 130.0, yd + 13.5, **A)
box(ax, 161.0, yd, 27.0, 27.0, "output", "Deployed", ["detector weights", "per-part thresholds", "0.30 to 0.50", "visible-part set V"],
    title_size=6.3, body_size=5.4)
arrow(ax, 157.0, yd + 13.5, 161.0, yd + 13.5, **A)
tag(45.5, yd + 29.6, "COCO init.", "#1baf7a")

# ---------------- report writer lane (y 1 to 36)
yr = 4.5
thumb(Image.open(cand["drought"][15]), X0 + 1.5, yr + 11.0, 14.0)
ax.text(X0 + 8.5, yr + 7.8, "image + label", ha="center", fontsize=5.4, color=INK2)
box(ax, 30.0, yr, 31.0, 27.0, "knowledge", "Report authoring", ["OpenAI o3 (vision)", "fixed label, visual-only", "2159 reports; 1364",
    "train, 212 validation,", "test photos dropped"], title_size=6.3, body_size=5.4)
arrow(ax, 26.5, yr + 13.5, 30.0, yr + 13.5, **A)
box(ax, 65.0, yr, 30.0, 27.0, "language", "MiniGPT-v2", ["stage-2 checkpoint", "EVA ViT-g/14 (frozen)", "LLaMA-2-7B (8-bit,", "frozen) + LoRA r 64",
    "+ linear projection"], title_size=6.3, body_size=5.4)
arrow(ax, 61.0, yr + 13.5, 65.0, yr + 13.5, **A)
box(ax, 99.0, yr, 27.0, 27.0, "data", "Optimization", ["10 epochs x 2000 it.", "1 GPU, batch 2,", "accumulation 16", "AdamW, cosine", "1e-5 to 5e-6"],
    title_size=6.3, body_size=5.4)
arrow(ax, 95.0, yr + 13.5, 99.0, yr + 13.5, **A)
box(ax, 130.0, yr, 27.0, 27.0, "gate", "Select checkpoint", ["lowest loss on the", "212 validation", "reports; prompt =", "inference system", "prompt"],
    title_size=6.1, body_size=5.4)
arrow(ax, 126.0, yr + 13.5, 130.0, yr + 13.5, **A)
box(ax, 161.0, yr, 27.0, 27.0, "output", "Deployed", ["LoRA adapters", "projection layer", "system prompt with", "d and V (Methods)"],
    title_size=6.3, body_size=5.4)
arrow(ax, 157.0, yr + 13.5, 161.0, yr + 13.5, **A)
tag(80.0, yr + 29.6, "LoRA + projection trained", TRAIN)

save(fig, "fig_training_pipelines")
print("ok")
