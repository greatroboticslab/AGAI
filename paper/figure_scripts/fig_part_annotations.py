"""Fig. Plant-part instance annotations (ground truth polygons) on four example images."""
import io
import json
import sys
import zipfile
from PIL import Image, ImageDraw
from matplotlib.patches import Rectangle

sys.path.insert(0, "D:/agai-paper/draft/figsrc")
from agstyle import *  # noqa

setup(7.0)
PARTS = ["leaf", "fruit", "stem", "flower", "soil", "root"]
PART_COLOR = {"leaf": "#2a78d6", "fruit": "#eda100", "stem": "#e87ba4", "flower": "#6250d6",
              "soil": "#eb6834", "root": "#1baf7a"}
z = zipfile.ZipFile("D:/agai-paper/labdata/LatestStrawberry archive.coco-segmentation.zip")
coco = json.loads(z.read("train/_annotations.coco.json"))
cats = {c["id"]: c["name"] for c in coco["categories"]}
imgs = {i["id"]: i for i in coco["images"]}
cand = json.load(open("D:/agai-paper/draft/notes/ann_cands.json"))
PICK = [4, 6, 3, 12]


def hex2rgb(h):
    return tuple(int(h[i:i + 2], 16) for i in (1, 3, 5))


def rle_counts(sv):
    """Decode a COCO compressed RLE counts string into run lengths."""
    counts, p, m = [], 0, 0
    while p < len(sv):
        x, k, more = 0, 0, 1
        while more:
            c = ord(sv[p]) - 48
            x |= (c & 0x1f) << (5 * k)
            more = c & 0x20
            p += 1
            k += 1
            if not more and (c & 0x10):
                x |= -1 << (5 * k)
        if m > 2:
            x += counts[m - 2]
        counts.append(x)
        m += 1
    return counts


def rle_mask(seg):
    h, w = int(seg["size"][0]), int(seg["size"][1])
    flat = np.zeros(h * w, dtype=np.uint8)
    pos, val = 0, 0
    for c in rle_counts(seg["counts"]):
        if val:
            flat[pos:pos + c] = 1
        pos += c
        val ^= 1
    return flat.reshape((h, w), order="F")


def render(iid, size=900):
    import cv2
    meta = imgs[iid]
    im = Image.open(io.BytesIO(z.read("train/" + meta["file_name"]))).convert("RGB")
    W0, H0 = im.size
    s = size / max(W0, H0)
    base = np.asarray(im.resize((int(W0 * s), int(H0 * s)), Image.LANCZOS)).astype(np.float32)
    anns = [a for a in coco["annotations"] if a["image_id"] == iid]
    anns.sort(key=lambda a: -float(a["area"]))
    counts = {}
    out = base.copy()
    edges = []
    for a in anns:
        p = cats[a["category_id"]]
        if p not in PART_COLOR:
            continue
        counts[p] = counts.get(p, 0) + 1
        mk = rle_mask(a["segmentation"])
        mk = cv2.resize(mk, (base.shape[1], base.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
        col = np.array(hex2rgb(PART_COLOR[p]), np.float32)
        alpha = 0.22 if p == "soil" else 0.38
        out[mk] = out[mk] * (1 - alpha) + col * alpha
        cs, _ = cv2.findContours(mk.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        edges.append((cs, tuple(int(v) for v in col)))
    out = out.clip(0, 255).astype(np.uint8)
    for cs, col in edges:
        cv2.drawContours(out, cs, -1, (255, 255, 255), 3, lineType=cv2.LINE_AA)
        cv2.drawContours(out, cs, -1, col, 2, lineType=cv2.LINE_AA)
    return Image.fromarray(out), counts


W = 190.0
pad, gap = 0.5, 1.6
n = len(PICK)
t = (W - 2 * pad - (n - 1) * gap) / n
leg_h = 6.5
H = leg_h + t + 5.5
fig, ax = canvas(W * MM, H * MM)
for k, ci in enumerate(PICK):
    iid = cand[ci]["iid"]
    img, counts = render(iid, size=620)
    w, h = img.size
    side = min(w, h)
    img = img.crop(((w - side) // 2, (h - side) // 2, (w - side) // 2 + side, (h - side) // 2 + side))
    x = pad + k * (t + gap)
    y = 5.0
    image_tile(ax, img, x, y, t, t, radius=1.0, border=AXIS, lw=0.4)
    label = f"({'abcd'[k]})  " + ", ".join(f"{counts[p]} {p}" for p in PARTS if p in counts)
    ax.text(x + 0.6, y - 1.0, label, ha="left", va="top", fontsize=6.0, color=INK2)

# legend row
lx = pad
ly = H - leg_h / 2
ax.text(lx, ly, "Plant parts:", ha="left", va="center", fontsize=7, color=INK, fontweight="bold")
lx += 17
for p in PARTS:
    ax.add_patch(Rectangle((lx, ly - 1.4), 2.8, 2.8, fc=PART_COLOR[p], ec="white", lw=0.5))
    ax.text(lx + 3.8, ly, p.capitalize(), ha="left", va="center", fontsize=7, color=INK)
    lx += 17.0
ax.text(W - pad, ly, "Ground-truth instance masks", ha="right", va="center",
        fontsize=6.3, color=MUTED)
save(fig, "fig_part_annotations")
print("ok", round(H, 1))
