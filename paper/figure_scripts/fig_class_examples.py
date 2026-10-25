"""Fig. Representative images of the seven plant-health conditions (two views per class)."""
import json
import sys
from PIL import Image, ImageOps
from matplotlib.patches import FancyBboxPatch, Rectangle, Circle

sys.path.insert(0, "D:/agai-paper/draft/figsrc")
from agstyle import *  # noqa

setup(7.0)
cand = json.load(open("D:/agai-paper/draft/notes/example_candidates.json"))

# (class, candidate index, crop cx, cy, frac, symptom text, markers [(x, y, r) in tile-relative coords])
TOP = [
    ("healthy", 13, 0.50, 0.50, 1.00, "Vigorous canopy with\nflowers and fruit", []),
    ("drought", 2, 0.50, 0.42, 1.00, "Wilting, drooping\nfoliage", []),
    ("frost", 4, 0.55, 0.50, 0.95, "Cold-injured leaflets\non a young plant", []),
    ("overwatered", 11, 0.42, 0.45, 0.95, "Collapsed crown in\nsaturated substrate", []),
    ("root_rot", 13, 0.50, 0.50, 1.00, "Reddened, necrotic\nleaves; decayed roots", []),
    ("gray_mold", 13, 0.50, 0.50, 1.00, "Rotted fruit hanging\nfrom the canopy", []),
    ("white_mold", 6, 0.50, 0.50, 1.00, "White fungal growth\non harvested fruit", []),
]
BOTTOM = [
    ("healthy", 6, 0.50, 0.50, 1.00, "Ripe fruit with\ngreen calyx", []),
    ("drought", 15, 0.50, 0.35, 1.00, "Desiccated, curled\nleaf blades", []),
    ("frost", 8, 0.47, 0.45, 0.80, "Blackened flower\ncenter", []),
    ("overwatered", 6, 0.50, 0.45, 1.00, "Wilted seedling on\nwaterlogged soil", []),
    ("root_rot", 7, 0.50, 0.45, 1.00, "Internal crown\ndiscoloration", [(0.31, 0.47, 0.12), (0.66, 0.50, 0.12)]),
    ("gray_mold", 6, 0.50, 0.50, 1.00, "Gray sporulation on\nfruit surface", []),
    ("white_mold", 2, 0.50, 0.45, 1.00, "Cottony white\nmycelium", []),
]

W = 190.0
left, right, gap = 8.0, 0.8, 1.4
t = (W - left - right - 6 * gap) / 7
hdr_g, hdr_c = 4.2, 5.2
H = hdr_g + hdr_c + 2 * t + gap + 1.0
fig, ax = canvas(W * MM, H * MM)

col_x = [left + i * (t + gap) for i in range(7)]
y_top_row = H - hdr_g - hdr_c - t
y_bot_row = y_top_row - gap - t

# group headers
groups = [("Reference", 0, 0), ("Abiotic stress", 1, 3), ("Biotic disorder", 4, 6)]
for name, a, b in groups:
    x0, x1 = col_x[a], col_x[b] + t
    yy = H - hdr_g + 0.6
    ax.plot([x0 + 0.5, x1 - 0.5], [yy, yy], color=AXIS, lw=0.6)
    ax.text((x0 + x1) / 2, yy + 1.6, name, ha="center", va="center", fontsize=7, color=INK2,
            fontweight="bold")

for i, cls in enumerate(CLASS_ORDER):
    x = col_x[i]
    yb = H - hdr_g - hdr_c
    ax.add_patch(Rectangle((x, yb + 0.9), t, 0.9, color=CLASS_COLOR[cls], lw=0))
    ax.text(x + t / 2, yb + 3.3, CLASS_LABEL[cls], ha="center", va="center", fontsize=7, color=INK,
            fontweight="bold")

ax.text(left / 2 - 0.4, y_top_row + t / 2, "Plant / canopy", rotation=90, ha="center", va="center",
        fontsize=6.5, color=INK2)
ax.text(left / 2 - 0.4, y_bot_row + t / 2, "Organ close-up", rotation=90, ha="center", va="center",
        fontsize=6.5, color=INK2)


def tile(spec, x, y):
    cls, idx, cx, cy, frac, text, marks = spec
    img = Image.open(cand[cls][idx])
    img = ImageOps.exif_transpose(img).convert("RGB")
    img = square_crop(img, cx, cy, frac).resize((400, 400), Image.LANCZOS)
    image_tile(ax, img, x, y, t, t, radius=1.0, border=AXIS, lw=0.4)
    band = 6.2
    ax.add_patch(FancyBboxPatch((x, y), t, band, boxstyle="round,pad=0,rounding_size=1.0",
                                fc=(0.06, 0.06, 0.06, 0.62), ec="none", zorder=4))
    ax.add_patch(Rectangle((x, y + band - 1.0), t, 1.0, fc=(0.06, 0.06, 0.06, 0.62), ec="none", zorder=4))
    ax.text(x + t / 2, y + band / 2, text, ha="center", va="center", fontsize=5.6, color="white",
            zorder=5, linespacing=1.05)
    for (mx, my, mr) in marks:
        ax.add_patch(Circle((x + mx * t, y + (1 - my) * t), mr * t, fc="none", ec="white", lw=0.9,
                            ls=(0, (2.0, 1.2)), zorder=6))


for i in range(7):
    tile(TOP[i], col_x[i], y_top_row)
    tile(BOTTOM[i], col_x[i], y_bot_row)

save(fig, "fig_class_examples")
print("ok", round(W), "x", round(H), "mm")
