"""Shared figure style for the AGAI paper (Computers and Electronics in Agriculture).

Final print widths (Elsevier): single column 90 mm, 1.5 column 140 mm, double column 190 mm.
Text in figures: Arial, 6.5-8 pt at final size. Colors: class palette validated for
colour-vision deficiency on adjacent pairs in CLASS_ORDER (dataviz validator, light mode,
white surface): worst adjacent CVD dE 9.1, normal-vision dE 22.9.
"""
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Circle
import numpy as np

MM = 1 / 25.4
W1, W15, W2 = 90 * MM, 140 * MM, 190 * MM

CLASS_ORDER = ["healthy", "drought", "frost", "overwatered", "root_rot", "gray_mold", "white_mold"]
CLASS_LABEL = {"healthy": "Healthy", "drought": "Drought stress", "frost": "Frost injury",
               "overwatered": "Overwatering", "root_rot": "Root rot", "gray_mold": "Gray mold",
               "white_mold": "White mold"}
CLASS_COLOR = {"healthy": "#008300", "drought": "#eda100", "frost": "#1baf7a",
               "overwatered": "#2a78d6", "root_rot": "#e34948", "gray_mold": "#6250d6",
               "white_mold": "#e87ba4"}

INK = "#1b1b1b"        # primary text
INK2 = "#52514e"       # secondary text
MUTED = "#898781"      # axis labels, notes
GRID = "#e1e0d9"       # hairline grid
AXIS = "#c3c2b7"       # baseline / axis

# Diagram roles: (fill tint, edge/accent). Text always stays in INK/INK2.
ROLE = {
    "data":      ("#f3f2ee", "#898781"),
    "classify":  ("#e7f0fb", "#2a78d6"),
    "detect":    ("#e2f5ee", "#1baf7a"),
    "language":  ("#ecebfa", "#6250d6"),
    "knowledge": ("#fdf2d9", "#c98500"),
    "output":    ("#e3f1e3", "#008300"),
    "gate":      ("#fbe7e6", "#e34948"),
    "neutral":   ("#ffffff", "#898781"),
}


def setup(base=7.0):
    mpl.rcParams.update({
        "font.family": "Arial",
        "font.size": base,
        "axes.titlesize": base + 0.5,
        "axes.labelsize": base,
        "xtick.labelsize": base - 0.5,
        "ytick.labelsize": base - 0.5,
        "legend.fontsize": base - 0.5,
        "axes.edgecolor": AXIS,
        "axes.labelcolor": INK,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "grid.color": GRID,
        "grid.linewidth": 0.5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.dpi": 600,
        "figure.dpi": 150,
        "text.color": INK,
        "mathtext.fontset": "custom",
        "mathtext.rm": "Arial",
        "mathtext.it": "Arial:italic",
        "mathtext.bf": "Arial:bold",
    })


def canvas(width_in, height_in):
    """A figure with one axes spanning it, in data units = millimetres."""
    fig = plt.figure(figsize=(width_in, height_in))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, width_in / MM)
    ax.set_ylim(0, height_in / MM)
    ax.axis("off")
    return fig, ax


def box(ax, x, y, w, h, role="neutral", title=None, lines=(), r=1.2, lw=0.7,
        title_size=None, body_size=None, align="center", fill=None, edge=None, title_color=INK,
        zorder=2, dashed=False):
    f, e = ROLE[role]
    f = fill or f
    e = edge or e
    p = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                       fc=f, ec=e, lw=lw, zorder=zorder, linestyle=(0, (2.2, 1.4)) if dashed else "-")
    ax.add_patch(p)
    base = mpl.rcParams["font.size"]
    ts = title_size or base
    bs = body_size or base - 0.8
    n = len(lines)
    gap = bs * 0.42  # mm per text line roughly (pt -> mm ~0.35, plus leading)
    block = (ts * 0.40 if title else 0) + n * gap
    cy = y + h / 2 + block / 2
    tx = x + w / 2 if align == "center" else x + 1.6
    ha = "center" if align == "center" else "left"
    if title:
        ax.text(tx, cy - ts * 0.20, title, ha=ha, va="center", fontsize=ts, fontweight="bold",
                color=title_color, zorder=zorder + 1)
        cy -= ts * 0.40
    for i, ln in enumerate(lines):
        ax.text(tx, cy - gap * (i + 0.5), ln, ha=ha, va="center", fontsize=bs, color=INK2,
                zorder=zorder + 1)
    return p


def arrow(ax, x0, y0, x1, y1, color=INK2, lw=0.8, style="-|>", ms=6, conn="arc3", zorder=3,
          dashed=False, shrink=0.0):
    a = FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style, mutation_scale=ms, color=color,
                        lw=lw, connectionstyle=conn, zorder=zorder, shrinkA=shrink, shrinkB=shrink,
                        linestyle=(0, (2.5, 1.5)) if dashed else "-")
    ax.add_patch(a)
    return a


def lane(ax, x, y, w, h, label, fill="#f7f7f5", edge="#e1e0d9", label_w=6.0, size=None):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.0",
                                fc=fill, ec=edge, lw=0.6, zorder=0))
    ax.add_patch(FancyBboxPatch((x, y), label_w, h, boxstyle="round,pad=0,rounding_size=1.0",
                                fc=edge, ec=edge, lw=0.6, zorder=0.5))
    ax.text(x + label_w / 2, y + h / 2, label, rotation=90, ha="center", va="center",
            fontsize=size or mpl.rcParams["font.size"] - 0.5, fontweight="bold", color=INK2)


def image_tile(ax, img, x, y, w, h, radius=1.0, border="#ffffff", lw=0.0, zorder=2):
    """Draw a PIL image into a mm box with rounded clipping."""
    arr = np.asarray(img)
    im = ax.imshow(arr, extent=(x, x + w, y, y + h), zorder=zorder, interpolation="none")
    clip = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={radius}",
                          transform=ax.transData, fc="none", ec="none")
    ax.add_patch(clip)
    im.set_clip_path(clip)
    if lw:
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={radius}",
                                    fc="none", ec=border, lw=lw, zorder=zorder + 0.5))
    return im


def square_crop(img, cx=0.5, cy=0.5, frac=1.0):
    """Crop a square around (cx, cy) in relative coords; frac = side / short side."""
    w, h = img.size
    s = int(min(w, h) * frac)
    x0 = int(min(max(cx * w - s / 2, 0), w - s))
    y0 = int(min(max(cy * h - s / 2, 0), h - s))
    return img.crop((x0, y0, x0 + s, y0 + s))


def panel_label(ax, x, y, text, size=None):
    ax.text(x, y, text, fontsize=size or mpl.rcParams["font.size"] + 0.5, fontweight="bold",
            ha="left", va="top", color=INK)


def save(fig, stem, outdir="D:/agai-paper/draft/figures"):
    fig.savefig(f"{outdir}/{stem}.pdf")
    fig.savefig(f"{outdir}/{stem}.png", dpi=300)
    plt.close(fig)
