"""Placeholder figures for results that wait on the clean cluster reruns.
usage: python fig_pending.py <stem> <height_mm> "<title>" "<line 1>" ["<line 2>" ...]
"""
import sys

sys.path.insert(0, "D:/agai-paper/draft/figsrc")
from agstyle import *  # noqa

setup(7.5)
stem, hmm, title, lines = sys.argv[1], float(sys.argv[2]), sys.argv[3], sys.argv[4:]
fig, ax = canvas(W2, hmm * MM)
box(ax, 1, 1, 188, hmm - 2, role="neutral", fill="#f6f6f4", edge="#c3c2b7", dashed=True, r=2.0)
ax.text(95, hmm / 2 + 2.5 * len(lines) / 2 + 2, "PENDING: " + title, ha="center", va="center", fontsize=9,
        fontweight="bold", color="#b23a3a")
for i, ln in enumerate(lines):
    ax.text(95, hmm / 2 + 2.5 * len(lines) / 2 - 3.2 - 4.2 * i, ln, ha="center", va="center", fontsize=7.2,
            color=INK2)
save(fig, stem)
print("ok", stem)
