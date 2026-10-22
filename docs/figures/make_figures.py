"""Figures for the docs/ pages. Every number is copied from cluster/EXPERIMENT_LOG.md (job IDs and scoring scripts
noted beside each block), so the figures can be regenerated without the cluster:
    python docs/figures/make_figures.py
Writes <name>_light.png and <name>_dark.png next to this file; the pages pick one with <picture>.
Palette: the validated categorical order blue / orange / aqua (all-pairs CVD-checked for <= 3 series per chart),
muted gray for the lab baseline. Text uses ink tokens, never series colors. Accuracy comparisons are dot + whisker
plots (no bars, so no truncated-baseline problem)."""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
THEMES = {
    "light": dict(surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e", muted="#898781", grid="#e1e0d9",
                  axis="#c3c2b7", s1="#2a78d6", s2="#eb6834", s3="#1baf7a"),
    "dark": dict(surface="#1a1a19", ink="#f0efec", ink2="#c3c2b7", muted="#898781", grid="#2c2c2a",
                 axis="#383835", s1="#3987e5", s2="#d95926", s3="#199e70"),
}
PX = 0.72            # one display pixel in points (figures are rendered at 200 dpi and shown at half size)
DOT = (9 * PX) ** 2  # marker area for a 9 px dot
plt.rcParams.update({"font.family": ["Segoe UI", "DejaVu Sans"], "font.size": 9.5, "axes.titlesize": 11,
                     "axes.titleweight": "semibold", "savefig.dpi": 200})


def frame(t, figsize):
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor(t["surface"])
    ax.set_facecolor(t["surface"])
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(t["axis"])
    ax.spines["bottom"].set_linewidth(PX)
    ax.tick_params(colors=t["muted"], labelcolor=t["ink2"], length=0)
    ax.xaxis.label.set_color(t["ink2"])
    ax.yaxis.label.set_color(t["ink2"])
    return fig, ax


def title(fig, ax, t, text, sub):
    """Title and subtitle at the figure's left edge (not the axes'), so long y labels never push them off-canvas."""
    h = fig.get_figheight()
    fig.text(0.012, 1 - 0.16 / h, text, color=t["ink"], fontsize=11, fontweight="semibold", va="top")
    fig.text(0.012, 1 - 0.42 / h, sub, color=t["ink2"], fontsize=8.5, va="top")
    fig._top = 1 - 0.68 / h


def dot(ax, t, x, y, color, **kw):
    ax.scatter(x, y, s=DOT, color=color, edgecolors=t["surface"], linewidths=2 * PX, zorder=4, **kw)


def save(fig, name, theme):
    fig.tight_layout(rect=(0, 0, 1, getattr(fig, "_top", 1)))
    fig.savefig(os.path.join(HERE, f"{name}_{theme}.png"), facecolor=fig.get_facecolor())
    plt.close(fig)


# ---- 1. classifier: split v1 test accuracy (log: "Final seed summary", recipe ablation jobs 72105-72112, ensembles)
CLS = [  # label, mean, sd (None = single evaluation), is_baseline
    ("ResNet-50, lab recipe (5 seeds)", 90.2, 1.5, True),
    ("ResNet-50 + lab GPT synthetic (5 seeds)", 90.4, 0.9, True),
    ("Qwen3-VL-32B QLoRA (1 seed)", 93.7, None, False),
    ("Qwen3-VL-8B LoRA (3 seeds)", 94.2, 0.3, False),
    ("SigLIP 2 so400m fine-tuned (3 seeds)", 95.6, 1.3, False),
    ("EVA-02-L fine-tuned (3 seeds)", 95.7, 1.1, False),
    ("SigLIP 2 + ChatGPT synthetic + low-quality aug. (3 seeds)", 96.3, 0.8, False),
    ("Ensemble E2: 8B LoRA + EVA-02 + SigLIP 2 probes (pre-declared)", 97.2, None, False),
    ("Ensemble X3: 6 fine-tuned backbones + 8B LoRA (exploratory)", 97.7, None, False),
]


def fig_classifier(theme):
    t = THEMES[theme]
    fig, ax = frame(t, (7.6, 4.3))
    for i, (lab, m, sd, base) in enumerate(CLS):
        c = t["muted"] if base else t["s1"]
        if sd:
            ax.plot([m - sd, m + sd], [i, i], color=c, lw=2 * PX, solid_capstyle="round", zorder=3)
        dot(ax, t, m, i, c)
        ax.text((m + sd if sd else m) + 0.35, i, f"{m:.1f}" + (f" ± {sd:.1f}" if sd else ""), va="center",
                color=t["ink2"], fontsize=8.5)
    ax.set_yticks(range(len(CLS)), [c[0] for c in CLS])
    ax.set_xlim(86, 100.5)
    ax.set_xlabel("Split v1 test accuracy, % (398 photos; mean ± sd over seeds)")
    ax.grid(axis="x", color=t["grid"], lw=PX)
    ax.set_axisbelow(True)
    ax.scatter([], [], s=DOT, color=t["muted"], label="lab baseline (ResNet-50)")
    ax.scatter([], [], s=DOT, color=t["s1"], label="revival models")
    ax.legend(loc="upper left", frameon=False, labelcolor=t["ink2"], fontsize=8.5)
    title(fig, ax, t, "Modern backbones close most of the gap to the label-noise ceiling",
          "Same data and training recipe as the lab's ResNet-50; label noise caps accuracy at ~97-98%")
    save(fig, "classifier", theme)


# ---- 2. synthetic data x low-quality augmentation (log: jobs 72084-72093, 72105-72112; cluster/classifier/eval_lowq.py)
ARMS = [  # label, clean, 128 px thumbnail
    ("real only", 95.6, 83.7),
    ("+ lab GPT synthetic (601)", 95.7, 85.8),
    ("+ ChatGPT synthetic, first 165", 96.0, 77.9),
    ("+ ChatGPT synthetic, all 235", 94.2, 83.0),
    ("+ ChatGPT synthetic, without frost 3-5", 96.1, 81.9),
    ("real only + low-quality aug.", 95.1, 90.0),
    ("+ ChatGPT synthetic + low-quality aug.", 95.7, 91.6),
    ("+ ChatGPT w/o frost 3-5 + low-quality aug.", 96.3, 92.1),
]


def fig_lowq(theme):
    t = THEMES[theme]
    fig, ax = frame(t, (7.6, 4.0))
    for i, (lab, clean, thumb) in enumerate(ARMS):
        y = len(ARMS) - 1 - i
        ax.plot([thumb, clean], [y, y], color=t["grid"], lw=2 * PX, zorder=2)
        dot(ax, t, thumb, y, t["s2"])
        dot(ax, t, clean, y, t["s1"])
        ax.text(thumb - 0.4, y, f"{thumb:.1f}", va="center", ha="right", color=t["ink2"], fontsize=8.5)
    ax.set_yticks(range(len(ARMS)), [a[0] for a in ARMS][::-1])
    ax.set_xlim(74, 98.5)
    ax.set_xlabel("Accuracy, % (SigLIP 2 fine-tunes, mean of 3 seeds, real split v1 test)")
    ax.grid(axis="x", color=t["grid"], lw=PX)
    ax.set_axisbelow(True)
    ax.scatter([], [], s=DOT, color=t["s1"], label="clean test photos")
    ax.scatter([], [], s=DOT, color=t["s2"], label="same photos as 128 px thumbnails")
    ax.legend(loc="lower left", frameon=False, labelcolor=t["ink2"], fontsize=8.5)
    title(fig, ax, t, "Robustness to low-detail photos comes from train-time degradation",
          "Clean accuracy barely moves; thumbnail accuracy rises 83.7 -> 92.1 with the augmentation")
    save(fig, "synthetic_lowq", theme)


# ---- 3. label-swap honesty test (log: test split, 398 photos x {true, wrong} label; cluster/vlm/score_honesty.py)
HON = [  # label, accepts true, rejects wrong, is_student (235B job 72137, GLM-4.6V job 72138)
    ("Qwen3-VL-8B", 72.4, 74.1, False),
    ("Qwen3-VL-32B (teacher)", 78.6, 76.6, False),
    ("Qwen3.8-27B", 90.5, 67.8, False),
    ("Qwen3-VL-235B", 85.9, 75.5, False),
    ("GLM-4.6V", 50.0, 78.1, False),
    ("student from base 8B", 79.3, 81.9, True),
    ("student from classification-tuned 8B", 79.9, 85.2, True),
]


def fig_honesty(theme):
    t = THEMES[theme]
    fig, ax = frame(t, (7.0, 4.6))
    ax.spines["left"].set_visible(True)
    ax.spines["left"].set_color(t["axis"])
    ax.spines["left"].set_linewidth(PX)
    x0, x1, y0, y1 = 45, 100, 62, 95
    for b in (65, 70, 75, 80, 85):   # iso-balanced-accuracy lines: (x + y) / 2 = b, labelled where they enter the plot
        ax.plot([2 * b - y1, x1], [y1, 2 * b - x1], color=t["grid"], lw=PX, zorder=1, clip_on=True)
        xe = max(x0, 2 * b - y1)
        ax.text(xe + 0.6, 2 * b - xe - 0.6, f"balanced {b}%", color=t["muted"], fontsize=7.5, va="top",
                ha="left", rotation=-45, rotation_mode="anchor", clip_on=True)
    offs = {"Qwen3.8-27B": (-0.9, 1.3, "right"), "Qwen3-VL-8B": (-0.9, -1.4, "right"),
            "Qwen3-VL-235B": (0.9, -1.4, "left"), "Qwen3-VL-32B (teacher)": (-0.9, 1.3, "right")}
    for lab, x, y, st in HON:
        dot(ax, t, x, y, t["s2"] if st else t["s1"])
        dx, dy, ha = offs.get(lab, (0.8, 0.0, "left"))
        ax.text(x + dx, y + dy, lab, color=t["ink2"], fontsize=8.5, va="center", ha=ha)
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_xlabel("Accepts the TRUE label, % of 398")
    ax.set_ylabel("Rejects a WRONG label, % of 398")
    ax.scatter([], [], s=DOT, color=t["s1"], label="zero-shot VLM")
    ax.scatter([], [], s=DOT, color=t["s2"], label="distilled 8B student")
    ax.legend(loc="lower left", frameon=False, labelcolor=t["ink2"], fontsize=8.5)
    title(fig, ax, t, "Filtered distillation makes an 8B more honest than its 32B teacher",
          "Label-swap test: every test photo shown once with its true label and once with a wrong one")
    save(fig, "honesty", theme)


# ---- 4. hallucination filters (common set: 562 human-rated claims, 515 real / 47 not there, with valid outputs from
#         every filter; consistency_pilot.py + judge_score.py scoring, 2026-10-21; judges = cluster/vlm/judge_claims.py,
#         each dropping the claims it calls 'incorrect' or 'cannot tell')
FILT = [  # label, real removed %, false removed %, group (0 consistency, 1 judge whole photo, 2 judge + zoom)
    ("cross-model agreement", 41.7, 59.6, 0),
    ("self-consistency (2 of 3 repeats)", 52.0, 66.0, 0),
    ("agreement, else self-consistency", 34.6, 51.1, 0),
    ("8B", 21.4, 59.6, 1),
    ("27B", 8.5, 31.9, 1),
    ("32B", 10.5, 38.3, 1),
    ("8B", 16.3, 48.9, 2),
    ("27B", 9.9, 36.2, 2),
    ("32B", 12.8, 44.7, 2),
    ("235B", 8.2, 29.8, 2),   # job 72140, zoom only
]


def fig_filters(theme):
    t = THEMES[theme]
    fig, ax = frame(t, (7.0, 4.6))
    ax.spines["left"].set_visible(True)
    ax.spines["left"].set_color(t["axis"])
    ax.spines["left"].set_linewidth(PX)
    ax.plot([0, 70], [0, 70], color=t["axis"], lw=PX, zorder=1)
    ax.text(56, 52.5, "random filter", color=t["muted"], fontsize=8, rotation=33, rotation_mode="anchor")
    cols = [t["s1"], t["s2"], t["s3"]]
    for lab, x, y, g in FILT:
        dot(ax, t, x, y, cols[g])
        dx, ha = (1.2, "left") if g == 0 else ((-1.0, "right") if g == 1 else (1.0, "left"))
        ax.text(x + dx, y, lab, color=t["ink2"], fontsize=8, va="center", ha=ha)
    ax.set_xlim(0, 70)
    ax.set_ylim(0, 75)
    ax.set_xlabel("Real claims removed, % of 515 (cost)")
    ax.set_ylabel("False claims removed, % of 47 (benefit)")
    ax.grid(color=t["grid"], lw=PX)
    ax.set_axisbelow(True)
    for c, lab in zip(cols, ("consistency checks (no extra look)", "judge (8B / 27B / 32B), whole photo",
                             "judge (8B / 27B / 32B / 235B), photo + zoomed crop")):
        ax.scatter([], [], s=DOT, color=c, label=lab)
    ax.legend(loc="lower right", frameon=False, labelcolor=t["ink2"], fontsize=8.5)
    title(fig, ax, t, "Judges that re-examine each claim remove false claims at far lower cost",
          "Hallucination filters on 562 human-rated claims; judges drop claims they call incorrect or "
          "cannot-tell; up and to the left is better")
    save(fig, "hallucination_filters", theme)


# ---- 5. examiner claim error rates (interim human ratings, 615 claims, one rater; rating_detail.py, Wilson CI)
EXAM = [  # label, rate %, lo, hi, n claims, photos
    ("Qwen3-VL-235B-A22B (AWQ)", 10.3, 6.8, 15.2, 204, 34),
    ("Qwen3.8-27B", 10.2, 6.7, 15.2, 197, 34),
    ("GLM-4.6V (FP8)", 5.0, 1.4, 16.5, 40, 9),
    ("Qwen3-VL-32B", 4.0, 2.0, 8.1, 174, 34),
]


def fig_examiners(theme):
    t = THEMES[theme]
    fig, ax = frame(t, (7.0, 2.6))
    for i, (lab, r, lo, hi, n, ph) in enumerate(EXAM):
        ax.plot([lo, hi], [i, i], color=t["s1"], lw=2 * PX, solid_capstyle="round", zorder=3)
        dot(ax, t, r, i, t["s1"])
        ax.text(hi + 0.5, i, f"{r:.1f}%  ({n} claims, {ph} photos)", va="center", color=t["ink2"], fontsize=8.5)
    ax.set_yticks(range(len(EXAM)), [e[0] for e in EXAM])
    ax.set_xlim(0, 28)
    ax.set_xlabel("Claims rated 'not there', % (95% Wilson interval)")
    ax.grid(axis="x", color=t["grid"], lw=PX)
    ax.set_axisbelow(True)
    title(fig, ax, t, "How often each captioner describes something that is not in the photo",
          "Interim: one blind rater, 615 of 845 claims; GLM-4.6V rated on 9 photos so far")
    save(fig, "examiner_errors", theme)


# ---- 6. test-time voting vs training (log: job 72142, consensus_score.py; majority of N sampled zero-shot diagnoses,
#         T 0.7, averaged over random subsets; the 32B has 19 valid samples per photo)
VOTE_N = {"Qwen3-VL-8B": [1, 3, 5, 10, 20], "Qwen3-VL-32B": [1, 3, 5, 10, 19], "Qwen3.8-27B": [1, 3, 5, 10, 20]}
VOTE = {"Qwen3-VL-8B": [67.1, 68.2, 68.8, 69.6, 70.1], "Qwen3-VL-32B": [66.9, 67.9, 68.0, 68.4, 68.8],
        "Qwen3.8-27B": [64.1, 65.8, 66.3, 67.1, 67.1]}


def fig_voting(theme):
    t = THEMES[theme]
    fig, ax = frame(t, (7.0, 4.2))
    ax.spines["left"].set_visible(True)
    ax.spines["left"].set_color(t["axis"])
    ax.spines["left"].set_linewidth(PX)
    for y, lab in ((96.3, "fine-tuned SigLIP 2 (one 10-minute training run)"), (94.2, "fine-tuned Qwen3-VL-8B (LoRA)")):
        ax.plot([0, 22], [y, y], color=t["muted"], lw=PX, zorder=1)
        ax.text(0.6, y + 0.6, lab, color=t["ink2"], fontsize=8, va="bottom")
    for (m, ys), c in zip(VOTE.items(), (t["s1"], t["s2"], t["s3"])):
        xs = VOTE_N[m]
        ax.plot(xs, ys, color=c, lw=2 * PX, solid_capstyle="round", zorder=3)
        dot(ax, t, xs[-1], ys[-1], c)
        ax.text(xs[-1] + 0.5, ys[-1], f"{m} {ys[-1]:.1f}", color=t["ink2"], fontsize=8, va="center")
    ax.scatter([20], [73.9], s=DOT, color=t["ink2"], edgecolors=t["surface"], linewidths=2 * PX, zorder=4)
    ax.text(19.6, 74.6, "3 models pooled, 60 votes: 73.9", color=t["ink2"], fontsize=8, ha="right", va="bottom")
    ax.set_xlim(0, 28)
    ax.set_ylim(60, 100)
    ax.set_xticks([1, 3, 5, 10, 20])
    ax.set_xlabel("Sampled zero-shot diagnoses per photo (majority vote)")
    ax.set_ylabel("Test accuracy, % (398 photos)")
    ax.grid(axis="y", color=t["grid"], lw=PX)
    ax.set_axisbelow(True)
    title(fig, ax, t, "Voting over samples does not substitute for training",
          "Majority of N sampled zero-shot answers vs fine-tuned models on the same test photos")
    save(fig, "voting", theme)


# ---- 7. answer format (log: jobs 72147-72149 + sample 0 of 72142; controlled, greedy, same stack)
FMT = [  # model, [text, bare, category-then-evidence], evidence-then-category
    ("Qwen3.8-27B", [67.1, 66.3, 69.3], 65.3),
    ("Qwen3-VL-32B", [74.4, 73.4, 72.4], 67.6),
    ("Qwen3-VL-8B", [76.9, 77.1, 77.1], 67.8),
]


def fig_format(theme):
    t = THEMES[theme]
    fig, ax = frame(t, (7.0, 2.6))
    for i, (m, firsts, ev) in enumerate(FMT):
        ax.plot([ev, max(firsts)], [i, i], color=t["grid"], lw=2 * PX, zorder=2)
        for x in firsts:
            dot(ax, t, x, i, t["s1"])
        dot(ax, t, ev, i, t["s2"])
        ax.text(ev - 0.4, i, f"{ev:.1f}", color=t["ink2"], fontsize=8.5, va="center", ha="right")
        ax.text(max(firsts) + 0.4, i, f"{min(firsts):.1f}-{max(firsts):.1f}", color=t["ink2"], fontsize=8.5,
                va="center")
    ax.set_yticks(range(len(FMT)), [f[0] for f in FMT])
    ax.set_xlim(60, 82)
    ax.set_xlabel("Zero-shot test accuracy, % (greedy, 398 photos)")
    ax.grid(axis="x", color=t["grid"], lw=PX)
    ax.set_axisbelow(True)
    ax.scatter([], [], s=DOT, color=t["s1"], label="answer first, or no explanation (3 formats)")
    ax.scatter([], [], s=DOT, color=t["s2"], label="one evidence sentence before the answer")
    ax.legend(loc="lower left", frameon=False, labelcolor=t["ink2"], fontsize=8, bbox_to_anchor=(0, 1.0), ncol=2)
    title(fig, ax, t, "Explaining before answering costs the Qwen3-VL models 5-9 points",
          "Same models, images and categories; only the answer format changes")
    save(fig, "answer_format", theme)


if __name__ == "__main__":
    for th in THEMES:
        fig_classifier(th)
        fig_lowq(th)
        fig_honesty(th)
        fig_filters(th)
        fig_examiners(th)
        fig_voting(th)
        fig_format(th)
    print("written to", HERE)
