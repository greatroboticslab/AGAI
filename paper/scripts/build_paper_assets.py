#!/usr/bin/env python3
"""
Generate current paper figures and LaTeX tables from local AGAI artifacts.

Assumptions
-----------
1. The JSON evaluation artifacts under ``evaluation/holdout`` are the canonical
   measured outputs for the current reported ResNet checkpoint evaluations.
2. The filesystem under ``plant_diagnostic/data`` is the canonical source of
   current dataset sizes after the active holdout release, training-root
   rebuild, external imports, and class-floor augmentation.
3. The dual-report pilot summary under ``evaluation/stage2_pilot`` is the
   current factual basis for the report-generation section of the manuscript.
4. This script should fail loudly if probabilities leave the physical interval
   [0, 1] or if expected classes/artifacts are missing.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[2]
PAPER_ROOT = REPO_ROOT / "paper"
FIG_ROOT = PAPER_ROOT / "figures" / "generated"
TEX_ROOT = PAPER_ROOT / "generated"

HOLDOUT_V4 = REPO_ROOT / "evaluation" / "holdout" / "resnet_v4_release10holdout_full_holdout.json"
DUAL_REPORT_SUMMARY = (
    REPO_ROOT
    / "evaluation"
    / "stage2_pilot"
    / "dual_report_focus20_tight"
    / "dual_report_focus20_tight_summary.json"
)
DUAL_REPORT_AUDIT = (
    REPO_ROOT
    / "evaluation"
    / "stage2_pilot"
    / "dual_report_focus20_tight"
    / "dual_report_focus20_tight_audit.jsonl"
)

TRAIN_ROOT = REPO_ROOT / "plant_diagnostic" / "data" / "train"
TRAIN_AUG_ROOT = REPO_ROOT / "plant_diagnostic" / "data" / "train_aug"
HOLDOUT_ROOT = REPO_ROOT / "plant_diagnostic" / "data" / "holdout"

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
CLASS_ORDER = [
    "drought",
    "frost",
    "gray_mold",
    "healthy",
    "overwatered",
    "root_rot",
    "white_mold",
]
DISPLAY_LABELS = {
    "drought": "Drought",
    "frost": "Frost",
    "gray_mold": "Gray mold",
    "healthy": "Healthy",
    "overwatered": "Overwatered",
    "root_rot": "Root rot",
    "white_mold": "White mold",
}


@dataclass(frozen=True)
class SplitCounts:
    train: dict[str, int]
    train_aug: dict[str, int]
    holdout: dict[str, int]


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def ensure_dirs() -> None:
    FIG_ROOT.mkdir(parents=True, exist_ok=True)
    TEX_ROOT.mkdir(parents=True, exist_ok=True)


def count_split(root: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    for label in CLASS_ORDER:
        class_dir = root / label
        if not class_dir.is_dir():
            raise FileNotFoundError(f"Missing expected class directory: {class_dir}")
        counts[label] = sum(
            1 for path in class_dir.rglob("*") if path.is_file() and path.suffix.lower() in IMG_EXTS
        )
    return counts


def collect_counts() -> SplitCounts:
    return SplitCounts(
        train=count_split(TRAIN_ROOT),
        train_aug=count_split(TRAIN_AUG_ROOT),
        holdout=count_split(HOLDOUT_ROOT),
    )


def validate_probability_range(values: Iterable[float], label: str) -> None:
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size == 0:
        raise ValueError(f"No probability values found for {label}")
    if np.any(~np.isfinite(arr)):
        raise ValueError(f"Non-finite probability values found for {label}")
    if np.any(arr < -1e-9) or np.any(arr > 1.0 + 1e-9):
        raise ValueError(f"Probability values outside [0, 1] found for {label}")


def set_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 180,
            "savefig.dpi": 300,
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "grid.linestyle": ":",
        }
    )


def tex_escape(text: str) -> str:
    repl = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
    }
    out = text
    for src, dst in repl.items():
        out = out.replace(src, dst)
    return out


def write_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def render_pipeline_overview() -> None:
    fig, ax = plt.subplots(figsize=(14.5, 7.0))
    ax.set_axis_off()
    ax.set_xlim(0, 15.5)
    ax.set_ylim(0, 8.5)

    def box(x: float, y: float, w: float, h: float, text: str, fc: str, fs: int = 12) -> None:
        patch = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.02,rounding_size=0.08",
            linewidth=1.2,
            edgecolor="#22313f",
            facecolor=fc,
        )
        ax.add_patch(patch)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", wrap=True, fontsize=fs)

    def arrow(x1: float, y1: float, x2: float, y2: float, style: str = "-") -> None:
        patch = FancyArrowPatch(
            (x1, y1),
            (x2, y2),
            arrowstyle="-|>",
            mutation_scale=14,
            linewidth=1.4,
            linestyle=style,
            color="#22313f",
        )
        ax.add_patch(patch)

    box(0.5, 4.9, 2.2, 1.5, "Input\nimage", "#e8f1ff", fs=13)
    box(3.1, 4.9, 2.9, 1.5, "ResNet-50\n2-view TTA\nTemperature scaling", "#fcefdc", fs=12)
    box(6.5, 4.9, 2.7, 1.5, "Class gate\nUnknown if confidence is low", "#f9e5e5", fs=12)
    box(9.8, 5.8, 2.8, 1.2, "RF-DETR part grounding\nCore runtime stage", "#e9f7ef", fs=11)
    box(9.8, 4.0, 2.8, 1.2, "Knowledge graph RAG\nOptional at runtime", "#eef2ff", fs=11)
    box(13.0, 4.9, 2.1, 1.5, "MiniGPT\nreport writer", "#fff4d6", fs=12)
    box(13.0, 2.9, 2.1, 1.2, "Hallucination retry\nPhrase heuristics", "#fde9f3", fs=11)
    box(13.0, 1.0, 2.1, 1.2, "Farmer-facing\nreport", "#e8f6f3", fs=12)

    box(3.1, 1.6, 2.8, 1.2, "Offline stage-2\ngeneration", "#f7f7f7", fs=12)
    box(6.4, 1.6, 2.9, 1.2, "ResNet p1/p2 metadata\nStrict ambiguity policy", "#f7f7f7", fs=11)
    box(9.8, 1.6, 2.8, 1.2, "GPT-5.4 dual-report\ntargets\nPilot completed", "#f7f7f7", fs=11)

    arrow(2.7, 5.65, 3.1, 5.65)
    arrow(6.0, 5.65, 6.5, 5.65)
    arrow(9.2, 5.65, 13.0, 5.65)
    arrow(12.6, 6.35, 13.0, 6.0)
    arrow(12.6, 4.6, 13.0, 5.2)
    arrow(14.05, 4.9, 14.05, 4.1)
    arrow(14.05, 2.9, 14.05, 2.2)

    arrow(5.9, 2.2, 6.4, 2.2, style="--")
    arrow(9.3, 2.2, 9.8, 2.2, style="--")
    arrow(12.6, 2.2, 13.0, 5.1, style="--")

    ax.text(
        0.5,
        8.1,
        "Current AGAI pipeline. Solid arrows denote deployed runtime flow. Dashed arrows denote the offline\n"
        "data-generation path used for the tightened dual-report pilot. Clean post-carve MiniGPT retraining\n"
        "and holdout explanation evaluation are still pending.",
        ha="left",
        va="top",
        fontsize=11,
    )

    for ext in ("png", "pdf"):
        fig.savefig(FIG_ROOT / f"pipeline_overview.{ext}", bbox_inches="tight")
    plt.close(fig)


def render_dataset_composition(counts: SplitCounts) -> None:
    labels = [DISPLAY_LABELS[x] for x in CLASS_ORDER]
    x = np.arange(len(CLASS_ORDER))
    width = 0.24

    fig, ax = plt.subplots(figsize=(11.8, 4.8))
    ax.bar(x - width, [counts.train[c] for c in CLASS_ORDER], width, label="train", color="#4c78a8")
    ax.bar(x, [counts.train_aug[c] for c in CLASS_ORDER], width, label="train_aug", color="#f58518")
    ax.bar(x + width, [counts.holdout[c] for c in CLASS_ORDER], width, label="holdout", color="#54a24b")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Image count")
    ax.set_title("Current dataset composition after the active holdout release, rebuild, and balancing pass")
    ax.legend(frameon=False, ncol=3)
    for ext in ("png", "pdf"):
        fig.savefig(FIG_ROOT / f"dataset_composition.{ext}", bbox_inches="tight")
    plt.close(fig)


def render_holdout_grid() -> None:
    fig, axes = plt.subplots(2, 4, figsize=(12.5, 6.8))
    axes = axes.ravel()
    for ax in axes:
        ax.axis("off")

    for idx, label in enumerate(CLASS_ORDER):
        image_paths = sorted(
            p for p in (HOLDOUT_ROOT / label).iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS
        )
        if not image_paths:
            raise FileNotFoundError(f"No holdout exemplars found for {label}")
        with Image.open(image_paths[0]) as img:
            axes[idx].imshow(np.asarray(img.convert("RGB")))
        axes[idx].set_title(DISPLAY_LABELS[label])
        axes[idx].axis("off")

    axes[-1].text(
        0.02,
        0.95,
        "Holdout set\nn = 70\n10 images per class\nBalanced across all\nseven disease states",
        va="top",
        ha="left",
        fontsize=11,
    )
    axes[-1].axis("off")

    fig.suptitle("Representative holdout images by class", y=0.98, fontsize=14)
    for ext in ("png", "pdf"):
        fig.savefig(FIG_ROOT / f"holdout_examples_grid.{ext}", bbox_inches="tight")
    plt.close(fig)


def render_confusion_matrix(v4: dict[str, Any]) -> None:
    cm = np.asarray(v4["confusion_matrix"], dtype=np.float64)
    if cm.shape != (len(CLASS_ORDER), len(CLASS_ORDER)):
        raise ValueError(f"Unexpected confusion matrix shape: {cm.shape}")

    fig, ax = plt.subplots(figsize=(7.4, 6.2))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(np.arange(len(CLASS_ORDER)))
    ax.set_yticks(np.arange(len(CLASS_ORDER)))
    ax.set_xticklabels([DISPLAY_LABELS[x] for x in CLASS_ORDER], rotation=30, ha="right")
    ax.set_yticklabels([DISPLAY_LABELS[x] for x in CLASS_ORDER])
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title("ResNet v4 confusion matrix on the balanced 70-image holdout")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            value = int(cm[i, j])
            color = "white" if value >= np.max(cm) * 0.5 else "#111111"
            ax.text(j, i, str(value), ha="center", va="center", color=color, fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    for ext in ("png", "pdf"):
        fig.savefig(FIG_ROOT / f"resnet_v4_confusion_matrix.{ext}", bbox_inches="tight")
    plt.close(fig)


def render_accuracy_and_calibration(v4: dict[str, Any]) -> None:
    v4_summary = v4["summary"]

    v4_per = [v4["per_class"][c]["top1_accuracy"] for c in CLASS_ORDER]
    validate_probability_range(v4_per, "per-class top-1 accuracy")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.0, 5.0))

    x = np.arange(len(CLASS_ORDER))
    ax1.bar(x, v4_per, width=0.58, color="#3182bd")
    ax1.set_ylim(0.0, 1.05)
    ax1.set_xticks(x)
    ax1.set_xticklabels([DISPLAY_LABELS[c] for c in CLASS_ORDER], rotation=25, ha="right")
    ax1.set_ylabel("Top-1 accuracy")
    ax1.set_title("Per-class top-1 accuracy on the active balanced holdout")
    ax1.text(
        0.02,
        0.97,
        (
            f"Overall top-1: {100*v4_summary['top1_accuracy']:.1f}%\n"
            f"Top-2: {100*v4_summary['top2_accuracy']:.1f}%\n"
            f"ECE: {100*v4_summary['ece']:.2f}%"
        ),
        transform=ax1.transAxes,
        va="top",
        ha="left",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.9, "edgecolor": "#cccccc"},
    )

    rel = v4["reliability"]
    xs = [row["mean_confidence"] for row in rel if row["count"] > 0]
    ys = [row["accuracy"] for row in rel if row["count"] > 0]
    weights = [row["count"] for row in rel if row["count"] > 0]
    validate_probability_range(xs, "reliability mean confidence")
    validate_probability_range(ys, "reliability bin accuracy")

    ax2.plot([0, 1], [0, 1], linestyle="--", color="#666666", linewidth=1.0, label="Perfect calibration")
    ax2.scatter(xs, ys, s=np.asarray(weights) * 12.0, color="#dd8452", alpha=0.85, label="Bin")
    for x_val, y_val, count in zip(xs, ys, weights):
        ax2.text(x_val + 0.01, y_val, str(count), fontsize=8, color="#333333")
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.set_xlabel("Mean predicted confidence")
    ax2.set_ylabel("Empirical accuracy")
    ax2.set_title(
        f"V4 reliability diagram, ECE = {100*v4_summary['ece']:.2f}%"
    )
    ax2.legend(frameon=False, loc="lower right")

    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(FIG_ROOT / f"resnet_accuracy_calibration.{ext}", bbox_inches="tight")
    plt.close(fig)


def render_dual_report_summary(summary: dict[str, Any], audit_rows: list[dict[str, Any]]) -> None:
    from collections import Counter

    mode_counts = Counter(row["report_mode"] for row in audit_rows)
    risk_counts = Counter(row["overclaim_risk"] for row in audit_rows)
    rule_counts = summary["rule_counts"]

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.6))

    axes[0].bar(list(rule_counts.keys()), list(rule_counts.values()), color="#4c78a8")
    axes[0].set_title("Selection rules in focused pilot")
    axes[0].tick_params(axis="x", rotation=35)

    axes[1].bar(list(mode_counts.keys()), list(mode_counts.values()), color="#59a14f")
    axes[1].set_title("Generated report modes")

    axes[2].bar(list(risk_counts.keys()), list(risk_counts.values()), color="#e15759")
    axes[2].set_title("Self-reported overclaim risk")

    for ax in axes:
        ax.set_ylabel("Count")
        ax.grid(axis="y", alpha=0.25)

    fig.suptitle("Tight dual-report pilot summary")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(FIG_ROOT / f"dual_report_pilot_summary.{ext}", bbox_inches="tight")
    plt.close(fig)


def write_dataset_counts_table(counts: SplitCounts) -> None:
    rows = []
    for label in CLASS_ORDER:
        rows.append(
            f"{tex_escape(DISPLAY_LABELS[label])} & {counts.train[label]} & {counts.train_aug[label]} & {counts.holdout[label]} \\\\"
        )
    rows.append(
        f"\\textbf{{Total}} & \\textbf{{{sum(counts.train.values())}}} & "
        f"\\textbf{{{sum(counts.train_aug.values())}}} & \\textbf{{{sum(counts.holdout.values())}}} \\\\"
    )
    body = "\n".join(rows)
    table = (
        "\\begin{tabular}{lrrr}\n"
        "\\toprule\n"
        "Class & train & train\\_aug & holdout \\\\\n"
        "\\midrule\n"
        f"{body}\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
    )
    write_text(TEX_ROOT / "dataset_counts_table.tex", table)


def write_resnet_summary_table(v4: dict[str, Any]) -> None:
    s4 = v4["summary"]
    table = f"""\\begin{{tabular}}{{lr}}
\\toprule
Metric & Current v4 \\\\
\\midrule
Holdout samples & {s4['samples']} \\\\
Top-1 accuracy & {100*s4['top1_accuracy']:.2f}\\% \\\\
Top-2 accuracy & {100*s4['top2_accuracy']:.2f}\\% \\\\
ECE & {100*s4['ece']:.2f}\\% \\\\
Temperature & {s4['temperature_used']:.4f} \\\\
Horizontal-flip TTA & yes \\\\
\\bottomrule
\\end{{tabular}}
"""
    write_text(TEX_ROOT / "resnet_summary_table.tex", table)


def write_per_class_table(v4: dict[str, Any]) -> None:
    lines = []
    for label in CLASS_ORDER:
        row = v4["per_class"][label]
        ci = row["top1_wilson95"]
        lines.append(
            f"{tex_escape(DISPLAY_LABELS[label])} & {row['support']} & "
            f"{100*row['top1_accuracy']:.1f}\\% & "
            f"[{100*ci['low']:.1f}, {100*ci['high']:.1f}] & "
            f"{100*row['top2_accuracy']:.1f}\\% \\\\"
        )
    table = (
        "\\begin{tabular}{lrrrr}\n"
        "\\toprule\n"
        "Class & $n$ & Top-1 & Wilson 95\\% CI & Top-2 \\\\\n"
        "\\midrule\n"
        + "\n".join(lines)
        + "\n\\bottomrule\n"
        "\\end{tabular}\n"
    )
    write_text(TEX_ROOT / "resnet_per_class_table.tex", table)


def write_status_table() -> None:
    rows = [
        ("Balanced holdout with provenance", "Done", "The active holdout now contains 10 images for each of the seven classes"),
        ("Compact-train release and train_aug rebuild", "Done", "Ten images per class were released from holdout back to train and the augmented root was rebuilt from scratch"),
        ("Augmented class-floor balancing", "Done", "Underrepresented classes in the active augmented split were raised to a minimum of 300 images"),
        ("Conservative external image additions", "Done", "Human-audited frost and drought additions were imported into train and propagated into train_aug"),
        ("ResNet retrain on the rebuilt current split", "Done", "A fresh v4 checkpoint was trained on the rebuilt train_aug root and calibrated on its internal validation split"),
        ("Latest ResNet checkpoint evaluation on active holdout", "Done", "The v4 checkpoint was evaluated on the active 70-image balanced holdout"),
        ("Runtime promotion with rollback path", "Done", "The runtime app defaults to the current v4 checkpoint and keeps an explicit override path"),
        ("p1/p2 runtime logging", "Done", "JSONL logging added for downstream calibration analysis"),
        ("Tight dual-report generation pilot", "Done", "Focused 20-image pilot completed with 36 generated reports"),
        ("Full stage-2 regeneration under tight policy", "Pending", "The current stage-2 JSON remains stale relative to the augmented training root"),
        ("MiniGPT retrain on regenerated stage-2", "Pending", "Current checkpoint predates the holdout carve and dual-report policy"),
        ("Clean holdout explanation evaluation", "Pending", "No post-regeneration holdout explanation metric exists yet"),
        ("RF-DETR on/off ablation on clean holdout", "Pending", "Planned but not executed"),
    ]
    body = "\n".join(
        f"{tex_escape(task)} & {status} & {tex_escape(note)} \\\\" for task, status, note in rows
    )
    table = (
        "\\begin{tabularx}{\\linewidth}{>{\\raggedright\\arraybackslash}p{0.33\\linewidth} "
        ">{\\raggedright\\arraybackslash}p{0.13\\linewidth} X}\n"
        "\\toprule\n"
        "Component & Status & Notes \\\\\n"
        "\\midrule\n"
        f"{body}\n"
        "\\bottomrule\n"
        "\\end{tabularx}\n"
    )
    write_text(TEX_ROOT / "status_table.tex", table)


def write_dual_report_table(summary: dict[str, Any], audit_rows: list[dict[str, Any]]) -> None:
    from collections import Counter

    mode_counts = Counter(row["report_mode"] for row in audit_rows)
    risk_counts = Counter(row["overclaim_risk"] for row in audit_rows)
    table = f"""\\begin{{tabular}}{{lr}}
\\toprule
Focused tight dual-report pilot statistic & Value \\\\
\\midrule
Selected images & {summary['num_selected']} \\\\
Generated records & {len(audit_rows)} \\\\
Hard-case selections & {summary['rule_counts'].get('top1_disagrees_gt_hard_case', 0)} \\\\
Close-margin selections & {summary['rule_counts'].get('top1_matches_gt_close_to_top2', 0)} \\\\
Standard reports & {mode_counts.get('standard', 0)} \\\\
Standard variants & {mode_counts.get('standard_variant', 0)} \\\\
Ambiguous reports & {mode_counts.get('ambiguous', 0)} \\\\
Self-reported low overclaim risk & {risk_counts.get('low', 0)} \\\\
Self-reported medium overclaim risk & {risk_counts.get('medium', 0)} \\\\
Fallback reports & {sum(int(bool(row.get('used_fallback'))) for row in audit_rows)} \\\\
\\bottomrule
\\end{{tabular}}
"""
    write_text(TEX_ROOT / "dual_report_table.tex", table)


def main() -> None:
    ensure_dirs()
    set_style()

    v4 = load_json(HOLDOUT_V4)
    dual_summary = load_json(DUAL_REPORT_SUMMARY)
    dual_audit = load_jsonl(DUAL_REPORT_AUDIT)
    counts = collect_counts()

    render_pipeline_overview()
    render_dataset_composition(counts)
    render_holdout_grid()
    render_confusion_matrix(v4)
    render_accuracy_and_calibration(v4)
    render_dual_report_summary(dual_summary, dual_audit)

    write_dataset_counts_table(counts)
    write_resnet_summary_table(v4)
    write_per_class_table(v4)
    write_status_table()
    write_dual_report_table(dual_summary, dual_audit)


if __name__ == "__main__":
    main()
