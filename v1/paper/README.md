# Active Paper

This directory is the active paper source for AGAI.

- Root editable entry point: [../RESEARCH_PAPER.tex](../RESEARCH_PAPER.tex)
- Root compiled PDF target: [../RESEARCH_PAPER.pdf](../RESEARCH_PAPER.pdf)
- Main manuscript: [main.tex](./main.tex)
- Sections: [sections/](./sections)
- Generated figures: [figures/generated](./figures/generated)
- Generated LaTeX tables: [generated](./generated)
- Bibliography: [references.bib](./references.bib)

## Build order

1. Regenerate figures and tables (run each script from the repo root):

```bash
# Pipeline figure, dataset counts, classifier confusion matrix,
# accuracy/calibration figure, dual-report pilot summary
python paper/scripts/build_paper_assets.py

# Bootstrap CIs, acceptance-gate replay, per-image misclass table
python paper/scripts/compute_extra_metrics.py

# Threshold sweep, risk-coverage curve, selective-accuracy table
python paper/scripts/compute_threshold_sweep.py

# McNemar paired test (v3 vs v4)
python paper/scripts/compute_mcnemar.py

# Per-image misclassification grid figure
python paper/scripts/render_misclass_grid.py

# RF-DETR per-class firing rates on the disease holdout, latency
# (run inside the rfdetr conda env)
conda activate rfdetr
python paper/scripts/run_rfdetr_on_holdout.py
```

2. Compile the manuscript:

```bash
cd paper
pdflatex -interaction=nonstopmode main.tex
bibtex main
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
cp main.pdf ../RESEARCH_PAPER.pdf
```

## Pipeline summary

The pipeline is described as a **three-stage** system in the manuscript:

1. ResNet-50 image classifier (calibrated, gated).
2. RF-DETR Small detector for plant-part visibility grounding.
3. LoRA-adapted MiniGPT-v2 report writer constrained by stages 1 and 2.

The phrase "dual report" in the manuscript refers to the offline
supervision policy that produces two report variants per image (a
primary report and a conditionally emitted ambiguity-aware second
report), not to a model count.

## Reproducibility manifest

See the appendix of [main.tex](./main.tex) for the pinned commit hash,
SHA-256 hashes of the deployed ResNet-50 and RF-DETR checkpoints, and
the JSON evaluation artifact paths used by all asset scripts above.
