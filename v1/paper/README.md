# `v1/paper/`: the April 2026 technical report of v1

Sources, generated tables and figures, build scripts and the compiled PDF of the technical report that describes v1,
"AGAI: A Classifier-Anchored, Detector-Grounded Pipeline for Strawberry Disease Diagnosis and Constrained Report
Generation", dated 25 April 2026 (v1, 2025 to April 2026). The v2 revival's 2026 journal manuscript is a different
paper: its LaTeX is on Overleaf, and its figure scripts are in
[`paper/figure_scripts/`](../../paper/figure_scripts/) at the top of the repository.

Next to this folder, [`../RESEARCH_PAPER.pdf`](../RESEARCH_PAPER.pdf) is a copy of [`main.pdf`](main.pdf) (the two files
are identical), and [`../RESEARCH_PAPER.tex`](../RESEARCH_PAPER.tex) is a one-line entry point,
`\input{paper/main.tex}`.

## Contents

| Path | Contents |
|---|---|
| [`main.tex`](main.tex) | preamble, title, abstract; inputs the eight section files and the bibliography (`unsrtnat`, 18 references) |
| [`sections/`](sections/) | the eight sections ([`sections/README.md`](sections/README.md)) |
| [`generated/`](generated/) | 11 LaTeX tables written by the scripts ([`generated/README.md`](generated/README.md)) |
| [`figures/`](figures/) | `figures/generated/`: 12 PDF figures, 10 of them in the report ([`figures/README.md`](figures/README.md)) |
| [`scripts/`](scripts/) | 7 Python scripts that write the tables and figures ([`scripts/README.md`](scripts/README.md)) |
| [`references.bib`](references.bib) | bibliography |
| [`main.pdf`](main.pdf) | the compiled report (pdfTeX, 25 April 2026) |
| `main.aux`, `main.bbl`, `main.blg`, `main.out` | LaTeX and BibTeX files of that build |
| `README.md` | this page |

## Build order

1. Regenerate figures and tables (from `v1/`; the scripts find their inputs and outputs from their own location, so
   the working directory does not matter):

```bash
cd v1

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

2. Compile the manuscript (from `v1/`; the section, table, figure and bibliography paths in `main.tex` are relative to
   `paper/`):

```bash
cd paper
pdflatex -interaction=nonstopmode main.tex
bibtex main
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
cp main.pdf ../RESEARCH_PAPER.pdf
```

Only three of the scripts can be run from this repository as it is: `compute_extra_metrics.py`,
`compute_threshold_sweep.py` and `compute_mcnemar.py` need just `../evaluation/holdout/`, and they reproduce the five
committed tables they write exactly (checked on 18 November 2026). The others need files that are not in the
repository ([`scripts/README.md`](scripts/README.md)). Every table and figure that the sections input is committed.

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

The appendix ([sections/08_appendix.tex](sections/08_appendix.tex)) lists a pinned commit hash, SHA-256 hashes of
the deployed ResNet-50 and RF-DETR checkpoints, and the JSON evaluation artifact paths used by the asset scripts. Its
paths are relative to the lab server's working copy, which has the layout of `v1/` (`evaluation/holdout/...` is
[`../evaluation/holdout/`](../evaluation/holdout/)). The commit it names (`6eaa403e...`) is not in this repository, and
neither are the two checkpoints or the dual-report pilot files (`evaluation/stage2_pilot/`).

## Known problems

- **Not this repository's photographs.** The dataset counts in the report (1,506 / 2,424 / 70 photographs in `train/`,
  `train_aug/` and `holdout/`) describe the lab server's folders in April 2026; `v1/plant_diagnostic/data/` holds
  1,540 / 2,159 / 20, and only 16 of the 70 holdout photographs.
- **Old status.** The report's status table and its "pending experiments" describe April 2026; the v2 revival's later
  measurements of these components are in [`../../docs/`](../../docs/README.md).
