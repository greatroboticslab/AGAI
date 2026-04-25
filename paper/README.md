# Active Paper

This directory is the active paper source for AGAI.

- Main manuscript: [main.tex](./main.tex)
- Generated figures: [figures/generated](./figures/generated)
- Generated LaTeX tables: [generated](./generated)
- Asset builder: [scripts/build_paper_assets.py](./scripts/build_paper_assets.py)

## Build order

1. Regenerate figures and tables:

```bash
cd ..
python paper/scripts/build_paper_assets.py
```

2. Compile the manuscript:

```bash
cd paper
pdflatex -interaction=nonstopmode main.tex
bibtex main
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
```

The current manuscript is a factual current-state technical paper. It explicitly marks what is done and what is still pending.
