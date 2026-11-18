# `paper/`: figure scripts of the 2026 journal manuscript

This folder holds only [`figure_scripts/`](figure_scripts/README.md), the Python scripts that draw the figures of the
journal manuscript comparing v1 (the original system, 2025 to April 2026) with v2 (the 2026 revival, October to
November 2026), titled *Leak-free re-evaluation of a strawberry plant health AI system* in its 24 October 2026 build.
The manuscript's LaTeX source is on Overleaf and is not in this repository. The April 2026 technical report on v1
(sources, build scripts and PDF) is in [`v1/paper/`](../v1/paper/), and its PDF is also
[`v1/RESEARCH_PAPER.pdf`](../v1/RESEARCH_PAPER.pdf).

| Folder | Contents |
|---|---|
| [`figure_scripts/`](figure_scripts/README.md) | one matplotlib script per figure (Figs. 1 to 7, Supplementary Figs. S1 to S4 and S6 to S9), the shared style `agstyle.py` and the helper `calib_to_eval.py`; committed from 24 October to 17 November 2026 |

The answers to the manuscript's bracketed notes, with text to paste and the evidence, are in
[`docs/MANUSCRIPT_QA.md`](../docs/MANUSCRIPT_QA.md); the result files behind its tables are in
[`evaluation/paper_data/`](../evaluation/paper_data/README.md).

## Known problems

- The scripts take their inputs from, and save the figures to, absolute paths on the machine they were written on
  (`D:/agai-paper/...`), so the paths need changing before they run elsewhere, as
  [`figure_scripts/README.md`](figure_scripts/README.md) says.
- Their docstrings cite line numbers of `docs/` pages and of `cluster/EXPERIMENT_LOG.md` in a local clone; the lab
  notebook is now [`v2/EXPERIMENT_LOG.md`](../v2/EXPERIMENT_LOG.md).
- The docstring of `fig_honesty.py` says that the re-verified run of the 8B student is not plotted, but the script's
  data row for that student is the verified run (80.9 / 85.4 / 83.2 %).
