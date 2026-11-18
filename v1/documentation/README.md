# `v1/documentation/`: a 2025 session note and the archived 2025 draft

Part of v1, the original system (2025 to April 2026). Two older documents: a session note on training commands and
data, and, under `archive/legacy_paper_2025/`, the thesis-style draft that the repository kept at its root as
`RESEARCH_PAPER.md` until 25 April 2026, when the April 2026 technical report
([`../RESEARCH_PAPER.pdf`](../RESEARCH_PAPER.pdf)) replaced it. No code uses them.

## Files

| File | What it is |
|---|---|
| [`session_2024_12_17.md`](session_2024_12_17.md) | session notes: the MiniGPT-v2 and ResNet-50 training commands, the training config's main values (up to 500 text tokens, LoRA r 64 / α 128, learning rate 1e-5, 20 epochs), the change of the 2,159 training records' paths from `/data/kiriti/MiniGPT-4` to `/data/AGAI/MiniGPT-4`, and the number of records per condition |
| [`archive/legacy_paper_2025/RESEARCH_PAPER_legacy_2025.md`](archive/legacy_paper_2025/RESEARCH_PAPER_legacy_2025.md) | "Dual-Model Architecture for Plant Disease Diagnosis": the ResNet-50 classifier, with MiniGPT-v2 explanations proposed as future work; reports 96.0 % ResNet-50 accuracy on a balanced 350-image test set (50 per condition); title page dated January 2025; identical to the root `RESEARCH_PAPER.md` it replaced |
| [`archive/legacy_paper_2025/RESEARCH_PAPER_legacy_2025.docx`](archive/legacy_paper_2025/RESEARCH_PAPER_legacy_2025.docx) | the same draft as a Word file (its document properties give 15 November 2025 as the creation date), without images |

## Notes

- **Dates.** The session note's title gives 17 December 2024, which is earlier than the repository's first commit
  (28 June 2025) and than the path change it records (committed on 17 October 2025); the note itself was committed on
  21 January 2026. Its counts per condition add up to the 2,159 records of
  [`stage2_train_7class_fixed.json`](../plant_diagnostic/datasets/stage2_train_7class_fixed.json).
- **Figures of the draft.** Its Figures 1 to 4 are in [`../figs/`](../figs/) (architecture diagram, sample
  images, per-class metrics, confusion matrix); the draft cites them by number only. The scripts it names for results
  and figures (`run_quick_tests.py`, `create_paper_figures.py`, `create_architecture_diagram.py`,
  `create_sample_images_grid.py`) are not in the repository.
- **Test sets.** The draft's 350-image test set is not the 70-photograph holdout of the April 2026 report or the
  leak-free split v1 of the 2026 revival ([BENCHMARK.md](../../docs/BENCHMARK.md)), so its accuracy cannot be compared
  with theirs.
