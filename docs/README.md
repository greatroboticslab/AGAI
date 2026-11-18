# `docs/`: write-ups of the 2026 revival

The documentation of v2, the 2026 revival (October to November 2026): one page per study, notes on compute and
reproducibility, the roadmap, and the answers to the bracketed notes of the journal manuscript that compares v1 with
v2. v1 (the original system, 2025 to April 2026) appears in these pages where a study measures or replaces one of its
parts; the guide to v1 itself (setup, demo, training) is [`v1/README.md`](../v1/README.md). The numbers come from the
lab notebook [`v2/EXPERIMENT_LOG.md`](../v2/EXPERIMENT_LOG.md), the code is in [`v2/`](../v2/README.md), and the
benchmark and result files are in [`evaluation/`](../evaluation/README.md).

| Page | Contents |
|---|---|
| [Benchmark](BENCHMARK.md) | split v1 (the revival's leak-free benchmark), auxiliary test sets, the low-quality test suite, statistics and protocol, known defects of the data |
| [Classifier](CLASSIFIER.md) | reproduction of the v1 ResNet-50, zero-shot VLMs, frozen backbones, fine-tuned backbones, ensembles, the resolution shortcut, error analysis |
| [Synthetic data](SYNTHETIC_DATA.md) | the lab's GPT-generated images, a synthetic set designed against the dataset's shortcuts, accuracy on low-detail photos, the roots shortcut |
| [VLM honesty](VLM_HONESTY.md) | annotation schema, the label-swap honesty test, the distilled 8B student, whether the order of the schema matters |
| [Captioning](CAPTIONING.md) | two-pass captioning, the captioner bake-off, blind human ratings (interim), hallucination filters |
| [Test-time compute](TEST_TIME_COMPUTE.md) | voting over sampled diagnoses, disagreement and debate between two models, reference images, answer format |
| [Detector](DETECTOR.md) | RF-DETR Small (v1) against Medium and Large, on the lab's validation split and on a test split that agrees with split v1 |
| [Compute](COMPUTE.md) | cluster hardware, Python environments, serving models larger than one GPU, reproducing a result |
| [Roadmap](ROADMAP.md) | what is done, running and next, with acceptance criteria and decisions (last updated 22 October 2026) |
| [Manuscript Q&A](MANUSCRIPT_QA.md) | the 34 bracketed notes of the journal manuscript's 24 October 2026 build: answers with text to paste and evidence, the notes still open, the result files |
| [`figures/`](figures/README.md) | the figures of these pages, each as a light and a dark PNG, and the script that draws them |

The first nine pages share a navigation bar on their first line: a link back to the repository's
[README](../README.md), then the nine pages in the order of this table, the current page in bold. Manuscript Q&A has
no navigation bar.
