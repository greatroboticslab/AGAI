# `docs/figures/`: figures of the documentation pages

The figures of the [documentation pages](../README.md) of v2, the 2026 revival, the script that draws them, and the
project logo. The figures were committed on 22 October 2026; the honesty figure was redrawn on 18 November 2026 with the
verified run of the student. v1 (the original system)
appears in them only as a baseline, for example the lab's ResNet-50 in the classifier figure.

| Figure (light and dark PNG) | What it shows | Shown in |
|---|---|---|
| `classifier_light.png`, `classifier_dark.png` | split v1 test accuracy by method, mean ± s.d. over seeds, from the lab's ResNet-50 (90.2 %) to two ensembles (97.2 and 97.7 %) | [Classifier](../CLASSIFIER.md), the [repository README](../../README.md) |
| `synthetic_lowq_light.png`, `synthetic_lowq_dark.png` | accuracy on clean test photographs and on 128-pixel thumbnails of them, for SigLIP 2 fine-tunes of eight training arms (synthetic data, low-quality augmentation) | [Synthetic data](../SYNTHETIC_DATA.md) |
| `honesty_light.png`, `honesty_dark.png` | label-swap test: share of true labels accepted against share of wrong labels rejected, for five zero-shot VLMs and two distilled 8B students | [VLM honesty](../VLM_HONESTY.md) |
| `hallucination_filters_light.png`, `hallucination_filters_dark.png` | real claims removed against false claims removed, for three consistency checks and seven judge settings, on 562 human-rated claims | [Captioning](../CAPTIONING.md) |
| `examiner_errors_light.png`, `examiner_errors_dark.png` | share of each of four captioners' claims rated "not there", with 95 % Wilson intervals (interim ratings, 615 of 845 claims) | [Captioning](../CAPTIONING.md) |
| `voting_light.png`, `voting_dark.png` | majority-vote accuracy against the number of sampled zero-shot diagnoses for three models, with two fine-tuned models as reference lines | [Test-time compute](../TEST_TIME_COMPUTE.md) |
| `answer_format_light.png`, `answer_format_dark.png` | zero-shot accuracy of three models under four answer formats: three that give the answer first or no explanation, one with an evidence sentence before the answer | [Test-time compute](../TEST_TIME_COMPUTE.md) |
| `AGAILogo.png` | the project logo, 1024 × 1024 pixels: a seedling whose roots are drawn as circuit traces. Added on 21 January 2026, moved here from the repository root on 18 November 2026; no page or script uses it | none |

Each page shows a figure through a `<picture>` element: the dark PNG when the reader's system uses a dark colour scheme,
the light PNG otherwise.

## Regenerating the figures

[`make_figures.py`](make_figures.py) draws all fourteen figure PNGs. It reads no data files: every number is written
into the script, copied from the lab notebook ([`v2/EXPERIMENT_LOG.md`](../../v2/EXPERIMENT_LOG.md)), with the job IDs
and scoring scripts in comments. It writes `<name>_light.png` and `<name>_dark.png` next to itself, whatever the working
directory:

```bash
python docs/figures/make_figures.py
```

It needs matplotlib. It asks for the Segoe UI font and falls back to DejaVu Sans, so the text can differ slightly from
machine to machine. The PNGs are stored in Git LFS (the `*.png` rule in `.gitattributes`).

## Known problems

- Comments in `make_figures.py` name files by their paths before 18 November 2026: `cluster/EXPERIMENT_LOG.md` is now
  `v2/EXPERIMENT_LOG.md`, and `cluster/<study>/<script>` is now `v2/<study>/<script>`.
