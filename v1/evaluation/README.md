# `v1/evaluation/`: the April 2026 holdout results of v1 and an older evaluator

Two things from v1 (2025 to April 2026): the measured ResNet-50 results on the 70-photograph holdout of the April 2026
technical report, from which the report's classifier tables and figures are built, and an evaluator for the ResNet-50
and MiniGPT-v2 pipeline that does not run as committed. The v2 revival's benchmark and result files are in the
top-level [`../../evaluation/`](../../evaluation/).

## Contents

| Path | Contents |
|---|---|
| [`holdout/`](holdout/) | per-photograph predictions and metrics of ResNet-50 v3 and v4 on the April 2026 holdout (25 April 2026); see [`holdout/README.md`](holdout/README.md) |
| [`eval_comprehensive.py`](eval_comprehensive.py) | evaluator of the ResNet-50 classifier and the MiniGPT-v2 report writer together (21 January 2026); does not run (Known problems) |
| `README.md` | this page |

## `eval_comprehensive.py`

What it is written to do: classify each photograph with `resnet_classifier.py`, ask MiniGPT-v2 for a report with the
ResNet-50 label in the prompt, and compute

- for the classifier: accuracy, per-class precision, recall and F1, macro and weighted averages, the confusion matrix
  and the mean top-1 probability;
- for the reports: how often a report names the ResNet-50 label or an alias ("agreement"), how often it contains
  diagnosis, visible-sign and recommendation keywords, and the generation time;
- for the system: the share of photographs where the classifier is right and the report agrees.

Intended use, from `v1/` (the script adds the folder above its own, `v1/`, to `sys.path` and imports `minigpt4` and
`resnet_classifier` from there):

```bash
python evaluation/eval_comprehensive.py --cfg-path eval_configs/minigptv2_eval.yaml \
    --resnet-path <ResNet-50 checkpoint> --image-list <photo> [<photo> ...] --ground-truth <label> [<label> ...]
# or, instead of --image-list and --ground-truth: --dataset <instruction JSON> --images-dir <folder> [--max-samples N]
```

Other options: `--output-dir` (default `evaluation/results`, relative to the working directory) and `--device`
(default `cuda:0`). It would write `comprehensive_evaluation_results.json`, `confusion_matrix.csv`,
`per_image_results.jsonl` and `evaluation_summary.txt` to the output folder. It needs a GPU environment with the v1
dependencies, a ResNet-50 checkpoint and the MiniGPT-v2 weights named in the config, none of which is in the
repository.

Nothing in the repository runs or imports it, and no result in the repository comes from it.

## Known problems

- **It stops at start-up.** It passes the config path as a string to `minigpt4.common.config.Config`, which expects an
  argument object with `cfg_path` and `options` (`AttributeError`). A few lines later it calls `setup_task`, which its
  imports do not provide: `from minigpt4.tasks import *` exports only `BaseTask` and `ImageTextPretrainTask`
  (`NameError`).
- **Placeholder labels.** With `--dataset`, the ground truth is `healthy` for every record whose conversation contains
  the word "diagnosis" and `unknown` otherwise; `--image-list` without `--ground-truth` gives `unknown` for all.
- **Class names.** Its class list spells `overwatering`, while `resnet_classifier.py` returns `overwatered`, so that
  class would be missing from the per-class metrics and the confusion matrix.
- **Lab-server paths.** `eval_configs/minigptv2_eval.yaml` names its checkpoint, the LLaMA weights and the training data
  under `/data/AGAI/MiniGPT-4/` ([`../README.md`](../README.md#paths)).
