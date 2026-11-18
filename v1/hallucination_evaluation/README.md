# `v1/hallucination_evaluation/`: hallucination checks of v1 reports (December 2025)

Part of v1, the original system (2025 to April 2026). Scripts that measure how often the MiniGPT-v2 report writer
names signs of a condition other than the diagnosed one or describes plant parts that are not in the photograph, with
keyword rules and with sentence embeddings, and their results of 16 and 17 December 2025. They run ResNet-50 and
MiniGPT-v2 with their own prompt and without the RF-DETR part detector (added in April 2026); neither the April 2026
report nor the 2026 revival uses this folder.

## Overview

The framework evaluates two types of hallucination:

1. **Wrong-disease hallucination (misdiagnosis text):** the report contains a phrase listed for a different condition
   than the one it was given. Example: "gray fuzzy growth" in a report on drought (the phrase is listed for gray mold).
2. **Visibility hallucination (non-apparent claims):** the report describes a plant region that is not visible in the
   photograph. Example: describing the roots when no roots are visible. What counts as visible is taken from the
   training report of a photograph with the same file name (see Metrics).

## Files

| File | Description |
|---|---|
| [`config.py`](config.py) | paths (relative to `v1/`), the seven condition names, label aliases, the wrong-condition phrases (`UNIQUE_INDICATORS`), plant-region keywords (`PLANT_REGIONS`), claim words, defaults (50 photographs, temperature 0.2, 500 new tokens) |
| [`hallucination_checker.py`](hallucination_checker.py) | `HallucinationChecker`: the three checks and the combined score for one report; `compute_summary()` for a list of results |
| [`evaluate.py`](evaluate.py) | main evaluation (GPU): ResNet-50 label (or the folder label), one or more MiniGPT-v2 reports per photograph, checks, summary → `results/eval_<timestamp>.json` (or `--output NAME`) |
| [`quick_test.py`](quick_test.py) | smoke test: the checker on a sample text, then, if CUDA is available, model loading and one report |
| [`analyze_training_responses.py`](analyze_training_responses.py) | text-only analysis of the 2159 training reports: symptom words per condition, overlaps between conditions, presence of the wrong-condition phrases (no GPU) → `results/training_analysis.json` |
| [`report_generator.py`](report_generator.py) | text, Markdown or CSV report from a result file, or a comparison of several, written to the working directory |
| [`semantic_evaluator.py`](semantic_evaluator.py) | `SemanticEvaluator`: sentence-embedding scores with `all-MiniLM-L6-v2` (similarity to training reports, coverage of training symptom sentences); its LLM-judge method is a stub |
| [`evaluate_semantic.py`](evaluate_semantic.py) | re-scores an `evaluate.py` result file with `SemanticEvaluator` → `<input>_semantic.json` |
| [`test_prompt_styles.py`](test_prompt_styles.py) | compares four prompt styles (two of them identical: no system prompt and no question) on up to two holdout photographs per condition by symptom-word overlap with the training reports → `results/prompt_comparison.json` |
| [`IMPROVEMENT_GUIDE.md`](IMPROVEMENT_GUIDE.md) | notes on the limits of these metrics (it quotes the 59 %, 18.7 % and 63.4 % of the files in `results/`) and ideas for better checks |
| [`__init__.py`](__init__.py) | package exports (`HallucinationChecker`, `compute_summary`, the phrase and region lists); importing the package fails, see Known problems |
| [`results/test_run.json`](results/test_run.json) | `evaluate.py`, 16 December 2025: 4 holdout photographs, one report each |
| [`results/improved_eval.json`](results/improved_eval.json) | `evaluate.py`, 16 December 2025: the 20 holdout photographs, one report each with the ResNet-50 label (19 of 20 correct); no wrong-condition or visibility flags, mean grounding score 0.59 |
| [`results/improved_eval_semantic.json`](results/improved_eval_semantic.json) | `evaluate_semantic.py` on `improved_eval.json`, 17 December 2025: training alignment 0.454, symptom grounding 0.187 |
| [`results/improved_eval_semantic_v2.json`](results/improved_eval_semantic_v2.json) | the same input, 17 December 2025: alignment 0.634, grounding 0.187 |
| [`results/improved_eval_semantic_training.json`](results/improved_eval_semantic_training.json) | the same input, 17 December 2025: alignment 0.634, grounding 0.415 |
| [`results/prompt_comparison.json`](results/prompt_comparison.json) | `test_prompt_styles.py`, 17 December 2025: four prompt styles on six holdout photographs |
| [`results/training_analysis.json`](results/training_analysis.json) | `analyze_training_responses.py`: the 2159 training reports (records per condition, frequent symptom words, overlaps) |

The result files were written on the lab server (their image paths start with `/data/AGAI/MiniGPT-4/`) and are stored
with Git LFS. Which version of `semantic_evaluator.py` produced each of the three semantic files is not recorded.

## Quick start

### 1. Without a GPU

```bash
cd v1/hallucination_evaluation
python analyze_training_responses.py        # patterns in the training reports
```

### 2. Quick test

```bash
python quick_test.py --images 5
```

### 3. Full evaluation (GPU)

```bash
# Standard evaluation: 50 samples, 3 runs each, ResNet-50 labels
python evaluate.py --num-samples 50 --runs-per-image 3

# Use ground truth (folder) labels instead of ResNet predictions
python evaluate.py --num-samples 50 --use-gt-label

# Use photographs of the training records instead of the holdout set
python evaluate.py --num-samples 100 --use-train
```

The holdout folder, [`../plant_diagnostic/data/holdout/`](../plant_diagnostic/data/holdout/), holds 20 photographs of
five conditions (no gray or white mold), so a holdout run uses at most 20.

### 4. Reports

```bash
python report_generator.py results/eval_YYYYMMDD_HHMMSS.json                     # text
python report_generator.py results/eval_YYYYMMDD_HHMMSS.json --format markdown   # or csv, all
python report_generator.py results/eval_1.json results/eval_2.json --compare
```

### 5. Semantic re-scoring

```bash
pip install sentence-transformers     # all-MiniLM-L6-v2 is downloaded on first use
python evaluate_semantic.py --results results/improved_eval.json
```

The scripts put their own folder and `v1/` on the import path, so `python hallucination_evaluation/evaluate.py` from
`v1/` works too. `evaluate.py`, `quick_test.py` and `test_prompt_styles.py` need the MiniGPT-v2 environment
([`../README.md`](../README.md#setup)); `analyze_training_responses.py` and `report_generator.py` need only the
standard library.

## Inputs

From [`config.py`](config.py), relative to `v1/`: the knowledge base
[`disease_knowledge_base.json`](../knowledge_graph/disease_knowledge_base.json), the 2159 training records
[`stage2_train_7class_fixed.json`](../plant_diagnostic/datasets/stage2_train_7class_fixed.json), the holdout and
`train_aug` photographs in [`../plant_diagnostic/data/`](../plant_diagnostic/data/),
[`../eval_configs/minigptv2_eval.yaml`](../eval_configs/minigptv2_eval.yaml) with the weights it names, and
`plant_diagnostic/models/resnet_strawberry.pth` (not in the repository). `evaluate.py` imports
[`../resnet_classifier.py`](../resnet_classifier.py) and [`../minigpt4/`](../minigpt4/).

## Metrics explained

### Wrong-disease hallucination rate
- **What it measures:** reports that contain a phrase listed for another condition in `UNIQUE_INDICATORS` (table
  below). A phrase is skipped when "not ", "no ", "without ", "absence of ", "lacks ", "don't ", "doesn't ", "isn't ",
  "aren't ", "never ", "avoid ", "prevent " or "unlike " occurs in the 30 characters before its first occurrence.
- **How it's calculated:** (reports with a wrong-condition phrase) / (all reports).

### Visibility hallucination rate
- **What it measures:** claims about a plant region (roots, crown, leaves, stems, fruit, flowers, soil) that the
  training report of the same file name does not mention. A claim is a claim word (shows, appears, visible, ...)
  followed within 30 characters by a keyword of the region.
- **How it's calculated:** (reports with such a claim) / (all reports).
- **Limitation:** only works for photographs with a training record of the same file name (an `_augNNN` suffix is
  ignored); in `results/improved_eval.json` that was 6 of the 20 holdout photographs.

### Grounding score
- **What it measures:** how many expected signs of the given condition the report mentions. Expected signs are the
  knowledge base's visual indicators plus its first five symptoms; a sign counts when one of its first three words
  longer than four letters appears in the report.
- **Range:** 0.0 to 1.0, computed as matches / min(number of expected signs, 10), capped at 1.0.

### Total hallucination score
- **Range:** 0.0 (no hallucination) to 1.0 (severe hallucination).
- **Calculation:** min(1, 0.3 × wrong-condition phrases + 0.2 × invisible-region claims + 0.3 × (1 − grounding
  score)).

### Semantic scores (`semantic_evaluator.py`)
- **Training alignment:** mean of the five highest cosine similarities between the report and up to 50 training
  reports of the same condition.
- **Symptom grounding:** share of up to 50 symptom sentences (taken from up to 30 training reports of the condition)
  that some sentence of the report matches with a cosine similarity above 0.5.
- **Semantic quality:** the mean of the two (the LLM-judge term is not implemented).

## Disease-specific unique indicators

The phrases in `UNIQUE_INDICATORS` of [`config.py`](config.py). They are not the 16 hallmark phrases of the
application's consistency check ([`../knowledge_graph/README.md`](../knowledge_graph/README.md)).

| Disease | Unique indicators |
|---------|-------------------|
| Drought | crispy brown edges, papery leaves, soil pulling away from pot |
| Overwatering | mushy crown, algae on soil, green moss on soil, yellowing lower leaves |
| Root Rot | red stele, phytophthora, pythium, root sloughing, black mushy roots |
| Frost Injury | black eye symptom, translucent tissue, glassy appearance, ice crystal damage |
| Gray Mold | gray fuzzy growth, dusty gray spores, tan mummified fruit, botrytis cinerea |
| White Mold | black sclerotia, cottony white mycelium, sclerotinia, hard black bodies |
| Healthy | none (left empty to avoid false positives) |

## Output structure

### Evaluation results JSON

The summary of `results/improved_eval.json`, shortened:

```json
{
  "metadata": {
    "timestamp": "20251216_230004",
    "num_samples": 20,
    "runs_per_image": 1,
    "total_evaluations": 20,
    "use_holdout": true,
    "use_resnet_label": true,
    "temperature": 0.2
  },
  "resnet_performance": {"accuracy": 0.95, "avg_confidence": 0.91, "samples": 20},
  "hallucination_summary": {
    "total_samples": 20,
    "wrong_disease_hallucination": {"count": 0, "rate": 0.0, "most_confused_pairs": []},
    "visibility_hallucination": {"count": 0, "rate": 0.0, "most_common_claims": []},
    "grounding": {"avg_score": 0.59, "low_grounding_count": 5},
    "overall": {"any_hallucination_count": 0, "any_hallucination_rate": 0.0, "avg_hallucination_score": 0.123},
    "per_class": {"drought": {"count": 4, "wrong_disease_rate": 0.0, "visibility_rate": 0.0, "avg_grounding": 0.4}}
  },
  "individual_results": [...]
}
```

## Extending the framework

### Adding new unique indicators

Edit `config.py` and add to the `UNIQUE_INDICATORS` dictionary:

```python
UNIQUE_INDICATORS = {
    "new_disease": [
        "unique_symptom_1",
        "unique_symptom_2"
    ],
    ...
}
```

### Adding new plant regions

Edit `config.py` and add to the `PLANT_REGIONS` dictionary:

```python
PLANT_REGIONS = {
    "new_region": ["keyword1", "keyword2"],
    ...
}
```

## Using the result in a script

```bash
# Run evaluation and fail if the hallucination rate is 10 % or more
python evaluate.py --num-samples 100 --runs-per-image 1 --output latest
python -c "
import json
with open('results/latest.json') as f:
    r = json.load(f)
rate = r['hallucination_summary']['overall']['any_hallucination_rate']
exit(0 if rate < 0.10 else 1)
"
```

## Troubleshooting

### CUDA out of memory
- Reduce `--runs-per-image` to 1
- Use a smaller batch of samples with `--num-samples 20`

### No test images found
- The holdout set is `../plant_diagnostic/data/holdout/`; most of its photographs are Git LFS files (`git lfs pull`)
- Use `--use-train` to sample from the training data instead

### Model loading errors
- Verify a GPU is available: `nvidia-smi`
- Check that the weights named in `../eval_configs/minigptv2_eval.yaml` (LLaMA-2-7B-chat and
  `output/minigpt_strawberry/checkpoint_best.pth`) and `../plant_diagnostic/models/resnet_strawberry.pth` exist; none
  of them is in the repository, and the config's paths are lab-server paths ([Paths](../README.md#paths))

## Known problems

- **Package import.** `__init__.py` imports `hallucination_checker.py`, which imports `config` as a top-level module,
  so `import hallucination_evaluation` from `v1/` fails with `ModuleNotFoundError`. Run the scripts by path, which
  puts this folder on the import path.
- **Grounding scores grow during a run.** `check_correct_symptoms()` extends the knowledge base's `visual_indicators`
  list in place, so every call adds five more expected signs for that condition and later reports of the same
  condition score higher for the same text (one drought text checked four times scores 0.4, 0.5, 0.6 and 0.7). Which
  code version produced the scores in `results/` is not recorded, so they may be affected as well.
- **Labels with `--use-train`.** Photographs from the `overwatered` and `frost` folders get the label `unknown`,
  because the condition names in `config.py` (`overwatering`, `frost_injury`) do not occur in those folder names.
- **No seed.** `evaluate.py` does not seed the choice of photographs or the sampling, and the 50 symptom sentences of
  `semantic_evaluator.py` are taken from a set, so with more than 50 candidates they can change from run to run.
- **Stale options and paths.** `--use-holdout` is always on (only `--use-train` changes the source).
  `evaluate_semantic.py`'s docstring shows `--run-fresh` and `--num-samples`, which it does not accept; without
  `--results` it re-scores the newest file in `results/` that is not a semantic one. `test_prompt_styles.py` uses only
  absolute lab-server paths (`/data/AGAI/MiniGPT-4/...`).
