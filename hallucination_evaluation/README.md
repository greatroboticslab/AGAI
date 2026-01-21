# Hallucination Evaluation Framework

A comprehensive evaluation framework for measuring hallucination rates in the MiniGPT-v2 plant diagnostic system.

## Overview

This framework evaluates two types of hallucinations:

1. **Wrong-Disease Hallucination (Misdiagnosis Text)**: When the model mentions symptoms that are unique to a disease different from the diagnosis.
   - Example: Saying "gray fuzz" when the diagnosis is "drought" (gray fuzz is unique to gray mold)

2. **Visibility Hallucination (Non-Apparent Claims)**: When the model describes plant parts that are not actually visible in the image.
   - Example: Describing root condition when roots aren't visible in the photo

## Files

| File | Description |
|------|-------------|
| `config.py` | Configuration settings, paths, and disease-specific indicators |
| `hallucination_checker.py` | Core hallucination detection logic |
| `evaluate.py` | Main evaluation script (requires GPU) |
| `analyze_training_responses.py` | Analyze training data patterns (no GPU needed) |
| `report_generator.py` | Generate reports from evaluation results |
| `quick_test.py` | Quick pipeline verification test |

## Quick Start

### 1. Verify Installation (No GPU Required)

```bash
cd /data/AGAI/MiniGPT-4/hallucination_evaluation

# Analyze training data patterns
python analyze_training_responses.py
```

### 2. Run Quick Test (Requires GPU)

```bash
python quick_test.py --images 5
```

### 3. Run Full Evaluation

```bash
# Standard evaluation: 50 samples, 3 runs each
python evaluate.py --num-samples 50 --runs-per-image 3

# Use ground truth labels instead of ResNet predictions
python evaluate.py --num-samples 50 --use-gt-label

# Use training data instead of holdout set
python evaluate.py --num-samples 100 --use-train
```

### 4. Generate Reports

```bash
# From evaluation results
python report_generator.py results/eval_YYYYMMDD_HHMMSS.json

# Markdown format
python report_generator.py results/eval_*.json --format markdown

# CSV export
python report_generator.py results/eval_*.json --format csv

# Compare multiple evaluations
python report_generator.py results/eval_1.json results/eval_2.json --compare
```

## Metrics Explained

### Wrong-Disease Hallucination Rate
- **What it measures**: Frequency of mentioning symptoms unique to other diseases
- **How it's calculated**: (Responses with wrong-disease symptoms) / (Total responses)
- **Target**: < 5%
- **Uses**: Knowledge base of disease-specific visual indicators

### Visibility Hallucination Rate
- **What it measures**: Frequency of describing non-visible plant parts
- **How it's calculated**: Cross-references generated text against training data descriptions
- **Target**: < 10%
- **Limitation**: Only works for images present in training data

### Grounding Score
- **What it measures**: How many correct symptoms for the diagnosed disease are mentioned
- **Range**: 0.0 (no correct symptoms) to 1.0 (all expected symptoms)
- **Target**: > 60%

### Total Hallucination Score
- **What it measures**: Combined severity score
- **Range**: 0.0 (no hallucination) to 1.0 (severe hallucination)
- **Calculation**: Weighted combination of wrong-disease, visibility, and grounding penalties

## Disease-Specific Unique Indicators

These symptoms are unique to each disease and should NEVER appear in descriptions of other diseases:

| Disease | Unique Indicators |
|---------|-------------------|
| Drought | cracked soil, limp petioles, dull foliage |
| Overwatering | mushy crown, algae growth, waterlogged |
| Root Rot | foul odor, red stele, brown roots |
| Frost Injury | black eye, translucent leaves, ice damage |
| Gray Mold | gray fuzz, dusty spores, botrytis |
| White Mold | white cotton, black sclerotia, sclerotinia |
| Healthy | glossy leaves, upright growth, white roots |

## Output Structure

### Evaluation Results JSON

```json
{
  "metadata": {
    "timestamp": "20231217_123456",
    "num_samples": 50,
    "runs_per_image": 3,
    "total_evaluations": 150
  },
  "resnet_performance": {
    "accuracy": 0.92,
    "avg_confidence": 0.87
  },
  "hallucination_summary": {
    "wrong_disease_hallucination": {
      "count": 5,
      "rate": 0.033,
      "most_confused_pairs": [["drought", "overwatering", 2]]
    },
    "visibility_hallucination": {
      "count": 8,
      "rate": 0.053
    },
    "grounding": {
      "avg_score": 0.72,
      "low_grounding_count": 3
    },
    "overall": {
      "any_hallucination_rate": 0.087,
      "avg_hallucination_score": 0.12
    }
  },
  "individual_results": [...]
}
```

## Extending the Framework

### Adding New Unique Indicators

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

### Adding New Plant Regions

Edit `config.py` and add to the `PLANT_REGIONS` dictionary:

```python
PLANT_REGIONS = {
    "new_region": ["keyword1", "keyword2"],
    ...
}
```

## Integration with CI/CD

The evaluation can be integrated into automated testing:

```bash
# Run evaluation and fail if hallucination rate > 10%
python evaluate.py --num-samples 100 --runs-per-image 1
python -c "
import json
with open('results/latest.json') as f:
    r = json.load(f)
rate = r['hallucination_summary']['overall']['any_hallucination_rate']
exit(0 if rate < 0.10 else 1)
"
```

## Troubleshooting

### CUDA Out of Memory
- Reduce `--runs-per-image` to 1
- Use smaller batch of samples with `--num-samples 20`

### No Test Images Found
- Ensure holdout set exists at `plant_diagnostic/data/holdout/`
- Use `--use-train` to sample from training data instead

### Model Loading Errors
- Verify GPU is available: `nvidia-smi`
- Check model weights exist in `checkpoints/`

## Citation

If using this framework, please cite the MiniGPT-v2 plant diagnostic paper.

