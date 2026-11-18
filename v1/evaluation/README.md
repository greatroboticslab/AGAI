# Evaluation Framework for Plant Diagnostic System

This directory contains comprehensive evaluation scripts for gathering metrics for the research paper.

## Quick Start

### Basic Usage

Evaluate on a list of images with ground truth labels:

```bash
python evaluation/eval_comprehensive.py \
  --cfg-path eval_configs/minigptv2_eval.yaml \
  --resnet-path plant_diagnostic/models/resnet_straw_final.pth \
  --image-list path/to/image1.jpg path/to/image2.jpg \
  --ground-truth healthy drought \
  --output-dir evaluation/results
```

### Using a Dataset

Evaluate on a test dataset:

```bash
python evaluation/eval_comprehensive.py \
  --cfg-path eval_configs/minigptv2_eval.yaml \
  --resnet-path plant_diagnostic/models/resnet_straw_final.pth \
  --dataset plant_diagnostic/jsondatasets/strawberry_stage2_val.json \
  --images-dir plant_diagnostic/data/holdout \
  --output-dir evaluation/results \
  --max-samples 100
```

## Output Files

The evaluation generates several output files in the specified output directory:

1. **`comprehensive_evaluation_results.json`**: Complete results in JSON format
2. **`confusion_matrix.csv`**: Confusion matrix as CSV
3. **`per_image_results.jsonl`**: Detailed results for each image
4. **`evaluation_summary.txt`**: Human-readable summary

## Metrics Collected

### ResNet Classification Metrics
- Overall accuracy
- Per-class precision, recall, F1-score
- Macro and weighted averages
- Confusion matrix
- Average confidence

### MiniGPT Explanation Metrics
- Diagnosis agreement rate (with ResNet)
- Structured output compliance
- Average inference time
- Per-image explanations

### System-Level Metrics
- End-to-end accuracy (both components correct)
- Component-wise breakdown
- Error categorization

## Next Steps

1. **Run baseline evaluation** on test set
2. **Implement hallucination detection** (see `PAPER_EVALUATION_FRAMEWORK.md`)
3. **Set up expert evaluation interface** for manual annotation
4. **Generate paper tables and figures** from results

## See Also

- `PAPER_EVALUATION_FRAMEWORK.md`: Complete evaluation framework documentation
- `eval_comprehensive.py`: Main evaluation script




