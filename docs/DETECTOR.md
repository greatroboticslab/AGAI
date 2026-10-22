[← README](../README.md) · [Benchmark](BENCHMARK.md) · [Classifier](CLASSIFIER.md) · [Synthetic data](SYNTHETIC_DATA.md) · [VLM honesty](VLM_HONESTY.md) · [Captioning](CAPTIONING.md) · [Test-time compute](TEST_TIME_COMPUTE.md) · **Detector** · [Compute](COMPUTE.md) · [Roadmap](ROADMAP.md)

# Plant-part detector

**Role in the pipeline.** RF-DETR detects which plant parts are visible (flower, fruit, leaf, root, soil, stem). The
detections constrain the explainer so it only describes parts that are in the photo, and drive part-specific
treatment advice from the knowledge graph ([v1 system](V1_SYSTEM.md)).

**Result.** Moving from RF-DETR Small to Large is a cheap, like-for-like upgrade: **+3.8 points mAP@50:95** (0.593 →
0.631) with the same code, data and settings, ~1.2 GPU-hours of training, same Apache-2.0 licence.

## Results

Evaluation with RF-DETR 1.11.2 on the lab's own 322-image validation split; every model trained with the lab's
configuration (`rfdetr_parts/rfdetr_config.yaml`) for 100 epochs on one RTX 5000 Ada
([`cluster/detector/rfdetr_size.py`](../cluster/detector/rfdetr_size.py)), with two changes forced by the newer RF-DETR: the yaml's
resolution of 560 px is rejected (patch 16 × 2 windows), so each size trains at its default resolution, and
`lr_min_factor` is not accepted and was dropped.

| Model | Input | mAP@50:95 | mAP@50 | mAP@75 | Training time |
|---|---|---|---|---|---|
| Small, the lab's checkpoint (v1, older RF-DETR) | 560 | 0.598 | 0.772 | 0.643 | — |
| Small, retrained (this work) | 512 | 0.593 | 0.765 | 0.639 | 0.73 h |
| Medium | 576 | 0.614 | 0.773 | 0.663 | 0.89 h |
| **Large** | 704 | **0.631** | **0.788** | **0.678** | 1.20 h |

The lab's Small checkpoint additionally scores mAR 0.781, precision 0.832, recall 0.665, F1 0.738. The retrained Small
reproduces it within 0.5 points, so the comparison across sizes is like-for-like. Large trains with batch 4 ×
gradient accumulation 4 to fit 32 GB.

## Caveats

- **There is no separate detector test split.** As in the lab's setup, the validation split is also the one the best
  checkpoint is selected on, so absolute numbers are optimistic; the *differences* between sizes are measured under
  the same protocol.
- Detection quality matters to the extent that it changes what the explainer says. The v1 paper's planned
  "RF-DETR on/off" ablation (does part grounding reduce hallucination?) has not been run yet.

## Next

| Task | Why |
|---|---|
| Carve a detector test split from new annotations | unbiased absolute numbers |
| Compare against SAM 3 (promptable concept segmentation; weights available) | zero-shot alternative, masks instead of boxes |
| RF-DETR on/off ablation on the honesty and claim-level metrics | measure the value of grounding, not just mAP |
