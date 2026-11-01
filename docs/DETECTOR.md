[← README](../README.md) · [Benchmark](BENCHMARK.md) · [Classifier](CLASSIFIER.md) · [Synthetic data](SYNTHETIC_DATA.md) · [VLM honesty](VLM_HONESTY.md) · [Captioning](CAPTIONING.md) · [Test-time compute](TEST_TIME_COMPUTE.md) · **Detector** · [Compute](COMPUTE.md) · [Roadmap](ROADMAP.md)

# Plant-part detector

**Role in the pipeline.** RF-DETR detects which plant parts are visible (flower, fruit, leaf, root, soil, stem). The
detections constrain the explainer so it only describes parts that are in the photo, and drive part-specific
treatment advice from the knowledge graph ([v1 system](V1_SYSTEM.md)).

**Result.** Moving from RF-DETR Small to Large is a cheap, like-for-like upgrade: **+3.8 points mAP@50:95** (0.593 →
0.631) on the lab's validation split and **+2.9 points** (0.564 → 0.593) on the clean test split, with the same code,
data and settings, ~1.2 GPU-hours of training, same Apache-2.0 licence. The clean test split (added 2026-10-30) also
shows that the validation-split figures are optimistic by about ten points: the lab's random split shares near-copies
with its training data.

## Results on the lab's validation split

Evaluation with RF-DETR 1.11.2 on the lab's own 322-image validation split; every model trained with the lab's
configuration (`rfdetr_parts/rfdetr_config.yaml`) for 100 epochs on one RTX 5000 Ada
([`cluster/detector/rfdetr_size.py`](../cluster/detector/rfdetr_size.py)), with two changes forced by the newer RF-DETR: the yaml's
resolution of 560 px is rejected (patch 16 × 2 windows), so each size trains at its default resolution, and
`lr_min_factor` is not accepted and was dropped. The lab's own checkpoint was trained at 512 px as well, not at the
yaml's 560: its position embeddings hold 32 × 32 patches of 16 px, and the lab's `train_rfdetr.py` builds
`RFDETRSmall()` at the library default without applying the yaml value (lab notebook, 2026-10-30).

| Model | Input | mAP@50:95 | mAP@50 | mAP@75 | Training time |
|---|---|---|---|---|---|
| Small, the lab's checkpoint (v1, older RF-DETR) | 512 | 0.598 | 0.772 | 0.643 | — |
| Small, retrained (this work) | 512 | 0.593 | 0.765 | 0.639 | 0.73 h |
| Medium | 576 | 0.614 | 0.773 | 0.663 | 0.89 h |
| **Large** | 704 | **0.631** | **0.788** | **0.678** | 1.20 h |

The lab's Small checkpoint additionally scores mAR 0.781, precision 0.832, recall 0.665, F1 0.738. The retrained Small
reproduces it within 0.5 points, so the comparison across sizes is like-for-like. Large trains with batch 4 ×
gradient accumulation 4 to fit 32 GB.

## Results on a clean test split (2026-10-30)

The lab's validation split is also the split the best checkpoint is selected on, and it shares near-copies with the
lab's training data, so a detector split that agrees with the benchmark's [split v1](BENCHMARK.md) was carved
([`cluster/detector/make_detector_split.py`](../cluster/detector/make_detector_split.py); report
[`evaluation/paper_data/detector_split_v1.json`](../evaluation/paper_data/detector_split_v1.json)). The 1075 annotated
images were matched to the 1576 condition photographs by perceptual hash (64-bit DCT pHash, Hamming distance ≤ 6; 646
matched, near-copies kept together); every annotated copy of a split v1 test photograph goes to the detector test set, the
others follow their photograph, and the 429 annotation-only images go to training: **train 845 / val 67 / test 163**.
The test images are copies of 151 split v1 test photographs (healthy 74, root rot 30, frost 24, gray mold 12, white mold
7, drought 2, overwatered 2), so the split is light on the molds and the water stresses. Small and Large were retrained
with the lab configuration (100 epochs, checkpoint chosen on the 67-image validation split) and scored with pycocotools
([`cluster/detector/eval_parts.py`](../cluster/detector/eval_parts.py); files `evaluation/paper_data/parts_*_splitv1_{valid,test}.json`).

| Model (163 test images) | mAP@50:95 | mAP@50 | mAP@75 | AP flower | fruit | leaf | root | soil | stem |
|---|---|---|---|---|---|---|---|---|---|
| Small, retrained | **0.564** | 0.716 | 0.608 | 0.687 | 0.702 | 0.578 | 0.288 | 0.660 | 0.471 |
| **Large**, retrained | **0.593** | 0.720 | 0.641 | 0.734 | 0.734 | 0.616 | 0.293 | 0.650 | 0.531 |
| Small, the lab's checkpoint (leaky) | 0.681 | 0.834 | 0.738 | 0.756 | 0.789 | 0.693 | 0.522 | 0.716 | 0.609 |

Roots and stems are the hard parts. The lab's checkpoint scores highest on these images only because its random
split had put near-copies of them into its training data (its own validation split gives it 0.598), so 0.564 (Small)
and 0.593 (Large) are the numbers to quote, and the earlier detector figures were optimistic by roughly ten points.

What the pipeline actually uses is image-level part presence (is a part visible at all, at the application's score
thresholds: flower and fruit 0.50, leaf, stem and soil 0.40, root 0.30). Presence F1 on the test split: Small
0.87 / 0.96 / 0.97 / 0.86 / 0.72 / 0.92 and Large 0.86 / 0.96 / 0.98 / 0.84 / 0.72 / 0.93 for flower / fruit / leaf /
root / soil / stem. Thresholds chosen by F1 on the 67-image validation split (4 to 25 images per part) do no better on
the test split, so the hand-set thresholds stay; the sweeps are in the `*_valid.json` files.

## Caveats

- **The validation-split table is optimistic.** The lab's random split is also the one the best checkpoint is selected
  on, and it leaks near-copies from training; the *differences* between sizes hold up on the clean split (+2.9 vs
  +3.8 points), the absolute numbers do not.
- **The clean test split is small for several conditions** (2 to 12 images of the molds and water stresses), because
  the annotated images overlap the condition photographs mostly through healthy, frost and root-rot photos.
- Detection quality matters to the extent that it changes what the explainer says, and for the v1 writer it changes
  little. The "RF-DETR on/off" ablation (lab notebook, 2026-10-31; `cluster/vlm/score_reports.py`) scored the report
  writer retrained without split v1 test images on 151 test photographs with part masks, with and without the detector's
  part lists in the prompt: the part-hallucination rate was 86.1 % without the lists and 92.1 % with them (paired exact
  McNemar P = 0.064), and the writer named a part the prompt declared not visible in 91.4 % of reports. Most false
  mentions are of soil in care advice; without soil the rates are 33.8 % vs 39.1 % (P = 0.28). The v1 writer does not
  follow the prompt's part constraints, so the detector's lists carry no measurable grounding benefit for it.
- **The application's detector call is slow for a reason unrelated to the model.** `grounding/detector.py` runs
  RF-DETR in a fresh interpreter for every photograph, so each call pays the interpreter start and the model load:
  15.9 s median per photograph on an RTX 5000 Ada against 0.016 s for the forward pass in-process (lab notebook,
  2026-10-31, job 72394; `cluster/detector/latency_rfdetr.py`, `cluster/vlm/latency_v1.py`). That is a quarter of
  v1's 62 s per photograph.

## Next

| Task | Why |
|---|---|
| Annotate parts on more mold and water-stress photographs | the clean test split has 2 to 12 images of those conditions |
| Compare against SAM 3 (promptable concept segmentation; weights available) | zero-shot alternative, masks instead of boxes |
| Grounding the explainer cannot ignore: constrain or check reports against the detector's parts after generation | the prompt-level constraint had no effect on the v1 writer (ablation, 2026-10-31) |
| Keep the detector resident in the application process (load once, call per photograph) | the per-call subprocess costs 15.9 s of v1's 62 s per photograph (latency measurement, 2026-10-31) |
