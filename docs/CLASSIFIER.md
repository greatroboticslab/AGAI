[← README](../README.md) · [Benchmark](BENCHMARK.md) · **Classifier** · [Synthetic data](SYNTHETIC_DATA.md) · [VLM honesty](VLM_HONESTY.md) · [Captioning](CAPTIONING.md) · [Test-time compute](TEST_TIME_COMPUTE.md) · [Detector](DETECTOR.md) · [Compute](COMPUTE.md) · [Roadmap](ROADMAP.md)

# Diagnosis: from ResNet-50 to modern backbones

**Question.** How much of the v1 classifier's error is the architecture, and how much is the data? We hold the data
and the lab's training recipe fixed, swap the backbone, and compare everything on split v1
([Benchmark](BENCHMARK.md)) with seeds and paired tests.

**Answer in one line.** A ~10-minute fine-tune of a 0.4B web-pretrained ViT with the lab's own recipe beats the
lab's ResNet-50 by ~5.4 points (95.6 ± 1.3 % vs 90.2 ± 1.5 %); ensembles reach 97.2-97.7 %, which is about the
ceiling that the current labels allow.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="figures/classifier_dark.png">
  <img alt="Dot plot of split v1 test accuracy by method: ResNet-50 90.2 ± 1.5, ResNet-50 plus lab synthetic 90.4 ± 0.9, Qwen3-VL-32B QLoRA 93.7, Qwen3-VL-8B LoRA 94.2 ± 0.3, SigLIP 2 95.6 ± 1.3, EVA-02-L 95.7 ± 1.1, SigLIP 2 with the recommended synthetic recipe 96.3 ± 0.8, ensemble E2 97.2, ensemble X3 97.7" src="figures/classifier_light.png" width="760">
</picture>

## 1. Reproduction of the v1 classifier

The paper's ResNet-50 v4 checkpoint (SHA-256 matches the paper manifest) reproduces the paper's predictions on 68/68
recoverable holdout images (max |Δp| 0.0015); its accuracy there is **60/68**. Retrained with the lab's own script on
split v1 train, ResNet-50 scores **90.2 ± 1.5 %** (5 seeds, macro 85.5 ± 2.0 %). The lab's inference path
(resize + centre crop) differs from its training path (squash to 256 × 256); scoring both over 5 seeds gives 90.2 vs
89.8 %, so the mismatch is harmless.

## 2. Zero-shot VLMs are not competitive diagnosers

Same 7-class prompt with class definitions for every VLM, greedy decoding
([`cluster/vlm/zeroshot_qwen3vl.py`](../cluster/vlm/zeroshot_qwen3vl.py)):

| Model | paper68 | leak-free 147 | all1,580 acc / macro | split v1 test |
|---|---|---|---|---|
| ResNet-50 v4 (lab, supervised) | 60/68 | 85.0 % | — (trained on them) | — |
| MiniGPT-v2, lab fine-tune | 61/68 (53 seen in training) | 73.5 % | — | — |
| MiniGPT-v2 official stage-3, zero-shot | 23/68 | 24.5 % | — | — |
| Qwen3-VL-8B-Instruct | 37/68 | 59.2 % | 74.6 % / 57.5 % | 77.1 % |
| Qwen3-VL-32B-Instruct (4-bit) | 40/68 | 65.3 % | 72.5 % / 58.1 % | 72.4 % |
| Qwen3.8-27B (4-bit, thinking off) | 34/68 | 54.4 % | 66.8 % / 56.3 % | 65.6 % |

- Signature errors: the 32B calls 188/222 white-mold photos "gray mold"; Qwen3.8-27B calls 165/650 gray-mold photos
  "healthy". On all1,580, 8B > 32B (p = 0.031) > Qwen3.8-27B (p = 1.1 × 10⁻⁵).
- **Thinking mode** (Qwen3.8-27B, 1,024-token budget): 87 → 93 / 161 on resnet161 (McNemar p = 0.24) at 20× the
  compute (0.62 → 12.3 s/image). Not worth it for classification.

## 3. Frozen backbones with a linear head

Frozen features + logistic regression, C chosen on val, test scored once
([`cluster/classifier/probe_backbones.py`](../cluster/classifier/probe_backbones.py)):

| Backbone | Test acc | Macro | vs ResNet-50 seed 42 (only this / only ResNet, p) |
|---|---|---|---|
| EVA-02-L/14 448 (MIM, IN-22k fine-tuned) | 95.0 % | 91.5 % | 19 / 6, **p = 0.015** |
| PE-Core-G/14 448 | 94.2 % | **93.2 %** | 18 / 8, p = 0.076 |
| SigLIP 2 so400m/16 512 | 93.7 % | 92.3 % | 23 / 15, p = 0.26 |
| DINOv3 ViT-L/16 | 93.7 % | 92.8 % | — |
| DINOv3 ViT-7B/16 | 93.2 % | 91.8 % | — |
| DINOv2-g/14 448 | 92.7 % | 86.2 % | 15 / 11, p = 0.56 |
| ConvNeXt V2-L 384 (IN-22k) | 92.5 % | 89.2 % | 16 / 13, p = 0.71 |
| BioCLIP 2 (ViT-L/14) | 90.7 % | 84.8 % | 18 / 22, p = 0.64 |

**5-fold grouped cross-validation over all 1,576 images** ([`cluster/classifier/cv_probes.py`](../cluster/classifier/cv_probes.py);
every image predicted once out-of-fold; Wilson 95 % CI) gives a more stable ranking:

| Model | CV accuracy (95 % CI) | Macro |
|---|---|---|
| Ensemble EVA-02 + PE-Core-G + SigLIP 2 | 96.8 % (95.8-97.5) | 94.5 % |
| PE-Core-G/14 448 | 96.3 % (95.2-97.1) | 93.2 % |
| SigLIP 2 so400m 512 | 95.7 % (94.6-96.6) | 92.6 % |
| EVA-02-L 448 | 95.6 % (94.5-96.5) | 92.3 % |
| DINOv3 ViT-L/16 | 95.1 % (93.9-96.1) | 92.2 % |
| ConvNeXt V2-L 384 | 94.3 % (93.0-95.3) | 90.1 % |
| DINOv3 ViT-7B/16 | 94.2 % | 89.9 % |
| PlantNet DINOv2-B (PlantCLEF 2024) | 93.5 % | 89.0 % |
| DINOv2-g | 93.4 % | 89.3 % |
| BioCLIP 2.5 ViT-H | 92.7 % | 88.6 % |
| BioCLIP 2 | 92.3 % | 86.4 % |

- PE-Core-G, SigLIP 2 and EVA-02 are statistically indistinguishable (pairwise p ≥ 0.20). PE-Core-G beats
  ConvNeXt V2 (p = 1.2 × 10⁻⁴) and DINOv2-g (p = 3.6 × 10⁻⁷); EVA-02 beats DINOv2-g (p = 1.3 × 10⁻⁴). The 3-way
  ensemble beats SigLIP 2 (p = 0.0046) and EVA-02 (p = 0.0039) but not PE-Core-G alone (p = 0.18).
- **Species-identification pretraining (BioCLIP, PlantNet) does not transfer to plant-health classes** better than
  general web-scale pretraining. DINOv3 improves on DINOv2 (L: 91.7 → 93.7 % on test), but the 7B is no better than
  the L with ~1,000 training images.

## 4. Fine-tuned backbones with the lab's recipe

[`cluster/classifier/train_backbone.py`](../cluster/classifier/train_backbone.py) reuses the lab's ResNet recipe (same augmentations,
loss, head-then-full schedule, temperature scaling), so the backbone is the only change.

| Method | Seeds | Test accuracy | Macro |
|---|---|---|---|
| EVA-02-L fine-tuned | 3 (96.2 / 94.5 / 96.5) | **95.7 ± 1.1 %** | 93.3 ± 1.4 % |
| SigLIP 2 so400m fine-tuned | 3 (96.7 / 94.2 / 95.7) | **95.6 ± 1.3 %** | 94.0 ± 2.4 % |
| SigLIP 2 + ChatGPT synthetic (w/o frost 3-5) + low-quality aug. | 3 | **96.3 ± 0.8 %** | 94.3 % |
| PE-Core-G/14 fine-tuned (1.88B params) | 1 | 96.7 % | 95.7 % |
| Qwen3-VL-8B LoRA (classification) | 3 (94.0 / 94.2 / 94.5) | 94.2 ± 0.3 % | 90.2 ± 1.7 % |
| Qwen3-VL-32B QLoRA (classification) | 1 | 93.7 % | 88.0 % |
| ResNet-50 + lab GPT synthetic | 5 | 90.4 ± 0.9 % | 84.2 ± 1.3 % |
| ResNet-50, lab recipe | 5 | 90.2 ± 1.5 % | 85.5 ± 2.0 % |

- **SigLIP 2 is the efficient pick**: 0.4B parameters, ~33 s per epoch on one A100, ~10 minutes per fine-tune. Its
  best seed vs ResNet-50 seed 42: 26 / 6, p = 0.00054. PE-Core-G matches it at 4.7× the parameters.
- **VLM fine-tunes** (LoRA r = 16 on all linear layers incl. the vision tower, answer-only loss,
  [`cluster/vlm/train_vlm_lora.py`](../cluster/vlm/train_vlm_lora.py)): the 32B buys nothing over the 8B (10 vs 9 discordant,
  p = 1) at ~4× the training compute. A VLM is worth having for *explanations*, not for the label
  ([VLM honesty](VLM_HONESTY.md)).
- The synthetic-data recipe in row 3 is explained in [Synthetic data](SYNTHETIC_DATA.md); its gain on clean photos is
  within seed noise, its gain on low-quality photos is not.

## 5. Ensembles

Probabilities: VLM first-token softmax over the 7 class names, probe/backbone softmax, ResNet with TTA + temperature
([`cluster/classifier/ensemble.py`](../cluster/classifier/ensemble.py)). Four ensembles were declared before any ensemble result was seen
(equal weights primary); later combinations are marked exploratory.

| Ensemble | Status | Test acc | Macro | vs best member (p) |
|---|---|---|---|---|
| E1: EVA-02 + SigLIP 2 probes | pre-declared | 96.0 % | 93.8 % | 4 / 0, p = 0.12 |
| **E2: 8B LoRA + EVA-02 + SigLIP 2 probes** | pre-declared | **97.2 %** | **96.1 %** | vs 8B 15 / 3, **p = 0.0075**; vs EVA-02 10 / 1, **p = 0.012** |
| E3: ResNet-50 + 8B LoRA | pre-declared | 93.5 % | 89.3 % | 5 / 8, p = 0.58 |
| E4: all five (ResNet, 8B, 32B, EVA-02, SigLIP 2) | pre-declared | 97.2 % | 96.1 % | = E2; val-tuned weights give ResNet weight 0 |
| X2: 6 fine-tuned backbones (SigLIP 2 + EVA-02, 3 seeds each) | exploratory | 97.2 % | 96.2 % | n.s. vs best seed |
| **X3: X2 + 8B LoRA** | exploratory | **97.7 %** | **97.7 %** | vs 8B 15 / 1, p = 0.0005 |

The fine-tuned VLM and the web-scale vision backbones make different mistakes; ResNet adds nothing on top. Seed
averaging removes the seed lottery (single seeds range 94.2-96.7 %).

## Resolution shortcut

Photo size is confounded with the label ([Benchmark](BENCHMARK.md#known-defects-of-the-data)): healthy photos are
mostly small web thumbnails, the mold classes are 512 × 512, the water-stress and frost classes are large phone photos.
A test proposed in the lab discussion: removing detail from a photo must not make the model *more* confident; if it
does, the model reads the degradation itself as evidence ([`cluster/classifier/eval_shortcut.py`](../cluster/classifier/eval_shortcut.py),
SigLIP 2, 3 seeds per training recipe, split v1 test).

| Training recipe (SigLIP 2, 3 seeds each) | Clean test | Healthy accuracy, thumbnails ≤ 300 px (n = 58) | Healthy accuracy, large ≥ 500 px (n = 14) | P(healthy) change when large healthy photos are shrunk to 275 px | Disease photos called healthy after shrinking to 275 px |
|---|---|---|---|---|---|
| real photos only | 95.6 % | 99 % | **64 %** | **+0.071** (rises on 14 / 14) | 0.2 → 1.0 % |
| + designed ChatGPT synthetic, no augmentation | 96.1 % | 98 % | 69 % | **+0.118** | 0.1 → 1.0 % |
| + lab GPT synthetic, no augmentation | 95.7 % | 98 % | **52 %** | **+0.164** | 0.0 → 0.2 % |
| + low-quality augmentation (p = 0.3) | 95.1 % | 97 % | 62 % | +0.055 | 0.5 → 0.4 % |
| + ChatGPT synthetic + augmentation p = 0.3 (recommended) | **96.3 %** | **100 %** | **74 %** | **+0.033** | 0.1 → 0.6 % |
| + ChatGPT synthetic + augmentation p = 0.6 | 95.6 % | 97 % | 67 % | +0.048 | 0.0 → 0.2 % |

- **The models partly learned "large, sharp phone photo → a stress condition".** Large healthy photos are misread as
  frost, drought or overwatered, the classes made of large phone photos, and shrinking them to thumbnail size makes
  every model more confident they are healthy.
- **Shrinking disease photos to thumbnail size rarely turns them "healthy"** (≤ 1 % at 275 px); the effect is larger
  for real-only models than for augmented ones (paired difference +0.004 to +0.005 in P(healthy), 95 % CI excludes 0)
  and grows at 128 px ([Synthetic data](SYNTHETIC_DATA.md#low-detail-robustness)).
- **Synthetic healthy images make it worse, not better.** Every synthetic image is 800-1672 px *in every class*, so the
  synthetic sets add at least as much "large → stress" as "large → healthy", and none adds a thumbnail of a sick plant.
- **More augmentation does not fix it either.** Doubling the degradation probability (0.6) only shrinks the already
  tiny disease-thumbnail effect and costs ~0.7 points of clean accuracy; it cannot show the model what a sharp,
  full-resolution healthy plant looks like.
- The recommended recipe reduces the shortcut most but does not remove it. Only 14 large healthy test photos exist
  (one photo = 7 points), so these accuracies are rough; the P(healthy) shift ranks the recipes the same way.
  **Full-resolution healthy photos from the field are the fix and the proper test** ([Roadmap](ROADMAP.md) C6).

## Error analysis and the label-noise ceiling

E2's 11 test errors, reviewed by eye (not by a pathologist) on a contact sheet
([`cluster/benchmark/error_sheet.py`](../cluster/benchmark/error_sheet.py)):

| Category | Count | Example |
|---|---|---|
| Clear label error | 1 | a white-mold photo with the overlaid caption "Botrytis gray mold"; every model says gray mold |
| Probable label problem | 3 | ripe fruit coated in ice crystals and plants in snow, both labeled healthy |
| Ambiguous photo | 4 | an uprooted drought plug with exposed roots (→ root rot); a collapsed overwatered plant |
| Model error | ~3 | a healthy white flower called frost (flower close-ups are mostly frost photos) |

**At ~97 % the remaining errors are mostly label noise or ambiguity.** Further gains will come from relabeling and
new data, not from model size ([Roadmap](ROADMAP.md)).
