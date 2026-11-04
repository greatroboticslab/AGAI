# AGAI revival: experiment log

Lab notebook for the AGAI strawberry-diagnosis revival on MTSU Hamilton. Newest entries at the bottom, never
rewritten (corrections are added as new dated notes). Canonical copy: `/projects/was4g/agai/MiniGPT-4/evaluation/EXPERIMENT_LOG.md`;
mirror: `cluster/EXPERIMENT_LOG.md` in the local clone.

- **Where things are:** repo `/projects/was4g/agai/MiniGPT-4` (branch `labbox` = the lab box's code + data);
  scripts in `cluster/`; every result file in `evaluation/` holds per-image predictions; Slurm logs in
  `/projects/was4g/agai/logs/<job-name>-<jobid>.out`.
- **Hardware:** research-gpu c3-c8 (1x RTX 5000 Ada 32 GB each), a100 c9 (A100 80 GB), 1 GbE between nodes.
- **Statistics:** paired exact McNemar tests on the same images; "macro" = mean of per-class accuracies.

## Evaluation sets

| name | images | definition |
|---|---|---|
| paper68 | 68 | the paper's holdout (68 of its 70 images could be recovered) |
| resnet161 | 161 | originals not in ResNet v4's training data by content hash (`evaluation/clean_eval_sets.json`) |
| leak-free 147 | 147 | resnet161 minus 14 visual near-copies of ResNet v4 training images (pHash distance <= 6) |
| all1580 | 1580 | every original image (no augmented copies, no synthetic images) |
| **split v1** | 1017 / 161 / 398 | grouped split of all1580, seed 0: near-duplicate groups never cross splits, 4 label-conflict images dropped; per class test >= 25%, val >= 10% (`evaluation/split_v1.json`, `cluster/make_split.py`). **The main benchmark from 2026-10-20 on.** |

Split v1 test per class: drought 17, frost 29, gray_mold 163, healthy 81, overwatered 19, root_rot 33, white_mold 56.

## 2026-10-19: reproduction of the lab's state

- Lab code + data rebuilt from Hongbo's export; git status matches the lab box (1148 deleted, 6 modified).
  Weight SHA-256s match the paper manifest (ResNet v4 `3d41c944...`, RF-DETR `0274206e...`).
- **ResNet v4 on paper68: predictions identical to the paper's on ~~all~~ 68/68 images (max |dp1| 0.0015), accuracy
  60/68** (`evaluation/holdout/repro_resnet_v4_hamilton.json`, job 71970), after recovering 3 paper-era images whose
  file names had since been reused for different photos. *(Corrected 2026-10-20 03:40, see note at the end.)*
- The lab's full pipeline (`test_grounding.py`: ResNet -> RF-DETR -> MiniGPT-v2 report) runs end to end (job 71977).
  Reports contain hallucinated details. MiniGPT needs transformers 4.45.2: under 4.41 the lab's `cache_position`
  shim makes `generate()` re-feed every token (quadratic KV cache, OOM on 32 GB, repetitive text).

## 2026-10-19/20: zero-shot VLMs vs the lab's models

Same 7-class prompt (with class definitions) for every VLM; greedy, 12 new tokens.

| model | paper68 | leak-free 147 | all1580 (acc / macro) |
|---|---|---|---|
| ResNet-50 v4 (lab) | ~~68/68~~ 60/68 | 85.0% | n/a (trained on them) |
| MiniGPT-v2 lab fine-tune | 61/68 (trained on 53 of them; 11/15 unseen) | 73.5% (62.5% on its 96 unseen) | n/a |
| MiniGPT-v2 official stage-3 [vqa], zero-shot | 23/68 | 24.5% | n/a |
| Qwen3-VL-8B-Instruct | 37/68 | 59.2% | 74.6% / 57.5% |
| Qwen3-VL-32B-Instruct (4-bit) | 40/68 | 65.3% | 72.5% / 58.1% |
| Qwen3.8-27B (4-bit, thinking off) | 34/68 | 54.4% | 66.8% / 56.3% |

- all1580 McNemar: 8B vs 32B p = 0.031; 8B vs Qwen3.8 p = 2e-12; 32B vs Qwen3.8 p = 1.1e-5.
- Signature errors: the 32B calls 188/222 white mold "gray mold"; Qwen3.8 calls 165/650 gray mold "healthy".
- Speed fix: Qwen-VL's patch-embed Conv3d (kernel == stride) replaced by an identical `F.linear`: 8B inference
  9.73 -> 0.22 s/img. Staging weights on node-local NVMe: load 8B 296 s -> 5 s, 32B 928 s -> 69 s.

## 2026-10-20: data hygiene

- pHash near-duplicate audit (`cluster/near_dupes.py`, `evaluation/near_dupes.json`): 2 groups filed under two
  different labels; 14 of the 161 "clean" test images have a near-copy in ResNet v4's training data.
- Built split v1 (above) so that no near-duplicate group spans train/val/test.

## 2026-10-20: split v1 baselines

| model | test acc | macro | drought | frost | gray_mold | healthy | overwatered | root_rot | white_mold |
|---|---|---|---|---|---|---|---|---|---|
| ResNet-50, lab script retrained on split v1 train (job 71993, seed 42) | 365/398 = 91.7% | 87.7% | 14/17 | 20/29 | 161/163 | 70/81 | 17/19 | 32/33 | 51/56 |
| Qwen3-VL-8B zero-shot | 307/398 = 77.1% | 61.7% | 8/17 | 10/29 | 150/163 | 73/81 | 3/19 | 32/33 | 31/56 |
| Qwen3-VL-32B zero-shot (4-bit) | 288/398 = 72.4% | 60.3% | 10/17 | 16/29 | 149/163 | 70/81 | 4/19 | 32/33 | 7/56 |
| Qwen3.8-27B zero-shot (4-bit) | 261/398 = 65.6% | 57.7% | 6/17 | 11/29 | 93/163 | 78/81 | 2/19 | 32/33 | 39/56 |

ResNet run: best val 92.5%, fitted temperature 0.748 (`/projects/was4g/agai/models/resnet50_split_v1.pth`).

## 2026-10-20: thinking mode (Qwen3.8-27B, budget 1024 tokens, job 71992)

- resnet161: 87/161 -> 93/161 with thinking (6 vs 12 discordant, McNemar p = 0.24); leak-free 147: 80 -> 85 (p = 0.33).
- Cost: 0.62 -> 12.3 s/img (20x), mean 464 thinking tokens, 20/161 answers forced after the budget ran out.
- **Conclusion:** no significant gain for 20x compute. Not worth it for classification.

## 2026-10-20: LoRA fine-tunes on split v1 (`cluster/train_vlm_lora.py`)

Same prompt as zero-shot; answer-only loss; LoRA r=16, alpha=32, dropout 0.05 on all linear layers (vision tower
included); lr 1e-4 cosine with 5% warmup; 3 epochs; best epoch picked on val macro; test scored once.

- **Qwen3-VL-8B, bf16, 1 GPU, effective batch 8 (job 72003):** val macro by epoch 83.2 / 89.6 / 88.3 -> epoch 2
  kept. **Test 374/398 = 94.0%, macro 89.5%** (drought 13/17, frost 26/29, gray_mold 161/163, healthy 77/81,
  overwatered 15/19, root_rot 33/33, white_mold 49/56). vs retrained ResNet: 14 vs 23 discordant, p = 0.19 (not
  significant). The two make different mistakes (8B better on frost/healthy, ResNet on overwatered/white mold), a
  hint that an ensemble may beat both. 0.57 s/img training, ~35 min total.
- **Qwen3-VL-32B QLoRA** (pre-quantized NF4 copy, effective batch 12 = 4 x 3 GPUs, job 72004): epoch 1 val
  145/161 = 90.1%, macro 82.6%. *Result pending.*
- **Multi-node data parallel works over 1 GbE for LoRA** (bf16 all-reduce of the 143M adapter weights per step):
  2 nodes 3.07 s/img/GPU (25% of it gradient sync); 3 nodes 3.02 s/img/GPU (35% sync), ~2.3x one GPU. The nodes'
  10 GbE ports have no IPs configured; asking HPC to configure them would cut sync time ~10x.

## 2026-10-20 night: in progress

- **Synthetic-data ablation:** the lab's 601 synthetic images (`plant_diagnostic/data/train_aug/<class>/synth_*GPT*`,
  generated with the OpenAI API in April) added to split v1 train (1 exact duplicate dropped; none is a near-copy of
  any val/test image). ResNet with the lab recipe, 5 seeds with vs without, real-only val/test
  (`cluster/build_synth_split.py`, `cluster/slurm/train_resnet_ablation.sbatch`, `evaluation/split_v1_gptsynth.json`).
- **Backbone probes** (frozen features + logistic regression; `cluster/probe_backbones.py`): DINOv2-g/L, SigLIP2-so400m,
  ConvNeXtV2-L, EVA02-L, BioCLIP 2, then BioCLIP 2.5, PE-Core-G, PlantNet DINOv2.
- **Ensembles:** per-class probabilities from the VLMs (`cluster/eval_vlm_scores.py`) and ResNet
  (`cluster/resnet_scores.py`); combination rules fixed on val (`cluster/ensemble.py`).
- **ResNet preprocessing mismatch:** the lab trains on images squashed to 256x256, but `resnet_classifier.py`
  (demo and paper eval) resizes and center-crops, cutting ~15% of the long edge at inference. Scoring both.
- **GPT annotation** (`cluster/annotate_gpt.py`, gpt-6.1-sol, structured JSON: caption, grounded findings, report,
  QA pairs incl. "not visible" questions, label-consistency audit): blocked, the OpenAI org has no credits
  (`credit_balance_exhausted`); no charges incurred.
- **Gated models:** DINOv3 and SAM 3 access requests submitted, awaiting Meta's manual review.

## 2026-10-20 03:40: correction

- The first version of this log said ResNet v4 scores 68/68 on paper68. Wrong: 68/68 is the *agreement* with the
  paper's own predictions (the reproduction check); its *accuracy* there is 60/68, the paper's own 88.6% on all 70
  minus the 2 images that could not be recovered. Fixed above with strike-through.

## 2026-10-20 night: results as they came in

- **DINOv3 and SAM 3 access granted** (HTTP 200 at 03:20); weights downloading (`dinov3-vitl16`, `-vith16plus`,
  `-vit7b16`, `sam3`).
- **Frozen backbone + logistic regression on split v1** (job 72020, A100; C picked on val macro; test once):

| backbone | test acc | macro | vs ResNet seed 42 (only this / only ResNet, p) |
|---|---|---|---|
| EVA-02-L/14 448 (MIM, IN-22k ft) | 378/398 = 95.0% | 91.5% | 19 / 6, **p = 0.015** |
| SigLIP 2 so400m/16 512 | 373/398 = 93.7% | **92.3%** | 23 / 15, p = 0.26 |
| DINOv2-g/14 448 | 369/398 = 92.7% | 86.2% | 15 / 11, p = 0.56 |
| ConvNeXt V2-L 384 (IN-22k) | 368/398 = 92.5% | 89.2% | 16 / 13, p = 0.71 |
| DINOv2-L/14 448 | 365/398 = 91.7% | 86.9% | 13 / 13, p = 1 |
| BioCLIP 2 (ViT-L/14 224) | 361/398 = 90.7% | 84.8% | 18 / 22, p = 0.64 |

  A frozen modern backbone with a linear head already matches or beats the fine-tuned ResNet; EVA-02 does so
  significantly. Biology-specific pretraining (BioCLIP 2, species ID) does not help for plant *health*.
- **ResNet-50 seed variance** (lab recipe, split v1, seeds 42/1/2/3/4): test 91.7 / 91.5 / 90.2 / 88.9 / 88.4% ->
  **90.2 +- 1.5%**, macro 85.5 +- 2.0%. The seed-42 run used as the reference so far is the best of five, so every
  "vs ResNet" comparison in this log is conservative.
- **Qwen3-VL-32B QLoRA (job 72004, 3 nodes):** val macro by epoch 82.6 / 89.0 / 93.6 -> epoch 3 kept; **test
  373/398 = 93.7%, macro 88.0%** (drought 10/17, frost 25/29, gray_mold 160/163, healthy 76/81, overwatered 16/19,
  root_rot 33/33, white_mold 53/56). **vs the 8B LoRA: 10 vs 9 discordant, McNemar p = 1.** On this task and data
  the 32B buys nothing over the 8B, at ~4x the training compute (3 GPUs x 53 min vs 1 GPU x 35 min) and 4-bit
  serving. Its higher val score (95.7 vs 93.8) did not carry over to test: 161 val images are too few to rank
  models that are this close.
- **More frozen probes (job 72030):** PE-Core-G/14 448: 375/398 = 94.2%, **macro 93.2% (best macro so far;**
  drought 15/17, frost 29/29), vs ResNet 18 / 8, p = 0.076. BioCLIP 2.5 ViT-H/14: 91.0% / 87.6%. PlantNet
  DINOv2-B (PlantCLEF 2024 species): 91.0% / 84.6%. Species-ID pretraining (BioCLIP, PlantNet) does not transfer
  to plant-health classes better than general web-scale pretraining.
- **ResNet + lab GPT synthetic, seeds 42/2/3/4:** 91.5 / 89.7 / 89.4 / 89.9% (seed 1 pending) vs 90.2 +- 1.5%
  without. No sign of a gain.
- **Ensembles, decided before seeing any ensemble result** (equal weights primary, val-tuned weights secondary):
  E1 EVA-02 probe + SigLIP 2 probe; E2 Qwen3-VL-8B LoRA + EVA-02 probe + SigLIP 2 probe; E3 ResNet + 8B LoRA;
  E4 everything (ResNet, 8B, 32B, EVA-02, SigLIP 2).
- **Ensemble results (`cluster/ensemble.py`, probabilities: VLM first-token softmax over the 7 class names,
  probes' logistic regression, ResNet's TTA + temperature):**

| ensemble | rule | test acc | macro | vs best member (only ens / only member, p) |
|---|---|---|---|---|
| E1 EVA-02 + SigLIP 2 probes | equal | 382/398 = 96.0% | 93.8% | vs EVA-02 4 / 0, p = 0.12 |
| **E2 8B LoRA + EVA-02 + SigLIP 2** | **equal** | **387/398 = 97.2%** | **96.1%** | vs 8B 15 / 3, **p = 0.0075**; vs EVA-02 10 / 1, **p = 0.012**; vs SigLIP 2 16 / 2, p = 0.0013 |
| E2 | val-tuned 0.1/0.3/0.6 | 385/398 = 96.7% | 95.5% | vs EVA-02 7 / 0, p = 0.016 |
| E3 ResNet (s42) + 8B LoRA | equal | 372/398 = 93.5% | 89.3% | vs 8B 5 / 8, p = 0.58 |

  E2 per class: drought 15/17, frost 29/29, gray_mold 162/163, healthy 76/81, overwatered 18/19, root_rot 33/33,
  white_mold 54/56. The fine-tuned VLM and the frozen web-scale vision backbones make different mistakes; ResNet
  adds nothing on top of the VLM. (E2 is one of four pre-declared ensembles; even with a Bonferroni x4 its gains
  over each member stay below p = 0.05.) The 8B's first-token probabilities give 375/398, one more than its greedy
  text output (374/398).
- **Error review of E2's 11 test errors** (contact sheet `logs/e2_errors.jpg`, `cluster/error_sheet.py`; reviewed by
  eye, not by a pathologist):
  - clear label error (1): `white_mold/WhiteMold0030.jpg` carries the overlaid caption "Botrytis gray mold"; every
    model says gray_mold.
  - probable label problems (3): `healthy/healthy_304.webp` (ripe fruit coated in ice crystals) and
    `healthy/healthy_308.jpg` (plants in snow), both labeled healthy and called frost; `white_mold/WhiteMold0095.jpg`
    (gray fuzz on fruit, looks like Botrytis).
  - ambiguous (4): `drought/umd_dry_plug_wilt_015.jpg` (uprooted plug, roots exposed -> root_rot), a gray_mold photo
    showing a cream fungal mass on soil, a field shot labeled healthy with dried leaves at the edge, a collapsed
    overwatered plant (-> root_rot).
  - real model errors (~3): healthy white flower -> frost (flower close-ups are mostly frost photos), healthy
    seedlings in dark wet-looking coir -> overwatered, a drought pot -> overwatered.
  - **Takeaway:** at ~97% the remaining test errors are mostly label noise or genuinely ambiguous photos; label
    quality and more data now matter more than model size. Web images with overlaid captions also let VLMs *read*
    the answer; they should be flagged (the annotation schema's `image_quality` field does this).
- **Fine-tuned SigLIP 2 so400m/16 512 with the lab's ResNet recipe (`cluster/train_backbone.py`, job 72031, A100,
  seed 0):** 3 head-only + 12 full epochs at 33 s/epoch, best = full epoch 4 (val 96.9% / 94.7%), T = 0.696.
  **Test 385/398 = 96.7%, macro 96.1%** (drought 15/17, frost 29/29, gray_mold 162/163, healthy 74/81,
  overwatered 19/19, root_rot 32/33, white_mold 54/56); vs ResNet seed 42: 26 / 6, **p = 0.00054**. One ~10-minute
  fine-tune of a 400M web-pretrained ViT, same augmentations and loss as the lab's ResNet, beats it by ~6 points.
- **E4 (pre-declared, all five: ResNet, 8B, 32B, EVA-02 probe, SigLIP 2 probe):** equal weights 387/398 = 97.2% /
  96.1%, identical to E2; val-tuned weights give ResNet and the 8B weight 0 (32B 0.1). Adding ResNet or the 32B adds
  nothing.
- **X1 (exploratory, chosen after seeing the results above): fine-tuned SigLIP 2 + 8B LoRA + EVA-02 probe,** equal
  weights: 389/398 = 97.7%, macro 97.0%; vs fine-tuned SigLIP 2 alone 5 / 1, p = 0.22 (not significant).
- 32B first-token probabilities: test 374/398 = 94.0%, macro 88.2% (its greedy text output: 373/398).
- **Exploratory seed ensembles (chosen after seeing results; equal weights):** X2 = the 6 fine-tuned backbones
  (SigLIP 2 s0/1/2 + EVA-02 s0/1/2): 387/398 = 97.2%, macro 96.2% (single seeds 94.2-96.7%; vs best seed n.s.).
  X3 = X2 + Qwen3-VL-8B LoRA: **389/398 = 97.7%, macro 97.7%** (drought 16/17, frost 29/29, gray_mold 162/163,
  healthy 76/81, overwatered 19/19, root_rot 33/33, white_mold 54/56); vs the 8B alone 15 / 1, p = 0.0005.
  Averaging seeds removes the seed lottery; ~97-98% looks like the ceiling the current labels allow.
- **5-fold grouped CV of the frozen probes over all 1,576 images** (`cluster/cv_probes.py`, job 72039; near-duplicate
  groups kept within a fold; C by inner 3-fold grouped CV; every image predicted once out-of-fold; Wilson 95% CI):

| model | accuracy (95% CI) | macro |
|---|---|---|
| ensemble EVA-02 + PE-Core-G + SigLIP 2 (equal) | 1525/1576 = 96.8% (95.8-97.5) | 94.5% |
| ensemble EVA-02 + PE-Core-G | 1521/1576 = 96.5% (95.5-97.3) | 94.3% |
| PE-Core-G/14 448 | 1517/1576 = 96.3% (95.2-97.1) | 93.2% |
| SigLIP 2 so400m 512 | 1508/1576 = 95.7% (94.6-96.6) | 92.6% |
| EVA-02-L 448 | 1507/1576 = 95.6% (94.5-96.5) | 92.3% |
| ConvNeXt V2-L 384 | 1486/1576 = 94.3% (93.0-95.3) | 90.1% |
| PlantNet DINOv2-B | 1474/1576 = 93.5% | 89.0% |
| DINOv2-g | 1472/1576 = 93.4% | 89.3% |
| DINOv2-L | 1471/1576 = 93.3% | 89.3% |
| BioCLIP 2.5-H | 1461/1576 = 92.7% | 88.6% |
| BioCLIP 2 | 1455/1576 = 92.3% | 86.4% |

  Paired McNemar on 1,576: top tier PE-Core-G / SigLIP 2 / EVA-02 indistinguishable (PE vs SigLIP 25/16, p = 0.21;
  PE vs EVA 30/20, p = 0.20; SigLIP vs EVA 32/31, p = 1); PE-Core-G beats ConvNeXt V2 (p = 1.2e-4) and DINOv2-g
  (p = 3.6e-7); EVA-02 beats DINOv2-g (p = 1.3e-4). The 3-way ensemble beats SigLIP 2 (25/8, p = 0.0046) and EVA-02
  (27/9, p = 0.0039) but not PE-Core-G alone (18/10, p = 0.18). Language-supervised / supervised-fine-tuned
  backbones (PE, SigLIP 2, EVA-02 IN-22k) beat self-supervised DINOv2 and species-ID models (BioCLIP, PlantNet) here.
- **Fine-tuned EVA-02-L/14 448 (lab recipe, Ada, gradient checkpointing, job 72032):** best = full epoch 6;
  **test 383/398 = 96.2%, macro 93.9%** (drought 14/17, frost 26/29, gray_mold 161/163, healthy 77/81,
  overwatered 18/19, root_rot 33/33, white_mold 54/56).
- **DINOv3 frozen probes (access granted tonight, job 72038, A100, 448 px, CLS + mean patch):** ViT-L/16 373/398 =
  93.7%, macro 92.8% (drought 16/17); ViT-H+/16 93.7% / 92.1%; **ViT-7B/16 371/398 = 93.2% / 91.8%** (554 s to
  embed 1,576 images x 2 views). DINOv3 clearly improves on DINOv2 (L: 91.7 -> 93.7) but the 7B is no better than
  the L with ~1,000 training images.
- **DINOv3 in the 5-fold grouped CV (job 72050, same folds):** ViT-L/16 1499/1576 = 95.1% (93.9-96.1), macro 92.2%;
  ViT-7B/16 94.2% / 89.9%; ViT-H+/16 93.9% / 90.1%. Adding DINOv3-L to the EVA-02 + PE-Core-G + SigLIP 2 ensemble:
  96.8% either way. **Final frozen-backbone ranking (CV):** PE-Core-G 96.3 > SigLIP 2 95.7 ~ EVA-02 95.6 >
  DINOv3-L 95.1 > ConvNeXt V2-L 94.3 ~ DINOv3-7B 94.2 > DINOv3-H+ 93.9 > PlantNet-DINOv2-B 93.5 ~ DINOv2-g 93.4 ~
  DINOv2-L 93.3 > BioCLIP 2.5 92.7 > BioCLIP 2 92.3.
- **Seed variance of the new models (split v1 test):** Qwen3-VL-8B LoRA seeds 0/1/2: 94.0 / 94.2 / 94.5% ->
  **94.2 +- 0.3%** (macro 90.2 +- 1.7). SigLIP 2 fine-tuned seeds 0/1: 96.7 / 94.2% (seed 1: overwatered 13/19) ->
  95.5 +- 1.8% so far; seed 2 running. Single-seed numbers move by up to ~2.5 points on this test set.
- **Final seed summary, split v1 test (mean +- sd over seeds; `cluster/leaderboard.py`,
  `evaluation/leaderboard_split_v1.md`):**

| method | seeds | accuracy | macro |
|---|---|---|---|
| EVA-02-L fine-tuned (lab recipe) | 3 (96.2 / 94.5 / 96.5) | **95.7 +- 1.1%** | 93.3 +- 1.4% |
| SigLIP 2 so400m fine-tuned (lab recipe) | 3 (96.7 / 94.2 / 95.7) | **95.6 +- 1.3%** | 94.0 +- 2.4% |
| Qwen3-VL-8B LoRA | 3 | 94.2 +- 0.3% | 90.2 +- 1.7% |
| Qwen3-VL-32B QLoRA | 1 | 93.7% | 88.0% |
| ResNet-50 + lab GPT synthetic | 5 | 90.4 +- 0.9% | 84.2 +- 1.3% |
| ResNet-50 lab recipe | 5 | 90.2 +- 1.5% | 85.5 +- 2.0% |

- **Fine-tuned PE-Core-G/14 448 (1.88B params, lab recipe, A100, grad checkpointing, 37 GiB peak, 138 s/epoch, job
  72067, seed 0):** best = full epoch 5; **test 385/398 = 96.7%, macro 95.7%** (drought 15/17, frost 29/29,
  gray_mold 162/163, healthy 74/81, overwatered 18/19, root_rot 33/33, white_mold 54/56). Same as the best SigLIP 2
  seed at ~4.7x the parameters, so SigLIP 2 (0.4B) is the efficient pick.
- **ResNet preprocessing mismatch, all 5 seeds (`cluster/resnet_scores.py`):** lab runtime (resize + center crop)
  90.2 +- 1.5% vs training-matched squash 89.8 +- 1.5%. The mismatch is harmless; no change needed.
- **ResNet + lab GPT synthetic, 4 of 5 seeds scored:** 90.1 +- 0.9% vs 90.2 +- 1.5% without. **No gain.** (Seed 1
  trained fine but segfaulted on exit before scoring; re-scored in job 72051. Both odd failures tonight were on c5.)
- **Honest-annotation track (OpenAI credits exhausted, so a local teacher):** spec in `cluster/annotation_spec.py`
  (label-conditioned; findings, label_assessment.consistency, differential, caption, grower report, QA pairs with
  "not visible" questions). Pilot with Qwen3-VL-32B (bnb4, transformers generate, job 72034): valid JSON, but
  109 s/image, so the full run moved to vLLM 0.32.0 (`envs/vllm`, schema-constrained decoding,
  `cluster/annotate_vllm.py`). **The pilot's first image already shows the honesty check working:**
  `drought/strlfscorch04-big-opt-vvea5b.jpg` (file name: strawberry *leaf scorch*, a fungal leaf spot) is filed as
  drought; the annotator answered `does_not_support`, listed the missing drought signs (dry cracked soil, crispy
  curled edges) and the conflicting evidence (moist soil, purple-brown irregular spots), and named a foliar leaf-spot
  disease in the differential. That matches the file name; the dataset label looks wrong.
- **Label-swap honesty test (designed tonight):** every val/test image is annotated twice, once told its true label
  and once a seeded wrong label. Honest = accept the true label (clearly/partially_supports) AND reject the wrong
  one (does_not_support/cannot_tell). Plan: score the 32B teacher, the base 8B, and an 8B LoRA-distilled on the
  teacher's train annotations (incl. ~35% wrong-label examples) (`cluster/train_vlm_reports.py`,
  `cluster/merge_lora.py`, `cluster/score_honesty.py`).
- **Honest-report students, final label-swap test (test split, 398 images x {true, wrong} = 796 verdicts;
  `evaluation/honesty_label_swap_test.md`; students: LoRA r16 on the teacher's train annotations filtered to correct
  verdicts, 795 true-label + 302 wrong-label examples, 2 epochs, ~1.5 s/example, jobs 72063/72066 + 72068):**

| annotator | accepts TRUE label | rejects WRONG label | balanced | true condition in differential (wrong label) |
|---|---|---|---|---|
| **8B student from the classification-tuned 8B** | 318/398 = 79.9% | 339/398 = 85.2% | **82.5%** | 389/398 = 97.7% |
| 8B student from the base 8B | 315/397 = 79.3% | 325/397 = 81.9% | 80.6% | 372/397 = 93.7% |
| 32B teacher (zero-shot) | 313/398 = 78.6% | 305/398 = 76.6% | 77.6% | 379/398 = 95.2% |
| base 8B (zero-shot) | 288/398 = 72.4% | 295/398 = 74.1% | 73.2% | 355/398 = 89.2% |

  Paired McNemar on verdicts: cls-student vs **teacher 70 / 31, p = 0.00013**; base-student vs teacher 65 / 41,
  p = 0.025; cls-student vs base-student 42 / 27, p = 0.091; cls-student vs base 8B 113 / 39, p = 1.5e-9.
  **Ground-truth-filtered distillation gives an 8B that is significantly more honest than its 32B teacher**, and
  starting from the classification-tuned 8B helps (trend). Caveat: the students practised this exact
  label-checking format in training and the zero-shot models did not; this measures verdict honesty, not every
  claim in the free-text report (that needs human or frontier-model grading, e.g. with the GPT pipeline once the
  org has credits). Validation split gave the same ordering (79.2 / 77.0 / 77.3 / 71.4%).
  Infrastructure note: the second student's merged checkpoint (tokenizer re-saved twice by transformers 4.57) failed
  to load in vLLM's transformers 5.x until the original tokenizer/processor files were copied back
  (`cluster/slurm/honesty_eval.sbatch`).
- **RF-DETR plant-part detector, the lab's Small checkpoint** evaluated by RF-DETR 1.11.2 on the lab's own 322-image
  valid split (job 72035; the split the checkpoint was selected on, so optimistic): **mAP@50:95 0.598, mAP@50 0.772,**
  mAP@75 0.643, mAR 0.781, precision 0.832, recall 0.665, F1 0.738. Medium and Large are training with the lab's
  yaml (jobs 72040/72041; `lr_min_factor` is not accepted by this RF-DETR version and was dropped). The training
  stack needed `pip install "rfdetr[train,loggers]==1.11.2"` in envs/rfdetr (torch left at 2.14.1+cu130).
- **RF-DETR Medium (576 px, lab yaml, 100 epochs, 0.89 h on one RTX 5000 Ada, job 72040):** valid mAP@50:95
  **0.614** (lab Small 0.598), mAP@50 0.773 (0.772), mAP@75 0.663 (0.643). A modest +1.6 points; Large still
  training, and a Small retrain with the same RF-DETR version queued for a like-for-like comparison.
- **RF-DETR Large (704 px, batch 4 x accum 4, 100 epochs, 1.20 h, job 72041):** valid **mAP@50:95 0.631**, mAP@50
  0.788, mAP@75 0.678: **+3.3 points over the lab's Small** (0.598), +1.7 over Medium. Same Apache-2.0 license and
  still ~30M parameters, so a cheap upgrade. (All three numbers are on the valid split used to pick the best
  checkpoint, as in the lab's own setup; there is no separate detector test split.)
- **RF-DETR Small retrained with the same RF-DETR 1.11.2 + lab yaml (0.73 h, job 72064):** mAP@50:95 0.593,
  mAP@50 0.765, mAP@75 0.639, reproducing the lab's checkpoint (0.598) within 0.5 points. **Like-for-like:
  Small 0.593 < Medium 0.614 < Large 0.631** (+3.8 points Small -> Large, same code, settings, data).
- **vLLM annotation runs fixed:** (1) vLLM spawns its engine process, so the script needs an `if __name__ ==
  "__main__"` guard; (2) the compute nodes have no CUDA toolkit, so FlashInfer's JIT sampler fails ->
  `VLLM_USE_FLASHINFER_SAMPLER=0`; (3) greedy decoding under the JSON grammar looped (`"leaf", "leaf", ...` until
  max_tokens: 56/96 base-8B outputs truncated) -> bounded arrays (maxItems) and strings (maxLength) plus
  repetition_penalty 1.05: 0 invalid of the next 192. Throughput: 8B ~1,000-1,300 output tok/s on an RTX 5000 Ada
  (~1 s per annotation), 32B bf16 ~480 tok/s on the A100 (~2 s), vs 109 s per annotation with transformers +
  bnb4.
- **Label-swap test, base Qwen3-VL-8B zero-shot, first 96 val images:** accepts the true label 63.5%, rejects a
  wrong label 61.5% -> balanced 62.5% (chance 50%). Out of the box the 8B mostly goes along with whatever label it
  is given. It still names the true condition in its differential 87.5% of the time when told a wrong label.
- **Label-swap test, full test split (398 images x {true, wrong} label = 796 verdicts, schema-constrained vLLM):**

| annotator | accepts TRUE label | rejects WRONG label | balanced | true condition in differential when told a wrong label |
|---|---|---|---|---|
| Qwen3-VL-32B-Instruct (bf16, zero-shot) | 313/398 = 78.6% | 305/398 = 76.6% | **77.6%** | 95.2% |
| Qwen3-VL-8B-Instruct (bf16, zero-shot) | 288/398 = 72.4% | 295/398 = 74.1% | **73.2%** | 89.2% |

  32B vs 8B paired over the 796 verdicts: 88 / 53, **p = 0.004**. So for *checking* a diagnosis against a photo
  without fine-tuning, the 32B is significantly more honest, unlike fine-tuned classification where 32B = 8B.
  Both still accept a wrong label about a quarter of the time. Validation split: 32B 77.3%, 8B 71.4%.
- **Label audit over all 1,576 images (`cluster/label_audit.py`, `evaluation/label_audit.csv`, contact sheet
  `logs/label_audit_top.jpg`):** the CV ensemble (EVA-02 + PE-Core-G + SigLIP 2, out-of-fold) disagrees with 51
  labels (15 with p(label) < 0.1); the 32B annotator rejects 340/1,576 true labels (22%, too strict alone); **both
  disagree on 11.** Eyeballing the 11 (not a pathologist): ~9 look mislabeled, e.g. `WhiteMold0030.jpg` (caption
  reads "Botrytis gray mold"), three stock photos of fuzzy gray Botrytis filed as white_mold (WhiteMold0093/0249/
  0255), snow-covered plants filed healthy, a root_rot field shot and a frost photo whose plants look healthy, a
  two-panel scientific figure filed overwatered. Ensemble-only flags are often model errors instead (a flooded
  field correctly labeled overwatered called frost; healthy white flowers called frost). The dataset also holds
  watermarked stock images (shutterstock / alamy / iStock) and text overlays that a VLM can read.
  **Action for the lab: have someone re-label the 11 "both" images (and skim the top 50 of the CSV).**
- **Student design change before training (filtered distillation):** in the training data we know which given
  labels are wrong; `train_vlm_reports.py` now keeps only teacher annotations whose verdict is correct (accepts a
  true label / rejects a swapped one), so the student does not learn the teacher's ~23% sycophantic verdicts. Two
  students queued: from the base 8B (job 72063) and from the classification-tuned 8B (LoRA merged,
  `models/Qwen3-VL-8B-Instruct-clsft`, job 72066), to test whether diagnostic skill carries over to honest reports.

## 2026-10-20 06:30: conclusions of the overnight run and recommended next steps

1. **Classifier: replace ResNet-50 with a modern backbone.** Same data, same lab recipe: fine-tuned EVA-02-L
   95.7 +- 1.1% and SigLIP 2 so400m 95.6 +- 1.3% (3 seeds each) vs ResNet-50 90.2 +- 1.5% (5 seeds). SigLIP 2 is
   the efficient pick (0.4B, ~10 min per fine-tune on the A100). A seed ensemble removes the seed lottery (97.2%);
   adding the fine-tuned 8B VLM gives the best test score seen, 97.7% (exploratory).
2. **VLM: use it to explain, not to classify; 8B is enough for classification.** Fine-tuned Qwen3-VL-8B 94.2 +- 0.3%;
   fine-tuned 32B 93.7% (vs 8B p = 1). Thinking mode and the 32B buy nothing for the label. For honest
   explanations, a ground-truth-filtered 8B student beats the zero-shot 32B (82.5% vs 77.6% on the label-swap
   test, p = 0.0001).
3. **Detector: RF-DETR Small -> Large** (+3.8 mAP@50:95 like-for-like, 1.2 h to train, same license).
4. **Data is now the bottleneck.** About 97-98% is the ceiling the current labels allow; the remaining test errors
   are mostly label noise or ambiguous photos. Fix the 11 doubly-flagged labels (`evaluation/label_audit.csv`), drop
   or flag stock photos with captions/watermarks, and collect real images for drought, frost, overwatered (no public
   strawberry data exists for drought, overwatering or root rot; a small greenhouse stress trial is the best source).
   The lab's 601 GPT-generated images did not help (5 seeds).
5. **Evaluation protocol going forward:** split v1 (leak-free, grouped) + 5-fold grouped CV for anything cheap;
   always several seeds; paired McNemar; pre-declare ensembles.
6. **Open items:** OpenAI org has no credits (GPT annotation pipeline ready, ~$0.02/image); rotate the HF token and
   the OpenAI key (both pasted in chat); ask HPC to configure the 10 GbE ports; commit `cluster/` to a branch.

## 2026-10-20 morning: synthetic data, mold toss-ups, low-quality robustness

- **The lab's GPT synthetic images, per class (fine-tuned SigLIP 2, 3 seeds each, real-only val/test; jobs
  72069-72071 vs the real-only runs).** Same tree as the ResNet ablation (`splits/v1_gptsynth`, +601 images).
  Overall null again: test 95.6 +- 1.3 -> 95.7 +- 1.0, val 96.5 -> 95.9; pooled paired McNemar over the 3 seed
  pairs 21 vs 23 discordant, p = 0.88. Per class it reproduces the ResNet pattern, so it is not noise:

| class (test n) | real only | + lab synth | diff | ResNet diff (5 seeds) |
|---|---|---|---|---|
| drought (17) | 92.2 | 76.5 | **-15.7** | -16.5 |
| overwatered (19) | 84.2 | 93.0 | **+8.8** | +8.4 |
| frost (29) | 97.7 | 100.0 | +2.3 | +0.7 |
| root rot (33) | 97.0 | 100.0 | +3.0 | -1.8 |
| gray mold (163) | 97.5 | 99.0 | +1.4 | +2.1 |
| healthy (81) | 93.0 | 90.1 | -2.9 | +1.0 |
| white mold (56) | 96.4 | 96.4 | 0.0 | -3.2 |

  Mechanism (confusions summed over 3 seeds): real-only calls overwatered photos drought 9 times and drought photos
  overwatered once; with the lab synth it is 1 and 7. **The synthetic images move the drought/overwatered boundary
  instead of sharpening it** (the overwatered gain is paid for by drought). This is probably why the lab remembers
  the synthetic data as a big win: its own holdouts contained synthetic images, and overwatered (a weak class) does
  improve. Two lab synthetic zips are still missing from what we have (`synthetic_data (1)-20260427T030414Z-3-001.zip`,
  986 MB, and `plant_diagnostic/data/latestsynthdata-20260427T133937Z-3-001.zip`); ask Hongbo / the Drive.
  Side note: the test split holds a .jpg and a .webp of the same photo (`strawberry-plants-are-dying-v0-wparrcp3m80f1`);
  both are in test, so no leak, but it counts twice.

- **Gray/white mold toss-ups (`cluster/mold_tossups.py`, 29 images where the out-of-fold CV ensemble gives the other
  mold >= 0.15).** Only one label is certain to be wrong: `WhiteMold0030.jpg` (its own caption says Botrytis gray
  mold) -> gray_mold. Twelve need a plant pathologist (8 white_mold, 1 gray_mold, 3 web images). Recorded in
  `evaluation/label_corrections_v1.json` with `applied_to_benchmark: false`; nothing was changed. The white_mold
  class itself looks mixed (Sclerotinia, Rhizopus leak and early Botrytis), which is the deeper problem.

- **Low-quality robustness test (`cluster/eval_lowq.py`, job 72072).** The real split v1 test images degraded in
  fixed, deterministic ways, each model then resizing as usual: 256 px + JPEG q50, 128 px thumbnail + JPEG q35,
  whole-frame defocus (blur 0.6% of the long side), darkened x0.55 + sensor noise. SigLIP 2 fine-tunes, 3 seeds:

| train data | clean | 256 px | 128 px | defocus | dark + noise | 128 px macro |
|---|---|---|---|---|---|---|
| real only | 95.6 +- 1.3 | 94.3 +- 1.9 | **83.7 +- 4.7** | 93.1 +- 0.6 | 93.6 +- 2.0 | 78.5 |
| + lab synth | 95.7 +- 1.0 | 95.1 +- 0.2 | 85.8 +- 0.8 | 93.9 +- 0.5 | 94.8 +- 1.0 | 78.2 |

  At thumbnail quality the weak classes collapse (real only: drought 59%, frost 62%, overwatered 68%; healthy and
  root rot stay ~97-99%). Defocus hurts drought (78%) and frost (82%) most. So low detail is a real failure mode,
  concentrated exactly in the classes that need data.

- **What the real root_rot class looks like (contact sheet of 48 of its 84 train images):** almost entirely dug-up
  roots and sliced crowns, many extension/scientific photos on plain backgrounds (panel letters, "HEALTHY/UNHEALTHY"
  captions, microscope shots); about 2 field photos. A classifier can therefore learn "roots visible -> root rot".
  Synthetic counterexamples must show healthy roots in the same compositions.

- **New synthetic set from ChatGPT (the user's web account, driven in their Chrome; training only, val/test stay
  real).** 10 numbered prompts per chat -> 10 images, downloaded as a series. Design, after review rounds:
  - nuisance factors spread across classes: hands/thumbs/gloves capped at ~25% of every class (`hand` recorded per
    image; `build_train_tree.py --hand-cap 0.33` enforces it), the same settings, viewpoints and lighting everywhere;
  - severity mix (mild / moderate / severe) and healthy look-alikes for each stress (wet soil after rain, dry
    topsoil, dew and yellow-centered flowers, an old red lower leaf, clean roots);
  - roots lying on the ground with no hands in frame for BOTH healthy and root rot, plus matching bench, in-hand and
    cut-crown shots (healthy crowns clean inside), so root condition, not "roots visible", decides the class;
  - v3 prompts: phone-camera deep focus, explicitly no bokeh / shallow depth of field / cinematic light (v1/v2
    images had a lot of it); ~40% "genuinely low quality" (low-res, noisy dusk, misfocused, smudged lens, motion
    blur, messaging-app compression, far away);
  - v3b (after the first v3 batches): mild or small cues (frost flower centers, a few spots, mild symptoms) are never
    combined with low quality or far/wide shots, which made the cue invisible (= label noise); frost leaf damage is
    described as blackened/water-soaked, not purple blotches (cold stress, ambiguous); root rot always shows roots,
    a crown or a field patch;
  - every image is reviewed by eye; failures go to `review.json` with a reason (so far: reddish leaves only, cue
    not visible, ambiguous frost purple, a dim single brown leaf). The importer (`cluster/import_synth.py`) drops
    them, random-crops to random aspect ratios, downscales and JPEG-compresses (raw outputs are pristine PNGs in a
    few fixed sizes, an easy shortcut), dedups by pHash against every real val/test image, and writes a manifest
    with the prompt, severity, hand, quality tier and group of each image.

## 2026-10-20 afternoon: ChatGPT synthetic set, first results, captioner bake-off setup

- **Generation, final:** 248 images in ChatGPT (user's web plan), 238 pass review (`review.json`, 10 excluded, 1 hand
  override). Mid-afternoon the 10-images-per-chat mode broke (frost_3/4, drought_4: "the generator repeated scene
  1", a 5x2 composite cut into tiles, 3 variations of prompt 1), at all thinking levels. Switched to ONE prompt per
  chat in Instant mode (~1 min per image, exact prompt-image mapping). Two 10-image batches came back out of prompt
  order (drought_3, root_rot_4): renamed by visual match (`*_ORDER_FIXED.txt`). frost_3 prompts were reordered
  (recorded in the v3b index). Final added to training (hand cap 0.33): drought +50, frost +46, healthy +50,
  overwatered +46, root rot +43 (235 synthetic vs 1,017 real).
- **First arm `chatgpt_a` (the ~165 images available at noon), SigLIP 2, 3 seeds, real test only:**
  test 96.0 +- 0.4 (real only 95.6 +- 1.3, lab synth 95.7 +- 1.0); per class drought 94 / 92 / 76 (ours / real / lab),
  healthy 95 / 93 / 90, root rot 100 / 97 / 100, overwatered 77 / 84 / 93 (n = 19: one image = 5.3 points).
  Not significant overall; unlike the lab's set it does not hurt drought.
- **Held-out synthetic probe (`cluster/eval_probe.py`, batches generated after `chatgpt_a` was trained; synthetic, so
  a diagnostic of shortcuts, not a benchmark):** 70 images: real only 57-66%, lab synth 74-83%, ours 89-96%. Healthy
  dug-up roots (healthy_5): real-only and lab-synth models call 0-1 of 6 healthy, all others "root rot"; ours 5-6 of 6,
  with rotted roots still 8/8. **Confirmed shortcut in the current models: roots visible -> root rot**, fixed by the
  healthy-roots images (the user's idea).
- **Low-quality robustness got WORSE with our synth** (128 px thumbnails: real only 83.7, lab 85.8, ours 77.9;
  macro 78.5 / 78.2 / 69.2). Errors go to "healthy" (193 vs 108 over 3 seeds): drought 59 -> 26, frost 62 -> 51.
  Working explanation: the look-alikes teach that dry/wet soil or visible roots alone are not disease, so when a
  thumbnail destroys the plant-level symptoms the model falls back to healthy, while the real-only model still
  guesses from soil context. The "genuinely low quality" prompts did not produce enough real low detail. Running
  now: train-time phone degradation for every class (`train_backbone.py --lowq-aug 0.3`), arms chatgpt / chatgpt_lq /
  lq (real only + aug), 3 seeds each (jobs 72084-72092, low-quality test 72093).
- **Captioner bake-off (Hongbo: no API spending, so local only; each model gets the whole cluster in turn):**
  `cluster/caption_bakeoff.py`, 38 hard real test images (`evaluation/bakeoff_set_v1.json`: faint mold patches,
  black frost centers, wilting stems, pineberry look-alikes, rotted crowns). Pass 1 sees the image without the label
  and lists findings with a point [x, y] and visibility; pass 2 is text-only (label + findings), so it cannot add
  visual details. Candidates: Qwen3.8-27B bf16 (Gus: newer, higher general vision benchmarks; zero-shot on our task
  it was the weakest Qwen: 66.8% vs 8B 74.6% / 32B 72.5%, thinking on 57.8% vs off 54.0% on resnet161, p = 0.24),
  Qwen3-VL-32B bf16, GLM-4.6V FP8 (106B, 12B active), Qwen3-VL-235B-A22B AWQ. The big two run with vLLM pipeline
  parallelism over all 6 RTX nodes on Ray (`slurm/vllm_ray.sbatch`); each node stages only its layers' weight files
  (`cluster/stage_pp_shard.py`, ~25-29 GB per node instead of 110-125 GB, because /projects is 1 GbE).
- **Network:** each research-gpu node has an unused Mellanox ConnectX-4 Lx linked at 10 Gb/s with no IP; c9 has
  10 GbE on both networks; no InfiniBand. Request sent via the user to IT: IPs on the 10G ports (storage network
  192.168.0.0/24), jumbo frames, RoCE v2 (mlx5_ib, rdma-core, nvidia-peermem).

## 2026-10-20 late afternoon: final synthetic arms + low-quality augmentation (jobs 72084-72093)

SigLIP 2 so400m fine-tunes, lab recipe, 3 seeds each, real split v1 test (398) and the low-quality test
(`cluster/eval_lowq.py`). McNemar = pooled paired discordant pairs vs real only over the 3 seed pairs.

| train data | clean test | macro | 128 px thumbnail (macro) | 256 px | defocus | dark+noise | vs real only |
|---|---|---|---|---|---|---|---|
| real only | 95.6 +- 1.3 | 94.0 | 83.7 (78.5) | 94.3 | 93.1 | 93.6 | - |
| + lab GPT synth (601) | 95.7 +- 1.0 | 93.6 | 85.8 (78.2) | 95.1 | 93.9 | 94.8 | 23 / 21, p = 0.88 |
| + ChatGPT synth, noon subset (~165) | 96.0 +- 0.4 | 94.1 | 77.9 (69.2) | 93.6 | 92.2 | 93.5 | 21 / 16, p = 0.51 |
| + ChatGPT synth, full (235) | **94.2 +- 1.5** | 92.2 | 83.0 (73.6) | 91.9 | 90.9 | 92.2 | 14 / 30, **p = 0.02** |
| + ChatGPT synth (235) + `--lowq-aug 0.3` | **95.7 +- 0.8** | 93.6 | **91.6 (89.6)** | 95.1 | 93.0 | 94.4 | 22 / 20, p = 0.88 |
| real only + `--lowq-aug 0.3` | 95.1 +- 1.0 | 93.3 | 90.0 (87.0) | 94.1 | 92.0 | 93.8 | 12 / 18, p = 0.36 |

- **Low-detail robustness comes from the train-time degradation, not from "low quality" prompts:** 128 px accuracy
  83.7 -> 90.0 (real + aug) -> 91.6 (real + ChatGPT synth + aug); per class at 128 px drought 59 -> 78, frost 62 -> 80,
  overwatered 68 -> 95; thumbnails wrongly called healthy 108 -> 8 (3 seeds). Clean accuracy unchanged.
- **The full ChatGPT set without augmentation hurts (p = 0.02),** unlike the noon subset: healthy -> frost 9 -> 12,
  healthy -> drought 6 -> 9, and a new frost -> gray mold (2 real frost photos). The 70 images added after noon are
  mostly the 30 one-per-chat frost images (black flower centers, browned blossoms). With the augmentation the effect
  disappears (healthy -> frost 7). Candidate ablation: full set minus frost_3-5.
- **Current best recipe:** real + ChatGPT synth + `--lowq-aug 0.3`: same clean accuracy as real only (95.7 vs 95.6,
  p = 0.88), far more robust to poor photos, and (shown on the noon arm with held-out synthetic batches) no longer
  calls healthy dug-up roots root rot. Caveat: the held-out roots probe exists only for the noon arm; the full arms
  trained on every synthetic batch.
- **Captioner bake-off, first two (A100, bf16, 38 photos):** Qwen3-VL-32B 5.3 claims/photo, 2.4 s/photo, its
  text-only verdict says "cannot tell" for 36/38 (over-cautious); Qwen3.8-27B 5.8 claims/photo, 1.4 s/photo, verdicts
  8 clearly / 22 partially supports, 7 cannot tell, 1 does not support. 32B sometimes switches language mid-claim
  (Chinese word in an English finding). Blind rating page: private claude.ai artifact, link not published (key in
  `bakeoff/bakeoff_v1/bakeoff_key.json`). c4 is held by another user, so the 235B and GLM-4.6V run on 5 RTX nodes
  (PP 5, layer splits 16,20,20,19,19 and 7,10,10,10,9; 17-26 GB of weights per GPU).
- **Ablation: full ChatGPT set minus the 30 one-per-chat frost images (frost_3-5), A100, 3 seeds (jobs 72105-72112):**
  without augmentation 96.1 +- 1.0 (macro 94.7; vs real only 23/16, p = 0.34), with `--lowq-aug 0.3`
  **96.3 +- 0.8** (macro 94.3; 21/12, p = 0.16; 128 px 92.1, macro 88.5; 256 px 96.1; dark+noise 95.8). Removing the
  late frost batch removes the harm (frost 98.9, healthy 93.0 / 95.5). The held-out frost_3-5 images are still called
  frost by 83-93% of models that never saw them (real-only models 43-83%), so they are not mislabeled; they
  over-weight "white flower with a dark center" and pull real healthy flowers toward frost.
  **Recommended training data: real + ChatGPT synth without frost_3-5 + `--lowq-aug 0.3`.** (Caveat: these two arms
  trained on the A100 without gradient checkpointing, the other synthetic arms on RTX nodes with it; same recipe and
  seeds.) Differences among the top arms (95.7-96.3) are within seed noise; robustness and the roots probe are not.
- **Captioner bake-off, Qwen3-VL-235B-A22B AWQ (PP 5 over c3/c5-c8, job 72115):** 38/38 parsed, 6.2 claims per photo,
  5.6 s per photo (one stage had only 0.4 GB KV cache), pass-2 verdicts 30 cannot tell / 8 clearly supports. Terse,
  precise claims (e.g. it separates yellow- from black-centered flowers on the frost photo). Launcher lessons:
  vLLM 0.32 orders PP ranks by driver then random Ray node ID (stage after `ray start`, `cluster/ray_pp_order.py`);
  it rejects an index naming absent files (per-node index); quantized models skip the missing-weights check (a wrong
  rank mapping would load garbage silently); Triton JIT cache on NFS home fails across nodes (moved to /scratch);
  Ray token auth now on.
- **Captioner bake-off, GLM-4.6V FP8 (PP 5 over c3/c5-c8, job 72119; 72117 had died on the NFS Triton cache):**
  38/38 parsed, 5.0 claims per photo (fewest of the four), 2.9 s per photo, pass-2 verdicts 19 cannot tell /
  18 partially / 1 clearly supports. All four examiners mark 94-98% of claims "clear" (GLM 98%, 235B 97%, 27B and 32B
  94%), so the visibility field barely separates confident from unsure claims; only human ratings can.
  Language slips: Qwen3-VL-32B has Chinese text in 13 entries over 15 of 38 photos, GLM 1, 235B and 27B none.
  The rating page now shows all four examiners (845 claims; version 3, same link). Aliases come from the fixed
  four-model list, so the alias key is unchanged and the 417 claims already rated (one rater) keep their meaning.
- **Bake-off ratings, interim (one rater, photos 0-24 of 38, 417 claims; GLM not yet rated; no healthy / root rot
  photos yet):** claims marked not there: Qwen3-VL-235B 12.0% (95% CI 7.7-18.2, 13 of 25 photos with one),
  Qwen3.8-27B 12.7% (8.2-19.1, 8 photos; 8 of its 18 on the two frost photos: frost coating called a white powdery
  coating, five dark spots that are not there), Qwen3-VL-32B 4.0% (1.7-9.0, 2 photos; but unsure 14% vs 5-6%).
  Per photo, 235B vs 32B p = 0.01 (sign test), 235B vs 27B p = 0.61, 32B vs 27B p = 0.18. Mold seen on the 13
  gray/white mold photos: 235B 11, 27B 11, 32B 9. Shared misses: ~~no examiner saw the water-soaked lesions on the
  overwatered fruit (img20)~~ *(corrected 2026-10-21: the 27B did see the lesion, "a slightly depressed,
  lighter-colored patch", rated real; the 235B and 32B stated there were no soft or water-soaked patches)*;
  235B and 27B called normal stem hairs cottony growth (img22). Caveat for the final
  score: claim length differs about 3x between examiners, so per-claim rates need normalising.
  (`rating_detail.py`, `mold_check.py` in the session scratchpad; to move into `cluster/` with the final score.)

## 2026-10-21

- **Hongbo's proposal (two captioners + a judge that checks consistency, captions when they disagree, and writes the
  final conclusion), checked against the interim ratings (417 claims, photos 0-24; `agreement_check.py`, session
  scratchpad).** A rated claim counts as corroborated when another examiner (GLM included) made a claim within 150
  of its point (0-1000 scale) with the same polarity and a shared condition keyword (crude matching).
  Corroborated claims: 232, not there 6.5%; solo claims: 185, not there 14.1% (corroborated by 2+ others: 4.2% vs
  12.8%). At a looser 250 the difference mostly disappears (9.3% vs 10.9%): agreement has to be same spot, same
  finding. **Agreement is a real signal but a poor filter:** keeping only corroborated claims drops 26 of the 41
  false claims and also 142 of the 342 real ones; mold was seen by at least one of the three rated examiners on
  13/13 mold photos, by two or more on 11/13, by each alone on 9-11. **Shared blind spots:** 15 of the 41 false
  claims had a matching claim from another model. On img22 (overwatered) three of four examiners, GLM (the only
  non-Qwen) included, called normal stem hairs fuzzy/cottony growth; on img20 three of four (235B, 32B, GLM)
  stated there were no soft or water-soaked patches and only the 27B saw the lesion (rated real). **On both photos a
  majority vote picks the wrong answer.** A text-only consistency judge would have accepted the false mold claim
  (the same error as the old pipeline's "white powdery spores" on an overwatered plant) and discarded the one
  correct lesion claim.
  Implication for the design: compare grounded claims rather than captions, have the judge re-examine disputed
  claims in a zoomed crop at the claimed point without seeing the label, write the final caption text-only from
  the verified claims (current pass 2), flag photos with heavy disagreement for review, and measure the judge's
  yes-bias with decoy claims from other photos. Pilot: judge the 845 bake-off claims and score it against the
  ratings.
- **Pre-registered: consistency-check pilot (Hongbo's two proposals; Gus: keep the final pipeline simple).** Same
  38 bake-off photos, ground truth = the human claim ratings (all that exist when scored). Signals for "this claim
  is real", each scored by the share of rated not-there claims it removes vs the share of rated real claims it
  removes: (a) cross-model agreement (done above, 26/41 false vs 142/342 real at distance 150); (b) self-consistency:
  each examiner repeats its label-blind pass 1 three times (`caption_bakeoff.py --samples 3`, temperature 0.7,
  seeds 1-3; the rated pass 1 was greedy, so plain re-runs would be identical), and a claim counts as stable when it
  recurs in >= 2 of 3 repeats (same matching rule as above: same polarity, shared condition keyword, point within
  150); (c) Hongbo's combined rule: a claim another model does not corroborate is kept only if it is stable in its
  own model's repeats, with a fixed cap of 3 repeats (no sampling until agreement, which would manufacture
  consensus) and unresolved claims sent to human review; (d) later: a judge re-checking disputed claims in a zoomed
  crop. Prediction: self-consistency catches one-off inventions but not stable misreadings (stem hairs read as
  mold by three different models on img22).
- **Bake-off ratings, interim 2 (one rater, 615 of 845 claims, 34 of 38 photos; GLM rated on 9 photos):** claims
  marked not there: Qwen3-VL-235B 10.3% (95% CI 6.8-15.2; 16 of 34 photos with one), Qwen3.8-27B 10.2% (6.7-15.2;
  10/34), Qwen3-VL-32B 4.0% (2.0-8.1; 4/34; unsure 10%), GLM-4.6V 5.0% (1.4-16.5; 2/9; unsure 18%). Per photo:
  235B vs 32B 14 worse / 4 better, p = 0.03; 235B vs 27B p = 0.65; 32B vs 27B p = 0.27. On the 9 photos rated for
  all four (frost 4, healthy 3, root rot 2) all four are alike (not there 3.6-5.6%), so the differences come from
  the mold and water-stress photos; there GLM marks 18% unsure vs 0-4%. Agreement check on the larger set: claims
  corroborated by another examiner 5.8% not there vs 11.0% solo; keeping only corroborated claims would drop 30 of
  50 false and 219 of 521 real claims. **Blinding caveat:** GLM was added to the page after photos 0-24 were rated,
  so on those photos the rater knows the unrated letter is the late examiner; GLM will be reported with and without
  photos 0-24. Left: GLM on photos 0-24, all four on photos 31-34 (healthy 2, root rot 2).
- **Pre-registered: (d) zoom-in judge (written before any judge output; `cluster/judge_claims.py`).** Judges:
  Qwen3-VL-32B, Qwen3.8-27B, Qwen3-VL-8B (A100, bf16, modes zoom and nozoom), then Qwen3-VL-235B AWQ and
  GLM-4.6V FP8 (RTX nodes, PP 5, zoom only for compute). Items: all 845 greedy pass-1 claims of the four examiners
  on the 38 bake-off photos, plus 152 decoys (4 per photo): a definite disease-sign claim (no hedges or negations)
  from a photo of another condition group (mold / wet / dry / frost / healthy), boxed where this photo's own claims
  put the same plant part, so the judge has to check the sign, not the part. The judge never sees the dataset
  label. Input: the whole photo (1792 px, as in pass 1) with a red box (side 35% of the short side) around the
  claimed point; zoom adds the box contents enlarged to 896 px. Output: what is visible in the box, then correct /
  incorrect / cannot tell. Primary metric on human-rated claims: not-there claims removed vs real claims removed
  when claims judged incorrect are dropped; also dropping incorrect + cannot tell; AUROC with correct = 1,
  cannot tell = 0.5, incorrect = 0. Secondary: decoy acceptance (an upper bound on yes-bias: a decoy can be true by
  chance), self-judging (judge = claimant) vs cross-judging, and judging only claims no other examiner
  corroborates. Predictions: zoom beats nozoom; the best judge beats agreement (AUROC 0.62) and self-consistency
  (0.55).
- **Result, consistency pilot (a)-(c), all four examiners (615 rated claims, 3 sampled repeats each at T 0.7,
  1 of 456 repeats unparsable):** (a) agreement removes 30/50 false and 219/521 real claims (not there among kept
  8.8% -> 6.2%); (b) self-consistency removes 32/50 false and 273/521 real (6.8%); (c) Hongbo's rule removes 25/50
  false and 181/521 real (6.8%). AUROC for telling real from not-there claims: agreement count 0.60, recurrence
  count 0.53 (chance 0.50), sum 0.57. Why self-consistency is weak here: 32% of REAL claims never recur in 3 repeats
  (the model mentions different details each time), so absence in a repeat says little. Prediction confirmed:
  one-off inventions drop out (four of the 27B's five phantom spots on img24 recur 0-1/3), stable misreadings
  survive every filter (235B "fuzzy or cottony growth" on the stem hairs of img22: 3/3 repeats and two other
  examiners agree; 27B frost coating as "white powdery coating" on img23: 3/3). **Conclusion: none of the cheap
  consistency checks is a usable hallucination filter; each loses a third to a half of the real details to remove
  half to two-thirds of the false ones.**
- **Result, label-swap teacher test, Qwen3.8-27B (job 72134, test split, 796 verdicts):** accepts the true label
  360/398 = 90.5%, rejects a wrong label 270/398 = 67.8%, balanced 79.1% (32B teacher 77.6%, McNemar 62/74,
  p = 0.35); true condition named in its differential when told a wrong label 75.1% (32B 95.2%). The filtered 8B
  student (82.5%) beats it (65/92, p = 0.038). The 27B is more agreeable, not more honest: it goes along with a
  wrong label 32% of the time (32B 23%) and rarely offers the right alternative. Not a better teacher.
- **Pre-registered: (B) zero-shot diagnosis consensus, Hongbo's "caption many times and vote" + Gus's "say both
  when the models disagree" (`cluster/consensus_dx.py`, split v1 test, 398 photos, labels as ground truth).**
  Qwen3-VL-32B, Qwen3.8-27B, Qwen3-VL-8B (A100, bf16): per photo 1 greedy + 20 sampled diagnoses (T 0.7, seed =
  sample number), each a one-sentence key evidence then one of the 7 categories (schema-enforced, zero-shot prompt
  of `zeroshot_qwen3vl.py`, photos at 1024 px). Metrics: accuracy and macro accuracy of greedy vs majority vote over
  N = 1, 3, 5, 10, 20 samples (averaged over random subsets; 95% CI by bootstrap over photos); two-model and
  three-model voting; when two models' greedy answers disagree, how often the true label is one of the two (the
  two-answer caption) vs each alone; selective prediction (accuracy vs coverage when abstaining below a vote share).
  Reference lines on the same test set: fine-tuned SigLIP 2 recipe 96.3%, fine-tuned Qwen3-VL-8B 94.2%.
  Prediction: voting adds a few points at most over greedy and stays far below the fine-tuned models; model
  disagreement and low vote share flag most errors.
- **Pre-registered: (B4/B5) resolving two-model disagreements (`cluster/resolve_dx.py`,
  `cluster/slurm/resolve_pair.sbatch`; pair Qwen3-VL-32B + Qwen3.8-27B; test photos where their greedy answers in
  (B) differ).** Each model, on those photos: *choose* (told the two candidate diagnoses, decide; control),
  *refs_real* (choose + 2 train-split photos of each candidate), *refs_synth* (choose + 2 images of each candidate
  from the lab's GPT synthetic set: Hongbo's synthetic-reference idea), and a 3-round *debate* (Hongbo + Gus: each
  model sees its own and the other's previous diagnosis with evidence and answers again; round 0 = greedy; agreed
  photos are settled). All greedy, label never shown. Metrics on the disagreement photos: each model's accuracy
  per method vs its greedy accuracy there; the ceiling "truth is one of the two" (Gus's two-answer caption);
  agreement and accuracy after each debate round, including convergence on a wrong answer; extra model calls per
  photo, compared with 20-sample voting. Prediction: references from real photos help more than synthetic ones
  (the late synthetic frost batch already pulled real flowers toward frost); debate converges mostly to the more
  confident model rather than to the truth.
- **Result, (d) zoom-in judges on the A100 (job 72139; per judge and mode 845 claims + 134-135 decoys; 0-7 unparsed
  per run).** AUROC on the human-rated claims (whole photo / zoom): Qwen3-VL-8B 0.68 / 0.66, Qwen3-VL-32B 0.63 / 0.66,
  Qwen3.8-27B 0.61 / 0.62. Trade-off on one common set (562 claims: 515 real, 47 not there; judge drops claims it
  calls incorrect or cannot tell), false removed / real removed: 8B 59.6 / 21.4 % (whole) and 48.9 / 16.3 % (zoom);
  27B 31.9 / 8.5 % and 36.2 / 9.9 %; 32B 38.3 / 10.5 % and 44.7 / 12.8 %; vs agreement 59.6 / 41.7 %,
  self-consistency 66.0 / 52.0 %, combined 51.1 / 34.6 %. Every judge removes 2.8-3.7 false claims per real claim
  lost, the consistency checks 1.3-1.5. **Pre-registered predictions:** "zoom beats nozoom" is **not supported**
  (AUROC +0.03 / +0.01 / -0.02 for 32B / 27B / 8B); "the best judge beats agreement and self-consistency" is
  **supported** (best 0.69 vs 0.61 and 0.53 on the common set). Self-judging (judge = claimant): 32B 0.43 / 0.54 on its
  own 153 claims (only 7 false) vs 0.67 / 0.69 cross; 27B 0.59 / 0.66 on its own 186 claims (20 false) vs 0.62 / 0.60
  cross, so no consistent self-verification deficit (an earlier informal reading of the 32B alone suggested one; not
  borne out). Decoys accepted as correct, all (whole / zoom): 8B 9 / 13 %, 27B 23 / 19 %, 32B 19 / 15 %; many generic
  decoys ("brown, necrotic tissue") are true on the target photo by chance (decoy design flaw). Clean subset chosen
  after seeing the results (mold signs on non-mold photos, n = 33): 8B 3.0 / 9.1 %, 27B 15.2 / 12.1 %, 32B 12.1 /
  9.1 %. All three judges in both modes accept the img22 stem-hair "cottony growth" claims and confirm the real img20
  lesion claim. Exploratory: drop a claim when >= 2 of the 3 judges call it incorrect: 36.2 % false / 7.2 % real
  removed (whole photo, ratio 5.0) and 31.9 / 7.2 % (zoom); needs a fresh rated set before use.
- **Docs:** README rewritten and `docs/` pages added (benchmark, classifier, synthetic data, VLM honesty, captioning,
  detector, compute, roadmap, v1 system) with figures from `docs/figures/make_figures.py`; local branch, not pushed.
- **Result, (B) zero-shot diagnosis consensus (job 72142, A100, 37 min for 3 models x 398 photos x 21 diagnoses;
  1 unparsable 32B sample).** Accuracy greedy -> majority of 20 samples (T 0.7): Qwen3-VL-32B 67.6 -> 68.8 % (19
  samples; McNemar 17 / 12, p = 0.46), Qwen3-VL-8B 67.8 -> 70.1 % (15 / 6, p = 0.08), Qwen3.8-27B 65.3 -> 67.1 %
  (16 / 9, p = 0.23); macro 58-60 %. Two models pooled (2 x 20 votes) 69.8-72.4 %; all three pooled 73.9 % (macro
  64.0 %). Reference: fine-tuned SigLIP 2 recipe 96.3 %, fine-tuned 8B 94.2 %. **Prediction supported: voting adds
  1-2 points (none significant) and stays ~22 points below a 10-minute fine-tune.** Where the samples help is
  uncertainty: answering only the 50 % of photos with the highest vote share gives 84-87 % accuracy (70 %: 78-79 %);
  the true label is among a model's 2 most-voted answers 80-82 % of the time. **Gus's two-answer captions:** when two
  models' greedy answers disagree (114-140 photos per pair), the truth is one of the two 68-74 % of the time vs
  30-40 % for either alone; but 45-58 photos per pair (11-15 % of all) are agreed-but-wrong, which no consensus can
  flag. Of the 32B's 129 greedy errors, 84 (65 %) fall on photos where the 27B disagrees. Top confusions of the best
  vote (8B): gray mold -> healthy 26, white mold -> gray mold 17, frost -> healthy 10, gray mold -> root rot 10.
  Side observation (not a controlled comparison: vLLM bf16 here vs transformers / 4-bit earlier): with the
  evidence-sentence-first JSON prompt, greedy zero-shot accuracy is lower than with the bare-answer prompt of
  2026-10-20 (8B 77.1 -> 67.8 %, 32B 72.4 -> 67.6 %, 27B 65.6 -> 65.3 %); worth a controlled test, since the
  annotation pipeline also asks for evidence before a verdict.
- **Pre-registered: answer-format ablation (controlled; `consensus_dx.py --prompt-style`, greedy only, split v1 test,
  same vLLM bf16 stack, same images at 1024 px and same category definitions).** Formats: *text* (free text "Reply with
  only the category name", the 2026-10-20 zero-shot prompt), *bare* (JSON with only the category), *after* (JSON:
  category, then one evidence sentence), *evidence* (JSON: evidence sentence, then category; = sample 0 of the
  consensus run above). Models: Qwen3-VL-8B, Qwen3-VL-32B, Qwen3.8-27B. Metric: accuracy and macro, paired McNemar of
  each format vs *evidence*. Prediction: *evidence* is below *text* / *bare* for the 8B and 32B (as the uncontrolled
  comparison suggests), *after* is close to *bare*. If confirmed, the annotation pipeline's evidence-before-verdict
  order needs its own test.
- **Result, label-swap teacher test, Qwen3-VL-235B-A22B AWQ (job 72137, PP 5 over the RTX nodes, 1 h 42 min for 796
  verdicts, ~120-130 output tok/s; 3 invalid, so 396 paired photos):** accepts the true label 340/396 = 85.9 %, rejects
  a wrong label 299/396 = 75.5 %, **balanced 80.7 %**; true condition in its differential when told a wrong label
  349/396 = 88.1 %. vs the 32B teacher (77.6 %): 68 / 44, **p = 0.029** (significantly more honest); vs Qwen3.8-27B
  (79.1 %): 67 / 55, p = 0.32; vs the filtered 8B student (82.5 %): 47 / 62, p = 0.18. The gain over the 32B is in
  accepting true labels (85.9 vs 78.6 %), not in rejecting wrong ones (75.5 vs 76.6 %), and it names the right
  alternative less often (88.1 vs 95.2 %). For filtered distillation this means more usable true-label examples per
  photo but no better wrong-label examples, at ~4x the GPU time of the 32B on the A100. The distilled 8B student remains
  at least as honest as every zero-shot model tested.
- **Result, (B4/B5) resolving two-model disagreements, Qwen3-VL-32B + Qwen3.8-27B (job 72143, A100, 29 min).** 140 of
  398 test photos have different greedy answers; there the 32B is right 40.0 %, the 27B 33.6 %, and the truth is one
  of the two answers 73.6 % of the time (a random pick between them would score 36.8 %). Accuracy on the 140
  (32B / 27B), photos where both end up agreeing (right / wrong), and the pair system on all 398 (32B's resolved
  answer on disagreements; 32B greedy alone 67.6 %):
  *choose* 28.6 / 39.3 %, agree 68 (22 / 46), system **63.6 %, significantly worse** (McNemar 17 / 33, p = 0.03);
  *refs_real* 37.1 / 43.6 %, agree 89 (40 / 49), 66.6 % (p = 0.69); *refs_synth* 40.0 / 46.4 %, agree 86 (43 / 43),
  67.6 % (p = 1.0); *debate* rounds 1-3: 32B 41.4 / 40.0 / 40.0 %, 27B 37.9 / 41.4 / 43.6 %, agree 43 -> 64 -> 71
  (after round 3: 36 right / 35 wrong), system 68.1 / 67.6 / 67.6 %. Flips since greedy after round 3: 27B 24
  wrong -> right, 10 right -> wrong (it mostly defers to the 32B); 32B 13 / 13. **Conclusions:** no method beats
  trusting the stronger model's greedy answer; arbitrating between two candidate answers is at chance level for
  these models; debate converges on a wrong answer about half the time (prediction "converges on the more confident /
  stronger model rather than the truth": consistent); narrowing to two candidates ("choose") hurts. Prediction "real
  references help more than synthetic ones": **not supported** (synthetic 40.0 / 46.4 % vs real 37.1 / 43.6 %, both
  n.s. vs greedy). Hongbo's synthetic-reference idea is harmless here but not useful. The honest output for a
  disagreement remains Gus's two-answer caption (right 73.6 % of the time as a pair).
- **Result, label-swap teacher test, GLM-4.6V FP8 (job 72138, PP 5, 37 min for 796 verdicts, ~165 output tok/s):**
  accepts the true label only 199/398 = 50.0 %, rejects a wrong label 311/398 = 78.1 %, **balanced 64.1 %**; true
  condition in its differential 78.4 %. Far below every Qwen model (vs 32B 74 / 182, p = 1.1e-11; vs 235B 68 / 198,
  p = 6.6e-16; vs the student 55 / 202, p = 6.4e-21). Verdict mix on TRUE labels: cannot tell 144, partially 121,
  clearly 78, does not support 55 (Qwen3-VL-32B: cannot tell 1; 235B: 2); on WRONG labels it still says "clearly
  supports" 24 times (Qwen 4-5). Over-cautious on correct labels and occasionally confidently wrong: **ruled out as a
  teacher.** Teacher ranking on this test: 235B 80.7 > 27B 79.1 > 32B 77.6 > base 8B 73.2 > GLM-4.6V 64.1
  (filtered 8B student 82.5).
- **Result, answer-format ablation (jobs 72147-72149, controlled: same vLLM bf16 stack, greedy, 398 test photos).**
  Accuracy text / bare JSON / category-then-evidence / **evidence-then-category**: Qwen3-VL-8B 76.9 / 77.1 / 77.1 /
  **67.8 %** (evidence vs text 16 / 52, p = 1.4e-5); Qwen3-VL-32B 74.4 / 73.4 / 72.4 / **67.6 %** (13 / 40,
  p = 0.0003); Qwen3.8-27B 67.1 / 66.3 / 69.3 / 65.3 % (vs text n.s.; vs after 12 / 28, p = 0.017). **Prediction
  confirmed: writing the evidence sentence BEFORE the answer costs 7-9 points for the Qwen3-VL models; writing it
  after costs nothing.** The text format reproduces the 2026-10-20 zero-shot result on the new stack (8B 76.9 vs 77.1 %),
  so the earlier gap was the format, not the stack. Mechanism: gray mold collapses (8B 93 -> 64 %, 32B 93 -> 77 %);
  the evidence sentence commits to a salient generic observation and often states the key sign is absent ("ripe red
  strawberry ... with no visible mold", "a dried, brown, shriveled flower calyx ... no signs of mold"), and the label
  follows the sentence (8B errors move to healthy 28 and root rot 18; 32B to drought 15 and frost 13). It helps white
  mold (32B 9 -> 30 %) but is net negative. **Implication for the annotation pipeline:** pass 2 decides from pass 1's
  written observations only, so a faint sign that pass 1 omits cannot support the label (the same failure mode);
  test verdict-first vs observations-first in the annotation schema, and keep an image-in-the-loop check (the judge)
  for absence claims.
- **Result, (d) zoom-in judge Qwen3-VL-235B-A22B AWQ (job 72140, PP 5, zoom mode only, 42 min for 980 items, 0
  unparsed):** AUROC 0.61 on the rated claims (0.608 on the 562-claim common set): with the 27B the weakest judge,
  below the 8B (0.66-0.69) and the 32B (0.63-0.66). Very permissive: 514 of 571 rated claims judged correct, only 37
  incorrect (8B 98-118, 32B 59); dropping incorrect + cannot tell removes 29.8 % of false and 8.2 % of real claims on
  the common set (ratio 3.65, same efficiency as the other judges, lowest recall). Decoys accepted 19 % (clean mold
  decoys 3/33 = 9.1 %). Self-judged 0.62 (196 claims, 21 false) vs cross-judged 0.61: again no self-verification
  deficit. Accepts the img22 stem-hair "cottony growth" claims, confirms the img20 lesion. **Judge quality does not
  grow with size here (8B >= 32B > 27B ~ 235B);** the best label-swap teacher (235B) is among the weakest verifiers,
  consistent with its tendency to accept (85.9 % of true labels, 90 % of claims).
- **Failed run, (d) zoom-in judge GLM-4.6V FP8 (job 72141, PP 5, zoom mode = two images per prompt):** the job
  completed, but all 980 outputs end after a few tokens (`{"visible!`) with finish reason `error`; `!` is token 0 of
  the vocabulary, the usual sign of NaN logits under a grammar mask. GLM-4.6V ran cleanly with one image per prompt
  (bake-off 38/38, label-swap 796/796), so this looks like a vLLM 0.32 problem with two images per prompt for this
  model under pipeline parallelism; not debugged. Rerun queued in the single-image (nozoom) mode (job 72150), which
  is still a fair cross-family test since zoom had no consistent effect for the three Qwen judges. No GLM judge
  numbers are reported from job 72141.
- **Result, (B) diagnosis consensus, Qwen3-VL-235B-A22B AWQ (job 72144, PP 5, 14 min for 398 x 11 diagnoses; 35 of
  4,378 outputs unparsable, cut at the 120-token limit by longer evidence sentences, so 3-10 valid samples per photo,
  median 10):** greedy 69.1 % (macro 61.9 %), the best greedy of the four with the evidence-first prompt (32B 67.6,
  8B 67.8, 27B 65.3 %); majority over all valid samples 70.1 % (12 / 8, p = 0.50); accuracy on the 50 % most unanimous
  photos 83.4 %. Same picture as the other models: voting +1 point, not significant. Pairs with the 235B: when the two
  greedy answers differ, the truth is one of them 66-77 % of the time (235B + 32B: 77.2 %, 101 photos). Scoring note:
  `consensus_score.py` now counts an unparsable greedy answer as wrong (it crashed on the 235B's one missing greedy
  answer) and its "majority of N" uses the minimum valid count across photos, so the 235B line above was recomputed
  over all valid samples per photo.
- **Result, (B) diagnosis consensus, GLM-4.6V FP8 (job 72145, PP 5, 13 min; 28 of 4,378 outputs empty):** greedy
  **75.6 %** (macro 62.2 %), the best zero-shot diagnoser of the five with the evidence-first prompt (235B 69.1, 8B
  67.8, 32B 67.6, 27B 65.3 %); majority over valid samples 76.1 % (16 / 14, p = 0.86). Per class (greedy): root rot
  100, gray mold 88, healthy 81, white mold 62, frost 48, drought 29, overwatered 26 %; top confusion white mold ->
  gray mold (21). **The best zero-shot diagnoser is the worst label-swap teacher (64.1 %, over-cautious): diagnosing a
  photo and judging whether a given label fits it are different skills in these models.** Voting adds nothing for
  any of the five models (+0.5 to +2.3 points, none significant). GLM was not run in the other answer formats.
- **Correction (2026-10-22) to the GLM-4.6V judge failure above:** the single-image rerun (job 72150) failed the same
  way (980 / 980), so the two-images explanation was wrong. vLLM logs "grammar rejected tokens [0] ... Terminating
  request" for every request, right after GLM writes `"visible` while spelling the JSON key `visible_in_box`; GLM
  handled other schemas (bake-off, label-swap) fine. The cause is the key's tokenization under grammar-constrained
  decoding, not the images. `judge_claims.py` gained `--desc-key` (the description field's name; outputs are
  normalised back to `visible_in_box`) and now retries failed records on resume. Rerun with `--desc-key description`,
  both modes (job 72159). Protocol deviation for GLM only: the description field is named `description`.
- **Result, (d) judge GLM-4.6V FP8 rerun with `--desc-key description` (job 72159, PP 5, both modes, 30 min):
  inconclusive.** Unparsed 127 / 980 (whole photo) and 85 / 980 (zoom), mostly degenerate text ("...enedlockenedlock
  enedlock...") ending in a grammar rejection, so the runtime (GLM-4.6V FP8 + vLLM 0.32 + pipeline parallel + grammar
  decoding) is not producing reliable text for this prompt. The parsed outputs say "correct" to nearly everything
  (491 / 519 and 505 / 527 rated claims): AUROC 0.51 / 0.51 (chance), 2-7 % of false claims removed, decoys accepted
  49 % / 61 %. Not reported as GLM's verification ability; GLM is not a usable judge on this stack. The cross-family
  question stays open (a non-Qwen judge on another runtime, e.g. transformers, would be needed).
- **Pre-registered: A8, annotation-schema order (`annotate_vllm.py --order verdict-first`).** Same prompt, model,
  test photos, seeded wrong labels and decoding settings as the label-swap runs of 2026-10-20; only the JSON property
  order changes: `label_assessment` (verdict first, then supporting / missing / conflicting evidence) moved ahead of
  image quality, parts and observations. Models: Qwen3-VL-32B (vs run v1, 77.6 %) and base Qwen3-VL-8B (vs run base,
  73.2 %), A100, bf16. Metric: balanced honesty, accept-true, reject-wrong, true condition in the differential;
  paired McNemar on the 796 verdicts. **No directional prediction:** the answer-format ablation (describing first
  anchors zero-shot classification on generic observations) predicts verdict-first is at least as honest; "look before
  judging" (observations first resist the given label) predicts the opposite.
- **Result, A8 annotation-schema order (jobs 72202-72203, A100, test split, 796 verdicts each).** Verdict-first vs
  observations-first (accept-true / reject-wrong / balanced / true condition in differential):
  Qwen3-VL-32B 92.5 / 65.3 / **78.9 %** / 95.7 % vs 78.6 / 76.6 / **77.6 %** / 95.2 % (overall McNemar 62 / 52,
  p = 0.40); true-label half 57 / 2, p = 6e-15; wrong-label half 5 / 50, p = 2e-10. Base Qwen3-VL-8B 76.1 / 76.1 /
  **76.1 %** / 85.7 % vs 72.4 / 74.1 / **73.2 %** / 89.2 % (79 / 56, p = 0.058); halves 45 / 30 (p = 0.11) and 34 / 26
  (p = 0.37). **Conclusion:** no general effect of order on balanced honesty (32B +1.3 n.s., 8B +2.9 borderline). For
  the 32B the order moves the acceptance threshold, not the discrimination: writing observations first makes it
  markedly more sceptical (goes along with a wrong label 23.4 % vs 34.7 % verdict-first) at the cost of rejecting more
  true labels; the 8B shows no such shift. For filtered distillation (wrong verdicts are dropped anyway) the order
  trades true-label yield (verdict-first) against wrong-label yield (observations-first); the current
  observations-first schema gives the student more examples of rejecting a wrong label. Unlike zero-shot
  classification (evidence-first -5 to -9 points), describing first does not hurt the verification verdict.
- **Exploratory (post hoc subgroup, 8 false claims only): judges on absence vs presence claims.** Of 571 rated
  claims, 104 assert that something is absent ("no visible ..."), 8 of them rated not there (7.7 %; presence claims
  42 / 467 = 9.0 %). Zoom raises every judge's AUROC on absence claims (32B 0.57 -> 0.66, 8B 0.61 -> 0.68, 27B 0.57 ->
  0.62) but not on presence claims (0.65 -> 0.66, 0.70 -> 0.66, 0.62 -> 0.62). Plausible (verifying that a sign is
  NOT there needs a close look) but resting on 8 events: a hypothesis for the next rated set, not a result. If it
  holds, the pipeline could zoom only for absence claims.
- **Pre-registered: answer-format extension to the large models.** Qwen3-VL-235B AWQ and GLM-4.6V FP8 (PP 5), greedy,
  `--prompt-style text` (bare category, the 2026-10-20 zero-shot prompt), compared with their evidence-first greedy
  answers (sample 0 of jobs 72144 / 72145). Prediction: evidence-first is lower for both, as for the Qwen3-VL 8B / 32B.
- **Result, answer-format extension (jobs 72204-72205).** Qwen3-VL-235B: plain-text answer **77.4 %** vs evidence-first
  69.1 % (47 / 14, p = 2.7e-5); gray mold 93 vs 67 %. **Prediction confirmed for the 235B: the evidence-first cost
  (8.3 points) replicates at 235B scale.** GLM-4.6V: the text run is **invalid** (17.6 %, 102 unparsed): its raw
  answers carry corrupted tokens ("graylock_mold"; 232 of 398 photos "frost"), the same glued "lock" artifact as in
  the failed judge runs.
- **Integrity check: GLM-4.6V runtime corruption tied to node c5 (investigating).** Every GLM run that produced
  degenerate text (judge 72141 / 72150 / 72159, text 72205) had pipeline rank 1 (layers 7-16) on c5; none of the clean
  GLM runs did (bake-off 72119, repeats 72136, label-swap 72138, voting 72145; a scan of all their outputs finds no
  "lock" artifact). c5 copied the rank-1 files in 72141 and later runs reused them. c5 was also the node behind two
  odd crashes on 2026-10-20 and the starved KV stage of the first 235B run. Hypothesis: a corrupted staged weight file
  on c5 (vLLM skips weight checks for quantized checkpoints), or a faulty node. Byte comparison of c5's staged GLM files
  against /projects running (job 72206). **Consequence: the GLM judge results (inconclusive) are explained by a broken
  runtime, not by GLM; GLM's label-swap, voting and bake-off results come from clean runs.**
- **Integrity finding: node c5's local disk (`/scratch`) returns corrupted data (jobs 72206-72208).** 22 of the 31
  GLM-4.6V weight files staged on c5 differ byte-for-byte from the originals at identical sizes. Hash test on one file
  (`model-00007`, 2.7 GB): c5 and c6 both read the /projects original as 3a4c5c73... (twice each), c6's staged copy
  matches it, c5's staged copy reads e8dfc7c1... (twice). So c5 reads NFS correctly but its stored copies are wrong
  and stable: the corruption happened on c5's write path or disk, not in the copy source. It also explains why some
  runs that used c5 were clean: right after staging, reads are served from the (correct) page cache; once evicted,
  later runs read the corrupted disk copy. Because vLLM does not check quantized weights, corrupted files load
  silently and produce degenerate text ("lock" tokens) or grammar errors. **Actions:** c5 excluded from jobs; full
  verification of every model staged on c5 running; results that used c5 to be reviewed (all clean-looking outputs so
  far: no artifact in any Qwen / 235B output, accuracies in expected ranges); report to IT; staging to verify file
  hashes against a per-model SHA256 manifest before loading.
- **c5 integrity review (2026-10-22, in progress).** The full check of c5's /scratch (job 72209, still running) finds
  corrupted copies in every model checked so far: GLM-4.6V 20/30, Qwen3-VL-235B AWQ 18/36, Qwen3-VL-32B 12/14 weight
  files. Yet 235B runs that used c5 stayed coherent and accurate as late as 05:13 today (format run 77.4 %), so the
  disk may be degrading progressively or corrupting intermittently, not only at write time. c5 ran many jobs since
  2026-10-19 (sacct): the v1 grounding smoke test, zero-shot runs 71980/71989, the 8B LoRA seed 0 (72003), ResNet
  ablation runs (three of which segfaulted there, 72009/72012/72016), VLM probability runs 72021/72022, the RF-DETR
  evaluation of the lab checkpoint (72035), the 5-fold CV probes (72039), fine-tuned backbone runs 72070/72075/72085/
  72091, the student honesty evaluation 72068, and rank(s) of every 5-node vLLM run. Cross-checks that already hold:
  zero-shot 8B 77.1 % (c5) reproduced at 76.9 % on the A100 tonight; RF-DETR lab checkpoint 0.598 (c5) vs retrained
  Small 0.593 elsewhere; classifier seeds run on c5 agree with seeds run elsewhere; no 235B / Qwen output shows the
  corruption artifact. **Re-check queued for the headline student result** (82.5 %, evaluated on c5): re-merge from the
  adapter on /projects and re-run the label-swap test on a healthy node (job 72215), compare verdicts. Staging now
  verifies against SHA256SUMS manifests (written tonight for GLM-4.6V, Qwen3-VL-235B, 32B, 27B, 8B; jobs 72210-72214).
- **Result, c5 integrity review completed.** Full check of c5's /scratch (job 72209): corrupted copies of 55 of 88
  weight files across every model that has an original on /projects (GLM-4.6V 20/30, Qwen3-VL-235B 18/36, Qwen3-VL-32B
  12/14, Qwen3-VL-8B 2/4, Qwen3-VL-8B-clsft 3/4; all sizes identical). **Headline student result re-verified on a
  healthy node** (job 72215, c3: re-merged from the adapter on /projects, label-swap test re-run): accepts true 80.9 %,
  rejects wrong 85.4 %, **balanced 83.2 %** (original c5 run 82.5 %), differential 97.7 %; 25 of 796 verdicts differ
  (15 / 10, p = 0.42, re-merge and batching noise); vs the 32B teacher 72 / 28, p = 1.3e-5 (originally 70 / 31,
  p = 0.00013). The result stands. Other c5 results are corroborated by runs elsewhere (zero-shot 8B, RF-DETR lab
  checkpoint, classifier seeds); the corrupted runs (GLM judge / text) were already excluded. c5 stays excluded until
  IT has checked it.

## 2026-10-22

- **Image resolution is confounded with the label (Gus, Discord 2026-10-22).** Long side of the lab's real photos
  (repo copy, train + holdout, synthetic excluded; to be confirmed on split v1): healthy n = 330, median 275 px, 78 %
  <= 300 px, 7 % >= 1000 px; overwatered median 1024 px (2 % <= 300); drought 773 (4 %); frost 768 (6 %); root rot 516
  (19 %); gray mold and white mold **all exactly 512 px** (resized source dataset). A model can therefore read
  "thumbnail -> healthy", "exactly 512 -> mold", "large phone photo -> water stress / frost" from upsampling blur,
  JPEG artefacts and sharpness. Existing evidence that the baseline uses it: on the 2026-10-20 low-quality test,
  real-only SigLIP 2 models called 128 px thumbnails of diseased plants healthy 108 times (3 seeds); with the
  low-quality augmentation applied to every class, 8. Planned direct tests (William's proposals): (1) downscale every
  non-healthy test photo to the healthy median size (275 px + JPEG) and measure P(healthy) and how often healthy
  becomes top-1 / top-2, real-only vs augmented models; (2) downscale the large healthy test photos to 275 px and
  compare their healthy top-1 confidence before / after (should drop slightly; a rise means the model rewards
  smallness); (3) natively small vs large photos within each disease class on clean test predictions.
- **Pre-registered: resolution-shortcut test (William's design; `cluster/eval_shortcut.py`).** Split v1 test, SigLIP 2
  fine-tunes, 3 seeds each of three training recipes: real only (s0-s2), real + low-quality augmentation (lq), and the
  recommended recipe (ChatGPT synth w/o frost 3-5 + low-quality augmentation, nof3_lq). Conditions: clean; thumb275 =
  long side 275 px (the healthy median) + JPEG q75; thumb275_raw = 275 px without JPEG. Full softmax probabilities saved
  (temperature-scaled, hflip TTA as in training). Tests: (A) non-healthy photos larger than 275 px: P(healthy) clean vs
  thumb275, and how often healthy becomes top-1 / enters the top-2; (B) healthy photos with long side >= 500 px:
  P(healthy) clean vs thumb275 (a rise = the model rewards smallness); (C) clean predictions, natively small (<= 300 px)
  vs large photos within each non-healthy class. **Control:** some rise of P(healthy) in (A) is legitimate (faint signs
  vanish at 275 px), so the shortcut is measured as the difference between real-only and augmented models.
  Predictions: (A) the rise in P(healthy) and in healthy top-1 is larger for real-only than for lq / nof3_lq;
  (B) P(healthy) rises for real-only and stays flat or drops for the augmented models.
- **Result, resolution-shortcut test (job 72311, A100, 7 min; 9 SigLIP 2 models; clean accuracies reproduce the saved
  results).** Split v1 confirms the confound: healthy photos <= 300 px 83 % of train / 72 % of test (median 275 px);
  white mold 100 % and gray mold 90-95 % exactly 512 px; overwatered / drought / frost medians 600-1024 px.
  (A) 309 non-healthy test photos shrunk to 275 px + JPEG: P(healthy) rises 0.005 -> 0.013 for real-only models
  (+0.008, 95 % CI +0.004 to +0.013), +0.003 / +0.004 for the augmented recipes (real-only minus augmented +0.004 to
  +0.005, CI excludes 0); healthy top-1 0.2 -> 1.0 % (real only), <= 0.6 % (augmented); healthy in the top 2 2.2 ->
  5.4 %. **Prediction confirmed in direction, but tiny at 275 px** (at 128 px the 2026-10-20 test found far more
  thumbnails called healthy). (B) 14 large healthy photos (>= 500 px) shrunk to 275 px: P(healthy) rises +0.071
  (CI +0.019 to +0.143, on 14 / 14 photos) real only, +0.055 (+0.020 to +0.100) with augmentation, +0.033 (+0.009 to
  +0.065, 13 / 14) recommended; healthy accuracy on them 64 -> 74 %, 62 -> 67 %, 74 -> 79 %. By William's criterion
  (removing detail must not raise confidence) this is a shortcut signature, in every recipe; **prediction for the
  augmented models (flat or lower) not supported**: augmentation shrinks the effect but does not remove it.
  **Clearest symptom (natural experiment, clean photos): healthy accuracy depends on photo size: 97-100 % for
  thumbnails (<= 300 px, n = 58) vs 62-74 % for large photos (>= 500 px, n = 14), misread as frost (6-10), drought
  (3-4) and overwatered (1-3) over 3 seeds: the classes made of large phone photos.** So the models partly learned
  "large sharp phone photo -> stress condition" (the mirror of Gus's "small -> healthy"). Caveats: 14 large healthy
  test photos; content differs too (e.g. white flowers, frost-like). (C) natively small disease photos: n = 11, too
  few. **Implications:** split v1 accuracy overstates performance on farmers' full-resolution phone photos of
  healthy plants, exactly where the errors concentrate; fixes: real large healthy photos (field holdout and training
  data), resolution randomised for every class in training (stronger than --lowq-aug 0.3), and a size-stratified
  report of test accuracy.
- **Result, shortcut test on the synthetic-data recipes without augmentation (job 72312).** Healthy accuracy on
  thumbnails / large photos, and the rise of P(healthy) when the 14 large healthy photos are shrunk to 275 px (3 seeds):
  real only 99 / 64 %, +0.071; + ChatGPT synth 98 / 69 %, **+0.118**; + lab GPT synth 98 / **52 %**, **+0.164**;
  + low-quality aug 97 / 62 %, +0.055; + ChatGPT synth + aug (recommended) 100 / 74 %, +0.033. **Synthetic healthy
  images alone make the resolution shortcut worse, not better** (both sets; the lab set's 97 healthy images at
  1254-1672 px lower large-healthy accuracy to 52 %). Every synthetic image is 800-1672 px in every class (drought,
  frost, overwatered too; lab overwatered +154), so the synthetic sets add at least as much "large -> stress" as
  "large -> healthy", and none of them adds a non-healthy thumbnail. Only synthetic data combined with the low-quality
  augmentation improves both measures. (14 large healthy photos: one photo = 7 points; the P(healthy)-rise ordering
  is the more stable measure.)
- **Pre-registered: augmentation dose (`--lowq-aug 0.6` vs 0.3).** Same command as the recommended recipe
  (`train_backbone.py --backbone siglip2-so400m-512 --train-root splits/v1_chatgpt_nof3/train --batch-size 16`, A100,
  seeds 0-2), only the degradation probability doubled; evaluated with eval_lowq.py and eval_shortcut.py.
  Prediction: P(healthy) rise for shrunk large healthy photos below +0.033 and large-healthy accuracy >= 74 %, with
  clean test accuracy within seed noise of 96.3 % (possibly up to 1 point lower).
- **Result, augmentation dose (jobs 72313-72316): prediction not supported.** `--lowq-aug 0.6` (ChatGPT synth recipe,
  3 seeds): clean 95.6 % (0.3: 96.3 %), 128 px 90.2 % (0.3: 92.1 %); healthy accuracy thumbnails 97 % / large 67 %
  (0.3: 100 / 74 %); P(healthy) rise for shrunk large healthy photos +0.048 (CI +0.019 to +0.080; 0.3: +0.033);
  disease photos shrunk +0.0018 (0.3: +0.0039). More degradation only shrinks the already tiny disease-thumbnail
  effect and costs ~0.7 points clean; it does not teach what a sharp, full-resolution healthy plant looks like, so the
  "large photo -> stress" direction stays. `--lowq-aug 0.3` remains the recipe. **Summary of the shortcut study:**
  the confound has two directions; augmentation addresses "thumbnail -> healthy" (disease thumbnails rarely flip at
  275 px), while "large photo -> stress" (large healthy photos 52-74 % correct across six recipes) needs real
  full-resolution healthy photos (roadmap C6); synthetic healthy images do not supply it (they made it worse).
- **Public release in the lab repository (`greatroboticslab/AGAI`, `main`).** The documentation (README, `docs/`)
  and the code behind every reported number were published. There the scripts are grouped by study
  (`cluster/benchmark/`, `classifier/`, `synthetic/`, `vlm/`, `detector/`, `infra/`; launchers in `cluster/slurm/`);
  the commands in this notebook keep the flat `cluster/<script>` layout of the cluster working copy. Not published:
  one-off diagnostics, superseded annotators, the per-experiment Slurm wrappers, and the blind-rating key (until the
  ratings are complete; the rating-page link in the 2026-10-20 entry is replaced by a placeholder in the public copy).

## 2026-10-30

- **Context.** Gus's manuscript draft (24 Oct build, "Leak-free re-evaluation of a strawberry plant health AI system")
  leaves the clean v1 numbers to William: calibration and acceptance gate of the classifier (its Table S7 / Fig. S9),
  a mirrored-view ablation, a detector test split, the report writer retrained without split v1 test images and the
  detector on/off ablation (Table S16), the v1 vs v2 end-to-end comparison, software versions. Answer sheet for the
  co-authors: private claude.ai page (link not published). Analyses below are registered before their results are seen.
- **Pre-registered: calibration and acceptance gate on split v1 (`cluster/calibration_metrics.py`).** Inputs: the saved
  per-image probability files, no new training: ResNet-50 lab recipe, 5 seeds (`scores_resnet50_split_v1_runtime_*`,
  `scores_resnet50_v1_s1-4_runtime_*`; runtime = the app's resize-294 / centre-crop-256 path) and the recommended
  SigLIP 2 recipe, 3 seeds (`ft_siglip2-so400m-512_chatgpt_nof3_lq_s0-2_*`). Unscaled probabilities are recovered
  exactly from the stored temperature-scaled ones (p_raw ∝ p_cal^T). Metrics, with and without temperature scaling:
  accuracy, macro accuracy, macro-F1, ECE (10 equal-width bins of the top-1 probability), AURC and the risk-coverage
  curve; the deployed v1 gate (p1 ≥ max(τ_c, 0.5), app thresholds healthy 0.40, drought / root rot 0.65, frost 0.70,
  others 0.60) and a validation-chosen gate (per predicted class, the smallest τ ≥ 0.5 on a 0.01 grid at which the
  accepted validation images of that class are ≥ 95 % correct; if never reached, the τ with the highest such accuracy);
  per-class precision / recall / F1, confusion matrix; 10,000-resample percentile bootstrap intervals over test images;
  seeds as mean ± sd. Predictions: temperature scaling lowers the ECE of the ResNet-50 (its fitted T ≈ 0.75 sharpens an
  under-confident model); the validation-chosen gate keeps ≥ 80 % of test images at ≥ 95 % accuracy for SigLIP 2 and
  less coverage for the ResNet-50. Gus's figure script consumes the per-image files `evaluation/calib_<name>_test_seed<k>.json`.
- **Pre-registered: mirrored-view ablation (`cluster/tta_ablation.py`).** ResNet-50, 5 seeds, split v1 test, runtime
  preprocessing: one view vs image + mirror (logits averaged), with the counts of predictions the mirror turns from
  right to wrong and from wrong to right (Shanmugam et al. 2021). Prediction: the mirror changes < 2 % of predictions
  and the net gain is within ±2 images per seed.
- **Report-writer retraining, data step (`cluster/filter_stage2_records.py`), descriptive.** The 2159 o3 reports
  are assigned to split v1 by their source photograph (augmented copies follow their original): train records train the
  writer, val records select the checkpoint, test records are dropped. Counts recorded when the script runs.
- **Finding: the 93 generated files with random names in the lab machine's `train_aug` (Gus's question).** They are
  UUID-named PNG files of 2.9-3.8 MB (ChatGPT web downloads carry such names), all dated 2026-04-27, in drought (50),
  frost (20) and gray mold (23); they carry the OpenAI C2PA record. Added two days after the paper's synthetic set (the
  601 `synth_*GPT*` files, 602 before one exact duplicate was dropped), on the day the lab's "synth" / "synthOOD"
  ResNet checkpoints v5-v12 start, so they belong to those later runs. The revival's synthetic-data arms used only the
  601 named files; the 93 enter no reported result.
- **Published-copy note.** The public copy of this notebook (lab repository, `cluster/EXPERIMENT_LOG.md`) replaces the
  blind-rating page link with a placeholder and omits nothing else; the canonical copy stays `evaluation/EXPERIMENT_LOG.md`
  on Hamilton.
- **Result, calibration and acceptance gate (jobs 72331 / 72332, research partition, `cluster/calibration_metrics.py`;
  files `evaluation/calib_<model>_test_seed<k>.json`, `evaluation/calib_<model>_summary.json`).** Split v1 test,
  mean ± sd over seeds, without -> with temperature scaling. ResNet-50 lab recipe (5 seeds, T = 0.72-0.78): accuracy
  90.2 ± 1.5, macro 85.5 ± 2.0, macro-F1 0.857 ± 0.023 (unchanged by T); ECE 16.7 ± 1.7 -> 6.7 ± 0.8 %; AURC 0.0249 ±
  0.0078 -> 0.0240 ± 0.0072; deployed v1 gate coverage 87.6 ± 1.1 % at 94.5 ± 0.6 % accuracy on accepted images;
  validation-chosen gate 84.2 ± 3.2 % at 95.7 ± 0.6 %. SigLIP 2 recommended recipe (3 seeds, T = 0.73-0.76): accuracy
  96.3 ± 0.8, macro 94.3 ± 0.7, macro-F1 0.936 ± 0.008; ECE 8.3 ± 1.3 -> 2.7 ± 0.3 %; AURC 0.0048 ± 0.0013 -> 0.0046 ±
  0.0016; v1 gate 97.7 ± 1.6 % at 97.5 ± 0.1 %; validation-chosen gate 96.6 ± 1.0 % at 97.9 ± 0.5 %. Prediction held
  (scaling sharpens an under-confident model and lowers ECE); the coverage prediction for SigLIP 2 held (≥ 80 % at
  ≥ 95 %) and the ResNet-50 also exceeds it (84 %). Caveat: the validation-chosen thresholds swing across seeds for the
  small conditions (drought 0.64-0.93) because 3-7 validation images are predicted as those conditions; report the
  deployed thresholds as v1's setting and the validation rule as the alternative with its variance.
- **Result, mirrored-view ablation (job 72330, c3, `cluster/tta_ablation.py`; `evaluation/tta_<ckpt>_split_v1_test.json`).**
  ResNet-50, 5 seeds (42, 1, 2, 3, 4), two views vs one: 91.7 / 91.0, 91.5 / 91.2, 90.2 / 90.0, 88.9 / 89.5, 88.4 / 87.9 %.
  Predictions changed: 6, 5, 5, 14, 11 of 398 (1.3-3.5 %); wrong -> right 4, 2, 3, 4, 5; right -> wrong 1, 1, 2, 6, 3
  (18 vs 13 over seeds; mean +0.25 points, range -0.5 to +0.8; n.s.). The "< 2 % changed" prediction failed for two
  seeds; the "net gain within ±2 images" prediction failed for seed 42 (+3) and seed 3 (-2 is within).
- **Finding: the v1 RF-DETR checkpoint was trained at 512 px, not the YAML's 560.** Its saved arguments hold no
  resolution field, but `backbone.0.encoder.encoder.embeddings.position_embeddings` has shape (1, 1025, 384): 1024 patch
  positions = 32 × 32 patches of 16 px = 512 px (`square_resize_div_64` on). `rfdetr_parts/train_rfdetr.py` builds
  `RFDETRSmall()` at the library default and the YAML value passed to `train()` did not rebuild the model. DETECTOR.md's
  "560" for the lab checkpoint is wrong and will be corrected; the retrained Small at 512 is therefore like for like.
- **Environments frozen** to `/projects/was4g/agai/envs/<env>.freeze.2026-10-30.txt` (minigptv: torch 2.5.1+cu121,
  transformers 4.45.2, peft 0.13.2 (COMPUTE.md says 0.11.1: correct it), bitsandbytes 0.43.3, timm 0.6.13, Python 3.9.23;
  rfdetr: rfdetr 1.11.2, torch 2.14.1, Python 3.11.17; vision: torch 2.9.1+cu128, timm 1.0.30, open_clip_torch 3.3.0,
  diffusers 0.41.0, transformers 5.20.2; vllm: vllm 0.32.0, torch 2.13.0, transformers 5.19.0, ray 2.59.0, xgrammar 0.2.8).
- **Report-writer data (`cluster/filter_stage2_records.py` -> `plant_diagnostic/datasets/split_v1/`):** of the 2159
  records, train 1364 (drought 192, frost 67, gray mold 399, healthy 208, overwatered 163, root rot 193, white mold 142),
  val 212, dropped 517 (test photographs) and 66 unmatched (16 photographs such as `drought_10.jpg`, `drought_23.jpg`,
  `whats-happening-to-her-...webp` and their augmented copies, which are no longer in `data/train`, so they are not in
  the 1580-image pool; dropped to be safe). Part detection with the v1 checkpoint on the 1576 train + val images: job 72335.
- **Detector split agreeing with split v1 (`cluster/make_detector_split.py` -> `rfdetr_parts/dataset_splitv1`,
  `evaluation/detector_split_v1.json`).** pHash (64-bit DCT, distance ≤ 6) matching of the 1075 annotated images to the
  1576 split v1 photographs; near-copies among annotated images kept together (none had to be moved). Test 163 images
  (matched photographs: healthy 81, frost 28, root rot 31, gray mold 12, white mold 7, drought 2, overwatered 2), 429
  annotation-only images unmatched (train). Train / val counts in the report file. Note for the paper: the annotated set
  overlaps the condition photographs mostly through healthy, frost and root-rot photos, so the detector test split is
  light on the molds and water stresses. Pre-registered: RF-DETR Small (job 72333) and Large (72334) retrained with the
  lab configuration (100 epochs, selection on the detector validation split), then `cluster/eval_parts.py` on the test
  split: per-part AP50 / AP50:95 and image-level presence precision / recall / F1, with per-part thresholds chosen by F1
  on the validation split. Prediction: mAP50:95 on the test split within ±3 points of the earlier validation-split
  values (Small 0.593, Large 0.631); AP lowest for flowers, roots and stems.
- **Report writer retrained on split v1: launched (job 72338, c7, `cluster/slurm/train_minigpt_splitv1.sbatch`,
  `train_configs/minigptv2_strawberry_splitv1.yaml`).** The lab's v2 recipe (LoRA r 64 / α 128 on q,v, visual
  projection trained, 8-bit LLaMA-2-7B-chat, 448 px, AdamW 1e-5 → 5e-6 cosine, label smoothing none, max 500 text
  tokens) with: training records = the 1364 split v1 training photographs and their augmented copies (the 617
  augmented copies missing from the cluster data were restored from the GitHub copy of `train_aug`, 31 MB); validation
  = the 212 records of split v1 validation photographs; the human turn = the application's system prompt and user text
  with the v1 detector's part lists (`cluster/detect_parts.py` on the 1576 train + val images, job 72336: 1357 of 1364
  training images had at least one part, 7 none; `cluster/build_stage2_prompts.py`), passed through unchanged by a
  new `identity_text` processor (the lab's `blip_caption` processor lowercases, strips punctuation and cuts at 50
  words: a first submission, job 72337, was cancelled after two minutes for this reason); one GPU, batch 2 ×
  accumulation 16 = effective 32, 10 epochs × 2000 iterations (half the lab's 20); checkpoint selection by validation
  loss (`minigpt4/tasks/image_text_pretrain.py` `evaluation()` patched to compute the loss over 106 validation batches
  and return exp(-loss) as the runner's metric; the lab's version returned 0 and never saved a best checkpoint, which
  is why the April writer is its last epoch).
- **Report test set (`plant_diagnostic/datasets/split_v1/report_test_set.json`):** the 163 detector test images are
  151 distinct split v1 test photographs with part masks (healthy 74, root rot 30, frost 24, gray mold 12, white mold 7,
  drought 2, overwatered 2; 1-5 annotated parts, median 3). The v1 detector's part lists for them: job 72339
  (`evaluation/parts_report_test.json`).
- **Pre-registered: detector on/off ablation of the report writer (manuscript Table S16; `cluster/minigpt_reports.py`,
  `cluster/score_reports.py`).** Each test photograph gets one report under the application's prompt with the
  detector's part lists and one under the same prompt without them (decoding as the app: temperature 0.2, top-p 0.9,
  repetition penalty 1.01, seed 0; the consistency check regenerates once with the stricter prompt). Scoring rules,
  fixed now: a report "asserts" a part when a part word (leaf/foliage, fruit/berry/calyx, stem/petiole/runner,
  flower/blossom/petal, soil/substrate/ground/mulch, root) occurs and is not negated within its sentence (the
  consistency check's negation cues plus "not visible / not seen"); H_part = share of reports asserting at least one
  part absent from the photograph's annotation (also reported counting every mention, and against the detector's
  list); condition consistency = the "Diagnosis:" line names the given condition; regeneration rate = share flagged at
  least once; length in words; bootstrap 95 % intervals over photographs; the with/without difference by exact McNemar
  on discordant photographs. Models: the retrained writer (best validation loss) and, as a labelled leaky reference,
  the April writer (`output/minigpt_strawberry/checkpoint_best.pth`, which trained on reports of most of these
  photographs). Predictions: with part lists H_part (asserted) is at most half of the value without them for the
  retrained writer; condition consistency ≥ 95 % in both arms; regeneration ≤ 15 %; the April writer's H_part is lower
  than the retrained writer's in both arms (it has seen these photographs).
- **Check: RF-DETR class ids (`cluster/check_class_ids.py`, lab checkpoint on the lab's 322-image validation split,
  detections ≥ 0.3 matched to ground-truth boxes at IoU ≥ 0.5).** Predicted id 0 → flower (229 of 247 matched), 1 →
  fruit (1592 / 1615), 2 → leaf (3964 / 4017), 3 → root (99 / 108), 4 → soil (166 / 174), 5 → stem (1153 / 1199): the
  0-based mapping of `rfdetr_parts/detect_images.py` (and of `cluster/detect_parts.py`, which fed the training
  prompts) is right; the COCO categories are 1..6, and an id 6 appears only below the 0.3 floor (a no-object slot).
  `cluster/eval_parts.py` shifts predictions by one and drops ids outside the categories; the first evaluation
  (job 72344) failed on this before the fix.
- **Report-writer training speed (job 72341, c7):** 1.7 s per iteration (batch 2) at 16.5 GB, so one epoch of 2000
  iterations takes about an hour and the 10-epoch run ends around 04:00 on 31 Oct. The April-writer reference run
  (72340) shares c3 with the Small detector training and generates about one report per 65-90 s.
- **Result, detector on the split that agrees with split v1 (jobs 72333 train / 72346 eval, `cluster/eval_parts.py`;
  `evaluation/parts_small_splitv1_{valid,test}.json`, `evaluation/parts_lab_small_on_splitv1_{valid,test}.json`).**
  RF-DETR Small retrained with the lab configuration (100 epochs, 47 min, selection on the 67-image detector
  validation split): test (163 images) mAP50:95 **0.564**, mAP50 0.716, mAP75 0.608; validation 0.499 / 0.691.
  Per-part test AP50:95 (AP50): flower 0.687 (0.844), fruit 0.702 (0.878), leaf 0.578 (0.769), root 0.288 (0.392),
  soil 0.660 (0.725), stem 0.471 (0.688): roots and stems lowest, as predicted. Image-level part presence on the test
  split at the application's thresholds, F1: flower 0.87, fruit 0.96, leaf 0.97, root 0.86, soil 0.72, stem 0.92; at
  thresholds chosen by F1 on the validation split (flower 0.60, fruit 0.25, leaf 0.20, root 0.55, soil 0.20, stem
  0.35): 0.83 / 0.95 / 0.96 / 0.71 / 0.63 / 0.93, i.e. no better than the hand-set values, because 67 validation
  images (4-25 per part) are too few to set six thresholds; keep the application's thresholds and say so.
  Prediction (within ±3 points of the earlier validation-split 0.593) held narrowly (-2.9). **Leakage check:** the
  lab's April checkpoint scored on the same test split reaches 0.681 / 0.834 (per part: flower 0.756, fruit 0.789,
  leaf 0.693, root 0.522, soil 0.716, stem 0.609), 11.7 points above the clean retrain, because the lab's random
  split put most of these photographs in its training data (they are near-copies of split v1 test photographs); the
  lab's own 322-image validation split gave it 0.598. So the earlier detector numbers were optimistic by roughly ten
  points, and 0.564 (Small) is the number to report for v1. Large: job 72334 / 72347, pending.
- **Result, RF-DETR Large on the split that agrees with split v1 (jobs 72334 train, 67 min / 72347 eval;
  `evaluation/parts_large_splitv1_{valid,test}.json`).** Test (163 images) mAP50:95 **0.593**, mAP50 0.720, mAP75
  0.641; validation 0.530 / 0.704. Per-part test AP50:95 (AP50): flower 0.734 (0.850), fruit 0.734 (0.891), leaf 0.616
  (0.782), root 0.293 (0.380), soil 0.650 (0.698), stem 0.531 (0.721). Small → Large: +2.9 points on the clean test
  split (0.564 → 0.593), in line with the +3.8 seen earlier on the lab's validation split; prediction (within ±3 of
  0.631) failed (-3.8), the validation-split figure having been inflated like the Small one. Presence F1 on test at
  the application's thresholds: flower 0.86, fruit 0.96, leaf 0.98, root 0.84, soil 0.72, stem 0.93 (chosen-on-val
  thresholds again no better). The detector numbers for the paper: Small 0.564, Large 0.593 on the clean test split.
- **April-writer reference run (72340) note:** with the application's long prompt, the April writer (trained with an
  empty instruction) sometimes echoes the user text ("Examine this image. The diagnosis ...") instead of starting with
  "Diagnosis:"; counted as it falls under the pre-registered rules (condition consistency reads the Diagnosis line).
- **Report-writer run: no epoch-end validation after all.** The lab's runner resolves `valid_splits=[]` whenever
  `evaluate` is false (training mode), whatever `val_splits` says (log: "Eval gate: evaluate=False, cfg.val_splits=['val'],
  resolved(valid_splits)=[]"), which is the mechanism behind the April writer being its last epoch. The run saves one
  checkpoint per epoch (`checkpoint_<epoch>.pth`, 685 MB each; epoch 0 train loss 0.120), so selection is done after
  training instead: `cluster/minigpt_val_loss.py` scores every epoch checkpoint on the 212 validation records (same
  prompted format as training) and copies the lowest-loss one to `checkpoint_best.pth`; `cluster/slurm/select_and_report.sbatch`
  chains that with the report generation (replaces job 72351).
- **2026-10-30, 21:30 CDT. (g) Documentation corrections in the working copy (uncommitted, for W.S. to review):**
  `docs/DETECTOR.md` (the lab checkpoint's input is 512 px, not 560; a clean-test-split section with the Small 0.564 /
  Large 0.593 / leaky lab-checkpoint 0.681 numbers, per-part AP and presence F1 from
  `evaluation/paper_data/parts_*_splitv1_test.json`; caveats and next steps updated), `docs/COMPUTE.md` (peft 0.13.2),
  `README.md` and `docs/V1_SYSTEM.md` (the knowledge base's treatment entries were "compiled from" the extension
  publications: there is no crawler in the repository; who wrote the entries, and whether a language model helped, is
  a question for W.S.). The co-authors' answer sheet carries the same status.
  Operational: the ssh master to Hamilton dropped at about 21:10 CDT before the overnight watcher could score anything;
  the Slurm chain (72340 April-writer reports, 72341 training, 72352 selection + reports) is unaffected, and
  `cluster/score_reports.py` runs once the connection is back.
- **2026-10-30, 22:00 CDT. Pre-registered: v1 vs v2 end-to-end latency and GPU memory on one RTX 5000 Ada
  (`cluster/latency_v1.py`, `latency_rfdetr.py`, `latency_siglip.py`, `latency_student.py`, `latency_summary.py`;
  `cluster/slurm/latency_v1_v2.sbatch`; written while the cluster was unreachable, to run when the connection is back).**
  Sample: 7 split v1 test photographs per condition, seeded (49), the same in every script; latency is a property of
  the system, so the sample is small, while accuracy and honesty come from all 398 test photographs as already recorded.
  v1 as deployed = demo_v5.py's path and settings (image encoding, ResNet-50 two views + gate, RF-DETR Small through the
  application's subprocess, MiniGPT-v2 report at temperature 0.2 / top-p 0.9 / repetition penalty 1.01 / up to 2000 new
  tokens, consistency check, one regeneration); also v1 with the detector in-process. v2 = fine-tuned SigLIP 2 (two
  views) + RF-DETR Large (split v1 retrain) + the distilled Qwen3-VL-8B student under vLLM (one request at a time, JSON
  schema, greedy), each stage in its own process on the same photographs, summed per photograph; batched throughput
  reported separately. Memory per process from nvidia-smi and torch peaks; the student's figure is the weights on disk
  plus the engine log, since vLLM pre-allocates. Two warm-up photographs excluded; medians and p90 reported.
  Predictions: (1) v1 as deployed is dominated by the detector subprocess (interpreter start plus model load, > 5 s per
  photograph) and the MiniGPT-v2 report (several hundred tokens at a 7B model's single-stream speed), so v1 as deployed
  takes longer per photograph than v2; (2) with the detector in-process, v1 and v2 are within a factor of two of each
  other, the report writers dominating both; (3) v2 needs more GPU memory than v1 (8B bf16 weights alone about 16 GB;
  MiniGPT-v2's 7B LLaMA-2 in 16-bit plus the ViT about 15 GB), both fitting a 32 GB card; (4) each classifier and
  detector takes well under 0.5 s per photograph in-process.
- **2026-10-31, 09:50 CDT. (i) Desk-check of the latency scripts before their first run (the ssh master was still down;
  the sentinel timed out at 09:15 CDT).** Checked against demo_v5.py, grounding/, resnet_classifier.py, the Chat class,
  backbones.py / eval_probe.py, train_backbone.py's checkpoint, annotate_vllm.py and annotation_spec.py: the gate (class
  thresholds; the margin rule is inactive because diagnose_or_none returns no p2; p1 < 0.50 -> unknown), the three
  prompts and the regeneration prompt, the decoding settings (temperature slider default 0.2, top-p 0.9 from
  answer_prepare, repetition penalty 1.01, 2000 new tokens, max_length 4000), the re-encoding before the regeneration,
  the checkpoint formats and the vLLM call all match. Fixed before any run: (1) the application does a debug forward
  pass (two views, softmax at T = 0.78, printed only) before diagnose_or_none and applies _postprocess_caption before
  the consistency check; latency_v1.py now does both, so "v1 as deployed" includes them. (2) grounding/detector.py
  activates a conda environment, which the cluster does not have (no branch of the repository has an alternative,
  although agai_env.sh exports RFDETR_PYTHON for it); the local copy now uses the interpreter named by RFDETR_PYTHON
  when the variable is set, otherwise conda as before, so the lab server's behaviour is unchanged; latency_v1.py raises
  on a failed detector call instead of timing the application's silent fall-back to the no-parts prompt; the job file
  checks RFDETR_PYTHON, the wrapper and every weight file before starting. The Hamilton copy of detector.py is to be
  compared with this one before anything is copied. (3) GPU memory: nvidia-smi's first row is not necessarily the job's
  GPU on a multi-GPU node, so each script now reports its own process's memory (compute-apps by PID on the GPU that
  CUDA_VISIBLE_DEVICES names) and the device's memory in use from torch's mem_get_info; the student reports the engine
  processes on its GPU; latency_summary.py sums the two v1 processes only when both figures exist. (4) latency_rfdetr.py
  guards a None prediction. Local checks: the helpers against a fake two-GPU nvidia-smi listing, _postprocess_caption
  against the demo's own on a sample, the seeded sampling, and latency_summary.py on mock stage files. The predictions
  of entry (h) are unchanged.
- **2026-10-31, 17:05 CDT. (k) Latency job submitted:** `cluster/slurm/latency_v1_v2.sbatch
  /projects/was4g/agai/adapters/Qwen3-VL-8B-Instruct-clsft_reports_v1` -> job 72394 on c6 (RTX 5000 Ada, 32 GB,
  driver 580.142). The paper's student sits on the classification-tuned 8B (`models/Qwen3-VL-8B-Instruct-clsft`; the
  adapter's config and job 72215's merge confirm it), so the job file now derives the base from the adapter's config
  instead of assuming the plain 8B. Pre-flight passed (RFDETR_PYTHON, the detector wrapper, every weight file). The
  Hamilton copy of `grounding/detector.py` already carried the RFDETR_PYTHON override (uncommitted there); the local copy
  was made identical to it (it also casts the paths to str). The deployed eval config loads the MiniGPT-v2 writer in
  8-bit (`low_resource: true`), so prediction (3) of entry (h) assumed 16-bit weights wrongly; the measurement reports
  what the application loads. The ssh master had come back at 16:57 CDT; entries (g)-(i) were appended then.
- **2026-10-31, 17:20 CDT. (j) Result, detector on/off ablation of the report writer (Table S16; jobs 72341 training,
  72352 selection + reports, 72340 April-writer reference; `cluster/score_reports.py`; files
  `evaluation/reports_splitv1.json`, `reports_april_writer.json`, `*_scores.json`, `minigpt_splitv1_val_loss.json`,
  copied to the repository's `evaluation/paper_data/` with the training log).**
  Training (10 epochs x 2000 iterations, batch 2 x 16 accumulation, 9 h 40 min on c7): train loss per epoch 0.120 ->
  0.064; validation loss on the 212 validation records 1.619, 1.421, 1.344, 1.305, 1.282, 1.272, 1.262, 1.257, 1.252,
  1.248, still falling at the last epoch, so the lowest-validation-loss rule keeps epoch 9 (`checkpoint_best.pth`). The
  20x gap between training and validation loss says the writer memorises its 1364 training reports (about 30 passes
  over them); the lab's schedule was kept for comparability. Reports: 151 split v1 test photographs with part masks,
  one report per condition (302), 47 s per report (238.5 min). **Retrained writer: H_part (asserted) 86.1 % (80.8-91.4)
  without the part lists vs 92.1 % (87.4-96.0) with them**; paired: with-parts only 14 / without only 5 of 151, exact
  McNemar P = 0.064. Counting every mention: 90.7 vs 92.1 %. A part the prompt declared NOT VISIBLE is named in 91.4 %
  of the with-parts reports. Condition consistency 100.0 vs 96.0 % (0 / 6 discordant, P = 0.031; the 6 reports omit the
  "Diagnosis:" line and their opening sentences describe the labelled condition, so this is format, not a wrong
  diagnosis). Regenerated at least once 15.2 vs 11.9 %; still flagged after regeneration 8 vs 6, all healthy
  photographs whose reports advise watching for "gray fuzz" ("inspect ... for any gray fuzz"; 15 of the 16 asserted
  mentions are such advice, the other a negated list that the five-word negation window misses), which the deployed
  check reads as an assertion: a limitation of the check, not a hallucination. Mean length 203 vs 200 words. April writer
  (leaky reference): H_part 85.4 vs 88.1 % (10 / 6, P = 0.45); it writes a "Diagnosis:" line in only 54 and 5 of 151
  reports, so its consistency figures (39.7 / 10.6 %) measure format; regenerated 17.9 / 6.6 %; length 186 / 174 words.
  Predictions vs results: (P1) "with part lists H_part at most half of the value without" **not supported**: the lists
  did not reduce part hallucination and the point estimate moved the other way; (P2) consistency >= 95 % in both arms:
  supported (100 / 96 %); (P3) regeneration <= 15 %: 15.2 and 11.9 %, borderline; (P4) the April writer's H_part below
  the retrained writer's in both arms: direction only (0.7 and 4.0 points, inside the intervals).
  Exploratory, post hoc (`cluster/score_reports_exploratory.py`): soil accounts for most false assertions, 113 / 127
  (without) and 127 / 127 (with) of the 127 photographs without annotated soil, nearly all in care advice (irrigation
  depth, "clean straw mulch"); "mulch" occurs in 75 vs 129 reports, so the NOT VISIBLE instruction for soil was followed
  by more mulch advice, not less (the soil regex counts mulch and substrate by the pre-registered rule). Without soil,
  H_part is 33.8 vs 39.1 % (17 / 25, P = 0.28); per part (false assertions / photographs without the part) leaf 11/21 vs
  11/21, fruit 2/54 vs 12/54, stem 10/41 vs 11/41, flower 7/124 vs 10/124, root 26/127 vs 20/127. The detector's lists
  are mostly right: they miss an annotated part in 9 and add an unannotated one in 18 of 151 photographs. A "Visible
  cues" section cannot be isolated (the heading appears in 2 and 0 reports), so a restriction to visual claims was not
  computable. Reading: the v1 writer (LLaMA-2 7B with a LoRA) does not obey the prompt's part constraints; the detector
  lists carry no measurable grounding benefit for this writer, and the April writer, which trained on reports of most
  of these photographs, does no better. For the manuscript: Table S16 takes the retrained writer's four rows with the
  paired tests and a leaky-reference note; 3.9 and the discussion sentence "no claim about reduced part hallucination
  can be made yet" become "the ablation found no reduction".
- **2026-10-31, 18:50 CDT. (l) Result, v1 vs v2 end-to-end latency and GPU memory (pre-registered in entry (h),
  2026-10-30 22:00; job 72394 on c6, RTX 5000 Ada 32 GB, driver 580.142, 1 h 30 min including staging and merging the
  student; `cluster/latency_v1.py`, `latency_rfdetr.py`, `latency_siglip.py`, `latency_student.py`,
  `latency_summary.py`; files `evaluation/latency_v1.json`, `latency_rfdetr.json`, `latency_siglip2.json`,
  `latency_student.json`, `evaluation/paper_data/latency_v1_v2.json`, copied to the repository's
  `evaluation/paper_data/`).**
  Protocol as pre-registered: 49 seeded split v1 test photographs (7 per condition, seed 0, the same in every script),
  two warm-up photographs before timing, each stage in its own process, CUDA synchronised; the one deviation is the one
  recorded in (k): the writer loads in 8-bit as the application does, not in 16-bit as prediction (3) assumed. Medians
  (p90) per photograph: **v1 as deployed 62.2 s (75.2)** = image encoding 0.06 + ResNet-50 two views and gate 0.03 +
  RF-DETR Small through the application's subprocess 15.9 (16.3) + MiniGPT-v2 report 46.2 (58.8; median 337 tokens,
  7.3 tokens/s over the run), no gate rejection, no consistency flag, no regeneration; **v1 with the detector
  in-process 46.3 s (59.0)** (RF-DETR Small in-process 0.016 s); **v2 21.7 s (24.9)** = SigLIP 2 two views 0.035 +
  RF-DETR Large 0.017 + student under vLLM, one request at a time, 21.6 (24.8; median 725 tokens, 33.5 tokens/s);
  batched, the student wrote the 49 reports in 40.9 s (1.20 photographs/s). The subprocess costs 15.9 s because
  `grounding/detector.py` starts an interpreter and loads the model for every call; the forward pass itself is 0.016 s.
  GPU memory (nvidia-smi per process; torch peak reserved): v1 writer + ResNet process 11828 MB (12.0 GB), RF-DETR Small
  process 1040 MB (0.70 GB), 12.9 GB together; v2 SigLIP 2 process 3116 MB (2.87 GB), RF-DETR Large process 1098 MB
  (0.76 GB), student weights 17.5 GB on disk (bf16), vLLM engine processes 26184 MB at gpu_memory_utilization 0.85
  (max_model_len 8192), 30.4 GB summed as configured. Load times: MiniGPT-v2 + ResNet 25 s, RF-DETR Small 0.5 s, Large
  2.2 s, SigLIP 2 101 s, vLLM engine 169 s (plus 213 s staging the base to /scratch and the LoRA merge).
  Predictions vs results: (1) v1 as deployed dominated by the detector subprocess (> 5 s) and the report, and slower
  than v2: **supported** (15.9 + 46.2 of 62.2 s; 2.9x v2). (2) with the detector in-process, v1 and v2 within a factor of
  two, the writers dominating both: **not supported on the factor** (46.3 vs 21.7 s, 2.1x); the writers do dominate
  (99.8 % and 99.7 % of the totals); the student writes twice the tokens at 4.6x the token rate (bf16 under vLLM against
  8-bit bitsandbytes in Hugging Face generate). (3) v2 needs more GPU memory than v1, both fitting a 32 GB card:
  **supported in direction**; the v1 figure is 12.9 GB rather than the ~16 GB estimated from 16-bit weights (8-bit
  load), and the v2 figure depends on the engine's reservation (26.2 GB at 0.85; the weights alone 17.5 GB); the stages
  ran one at a time, so co-residence of v2's three processes on one card was not tested. (4) each classifier and
  detector well under 0.5 s in-process: **supported** (0.02-0.06 s).
  Side counts, not pre-registered: the SigLIP 2 classifier was right on 47 of 49 (`images.jpeg`, overwatering, read as
  drought; `overwatered_47.jpg` read as frost injury); all 49 student answers were valid JSON; verdicts clearly supports
  3, partially supports 35, does not support 11, so 38 given labels accepted: 10 of the 47 correct labels rejected and 1
  of the 2 wrong labels rejected, in line with the test-set honesty figures (the sample is for timing, not honesty). The
  manuscript's 2.13 lab-server timings (report median 25.3 s over 16 reports, classifier 9.1 ms, detector forward
  18.4 ms) come from the lab GPU; the writer's 46 s here is the 8-bit model at 7.3 tokens/s on the Ada card, so the two
  need not agree. For the manuscript: 3.9 gets the latency and memory paragraph and table; the abstract/conclusion
  sentence reads 22 s per photograph for v2 against 62 s for v1 (46 s with v1's detector kept in memory);
  `docs/DETECTOR.md` and `docs/V1_SYSTEM.md` note the subprocess cost. Also placed in the repository: the four
  environment freezes of 2026-10-30 as `cluster/infra/requirements/<env>.freeze.2026-10-30.txt` (named in
  `docs/COMPUTE.md`).
- **2026-10-31, 19:10 CDT. (m) Provenance answers from W.S. (chat, 2026-10-31) to the manuscript's bracketed
  questions.** (1) 2.13, lab server GPU: the laboratory's pre-cluster server had two NVIDIA GeForce RTX 3090 (24 GB
  each), so "two GPUs with 24 GB of memory each" stands and the model is the RTX 3090. (2) 2.2, April synthetic
  prompts: the text prompts used to generate the April synthetic images were not kept; the manuscript states that the
  prompts were not retained. (3) 2.11, knowledge base: the entries of `knowledge_graph/disease_knowledge_base.json`
  are a corpus of information pulled from university extension publications, as the README's "compiled from" wording
  (entry (g)) says; no crawler existed. Whether a language model helped with the wording is not yet stated. The
  co-authors' answer sheet carries the same answers.
- **2026-10-31, 19:55 CDT. (n) Further answers from W.S. (chat, 2026-10-31).** (1) 2.11: a language model,
  Anthropic's Claude (the current model at the time, autumn 2025; the version was not recorded), helped adjust the
  wording of the knowledge-base entries; the content came from the university extension publications (entry (m)); the
  manuscript's generative-AI statement should say so. (2) 2.5: W.S. does not know the 93 generated files of 27 April
  2026 (UUID-named PNGs, drought 50 / frost 20 / gray mold 23, OpenAI C2PA record, in the lab machine's training
  folders); the question goes to G.M. and H.Z., since the files appeared on the day the lab's synth / synthOOD ResNet
  checkpoints begin; no reported result uses them. (3) Author contributions: W.S. says the manuscript's line covers it
  for the most part and points to the commits; the GitHub history agrees. His commits: 2025-09-29 "MiniGPT-v2 + ResNet
  strawberry diagnostician", 2025-10-04 "Complete implementation with ResNet-50 + MiniGPT-v2", 2026-04-12 "Add RF-DETR
  grounding pipeline, update knowledge base", 2026-04-25 paper refresh, 2026-10-21/22 revival (leak-free benchmark,
  experiment code, lab notebook, documentation). G.M.'s commits: 2025-10-06 demo security / reliability / UX,
  2025-10-17 demo_v5 with webcam, tests folder, knowledge-graph folder, 2025-10-21 system improvements and test suite,
  2026-01-21 YOLOv8 detection project. Entries (m) and (n) are in the local notebook only until the ssh master is back
  (it closed at about 19:10 CDT).
- **2026-10-31, 20:20 CDT. (o) The 93 generated files of 27 April 2026 (manuscript 2.5), from W.S. (chat,
  2026-10-31); supersedes item (2) of entry (n).** Shown the per-condition counts of Fig. S2(a) (real 324 / 65 / 115 /
  73 / 131 / 650 / 222 for healthy, drought, frost, overwatering, root rot, gray mold, white mold; synthetic in the 601
  named files 97 / 76 / 52 / 154 / 30 / 99 / 93; the 93 add drought 50, frost 20, gray mold 23), W.S. recalls
  generating them as further clear and ambiguous examples of drought stress, frost injury and gray mold, the
  conditions the April classifier was confusing, to try to fix those confusions; the lab's "synth" and "synthOOD"
  ResNet checkpoints (v5 to v12) begin the same day. Recalled, not documented. Drought and frost are among the three
  smallest real classes and gray mold the largest, so the batch targeted confusions, not class size. The files enter
  no reported result (the study's synthetic arm uses the 601 named files). Local notebook only until the ssh master is
  back.
- **2026-10-31, 20:45 CDT. (p) Committed and pushed to the lab repository (greatroboticslab/AGAI, main) with W.S.'s
  authorization of 2026-10-31 ("u have final commit push perm"); no Claude co-author trailer, per W.S.** Six commits on
  top of G.M.'s 3b0a31f (figure scripts), fast-forwarded first: ee006d9 report writer retrained on split v1 (training
  fixes, matched prompts, Table S16 scoring, result files; `.gitattributes` keeps `evaluation/paper_data/*.json` out of
  LFS); 53c59d5 detector test split, RF-DETR Small / Large retrains, per-part AP; 340c148 classifier calibration, gate
  metrics, mirrored-view ablation; 6f055df latency and GPU memory scripts, job file, RFDETR_PYTHON in
  `grounding/detector.py`, result files; 3a493fd consistency-check fixes (negation-aware matching, retry without parts,
  16-phrase list); 3916082 docs (`docs/MANUSCRIPT_QA.md`, DETECTOR / V1_SYSTEM / COMPUTE corrections, README,
  environment freezes, this notebook through entry (o)). Pre-commit checks: all 24 changed or new Python files compile;
  no secrets or machine-specific paths in the changes. This entry is committed separately. The Hamilton working copy
  (branch labbox, flat `cluster/` layout) is unchanged; its two notebook copies carry the same entries.
- **2026-11-04, 13:45 CST. (q) Pre-registered: claim-level faithfulness of v1 against v2 reports (manuscript 3.9),
  blind human ratings.** Sample: the 49 seeded split v1 test photographs of the latency measurement (entry (h); 7 per
  condition, seed 0) and the reports both systems wrote for them as deployed (entry (l)): v1 = the `report` field of
  `evaluation/paper_data/latency_v1.json` (retrained writer, ResNet-50 label, detector part lists, consistency check
  applied), v2 = the `report` field of `latency_student.json` (distilled 8B student, SigLIP 2 label, vLLM, JSON
  schema). Chosen over the 38 bake-off photographs because both systems' reports already exist for these and the
  sample is representative (7 per condition) rather than hard; the bake-off set served the captioner comparison, not
  the deployed explainers. Unit: one claim = one sentence of the report (split at sentence punctuation and line
  breaks; list numbering and the section labels "Visible cues:" / "Recommendation:" removed; v1's "Diagnosis: <label>"
  line excluded because it restates the label; fragments under 20 characters dropped): 696 claims, v1 445 (9.1 per
  report), v2 251 (5.1 per report). Blinding: per photograph the claims of both reports are pooled and shuffled with
  a seeded RNG (`faith-v1v2:<photo index>`); the rater sees the photograph, the dataset label on demand, and each
  claim's text; the system key (`key.json`) stays outside the page until the ratings are complete. Limitation:
  writing style can reveal the system (v1's symptom-and-treatment prose, v2's hedged sentences); recorded, not
  fixable. Ratings per claim: Supported (the photograph shows it), Not supported (the photograph shows otherwise, or
  the thing is not there), Unsure (cannot tell from the photograph), Not a visual claim (advice, generic statements,
  diagnostic judgement). Raters: W.S. all 49 photographs; G.M. at least 10 photographs for agreement (Cohen's kappa on
  Supported / Not supported) if available by Saturday 2026-11-07. Metrics: (M1) unsupported-claim rate per system =
  Not supported / (Supported + Not supported), bootstrap 95 % intervals over photographs; (M2) share of reports with
  at least one unsupported claim; (M3) visual and non-visual claims per report; (M4, exploratory) unsupported claims by
  plant part and by whether the part was in the detector's list. Tests, paired over photographs: Wilcoxon signed-rank
  on the per-photograph difference in unsupported-claim rate, exact McNemar on M2. Predictions: (P1) v2's
  unsupported-claim rate is at most half of v1's; (P2) v1 makes more visual claims per report than v2 (it lists
  symptoms for every part in its prompt) and more non-visual ones (treatment advice); (P3) v1's unsupported claims
  concentrate on parts the photograph does not show (soil, roots, flowers), as in Table S16; (P4) v2 reports with a
  "does not support" verdict have no more unsupported claims than its other reports (the rejections are grounded, not
  evasive). The scoring script is written before any rating is read. Storage: the rating page's database, one
  document per rater and claim; claims file `faith/claims.json` and key `faith/key.json` in the session scratchpad,
  to be added to `evaluation/paper_data/` with the ratings.
- **2026-11-04, 14:25 CST. (r) Correction: the report-generation script was never lost.**
  `plant_diagnostic/scripts/gptscripts/json_generation/generate_json_openai.py` (committed 2025-09-29, updated 2025-10-04
  and 2025-10-17) is the stage-2 generator: OpenAI o3 (default of `OPENAI_VISION_MODEL`), one image per record, label
  = folder name, system prompt fixing the label, visible-only content, one paragraph of 6-9 sentences starting
  "Diagnosis: <label>." with 3-6 visual clues, a cause, 2-3 actionable steps with measurements and 2-3 monitoring
  checks for the next 7 days, strict JSON {"report": ...}; a templated fallback per label when the call fails; the
  "Diagnosis: <label>" prefix enforced; records {image, conversations [human "<Img><ImageHere></Img>", assistant
  report]}, the dataset's exact format. The empty `generate_stage2_jsonl.py` is a stub from the 2025-06-28 "clean
  snapshot" commit. Checks on `plant_diagnostic/datasets/stage2_train_7class.fixed.jsonl` (2159 records, identical to
  the `_fixed.json`): all begin "Diagnosis:", all distinct (fallbacks are identical per label, so none is in the set),
  mean 207 words, 823 mention the next 7 days. The answer sheet's statement of 2026-10-30 that the generator was lost
  (made from the empty file alone) is withdrawn; the sheet, `docs/MANUSCRIPT_QA.md` and the manuscript's 2.7 are
  corrected, and H.Z.'s choice of 2026-11-04 to reconstruct the script is moot. Also recorded from H.Z. the same day:
  the SigLIP 2 recipe is the chosen classifier of Table S7. Entries (q) and (r) stay local until the ssh master is
  back.
- **2026-11-04, 14:50 CST. (s) Results: honesty table from the verified verdicts; mirror-aware near-duplicate audit
  of split v1 and its sensitivity analysis.** (1) `cluster/score_honesty.py` over the verified student run (job 72215,
  `annotations/recheck_Qwen3-VL-8B-Instruct-clsft-reports_v1/test_both.jsonl`) and the other test runs: the verified
  student accepts the true label 322/398 (80.9 %), rejects the wrong one 340/398 (85.4 %), balanced 83.2 %,
  differential 389/398 (97.7 %); paired exact McNemar vs its first (c5) run 15/10, P = 0.42; vs the student from the
  base 8B 43/23, P = 0.019 (first run 42/27, P = 0.091); vs Qwen3-VL-235B 63/43, P = 0.064 (62/47, P = 0.18); vs
  Qwen3.8-27B 91/59, P = 0.011 (92/65, P = 0.038); vs the 32B teacher 72/28, P = 1.3e-5; vs zero-shot 8B 112/33,
  P = 3e-11; vs GLM-4.6V 206/54, P = 4.3e-22. Output `evaluation/paper_data/honesty_verified_comparisons.txt`; the
  manuscript's Table S12, the text of 3.6 and `docs/VLM_HONESTY.md` now carry the verified run. (2) Mirror-aware
  audit (`cluster/benchmark/mirror_dupes.py`, the manuscript's audit rule: pHash within 8 and dHash within 10 bits on
  the image or its mirror after EXIF orientation; `evaluation/paper_data/mirror_dupes_v1.json`): 74 near-duplicate
  pairs among the 1576 split v1 images (18 mirrored); 14 pairs cross the sets (12 mirrored, all gray mold); 10 test
  images (9 gray mold, 1 drought) have a copy in training or validation and 3 validation images have one in training,
  which reproduces G.M.'s count of 10. (3) Sensitivity, post hoc (`cluster/benchmark/sensitivity_mirror.py`,
  `evaluation/paper_data/sensitivity_mirror_v1.txt`): dropping the 10 test images lowers accuracy by 0.1 to 0.3
  points for every one of the 22 result files rescored (ResNet-50 seeds 91.7 / 91.5 / 90.2 / 88.9 / 88.4 -> 91.5 /
  91.2 / 89.9 / 88.7 / 88.1; SigLIP 2 recipe 95.5 / 96.5 / 97.0 -> 95.4 / 96.4 / 96.9; EVA-02-L, PE-Core-G, the two
  probes, the 8B and 32B fine-tunes, zero-shot 8B 77.1 -> 76.8; ensembles E2 97.2 -> 97.2 and X3 97.7 -> 97.7) and
  macro accuracy by at most 0.5: the affected images are mostly correctly classified gray mold photographs, so no
  comparison in the paper changes. A full regrouping and retraining is not needed for the conclusions; the manuscript's
  regrouping note is answered with this result.
- **2026-11-04, 14:55 CST. (t) Pre-registered: ResNet-50 at 512 pixels, the resolution control for the backbone
  comparison (manuscript 3.3).** The lab's own script and recipe (`cluster/slurm/train_resnet_split.sbatch` ->
  `train_resnet_straw5.py`, split v1 train/val folders, defaults) with `--img_size 512` (random 90-100 % crop resized
  to 512; validation resize to 589 and center crop 512), seeds 42, 1, 2, 3 and 4, one RTX 5000 Ada each; to be scored on
  the 398 split v1 test photographs with the deployed two-view path at 512 (short side 589, center crop 512) and the
  checkpoint's temperature (`cluster/classifier/resnet_scores.py` gains an image-size option before scoring).
  Comparison: mean +- s.d. over five seeds against the 256-pixel ResNet-50 (90.2 +- 1.5 %) and the fine-tuned SigLIP 2
  (95.6 +- 1.3 %) and EVA-02-L (95.7 +- 1.1 %), with paired exact McNemar tests of the best seeds. Predictions: (P1)
  the 512-pixel ResNet-50 gains less than half of the 5.4-point backbone gap (mean below 92.9 %), so most of the gain
  stays with the backbone; (P2) its macro accuracy gains more than its accuracy, because the small classes are the
  large phone photographs; (P3) the mirror-sensitivity result holds for it too. Jobs 80917 80918 80919 80920 80921; checkpoints
  `/projects/was4g/agai/models/resnet50_512_s<seed>.pth`, logs `agai-resnet-train-<job>.out`.
