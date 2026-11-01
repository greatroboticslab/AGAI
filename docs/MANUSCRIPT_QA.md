# Manuscript Q&A: the bracketed notes of the 24 October build

Answers to every bracketed note addressed to William in the journal manuscript *Leak-free re-evaluation of a strawberry
plant health AI system* (24 October 2026 build, 31 pages), with paste-ready text, the evidence behind each answer, and
the notes that remain for Dr. Zhang and Gus. Compiled by W.S. on 30 and 31 October 2026 from the repository, the
cluster runs recorded in the lab notebook ([`cluster/EXPERIMENT_LOG.md`](../cluster/EXPERIMENT_LOG.md), entries dated
30 and 31 October) and the result files in [`evaluation/paper_data/`](../evaluation/paper_data/). Status as of the
evening of 31 October 2026.

## How to use this file

For a person: find the manuscript location in the checklist, read the item, paste the text from its block.

For Claude or another assistant applying the answers to the LaTeX source:

1. Work through the checklist in order. Each row names the manuscript section, quotes the bracketed note and points to
   the item that answers it.
2. Replace a bracketed note with the matching "Text for ..." block where one is given; otherwise edit the sentence so
   that it states the fact in the item. Keep the manuscript's own style: it writes percentages with the % sign; the
   paste texts spell out "percent" and should be converted on paste.
3. Do not change any number. Numbers marked exploratory or post hoc stay labelled as such. Intervals are 95 percent
   bootstrap intervals over photographs unless stated otherwise.
4. Leave the notes marked "Open" in place; they belong to Dr. Zhang or Gus.
5. Tables: Table S2 (classifier column), Table S7 and Fig. S9 (calibration and the gate), Table S9 (new detector
   rows), Table S16 (grounding ablation) and the new latency table of 3.9 take the values given in the items; the
   result files named in each item hold the per-image data.

## Checklist: the 34 bracketed notes

26 of the 34 notes are answered or resolved below; 8 are open and belong to Dr. Zhang or Gus.

| # | Manuscript location | Bracketed note | Status | Answer |
|---|---|---|---|---|
| 1 | 1 Introduction | Gus needs to fill in the 2024 world production from FAOSTAT, with the access date | Open: Gus | Not a William item; Gus fills in the FAOSTAT figure |
| 2 | 2.1 System overview | William needs to answer this: will the report writer be retrained without the split v1 test images? The April report writer saw reports for 228 of the 231 April holdout images. | Done | 2.1 Will the report writer be retrained without the split v1 test images? |
| 3 | 2.1 System overview | William needs to supply the v1 versus v2 end-to-end results (accuracy, honesty, claim faithfulness, latency and memory) when roadmap item B2 is done | Numbers in; claim faithfulness next week | 3.9, 2.1, abstract and conclusion: v1 versus v2 end to end |
| 4 | 2.2 Data collection | Dr. Zhang needs to decide: most white mold images show harvested fruit with white mycelium, which may be Rhizopus or Mucor rot rather than Sclerotinia. Keep this definition or rename the class? | Open: Dr. Zhang | Decision on the white mold class definition |
| 5 | 2.2 Data collection | Dr. Zhang needs to answer this: was any sample of labels checked by an extension specialist or plant pathologist? If not, it goes in the limitations. | Open: Dr. Zhang | Whether a specialist checked a sample of labels; otherwise a limitation |
| 6 | 2.2 Data collection | William needs to answer this: what prompt was used for the April synthetic images? The prompts are not stored in the repository. | Answered: the prompts were not kept | 2.2 Prompts of the April synthetic images |
| 7 | 2.3 Data processing pipeline | William needs to check this subsection; it was drafted from his docs/BENCHMARK.md. | Checked | 2.3 Data processing pipeline |
| 8 | 2.4 Condition classifier | William needs to confirm every setting in this paragraph and in Supplementary Table S2 against the final cluster runs. | Checked, corrections given | 2.4 Classifier settings, and the resize / centre-crop remark |
| 9 | 2.4 Condition classifier | William needs to fix this before the final runs: the deployed app resizes to 294 pixels and center-crops to 256, while training squashes to 256. | Resolved: text fix, no code change | 2.4 Classifier settings, point 2 (validation and deployment input) |
| 10 | 2.4 Condition classifier | William needs to confirm these were set on validation images in the final runs | Answered: no, set by hand; validation-based alternative supplied | 2.4 Acceptance-gate thresholds; 3.3 |
| 11 | 2.4 Condition classifier | William needs to answer this: how were these thresholds chosen? | Answered: by hand in October 2025 | 2.4 Acceptance-gate thresholds |
| 12 | 2.5 Synthetic data and degradation | William needs to answer this: what are the 93 further files in train_aug with random names, which carry the same OpenAI C2PA record? Data collection counts 694 synthetic images. | Explained; origin recalled by William | 2.5 The 93 further generated files with random names |
| 13 | 2.6 Plant-part detector | William needs to confirm this: the checkpoint metadata records 512 pixels, while his docs/DETECTOR.md lists the lab checkpoint at 560 | Confirmed: 512 pixels | 2.6 Detector input resolution: 512 or 560 |
| 14 | 2.6 Plant-part detector | William needs to add a detector test split that agrees with split v1. | Done | 2.6 / 3.5 A detector test split that agrees with split v1 |
| 15 | 2.6 Plant-part detector | William needs to answer this: how were these thresholds chosen? A sweep script exists (eval_thresholds.py), but no output was saved. | Answered: hand-set after a sweep; sweep rerun and saved | 2.6 Part-presence thresholds |
| 16 | 2.7 Grounded report generation | William needs to fix this: the script that generated this file is empty in every commit, so it has to be recovered or rewritten before the procedure can be published. | Facts given; option (a) or (b) is Dr. Zhang's decision | 2.7 The script that generated the instruction reports |
| 17 | 2.7 Grounded report generation | William needs to fix these in the final run: (1) no validation was run, so the kept checkpoint is simply the last epoch; (2) the training instruction q was empty, while at inference the model receives a long system prompt, so training and inference prompts should be made to match. Also, 228 of the 231 current holdout images appear in this instruction data, so the report writer must be retrained without the test images before reports are evaluated. | Done: retrained with validation selection, matched prompt, no test images | 2.7 Report-writer training; 2.1 |
| 18 | 2.7 Grounded report generation | William needs to fix this before the final evaluation: the check is a plain substring match with no negation handling, so a sentence such as “no gray fuzz is present” is flagged; and the retry path fails when no parts were detected (demo_v5.py, lines 1020 to 1021). William also needs to decide whether brown roots and botrytis stay in the hallmark list (knowledge_graph/rag_retriever.py), since they cause the false flags described above. | Done: fixes implemented, awaiting review before commit | 2.7 Consistency check: negation, retry path, hallmark list |
| 19 | 2.9 Claim-level faithfulness and | William needs to say who rated the claims; a second rater is planned for inter-rater agreement. | Answered | 2.9 Who rated the claims |
| 20 | 2.11 Knowledge base and follow-up | William needs to answer this: who compiled the knowledge base, and how? The README says the treatments were web-crawled and curated, but no crawler is in the repository. | Confirmed | 2.11 Who compiled the knowledge base, and how |
| 21 | 2.12 Evaluation protocol and statistical | Dr. Zhang needs to decide whether growers or an extension specialist will also rate full reports. | Open: Dr. Zhang | Whether growers or an extension specialist rate full reports |
| 22 | 2.13 Implementation | William needs to answer this: which GPU model is in the lab server? The old README says NVIDIA RTX 3090. | Answered: two NVIDIA RTX 3090, 24 GB each | 2.13 Lab server GPU |
| 23 | 2.13 Implementation | William needs to list the software versions used for the re-evaluation. | Done | 2.13 Software versions of the re-evaluation |
| 24 | 2.14 Plant material | Dr. Zhang needs to confirm this wording; Scientific Reports asks every study on plants to state that it complies with institutional, national and international guidelines. | Open: Dr. Zhang | Plant-material wording |
| 25 | 3.3 Condition classification | William needs to supply the per-image probabilities of the chosen classifier on the split v1 validation and test images. ECE before and after temperature scaling, AURC, the coverage and accuracy of the acceptance gate, per-class precision, recall and F1, and the confusion matrix follow from them (Supplementary Table S7 and Supplementary Fig. S9). | Done | 3.3 Per-image probabilities, ECE, AURC and the gate (Table S7, Fig. S9) |
| 26 | 3.5 Plant-part detection | William needs to supply per-part AP and results on a detector test split that agrees with split v1. | Done | 2.6 / 3.5 A detector test split that agrees with split v1 |
| 27 | 3.9 End-to-end evaluation | William needs to supply the end-to-end comparison of v1 and v2 on the split v1 test images once v2 is assembled (roadmap B1 and B2): accuracy and macro accuracy, label-swap honesty, claim-level faithfulness, latency and GPU memory. The planned detector on/off ablation (roadmap B3) gives the part-hallucination rate of equation (11) with and without the part lists (Supplementary Table S16). | Numbers in (accuracy, honesty, latency, memory, Table S16); claim faithfulness next week | 3.9, 2.1, abstract and conclusion: v1 versus v2 end to end; 2.7 Report-writer training (Table S16) |
| 28 | 4 Discussion | William needs to say whether the mirrored view was ablated on split v1 | Done | 4 Was the mirrored view ablated on split v1? |
| 29 | 4 Discussion | Gus and William need to check every row of Supplementary Table S17 against the papers themselves before submission. | Checked; Gus to confirm the Zhang and Wu author list on the MDPI page | 4 Supplementary Table S17 |
| 30 | 4 Discussion | Dr. Zhang needs to confirm the rover plan. | Open: Dr. Zhang | Rover plan |
| 31 | 5 Conclusion | Dr. Zhang needs to decide what can be shared. Images taken from the web cannot be redistributed, but the source URLs, labels, plant-part annotations, split files and code can. | Open: Dr. Zhang | What can be shared |
| 32 | Funding | Dr. Zhang needs to answer this: funding sources, grant numbers and computing resources. | Open: Dr. Zhang | Funding sources, grant numbers, computing resources |
| 33 | Author contributions | William and Dr. Zhang need to check their own lines; William may rewrite his in his own words. | W.S. line stands, checked against the commits; Dr. Zhang's own line remains for Dr. Zhang to check | Author contributions, W.S. line |
| 34 | Supplementary Table S2 (training settings of the three v1 models) | William needs to update this table with the final cluster runs. | Covered by items 2.4 (classifier column), 2.6 (512 pixels) and 2.7 (report-writer schedule) | 2.4 Classifier settings; 2.6 Detector input resolution; 2.7 Report-writer training |

**Which build.** Two PDFs were in circulation: the 22 October build (*A calibrated classifier and a plant-part detector constrain vision-language reports*, 22 pages, v1 only) and the 24 October build (*Leak-free re-evaluation of a strawberry plant health AI system*, 31 pages), which adds the re-evaluation and is the later one. This document follows the 24 October build. The items the two builds share are answered the same way; the items unique to the earlier build (random-split comparison, report-writer numbers in Section 3) are covered under the plan at the end.

## Status at a glance

Status chips are given in square brackets. Every item is answered below in manuscript order.

| Item (manuscript location) | Status | When |
|---|---|---|
| 2.3 Data processing pipeline check | [Done] | below |
| 2.4 Classifier settings and Table S2, ResNet column | [Done] | below |
| 2.4 Resize / centre-crop remark | [Done] | text fix, no code change |
| 2.4 Acceptance-gate thresholds: origin | [Done] | below |
| 2.4 Thresholds chosen on validation; 3.3 per-image probabilities, ECE, AURC, gate (Table S7, Fig. S9) | [Done] | numbers below; files in the repository |
| 4 Mirrored-view (test-time augmentation) ablation | [Done] | below |
| 2.6 Detector input resolution 512 vs 560 | [Done: 512] | below |
| 2.6 Part thresholds: origin and a validation-based choice | [Done] | below |
| 2.6 / 3.5 Detector test split agreeing with split v1, per-part AP | [Done] | below |
| 2.7 Lost report-generation script | [Facts] [Decision] | below |
| 2.7 Report writer retrained without test images; S16 grounding ablation | [Done] | below: the part lists did not reduce part hallucination |
| 2.7 Consistency-check fixes (negation, retry path, hallmark list) | [Done] | committed 31 October |
| 2.9 Who rated the claims | [Done] | below |
| 2.11 Who compiled the knowledge base | [Confirmed] | below; a Claude model helped with the wording |
| 2.13 Lab server GPU | [Two RTX 3090] | below |
| 2.13 Software versions of the re-evaluation | [Done] | below |
| 2.2 / 2.5 April synthetic prompts; the 93 extra generated files | [93 files explained] [Prompts not kept] [Origin of the 93 recalled] | below |
| 2.1 / 3.9 / abstract / conclusion: v1 versus v2 end to end | [Numbers in] [Faithfulness next week] | accuracy, honesty, latency and memory below; claim faithfulness needs the blind ratings |
| 4 Supplementary Table S17 checked against the papers | [Done] | below |
| Author contributions, W.S. line | [Checked against the commits] | below |

## Answers: Methods (Section 2)

### 2.1 Will the report writer be retrained without the split v1 test images?

Status: Done.

Yes. The v1 report writer is retrained on the existing o3 reports restricted to photographs in the training portion of split v1. Each of the 2159 instruction records points at a `train_aug` file, which is either an original photograph or an offline augmented copy named `<original>_augNNN`; a record is kept only when its source photograph is a split v1 training image (near-duplicate groups included), so no validation or test photograph, nor a copy of one, reaches the writer. The reports of the split v1 validation photographs select the checkpoint (lowest validation loss), and the training instruction is the inference system prompt itself, built per training image from the parts the detector finds. The retrained writer is then scored on the split v1 test photographs that carry part masks. Data step done: of the 2159 records, 1364 (training photographs and their augmented copies) train the writer, 212 (validation photographs) select the checkpoint, 517 (test photographs) are dropped, and 66 records of 16 photographs that are no longer in the condition collection are dropped as well. Part detection for the 1576 training and validation images is done and the training prompts carry the detector's part lists. Training ran on the night of 30 to 31 October (ten epochs of 2000 iterations, effective batch 32, one RTX 5000 Ada, 9 h 40 min). The validation loss fell every epoch (1.619 to 1.248) and was still falling at the tenth, so the tenth-epoch checkpoint is the one kept; the training loss (0.120 to 0.064) is twenty times lower, so the writer largely memorises its 1364 training reports, as the lab's schedule (about thirty passes over them) implies.

**Text for 2.1**

```text
For this study the report writer was retrained on the records whose source photograph lies in the training portion of split v1 (1364 records, the originals and augmented copies of the split v1 training photographs), with the 212 records of the validation photographs used to select the checkpoint (lowest validation loss, reached at the last of ten epochs) and with the training instruction equal to the system prompt used at inference.
```

### 2.2 Prompts of the April synthetic images

Status: Answered: not kept.

> Manuscript note: "What prompt was used for the April synthetic images? The prompts are not stored in the repository."

Facts to hand: the GitHub copy of `train_aug` holds no generated images at all; the 601 `synth_*GPT*` files exist only in the lab machine's data and in the cluster copy made from it. The prompts of the April set were not saved with the images, and William confirmed on 31 October that no copy exists, so they cannot be recovered. The prompts of the designed October set are all kept (248 prompts, `synth/prompt_index/batches_v1 to v3b.json` on the cluster) and will be added to the repository with the data-availability files.

**Text for 2.2**

```text
The prompts of the April set were not kept. The prompts of the designed set are listed in the repository.
```

### 2.5 The 93 further generated files with random names

Status: Explained; Origin: William.

Settled from the cluster copy of the lab machine's data: the 93 are PNG files with random UUID names (2.9 to 3.8 MB each, the form ChatGPT's web downloads take), all dated 27 April 2026, in drought (50), frost (20) and gray mold (23), and they carry the OpenAI C2PA record. They were added two days after the 601-image set, on the day the lab's "synth" and "synthOOD" ResNet checkpoints (v5 to v12) begin, so they belong to those later experiments. The re-evaluation used only the 601 files named `synth_*GPT*` (602 before one exact duplicate was dropped), so the 93 enter no reported result. William (31 October), shown the per-condition counts, recalls generating them as further clear and ambiguous examples of drought stress, frost injury and gray mold, the conditions the April classifier was confusing, to try to fix those confusions. That matches the day's "synth" and "synthOOD" ResNet checkpoints (v5 to v12). Recalled, not documented. For context, drought and frost are among the three smallest real classes (65 and 115 images) while gray mold is the largest (650), so the batch was about confusions, not class size. Nothing in the paper depends on it, since the files enter no reported result.

**Text for 2.5**

```text
A further 93 generated images of drought stress, frost injury and gray mold were added to the lab's training folder on 27 April 2026 as additional clear and ambiguous examples of conditions the April classifier confused, for later experiments not reported here; they were excluded from the synthetic-data arms of this study.
```

### 2.3 Data processing pipeline

Status: Checked.

> Manuscript note: "William needs to check this subsection; it was drafted from his docs/BENCHMARK.md."

Checked against `cluster/benchmark/make_split.py`, `near_dupes.py`, `clean_eval_sets.py` and `cluster/synthetic/import_synth.py`. Every statement holds: byte-identical copies counted once by SHA-256; 64-bit DCT perceptual hash (pHash) on the RGB image; pairs within Hamming distance 6 joined into groups by union-find; the two groups with conflicting labels (4 images) removed; per condition the groups shuffled with seed 0 and filled into test until at least 25 percent, then validation until at least 10 percent, then training; 1017 / 161 / 398; no group crosses sets. Synthetic images within distance 6 of a validation or test photograph are dropped on import (the designed set); the April set was checked once the same way (one exact duplicate dropped, no near-copies). The low-quality suite, the 38-photograph bake-off set and the grouped five-fold cross-validation are described correctly. One wording suggestion: "exact copies" could read "byte-identical copies", since resized copies are the near-duplicates handled next.

### 2.4 Classifier settings, and the resize / centre-crop remark

Status: Checked, corrections below.

> Manuscript note: "Confirm every setting in this paragraph and in Supplementary Table S2 against the final cluster runs." and "Fix this before the final runs: the deployed app resizes to 294 pixels and center-crops to 256, while training squashes to 256."

The split v1 ResNet-50 runs used the lab's own script `plant_diagnostic/scripts/resnetscripts/train_resnet_straw5.py` with its default settings, five seeds (42, 1, 2, 3, 4), the split v1 training and validation folders, and no warm start. Against the script, the paragraph needs these changes:

1. **Training input.** Not "resized to 256 × 256 without cropping". The training transform takes a random crop covering 90 to 100 percent of the image area with an aspect ratio between 3:4 and 4:3 and resizes it to 256 × 256 (so the crop is squashed), then applies rotation within ±3°, a horizontal flip with probability 0.5, perspective distortion (scale 0.10, probability 0.15), colour jitter (brightness 0.10, contrast 0.10, saturation 0.05, hue 0.02), Gaussian blur (kernel 3, σ from 0.1 to 1.5, always applied), automatic contrast (probability 0.20) and, after normalisation, random erasing of 2 to 80 percent of the image with probability 0.15.
1. **Validation and deployment input.** The script's own validation transform resizes the short side to 294 pixels (1.15 × 256) and centre-crops 256 × 256; the application uses exactly this transform. So the app does not deviate from the recipe: the difference is between the training augmentation (random, squashed crop) and the validation path (centre crop), which is the script's design. Scoring the test images by squashing instead gave 89.8 percent against 90.2 (lab notebook), so no code change is needed; the remark can be removed and the text corrected as above.
1. **Frozen parts.** In phase 2 the four residual stages and the output layer train; the stem (first convolution and its batch normalisation) stays frozen, and every batch-normalisation layer is kept in evaluation mode with frozen affine parameters in both phases. The paragraph says this correctly.
1. **Optimiser and schedule.** AdamW with weight decay 0.05 (no decay on biases and one-dimensional parameters), base rate 1e-3 for the head phase and 1e-4 for the full phase with the stage multipliers 0.25, 0.5, 0.75 and 1.0 (output layer 1.0); phase 2 has a two-epoch linear warm-up from a tenth of the rate followed by cosine decay over the remaining thirteen epochs; mixed precision (float16 autocast with loss scaling); gradient norm clipped at 1.0; batch 32; the checkpoint with the best validation top-1 accuracy is kept. No weight averaging (the script's EMA option is off by default).
1. **Loss.** Cross-entropy with inverse-frequency class weights and label smoothing 0.05, as stated.
1. **Calibration.** After training, a single temperature is fitted on the validation set by L-BFGS on the negative log-likelihood (learning rate 0.01, 50 iterations) and stored in the checkpoint; for seed 42 it is 0.748, so the scaling sharpens rather than softens the probabilities. Inference averages the logits of the image and its mirror before scaling, as stated.
1. **Initialisation.** torchvision's ImageNet weights (ResNet50_Weights.IMAGENET1K_V2).

**Replacement for the first paragraph of 2.4 (training part)**

```text
Training images were presented as a random crop covering 90 to 100 percent of the image, with an aspect ratio between 3:4 and 4:3, resized to 256 × 256 pixels, and augmented online with small rotations (±3°), horizontal flips, mild perspective distortion, colour jitter, Gaussian blur, automatic contrast adjustment and random erasing (Supplementary Table S2). At validation and in the application, the short side is resized to 294 pixels and the centre 256 × 256 pixels are kept.
```

**Table S2, condition-classifier column:** Input size "256 × 256 (random 90 to 100 percent crop, squashed; validation: resize 294, centre crop 256)"; Augmentation add "random erasing (p 0.15)"; Schedule "phase 2: 2-epoch linear warm-up from 0.1 ×, then cosine"; Regularisation add "batch normalisation frozen, stem frozen"; Seed "42 (replicated with seeds 1 to 4)"; add a row Checkpoint "best validation top-1" and a row Calibration "temperature fitted on validation (L-BFGS)". The rest of the column is right.

### 2.4 Acceptance-gate thresholds

Status: Origin; Validation-based values in 3.3.

> Manuscript note: "Confirm these were set on validation images in the final runs" and "How were these thresholds chosen?"

They were not set on validation images. The values (0.40 healthy, 0.60 default, 0.65 drought and root rot, 0.70 frost) were chosen by hand in October 2025 while the first demo was built (`resnet_classifier.py` and `demo_v5.py`, commit of 4 October 2025; the code comments read "lowered to catch more healthy predictions" and "lowered for better sensitivity"), by looking at demo outputs on the holdout of that time. For the paper I will report two things from the split v1 runs, both computed from the per-image probability files: (a) coverage and accuracy of the deployed v1 thresholds on the test set, as the setting the application ships with; and (b) thresholds chosen on the split v1 validation set by a stated rule (per predicted condition, the smallest threshold of at least 0.5 at which the accepted validation images of that condition are 95 percent correct), with their test coverage and accuracy. Gus can then keep (b) as the final setting and mention (a) in one sentence.

**Text for 2.4 if (b) is adopted**

```text
The class thresholds τ_c were chosen on the validation images as the smallest value of at least 0.5 at which 95 percent of the accepted validation images of that condition are correct; the thresholds shipped in the v1 application had been set by hand on the earlier holdout.
```

### 2.6 Detector input resolution: 512 or 560

Status: Done.

**512.** The checkpoint's stored arguments hold no resolution field at all, but its DINOv2 position embeddings have 1024 patch positions, 32 × 32 patches of 16 pixels, which is a 512-pixel input; the saved arguments also show `square_resize_div_64` on, and the current library loads the checkpoint at 512 without reshaping. The configuration file's 560 (`rfdetr_parts/rfdetr_config.yaml`) was never applied: `train_rfdetr.py` builds `RFDETRSmall()` at the library default and the value passed to the training call did not rebuild the model. Table S2 and the sentence in 2.6 should say 512; docs/DETECTOR.md has been corrected.

**Text for 2.6**

```text
The input resolution is 512 pixels; the configuration's request of 560 pixels was not applied by the training script.
```

### 2.6 Part-presence thresholds

Status: Origin; Sweep done.

> Manuscript note: "How were these thresholds chosen? A sweep script exists (eval_thresholds.py), but no output was saved."

The values live in `rfdetr_parts/detect_images.py` (flower and fruit 0.50, leaf, stem and soil 0.40, root 0.30). They were set by hand in April 2026 after running the sweep script on random validation images and reading its precision and recall per threshold; the output was not kept. For the paper, the sweep was rerun on the detector's new validation split (next item) with the thresholds chosen to maximise image-level presence F1 per part; on the test split they do no better than the hand-set values (only 4 to 25 validation images per part), so the hand-set thresholds stay, and the sweep output is saved with the result files (`evaluation/paper_data/parts_*_valid.json`).

### 2.6 / 3.5 A detector test split that agrees with split v1

Status: Done.

Built and run. The 1075 annotated images were matched to the condition photographs by perceptual hash (646 matched, 645 of them at Hamming distance 2 or less; the 429 annotation-only images went to training): 845 training, 67 validation and 163 test images, the test images being annotated copies of 151 split v1 test photographs (healthy 74, root rot 30, frost 24, gray mold 12, white mold 7, drought 2, overwatered 2), so the detector test split leans on healthy, frost and root-rot photographs. RF-DETR Small retrained with the lab configuration (100 epochs, checkpoint chosen on the detector validation split) reaches **mAP50:95 0.564**, mAP50 0.716 and mAP75 0.608 on the test split; per part, AP50:95 is flower 0.687, fruit 0.702, leaf 0.578, root 0.288, soil 0.660 and stem 0.471, so roots and stems are the hard parts, as the manuscript expects. Image-level part presence at the application's thresholds gives F1 0.87 (flower), 0.96 (fruit), 0.97 (leaf), 0.86 (root), 0.72 (soil) and 0.92 (stem); thresholds chosen by F1 on the 67-image validation split do no better (4 to 25 validation images per part), so the hand-set thresholds stay, with that sentence in 2.6.

One finding for 3.5 and the leakage discussion: the lab's April checkpoint scored on this same test split gives mAP50:95 0.681, because the lab's random split had put near-copies of these photographs into its training data. The earlier validation-split figures (0.598 for the lab checkpoint, 0.593 for the like-for-like retrain) were therefore optimistic by about ten points; 0.564 is the clean v1 number. **RF-DETR Large** trained the same way reaches mAP50:95 **0.593** (mAP50 0.720, mAP75 0.641) on the test split, 2.9 points above Small, in line with the 3.8-point gap seen earlier on the validation split; per part: flower 0.734, fruit 0.734, leaf 0.616, root 0.293, soil 0.650, stem 0.531, with presence F1 0.86 / 0.96 / 0.98 / 0.84 / 0.72 / 0.93 at the application's thresholds. Table S9 can take two new rows (Small 0.564, Large 0.593 on the clean test split) next to the validation-split rows, labelled as such. Files: `evaluation/paper_data/parts_*.json` and `detector_split_v1.json`.

**Text for 3.5**

```text
On a detector test split of 163 annotated images that agrees with split v1 (no test photograph or near-copy of one in detector training), RF-DETR Small retrained with the lab configuration reached a mAP50:95 of 0.564 (mAP50 0.716). Average precision was highest for fruit and flowers (0.70 and 0.69) and lowest for roots and stems (0.29 and 0.47), while image-level presence, which is what the report prompt uses, reached an F1 of 0.86 to 0.97 for every part except soil (0.72). RF-DETR Large reached 0.593 (mAP50 0.720), 2.9 points above Small. The lab's original checkpoint scores 0.681 on the same images, 12 points higher, because its random split had placed near-copies of them in training; the validation-split figures reported earlier were optimistic by about this much.
```

### 2.7 The script that generated the instruction reports

Status: Facts; Decision for Dr. Zhang.

> Manuscript note: "The script that generated this file is empty in every commit, so it has to be recovered or rewritten before the procedure can be published."

Confirmed: `plant_diagnostic/scripts/gptscripts/json_generation/generate_stage2_jsonl.py` is zero bytes in every commit, and it is absent from the lab machine's exported working tree as well, so it cannot be recovered. The reports themselves exist (2159 records in `plant_diagnostic/datasets/stage2_train_7class_fixed.json`; every report begins "Diagnosis: <label>."), but the prompt is not stored with them. Two honest options: (a) state that the generation script was lost and describe the procedure from the rules recorded at the time and from the reports; (b) additionally provide a re-implementation in the repository, documented as a reconstruction and not rerun (there are no API credits, and v2 draws its annotations from local models anyway). I recommend (b).

**Text for 2.7**

```text
The script that produced the reports was not preserved; the procedure is described from its recorded rules and its outputs, and a reconstruction of the script is provided in the repository for reference.
```

### 2.7 Report-writer training: validation, matching prompts, no test images; Table S16

Status: Done.

All three fixes are in the retraining plan of item 2.1: records filtered to split v1 training photographs; checkpoint chosen by validation loss on the records of the validation photographs; the training instruction equal to the inference system prompt, built per image from the detector's visible parts. Training the LoRA for the published schedule (20 epochs of 2000 iterations, effective batch 32) takes most of a day on one GPU, so the first run uses ten epochs and the validation loss will say whether more are needed. Training ran on the night of 30 to 31 October (9 h 40 min on one RTX 5000 Ada; see 2.1 for the loss curves). Two defects of the lab's training code were found and fixed on the way, both recorded in the notebook: the runner's evaluation gate drops the validation split whenever it runs in training mode, which is why the April writer is simply its last epoch, and the loss call passes a keyword argument that the LLaMA wrapper on the cluster rejects. Because of the first, the ten epoch checkpoints were scored on the 212 validation records after training and the lowest-loss one kept.

**Table S16 result.** The retrained writer wrote one report per prompt condition for each of the 151 split v1 test photographs that carry part masks (302 reports, the application's decoding, the consistency check with one regeneration). The part lists did not reduce part hallucination: the part-hallucination rate of equation (11) was 86.1 percent without them and 92.1 percent with them (paired exact McNemar on the 19 discordant photographs, 14 against 5, P = 0.064), and the writer named a part the prompt had declared not visible in 91.4 percent of the with-parts reports. Most false mentions are of soil, nearly always in care advice (irrigation depth, straw mulch; the pre-registered rule counts mulch and substrate as soil): 113 and 127 of the 127 photographs without annotated soil. Without soil the rates are 33.8 against 39.1 percent (P = 0.28), so the conclusion does not depend on that rule. Condition consistency was 100 and 96.0 percent (the six exceptions omit the "Diagnosis:" line but describe the labelled condition), 15.2 and 11.9 percent of reports were regenerated once, and reports average about 200 words. The April writer, scored on the same photographs as a labelled leaky reference, gives 85.4 and 88.1 percent; it rarely writes a "Diagnosis:" line (54 and 5 of 151 reports), so its consistency figures measure format. Files: `evaluation/paper_data/reports_splitv1.json`, `reports_april_writer.json`, the `*_scores.json` summaries and `minigpt_splitv1_val_loss.json`; scoring script `cluster/vlm/score_reports.py`, exploratory breakdown `score_reports_exploratory.py`.

| Measure (151 photographs, retrained writer) | Without part lists | With part lists | Paired test |
|---|---|---|---|
| Part-hallucination rate H_part (%) | 86.1 (80.8 to 91.4) | 92.1 (87.4 to 96.0) | 14 / 5, P = 0.064 |
| H_part without soil (%), exploratory | 33.8 | 39.1 | 17 / 25, P = 0.28 |
| Names a part listed as not visible (%) | n/a | 91.4 |   |
| Condition consistency (%) | 100.0 | 96.0 (92.7 to 98.7) | 0 / 6, P = 0.031 |
| Reports regenerated at least once (%) | 15.2 (9.9 to 21.2) | 11.9 (7.3 to 17.2) |   |
| Mean report length (words) | 203 | 200 |   |
| April writer (leaky reference): H_part (%) | 85.4 (79.5 to 90.7) | 88.1 (82.8 to 92.7) | 10 / 6, P = 0.45 |

Note: Intervals are bootstrap 95 percent intervals over photographs; the paired column gives the discordant photographs (with-parts only / without-parts only) and the exact McNemar P. The pre-registered prediction that the lists would at least halve H_part was not supported; the measure, rule and predictions were fixed in the notebook before the runs.

**Text for 3.9 and Table S16**

```text
With the detector's part lists in the prompt, the retrained writer's part-hallucination rate was 92.1 percent against 86.1 percent without them (151 photographs, paired exact McNemar P = 0.064), so the lists did not reduce part hallucination; the writer named a part the prompt had declared not visible in 91 percent of reports. Most false mentions are of soil, nearly always in care advice; excluding soil, the rates are 39.1 against 33.8 percent (P = 0.28). Condition consistency was 100 and 96 percent, and 15 and 12 percent of reports were regenerated once by the consistency check.
```

**Discussion sentence (replaces "the planned ablation of detector grounding has not been run, so no claim about reduced part hallucination can be made yet")**

```text
The ablation of detector grounding found no reduction in part hallucination: the v1 writer does not follow the prompt's part constraints, and the v2 explainer is not instructed through part lists at all.
```

### 2.7 Consistency check: negation, retry path, hallmark list

Status: Done.

Confirmed in the code. `knowledge_graph/rag_retriever.py` (check_definite_hallucination) tests each other condition's visual indicators as case-insensitive substrings, so "no gray fuzz is present" is flagged. In `demo_v5.py` the regeneration branch rebuilds the stricter prompt from the visible and hidden part lists, which are only defined when at least one part was detected; with no parts the retry raises an error. Fixes, implemented in `knowledge_graph/rag_retriever.py` and `demo_v5.py` and checked on ten test sentences (negated mentions are now reported separately as `negated_mentions`); committed on 31 October 2026 with William's approval:

- Negation-aware matching: a phrase is ignored when, within the same sentence, it is preceded within five words by no, not, without, free of, absence of, lacks, rather than or instead of.
- The part lists are defined before the generation loop (empty when nothing was detected), so the retry uses the short prompt instead of failing.
- Hallmark list: remove "brown roots" (listed as a sign of overwatering) and "botrytis" (named as a secondary-infection risk under frost injury, overwatering and root rot), leaving 16 phrases; "waterlogged" and "gray fuzz" stay, since the negation rule handles the contrastive uses. Figure S7 and the two sentences that count phrases ("18 phrases", "four of them") change accordingly.

### 2.9 Who rated the claims

Status: Answered.

**Text for 2.9**

```text
All claims were rated by W.S., blind to the captioner. G.M. will rate the claims of at least ten photographs as a second rater, and inter-rater agreement (Cohen's κ) will be reported.
```

### 2.11 Who compiled the knowledge base, and how

Status: Confirmed.

> Manuscript note: "The README says the treatments were web-crawled and curated, but no crawler is in the repository."

There is no crawler and there never was one in the repository. The knowledge base (`knowledge_graph/disease_knowledge_base.json`, seven conditions, 24 source citations) was written by hand from the cited extension pages when the first demo was built, in September and October 2025; the repository history shows it in the October 2025 commits by W.S., moved into its own folder by G.M. on 17 October 2025 and revised on 12 April 2026. The README's "web-crawled" wording is wrong and has been corrected in the README and docs/V1_SYSTEM.md. William confirmed on 31 October that the knowledge base is a corpus of information pulled from university extension publications and that no crawler was involved. William also confirmed that a language model helped adjust the wording of the entries: Anthropic's Claude, the current model at the time (autumn 2025); the version was not recorded. The generative-AI statement should say so.

**Text for 2.11**

```text
The entries were compiled by the first author from the cited university extension publications; a language model (Anthropic's Claude; the version was not recorded) helped adjust their wording. No automated collection was used.
```

### 2.13 Lab server GPU

Status: Two RTX 3090.

William (31 October): the laboratory's server before the cluster had two NVIDIA GeForce RTX 3090, 24 GB each, so the manuscript's "two GPUs with 24 GB of memory each" stands, and the old README's RTX 3090 was the card itself, not only a recommendation.

**Text for 2.13**

```text
The v1 models were trained and served on the laboratory's GPU server, which has two NVIDIA GeForce RTX 3090 GPUs with 24 GB of memory each.
```

### 2.13 Software versions of the re-evaluation

Status: Done.

Frozen from the four cluster environments on 30 October (full lists in `cluster/infra/requirements/`, one `<env>.freeze.2026-10-30.txt` per environment):

- **v1 pipeline** (Python 3.9.23): PyTorch 2.5.1 with CUDA 12.1, torchvision 0.20.1, Transformers 4.45.2, PEFT 0.13.2, bitsandbytes 0.43.3, accelerate 0.34.2, timm 0.6.13, NumPy 1.26.4.
- **Detector** (Python 3.11.17): rfdetr 1.11.2, PyTorch 2.14.1, torchvision 0.29.1, Transformers 5.20.2, NumPy 2.4.6.
- **Backbones and synthetic images** (Python 3.11.17): PyTorch 2.9.1 with CUDA 12.8, torchvision 0.24.1, timm 1.0.30, open_clip_torch 3.3.0, diffusers 0.41.0, Transformers 5.20.2, scikit-learn 1.9.1.
- **Vision-language serving** (Python 3.11.17): vLLM 0.32.0, PyTorch 2.13.0, Transformers 5.19.0, Ray 2.59.0, xgrammar 0.2.8.

Note: docs/COMPUTE.md lists PEFT 0.11.1 for the v1 environment; the installed version is 0.13.2 and the page has been corrected. The versions used on the lab machine in April are a separate question for William (2.13, first paragraph).

## Answers: Results and discussion (Sections 3 to 5) and author contributions

### 3.3 Per-image probabilities, ECE, AURC and the gate (Table S7, Fig. S9)

Status: Done.

Computed from the saved per-image probability files (no new training): the five ResNet-50 seeds of the lab recipe and the three seeds of the recommended SigLIP 2 recipe, validation and test. Unscaled probabilities were recovered exactly from the stored temperature. Split v1 test, 398 photographs, mean ± s.d. over seeds:

| Metric | ResNet-50 v1, no temperature | with temperature | SigLIP 2 recipe, no temperature | with temperature |
|---|---|---|---|---|
| Accuracy (%) | 90.2 ± 1.5 | same | 96.3 ± 0.8 | same |
| Macro accuracy (%) | 85.5 ± 2.0 | same | 94.3 ± 0.7 | same |
| Macro-F1 | 0.857 ± 0.023 | same | 0.936 ± 0.008 | same |
| ECE (%) | 16.7 ± 1.7 | 6.7 ± 0.8 | 8.3 ± 1.3 | 2.7 ± 0.3 |
| AURC | 0.0249 ± 0.0078 | 0.0240 ± 0.0072 | 0.0048 ± 0.0013 | 0.0046 ± 0.0016 |
| Deployed v1 gate: coverage / accuracy on accepted (%) | 87.6 ± 1.1 / 94.5 ± 0.6 |   | 97.7 ± 1.6 / 97.5 ± 0.1 |   |
| Validation-chosen gate: coverage / accuracy on accepted (%) | 84.2 ± 3.2 / 95.7 ± 0.6 |   | 96.6 ± 1.0 / 97.9 ± 0.5 |   |

Fitted temperatures are 0.72 to 0.78 for every seed, so scaling sharpens an under-confident model: ECE falls by ten points for the ResNet-50 and by 5.6 for SigLIP 2, while accuracy, ranking and AURC are unchanged, as the paper says. The validation-chosen thresholds (per predicted condition, the smallest value of at least 0.5 at which 95 percent of accepted validation images are correct) vary across seeds for the small conditions (drought 0.64 to 0.93) because only three to seven validation images are predicted as those conditions, so the deployed thresholds are the better setting to report as v1's, with the validation rule as the principled alternative and its seed variance stated. ECE uses ten equal-width bins of the top-1 probability; AURC follows Geifman and El-Yaniv; intervals are 10,000-resample bootstraps over photographs.

Files for Gus's figure script: `evaluation/calib_<model>_test_seed<k>.json` (per photograph: image, truth, prediction, probabilities with and without temperature, top-1 probability, both gate decisions) and `evaluation/paper_data/calib_<model>_summary.json` (per-class precision, recall and F1, confusion matrices, risk-coverage curves, bootstrap intervals, chosen thresholds). Which model is "the chosen classifier" of Table S7 remains Dr. Zhang's call; both are supplied.

### 4 Was the mirrored view ablated on split v1?

Status: Done.

Now it has been (ResNet-50, five seeds, 398 test photographs, saved checkpoints). Two views against one: 91.7 vs 91.0, 91.5 vs 91.2, 90.2 vs 90.0, 88.9 vs 89.5 and 88.4 vs 87.9 percent. The mirror changed 6, 5, 5, 14 and 11 predictions per seed (1.3 to 3.5 percent), turning 4, 2, 3, 4 and 5 photographs from wrong to right and 1, 1, 2, 6 and 3 from right to wrong: 18 against 13 over five seeds, a mean gain of 0.25 points (range −0.5 to +0.8), not significant.

**Text for 4**

```text
On split v1 the mirrored view changed one to four percent of the ResNet-50's predictions per seed and turned almost as many right predictions wrong as the reverse (13 against 18 photographs over five seeds), for a mean gain of 0.3 points.
```

### 3.9, 2.1, abstract and conclusion: v1 versus v2 end to end

Status: Table S16 done; Latency and memory done; Faithfulness next week.

What exists: diagnosis accuracy (v1 ResNet-50 90.2 ± 1.5 percent over five seeds; v2 recipe 96.3 ± 0.8 over three) and honesty (v1 cannot reject a label by design, since the writer is told the diagnosis is correct; the v2 student rejects 85.2 percent of wrong labels, 82.5 percent balanced). What the weekend adds: the assembled v2 pipeline (SigLIP 2 recipe, RF-DETR Large from the new detector split, the 8B student served with vLLM) and v1 as deployed (the application's own path and settings, its detector subprocess included) timed on one RTX 5000 Ada over a seeded sample of 49 test photographs (7 per condition), with latency per stage and per photograph and GPU memory per process. The sample is small because latency is a property of the system; accuracy and honesty come from all 398 test photographs. The measurement was pre-registered in the notebook (predictions included) and ran on the cluster on 31 October (job 72394, 1 h 30 min); the numbers are below. The detector on/off ablation for Table S16 is done (item 2.7). What needs people: claim-level faithfulness of v1 against v2 reports requires blind ratings; I propose the 38 bake-off photographs rated by G.M. and W.S. next week.

**Latency and GPU memory result.** One RTX 5000 Ada (32 GB), 49 seeded split v1 test photographs (7 per condition), two warm-up photographs, each stage in its own process, medians with 90th percentiles. v1 as deployed answered in 62.2 s per photograph (p90 75.2): 15.9 s go to the detector, because the application runs RF-DETR in a fresh interpreter for every photograph and reloads the model each time (the forward pass itself takes 0.016 s), and 46.2 s to the MiniGPT-v2 report (median 337 tokens at 7.3 tokens per second from the 8-bit 7B model); image encoding, the two-view ResNet-50 pass and the gate take under 0.1 s together. With the detector kept in memory, v1 would take 46.3 s. v2 answered in 21.7 s (p90 24.9), almost all of it the student's report (median 725 tokens at 33.5 tokens per second under vLLM, one request at a time); the SigLIP 2 classifier takes 0.04 s and RF-DETR Large 0.02 s. Served in a batch, the student wrote 1.2 reports per second. Memory: v1's writer-plus-classifier process held 11.8 GB and its detector process 1.0 GB, 12.9 GB together; v2's classifier process held 3.1 GB, its detector process 1.1 GB, and the vLLM engine 26.2 GB as configured (85 percent of the card; the student's bf16 weights are 17.5 GB). The stages ran one at a time, so the v2 sum (30.4 GB) is the configured footprint, not a tested co-residence. Of the four pre-registered predictions three held (v1 slower than v2 and dominated by the subprocess and the report; v2 using more memory; classifiers and detectors under 0.5 s) and one did not: with the detector in-process, v1 is 2.1 times slower than v2, just outside the predicted factor of two. Note for 2.13: the manuscript's lab-server figure of 25.3 s per report was measured on the lab GPU; the 46 s here is the 8-bit writer on the Ada card, and the two need not agree.

| Stage or pipeline (49 photographs, RTX 5000 Ada) | Median s | p90 s | GPU memory |
|---|---|---|---|
| v1 MiniGPT-v2 image encoding | 0.06 | 0.06 | 11.8 GB (one process) |
| v1 ResNet-50, two views and gate | 0.03 | 0.04 | same process |
| v1 MiniGPT-v2 report, 8-bit, median 337 tokens | 46.2 | 58.8 | same process |
| v1 RF-DETR Small, application subprocess | 15.9 | 16.3 | 1.0 GB |
| v1 RF-DETR Small, in-process | 0.02 | 0.02 |   |
| **v1 as deployed** | **62.2** | **75.2** | **12.9 GB** |
| v1 with the detector in-process | 46.3 | 59.0 |   |
| v2 SigLIP 2 classifier, two views | 0.04 | 0.04 | 3.1 GB |
| v2 RF-DETR Large, in-process | 0.02 | 0.02 | 1.1 GB |
| v2 Qwen3-VL-8B student under vLLM, one request, median 725 tokens | 21.6 | 24.8 | 26.2 GB as configured; weights 17.5 GB |
| **v2** | **21.7** | **24.9** | **30.4 GB as configured** |
| v2 student, batched throughput | 1.20 photographs per second |   |   |

Note: Memory is nvidia-smi's figure for each process; the vLLM engine pre-allocates gpu_memory_utilization = 0.85 of the card, so its figure is a setting, not a need. Load times: MiniGPT-v2 25 s, SigLIP 2 101 s, vLLM engine 169 s. On this sample the SigLIP 2 classifier was right on 47 of 49 photographs and the student accepted 38 of the 49 proposed labels (10 of the 47 correct ones rejected, 1 of the 2 wrong ones rejected), in line with the test-set honesty figures; the sample is for timing, not honesty. Files: evaluation/paper_data/latency_v1_v2.json and the four stage files beside it.

**Text for 3.9, latency and memory**

```text
Latency and GPU memory were measured on one NVIDIA RTX 5000 Ada (32 GB) over a seeded sample of 49 split v1 test photographs (7 per condition), each system run as deployed, after two warm-up photographs. v1 answered in a median of 62.2 s per photograph (90th percentile 75.2 s): 15.9 s for the detector, which the application runs in a separate interpreter that reloads RF-DETR for every photograph, and 46.2 s for the MiniGPT-v2 report (median 337 tokens from the 8-bit model); the classifier and the image encoding took under 0.1 s. With the detector kept in memory, v1 would take 46.3 s. v2 answered in a median of 21.7 s (90th percentile 24.9 s), almost all of it the student's report (median 725 tokens under vLLM); the classifier and detector took 0.04 and 0.02 s. In batched serving the student produced 1.2 reports per second. The v1 processes held 12.9 GB of GPU memory together (11.8 GB for the writer and classifier, 1.0 GB for the detector); v2's classifier and detector processes held 3.1 and 1.1 GB, and the student's weights occupy 17.5 GB, with the vLLM engine configured to reserve 26.2 GB. All measurements were made on a single 32 GB GPU; v2's three processes were timed one at a time.
```

**Sentence for the abstract and conclusion**

```text
End to end on the same test photographs, v2 raised diagnosis accuracy from 90.2 to 96.3 percent, rejected 85.2 percent of wrong labels where v1 could reject none, and answered in 22 seconds per photograph against 62 for v1 on the same GPU.
```

Note: 22 and 62 are the medians above (21.7 and 62.2 s); with its detector kept in memory v1 would be 46 s. 85.2 percent is the student's label-swap rejection rate from the manuscript's honesty table. A claim-faithfulness clause waits for the blind ratings.

### 4 Supplementary Table S17

Status: Checked.

All seven rows hold. Five were checked against the papers themselves, two against their bibliographic records and keywords because the full text is closed access.

- **Afzaal et al. (2021).** Confirmed: 2500 images of seven strawberry diseases, Mask R-CNN with a ResNet backbone, instance segmentation (Sensors 21, 6565).
- **Kim et al. (2021).** Confirmed: a two-stage cascade detector with a PlantNet backbone pretrained on PlantCLEF, 91.65 percent mAP for the cascade (Frontiers in Plant Science 11, 559172, published 11 January 2021 in the 2020 volume).
- **Bhujel et al. (2022).** Confirmed: U-Net semantic segmentation of gray mold on inoculated plants, compared with XGBoost, k-means and Otsu thresholding, with severity from the segmented area (J. Plant Dis. Prot. 129, 579).
- **Zhang and Wu (2026).** Confirmed: RT-DETR-AGWZ, built on RT-DETR-R50, on a self-collected strawberry disease dataset of six disease categories with a "frozen group-aware data partition" (Agronomy 16, 1830; submitted 29 July, published 17 September 2026). The author list (Mingzhen Zhang, Liuai Wu) comes from a mirror; Gus should confirm it on the MDPI page, which refused automated access.
- **Li, Yang and Li (2026).** Title, authors and venue confirmed (Comput. Electron. Agric. 244, 111484, published 27 January 2026). The abstract is closed access; the indexed keywords (crop disease detection, large language models, LLM hallucination, retrieval-augmented generation, multimodal knowledge graphs, disease severity assessment, plant disease prevention) support the row's "identification and prevention advice; vision model with a large language model". If Gus has the full text, "retrieval-augmented" could be added to the approach.
- **Liu et al. (2026).** Confirmed: TARAG, a time-aware knowledge base, hybrid retrieval with time-aware re-ranking and time-based generation, over 2000 pest and disease entities (Comput. Electron. Agric. 248, 111786).
- **Workneh et al. (2027).** Title, authors and venue confirmed (Comput. Electron. Agric. 256, 112464; online 22 September 2026, in the January 2027 volume, so the 2027 year is right). Abstract closed access; the keywords (temperature scaling, probability calibration, selective prediction, explainable AI, domain generalisation) support "calibrated, explainable CNN with leakage control".

### Author contributions, W.S. line

Status: Checked against the commits.

William (31 October): the manuscript's line covers it for the most part, and the commit history can be checked. Checked: William's commits hold the v1 system (September to October 2025: the MiniGPT-v2 and ResNet-50 diagnostician and the knowledge base), the RF-DETR grounding pipeline and knowledge-base update (12 April 2026), the paper refresh (25 April 2026) and the revival (21 and 22 October 2026: leak-free benchmark, experiment code, lab notebook, documentation). Gus's commits hold the demonstration application work (security, reliability and UX improvements, demo_v5 with webcam support, the test suite, the knowledge-graph folder; October 2025) and the YOLOv8 detection project (January 2026). The manuscript's W.S. line agrees with that history and can stand. The wording below is an alternative in William's own words for the re-evaluation part, to use or ignore.

**Alternative wording**

```text
W.S. designed and ran the re-evaluation: the leak-free benchmark, the backbone, synthetic-data and detector studies, the label-swap test and the student distillation, the captioner bake-off and its blind ratings, and the test-time-compute study; wrote the training and evaluation code; and kept the lab notebook.
```

## Plan for the pending numbers

| When | Work | Feeds |
|---|---|---|
| Friday night, 30 October | Probability files with and without temperature scaling; ECE, AURC, gate coverage and accuracy; thresholds chosen on validation; per-class precision, recall, F1; confusion matrices; bootstrap intervals; mirrored-view ablation; checkpoint resolution read; environment freeze; listing of the 93 files | Table S7, Fig. S9, 2.4, 2.6, 2.13, 4 |
| Saturday, 31 October | Detector split agreeing with split v1; RF-DETR Small and Large retrained; per-part AP and presence precision and recall; threshold sweep on validation. Report-writer data filtered, prompts matched, training started | 3.5, Table S9 (new rows), 2.6 |
| Saturday night, 31 October | Report writer scored with and without part lists (done); v1 and v2 end to end: latency and memory (done), accuracy and honesty (in hand); consistency-check fixes committed | Table S16, 3.9, abstract, conclusion, 2.7 |
| Week of 3 November | Blind ratings of v1 against v2 reports (G.M. and W.S.); second rater on the claim set | 3.9 faithfulness, 2.9 agreement |

Note: Every run is recorded in the lab notebook (cluster/EXPERIMENT_LOG.md) with its job ID and result file, and the result files are added to the repository so the tables can be recomputed.

## Open items, by person

- **William:** nothing open. Answered on 31 October: the April prompts were not kept (2.2); the lab server had two RTX 3090 (2.13); the knowledge base is a corpus pulled from university extension publications, with a Claude model helping with the wording (2.11); the author-contribution line stands, checked against the commits; the 93 files of 27 April were William's extra drought, frost and gray mold examples, made to fix confusions (2.5).
- **Dr. Zhang or Gus:** which classifier is "the chosen classifier" of Table S7 (v1 ResNet-50 or v2 SigLIP 2 recipe); option (a) or (b) for the lost generation script (2.7); which PDF build Gus is editing, so the right one gets these changes.

## Result files in the repository

Every number above can be recomputed from these files (per-image records where the item says so):

- `evaluation/paper_data/calib_resnet50_v1_summary.json`
- `evaluation/paper_data/calib_siglip2_recipe_summary.json`
- `evaluation/paper_data/detector_split_v1.json`
- `evaluation/paper_data/latency_rfdetr.json`
- `evaluation/paper_data/latency_siglip2.json`
- `evaluation/paper_data/latency_student.json`
- `evaluation/paper_data/latency_v1.json`
- `evaluation/paper_data/latency_v1_v2.json`
- `evaluation/paper_data/minigpt_splitv1_train_log.txt`
- `evaluation/paper_data/minigpt_splitv1_val_loss.json`
- `evaluation/paper_data/parts_lab_small_on_splitv1_test.json`
- `evaluation/paper_data/parts_lab_small_on_splitv1_valid.json`
- `evaluation/paper_data/parts_large_splitv1_test.json`
- `evaluation/paper_data/parts_large_splitv1_valid.json`
- `evaluation/paper_data/parts_small_splitv1_test.json`
- `evaluation/paper_data/parts_small_splitv1_valid.json`
- `evaluation/paper_data/reports_april_writer.json`
- `evaluation/paper_data/reports_april_writer_scores.json`
- `evaluation/paper_data/reports_splitv1.json`
- `evaluation/paper_data/reports_splitv1_scores.json`

The lab notebook entries behind the answers are dated 30 and 31 October 2026 in
[`cluster/EXPERIMENT_LOG.md`](../cluster/EXPERIMENT_LOG.md): pre-registrations, results against predictions, and the
provenance answers of 31 October (entries (m) to (o)).
