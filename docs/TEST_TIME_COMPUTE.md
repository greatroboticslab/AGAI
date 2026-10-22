[← README](../README.md) · [Benchmark](BENCHMARK.md) · [Classifier](CLASSIFIER.md) · [Synthetic data](SYNTHETIC_DATA.md) · [VLM honesty](VLM_HONESTY.md) · [Captioning](CAPTIONING.md) · **Test-time compute** · [Detector](DETECTOR.md) · [Compute](COMPUTE.md) · [Roadmap](ROADMAP.md)

# Test-time compute: voting, debate, references and answer format

**Question.** Can spending more inference compute on *zero-shot* VLMs (sampling many answers, letting two models
debate, showing reference images, asking for an explanation first) replace training, or at least make their
diagnoses trustworthy? These experiments test proposals from the lab's discussion: caption many times and take the
most likely disease; when two models disagree, keep both answers; let models re-think after seeing each other's
diagnosis; show each model a synthetic image of its predicted disease.

**Answer.** No: none of these closes the gap to training. Majority voting adds 1-2 points (not significant), debate and
reference images add nothing, and asking for an explanation before the answer *costs* 5-9 points. What sampling does
provide is a usable **uncertainty signal**: vote share and model disagreement identify most of the photos a model
gets wrong.

All experiments: split v1 test (398 photos, dataset labels as ground truth), Qwen3-VL-8B, Qwen3-VL-32B and
Qwen3.8-27B in bf16 with vLLM (voting also with Qwen3-VL-235B AWQ and GLM-4.6V FP8), the 7-class zero-shot prompt
with class definitions, pre-registered in [`cluster/EXPERIMENT_LOG.md`](../cluster/EXPERIMENT_LOG.md).

## 1. Voting over sampled diagnoses

Each model diagnosed every photo once greedily and 20 more times with sampling (T = 0.7)
([`cluster/vlm/consensus_dx.py`](../cluster/vlm/consensus_dx.py)); majority votes over N samples are averaged over random
subsets of the 20.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="figures/voting_dark.png">
  <img alt="Line chart of majority-vote accuracy versus number of sampled diagnoses: all three models rise by 2-3 points from 1 to 20 samples, ending at 67-70 percent; three models pooled reach 73.9 percent; fine-tuned SigLIP 2 is at 96.3 and fine-tuned Qwen3-VL-8B at 94.2 percent" src="figures/voting_light.png" width="700">
</picture>

| Model | Greedy | Majority of 20 | vs greedy (McNemar) | Truth among the 2 most-voted | Accuracy on the 50 % most unanimous photos |
|---|---|---|---|---|---|
| Qwen3-VL-8B | 67.8 % | 70.1 % | 15 / 6, p = 0.08 | 79.6 % | 86.9 % |
| Qwen3-VL-32B | 67.6 % | 68.8 % (19 samples) | 17 / 12, p = 0.46 | 81.9 % | 87.4 % |
| Qwen3.8-27B | 65.3 % | 67.1 % | 16 / 9, p = 0.23 | 80.4 % | 84.4 % |
| Qwen3-VL-235B-A22B (AWQ) | 69.1 % | 70.1 % (≤ 10 samples) | 12 / 8, p = 0.50 | 79.6 % | 83.4 % |
| **GLM-4.6V (FP8)** | **75.6 %** | **76.1 %** (≤ 10 samples) | 16 / 14, p = 0.86 | 86.7 % | 87.9 % |
| 8B + 32B + 27B pooled (60 votes) | — | 73.9 % | — | — | — |
| *fine-tuned SigLIP 2 (reference)* | 96.3 % | | | | |

The two large models were sampled 10 times per photo (pipeline-parallel on five GPUs); 0.6-0.8 % of their outputs
were unparsable, so each photo has 3-10 valid samples (median 10). **GLM-4.6V is the best zero-shot diagnoser here
and the worst label-swap teacher** (64.1 %, [VLM honesty](VLM_HONESTY.md)): naming the disease in a photo and judging
whether a *given* label fits it are different skills in these models. For every one of the five models voting adds
0.5-2.3 points, none significant.

The prompt here asks for one evidence sentence before the category; with an answer-first format greedy accuracy is
higher (§ 3), but the conclusion is the same: voting moves accuracy by 1-2 points and a 10-minute fine-tune is
~20 points better (best zero-shot vote: GLM-4.6V, 76.1 %).

## 2. Two models: disagreement, two-answer captions, debate and references

For a pair of models, photos where their greedy answers differ are where errors concentrate. Pair Qwen3-VL-32B +
Qwen3.8-27B: they agree on 258 photos (82.6 % correct) and disagree on 140, where the 32B is right 40.0 %, the 27B
33.6 %, and **the truth is one of the two answers 73.6 % of the time**. Across the three possible pairs this coverage
is 68-74 %, against 30-40 % for either model alone, so a caption that names both diagnoses and what would tell them
apart is correct about twice as often as either single answer. But 11-15 % of all photos are agreed-but-wrong, which
no consensus can flag.

Methods to resolve the 140 disagreements ([`cluster/vlm/resolve_dx.py`](../cluster/vlm/resolve_dx.py)); "pair system" =
the 32B's resolved answer on disagreements, the common answer elsewhere (32B greedy alone: 67.6 %):

| Method | 32B / 27B accuracy on the 140 | Both agree afterwards (right / wrong) | Pair system on all 398 | vs 32B greedy |
|---|---|---|---|---|
| greedy (no resolution) | 40.0 / 33.6 % | — | 67.6 % | — |
| *choose*: "one says X, the other Y; decide" | 28.6 / 39.3 % | 68 (22 / 46) | **63.6 %** | 17 / 33, **p = 0.03 (worse)** |
| *choose* + 2 real reference photos per candidate | 37.1 / 43.6 % | 89 (40 / 49) | 66.6 % | p = 0.69 |
| *choose* + 2 synthetic reference images per candidate | 40.0 / 46.4 % | 86 (43 / 43) | 67.6 % | p = 1.0 |
| debate, round 1 | 41.4 / 37.9 % | 43 (21 / 22) | 68.1 % | p = 0.80 |
| debate, round 3 | 40.0 / 43.6 % | 71 (36 / 35) | 67.6 % | p = 1.0 |

- **Arbitration is at chance.** A random pick between the two answers would score 36.8 % on these photos; every
  method lands between 28.6 and 46.4 %.
- **Debate converges, but on a wrong answer half the time** (36 right / 35 wrong after three rounds). The weaker model
  mostly defers to the stronger one (27B: 24 wrong → right, 10 right → wrong; 32B: 13 each way).
- **Narrowing to two candidates hurts** the stronger model (32B 40.0 → 28.6 %).
- **Synthetic references are not worse than real ones here** (the pre-registered prediction that real references
  would help more was not supported), but neither kind helps.

## 3. Explaining before answering hurts

A controlled ablation with the same stack, images, categories and greedy decoding; only the answer format changes.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="figures/answer_format_dark.png">
  <img alt="Dot plot of zero-shot accuracy by answer format: with an evidence sentence before the answer, Qwen3-VL-8B scores 67.8 and Qwen3-VL-32B 67.6 percent, against 76.9-77.1 and 72.4-74.4 percent for the three answer-first formats; Qwen3.8-27B changes little" src="figures/answer_format_light.png" width="700">
</picture>

| Model | Plain text | JSON, category only | JSON, category then evidence | **JSON, evidence then category** | Evidence-first vs plain text |
|---|---|---|---|---|---|
| Qwen3-VL-8B | 76.9 % | 77.1 % | 77.1 % | **67.8 %** | 16 / 52, p = 1.4 × 10⁻⁵ |
| Qwen3-VL-32B | 74.4 % | 73.4 % | 72.4 % | **67.6 %** | 13 / 40, p = 0.0003 |
| Qwen3.8-27B | 67.1 % | 66.3 % | 69.3 % | 65.3 % | 15 / 22, p = 0.32 |

**Mechanism.** The damage is concentrated in gray mold (8B 93 → 64 %, 32B 93 → 77 %). The evidence sentence commits to
a salient, generic observation and often states that the key sign is absent ("a ripe red strawberry … with no visible
mold", "a dried, brown, shriveled flower calyx … no signs of mold"), and the category then follows the sentence:
the 8B's new errors go to healthy (28) and root rot (18), the 32B's to drought (15) and frost (13). Writing the same
sentence *after* the category costs nothing. The plain-text format reproduces the earlier zero-shot results on this
stack (8B 76.9 vs 77.1 %).

**Why it matters beyond zero-shot classification.** The annotation pipeline's second pass decides from the first
pass's *written* observations ([Captioning](CAPTIONING.md#1-two-pass-protocol)). A faint sign the first pass leaves
out cannot support the label there either. Two follow-ups are queued in the [Roadmap](ROADMAP.md): test
verdict-first vs observations-first in the annotation schema, and keep an image-in-the-loop check (the claim judge)
for statements that a sign is absent.

## Not yet run

The answer-format ablation for Qwen3-VL-235B and GLM-4.6V (only the evidence-first format was run for them), and
disagreement resolution for pairs other than Qwen3-VL-32B + Qwen3.8-27B.
