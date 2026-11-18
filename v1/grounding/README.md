# `v1/grounding/`: plant-part grounding for the v1 report writer

Part of v1, the original system (2025 to April 2026). This Python package, added on 12 April 2026, runs the RF-DETR
Small part detector of [`../rfdetr_parts/`](../rfdetr_parts/) for the Gradio application and builds MiniGPT-v2
prompts that list the detected plant parts as visible and the others as not visible. One revival-era change: on
31 October 2026 `detector.py` gained the `RFDETR_PYTHON` override, so the detector can run without conda on the
cluster.

## Files

| File | What it is |
|---|---|
| [`__init__.py`](__init__.py) | package docstring; exports `analyze`, `GROUND_PARTS`, `PLANT_PARTS` and `DISEASE_CLASSES` |
| [`config.py`](config.py) | constants: `PROJECT_ROOT = /data/AGAI/MiniGPT-4` (the lab server) and the paths built on it (`TRAIN_DIR`, the detector script, an unused `RFDETR_CHECKPOINT`), the relative MiniGPT config path `eval_configs/minigptv2_eval.yaml`, the conda environment name `rfdetr`, the seven condition labels, the six parts (flower, fruit, leaf, root, soil, stem) and their synonyms, the switch `GROUND_PARTS = True`, the calyx rule for the two molds, `SEED = 42` |
| [`detector.py`](detector.py) | `run_rfdetr(image_paths)`: runs [`../rfdetr_parts/detect_images.py`](../rfdetr_parts/detect_images.py) in a separate interpreter (`$RFDETR_PYTHON` if set, else `conda activate rfdetr`), with `PROJECT_ROOT` as working directory, and returns its JSON: per photograph the parts found with their best score, every box, and the number of boxes; `{}` on any failure |
| [`inference.py`](inference.py) | `load_minigpt(gpu_id)` loads MiniGPT-v2 from `eval_configs/minigptv2_eval.yaml` and returns a `Chat` object and a `[INST]` conversation template; `build_prompt(label, detected_parts)` writes the grounded prompt (VISIBLE and NOT VISIBLE part lists) or, without parts, a free prompt; `run_minigpt(...)` generates one report (temperature 0.6, up to 512 new tokens) |
| [`analyzer.py`](analyzer.py) | `analyze(detected_parts, text, disease)`: regular-expression check of which parts a report mentions; drops sentences that negate a part, give care advice or use a part word only in `root rot` or `above-ground`, counts a leaf mention in a sentence with a fruit word as fruit (the calyx) for gray and white mold, and compares the result with the detected parts (grounded, mentioned but not detected, detected but not mentioned) |
| [`report.py`](report.py) | terminal output for [`../test_grounding.py`](../test_grounding.py) |

The application, [`../demo_v5.py`](../demo_v5.py), imports only `detector.run_rfdetr` and `config.PLANT_PARTS`; it
writes its own prompts and does not use `inference.py` or `analyzer.py`. When the detector finds nothing or fails, the
application uses its prompt without part lists.

## Running

[`../test_grounding.py`](../test_grounding.py) exercises the whole package: it picks one random photograph per
condition from `TRAIN_DIR` (seed 42), detects the parts, generates a report with the part lists in the prompt (or
without them, `--no-ground`), runs `analyze()` and prints the result.

```bash
cd v1                                   # inference.py opens eval_configs/ and imports minigpt4/ from here
python test_grounding.py                # all seven conditions, part lists in the prompt
python test_grounding.py --classes gray_mold white_mold --no-ground
```

Two environments are involved: the MiniGPT-v2 one for the main process (`minigptv`; see
[`../README.md`](../README.md#setup)) and one with the `rfdetr` package (Python 3.11) for the detector subprocess.
To use an interpreter directly instead of `conda activate rfdetr`:

```bash
export RFDETR_PYTHON=/path/to/rfdetr-env/bin/python
```

## Dependencies

- **Uses:** [`../rfdetr_parts/detect_images.py`](../rfdetr_parts/detect_images.py) and its checkpoint
  `rfdetr_parts/output/checkpoint_best_total.pth` (not in the repository);
  [`../eval_configs/minigptv2_eval.yaml`](../eval_configs/minigptv2_eval.yaml) and, through it,
  [`../minigpt4/`](../minigpt4/), [`../configs/`](../configs/), the LLaMA-2 weights
  ([`../llama_weights/`](../llama_weights/)) and the report-writer checkpoint.
- **Used by v1:** [`../demo_v5.py`](../demo_v5.py), [`../test_grounding.py`](../test_grounding.py), and the
  recovered April 2026 script
  [`run_minigpt_on_holdout.py`](../recovered/labbox_2026-10-19/paper/scripts/run_minigpt_on_holdout.py)
  (`load_minigpt`, `run_minigpt`).
- **Used by v2**, imported from the working folder that
  [`../../v2/infra/make_workdir.sh`](../../v2/infra/make_workdir.sh) builds:
  [`../../v2/vlm/latency_v1.py`](../../v2/vlm/latency_v1.py) (`run_rfdetr` and `load_minigpt`: v1 as deployed),
  [`../../v2/vlm/minigpt_reports.py`](../../v2/vlm/minigpt_reports.py),
  [`../../v2/vlm/minigpt_val_loss.py`](../../v2/vlm/minigpt_val_loss.py) and
  [`../../v2/benchmark/minigpt_diagnose.py`](../../v2/benchmark/minigpt_diagnose.py) (`load_minigpt`, then each
  loads other report-writer weights). On the cluster [`../../v2/infra/agai_env.sh`](../../v2/infra/agai_env.sh) sets
  `RFDETR_PYTHON`, and [`../../v2/slurm/latency_v1_v2.sbatch`](../../v2/slurm/latency_v1_v2.sbatch) stops if
  `detector.py` does not honour it.

## Known problems

- **Lab-server paths.** The detector script, the subprocess's working directory and `TRAIN_DIR` all derive from
  `PROJECT_ROOT = /data/AGAI/MiniGPT-4`. Elsewhere, edit `config.py` or link that path to `v1/`
  ([Paths](../README.md#paths)).
- **Slow detector call.** Every call starts a new interpreter that loads the model again: 15.9 s of v1's 62 s per
  photograph on an RTX 5000 Ada, against 0.016 s for the forward pass in one process
  ([DETECTOR.md](../../docs/DETECTOR.md), Caveats).
- **Silent fallback.** A failed call (for example a missing `rfdetr` environment) only prints to stderr and returns
  `{}`, so the application writes an ungrounded report without saying so.
- **The part lists have no measured effect.** In the revival's detector on/off ablation (31 October 2026) the report
  writer retrained on split v1 did not follow the lists in the prompt (part hallucination 86.1 % without them,
  92.1 % with them on 151 split v1 test photographs; [DETECTOR.md](../../docs/DETECTOR.md), Caveats).
- **transformers version.** `load_minigpt()` wraps every module's `forward` to drop `cache_position`. Under
  transformers 4.41 this hides the forward signature and generation runs out of memory on a 32 GB GPU; 4.45.2 works
  (comment in [`../../v2/infra/hamilton_setup_envs.sh`](../../v2/infra/hamilton_setup_envs.sh)).
