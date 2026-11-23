# `v2/vlm/`: vision-language model studies and the v2 report writer

Part of v2, the lab's 2026 revival of AGAI (October-November 2026). Zero-shot and fine-tuned VLM diagnosis, the
structured annotation schema and the label-swap honesty test, the distilled Qwen3-VL-8B student that writes v2's
reports, captioning and claim judges, voting and disagreement resolution, the manuscript's retraining and scoring of
v1's MiniGPT-v2 report writer, and the v1 versus v2 comparison of latency and claim faithfulness.

Results: [VLM honesty](../../docs/VLM_HONESTY.md), [Captioning](../../docs/CAPTIONING.md),
[Test-time compute](../../docs/TEST_TIME_COMPUTE.md), [Classifier](../../docs/CLASSIFIER.md) (zero-shot and LoRA
diagnosis), [Detector](../../docs/DETECTOR.md) (the detector on/off ablation of the report writer) and
[Manuscript Q&A](../../docs/MANUSCRIPT_QA.md) (report-writer retrain and Table S16 in 2.7, v1 versus v2 in 3.9). Every
run is recorded in the lab notebook, [`../EXPERIMENT_LOG.md`](../EXPERIMENT_LOG.md), mostly under the scripts' flat
names (`cluster/annotate_vllm.py`).

## Files

Environment: the venv under `/projects/$USER/agai/envs/` that a Slurm file in [`../slurm/`](../slurm/) or the
script's docstring names (the docstrings of `annotate_vllm.py`, `caption_bakeoff.py`, `consensus_dx.py` and
`judge_claims.py` point to `slurm/vllm_ray.sbatch`, which runs them in `envs/vllm`); "(inferred)" means no file names
one and the entry is an environment that has the packages the script imports; stdlib means standard library only
(any Python 3). † not in this repository.

### Annotation schema, honesty test and the student

| File | Purpose | Environment | Inputs → outputs (defaults) |
|---|---|---|---|
| [`annotation_spec.py`](annotation_spec.py) | shared module, no command line: the 7 conditions, the annotator's system prompt, the JSON schema (`bounded_schema()` adds length limits), `user_text()`, `validate()` | imported | none |
| [`annotate_vllm.py`](annotate_vllm.py) | schema-constrained structured annotations with vLLM; each task is `SPLIT:MODE`, with MODE `true` (the dataset label), `swap` (a wrong label) or `both` (the label-swap test); `--pp` for several nodes, `--order verdict-first` for the schema-order ablation | `vllm` | `--model` (required), `--tasks val:both`, `--split-json evaluation/split_v1.json`, `--run v1` → `/projects/$USER/agai/annotations/<run>_<model>/<split>_<mode>.jsonl` (resumable) |
| [`score_honesty.py`](score_honesty.py) | label-swap honesty table (accepts the true label, rejects the wrong one) and paired exact McNemar tests between annotation runs | `vision` (stdlib) | positional `annotate_vllm.py` JSONL files → stdout |
| [`train_vlm_reports.py`](train_vlm_reports.py) | LoRA distillation of a Qwen3-VL-8B student on a teacher's annotations (true and wrong labels), keeping only the teacher's correct verdicts unless `--keep-teacher-mistakes` | `vlm` | `--model`, `--train` (comma-separated JSONL), `--val` (all required), `--tag reports_v1` → adapter `/projects/$USER/agai/adapters/<model>_<tag>/` |
| [`merge_lora.py`](merge_lora.py) | merges a LoRA adapter into its base model and saves a plain checkpoint for vLLM | `vlm` | positional `BASE_DIR ADAPTER_DIR OUT_DIR` → `OUT_DIR` |

### Diagnosis

| File | Purpose | Environment | Inputs → outputs (defaults) |
|---|---|---|---|
| [`zeroshot_qwen3vl.py`](zeroshot_qwen3vl.py) | zero-shot 7-class diagnosis with transformers (optional thinking mode) on `paper68`, `current`, `resnet161` or `all1580`; the source of the class list and prompt that `annotation_spec.py`, `eval_vlm_scores.py`, `train_vlm_lora.py` and `benchmark/minigpt_diagnose.py` copy | `vlm` (`envs/vlm5` for Qwen3.8 models, which need the transformers 5 that `infra/hamilton_setup_envs.sh` installs there) | `--model` (required), `--quant none`, `--set paper68`; `evaluation/holdout/repro_resnet_v4_hamilton.json`† (paper68), `evaluation/clean_eval_sets.json`† (resnet161, all1580), `plant_diagnostic/data/holdout/<class>/` (current) → `evaluation/zeroshot_<model>_<quant>[_thinking<N>]_<set>.json` |
| [`train_vlm_lora.py`](train_vlm_lora.py) | LoRA or QLoRA fine-tune of a VLM as a 7-class diagnoser on split v1 with the zero-shot prompt, on one GPU or one rank per node through `infra/ddp_task.sh` | `vlm` (`VLM_ENV` selects another) | `--model` (required), `--quant none`, `--split evaluation/split_v1.json`, `--tag lora`, `--out-dir /projects/$USER/agai/adapters` → adapter `<out-dir>/<model>_<tag>/`, `evaluation/finetuned_<model>_<tag>_test.json` |
| [`eval_vlm_scores.py`](eval_vlm_scores.py) | per-class probabilities of a VLM (with an optional LoRA adapter) on split v1 validation and test, from the first-token logits of the class names, for ensembles and calibration | `vlm` (inferred) | `--model` (required), `--adapter`, `--tag lora`, `--split-json evaluation/split_v1.json`, `--splits val,test` → `evaluation/scores_<model>_<tag>_split_v1_<split>.json` |
| [`consensus_dx.py`](consensus_dx.py) | zero-shot diagnosis sampled several times per photograph (one greedy answer plus `--samples 20`) for voting, vote share and a ranked differential; `--prompt-style` for the answer-format ablation | `vllm` | `--model` (required), `--split-json evaluation/split_v1.json`, `--split test`, `--run v1` → `/projects/$USER/agai/consensus/<run>/<model>.jsonl` (resumable) |
| [`resolve_dx.py`](resolve_dx.py) | photographs where two zero-shot models disagree: choose between the two answers, with real or synthetic reference images, or debate over several rounds | `vllm` (inferred) | `--model`, `--peer` (both required), `--methods choose,refs_real,refs_synth,debate1`, `--split-json evaluation/split_v1.json`, `--synth-glob plant_diagnostic/data/train_aug/{c}/*synth*GPT*`†, `consensus_dx.py` outputs → `/projects/$USER/agai/consensus/<run>/resolve/<model>.<method>.jsonl` |

### Captioning

| File | Purpose | Environment | Inputs → outputs (defaults) |
|---|---|---|---|
| [`caption_bakeoff.py`](caption_bakeoff.py) | two-pass captioning with each candidate VLM: findings with a point each, without the label, then a text-only reconciliation with the label; `--samples N` repeats pass 1 for the self-consistency test | `vllm` | `--model` (required), `--set evaluation/bakeoff_set_v1.json`, `--run bakeoff_v1` → `/projects/$USER/agai/bakeoff/<run>/<model>.jsonl` and `<model>.samples.jsonl` |
| [`judge_claims.py`](judge_claims.py) | zoom-in claim judge: re-checks every pass-1 claim of the bake-off at the boxed location, without the label, together with decoy claims; `--modes zoom,nozoom`; `--dry-run DIR` renders inputs without a model | `vllm` | `--model` (required), `--set evaluation/bakeoff_set_v1.json`, `--run bakeoff_v1`, `caption_bakeoff.py` outputs in `/projects/$USER/agai/bakeoff/<run>/` → `/projects/$USER/agai/bakeoff/<run>/judge/<judge>.<mode>.jsonl` |

### v1's report writer, retrained on split v1 (manuscript 2.7, Table S16)

| File | Purpose | Environment | Inputs → outputs (defaults) |
|---|---|---|---|
| [`filter_stage2_records.py`](filter_stage2_records.py) | splits the writer's 2,159 instruction records by split v1: training photographs to training records, validation to validation, test and unmatched records dropped | stdlib | `--records` (e.g. `plant_diagnostic/datasets/stage2_train_7class_fixed.json`), `--image-root`, `--out-dir` (all required; the notebook's run used `plant_diagnostic/datasets/split_v1`), `--split evaluation/split_v1.json`, `--old-root /data/AGAI/MiniGPT-4` → `<out-dir>/stage2_train.json`, `stage2_val.json`, `stage2_filter_report.json` |
| [`detect_parts.py`](detect_parts.py) | visible plant parts of each photograph with v1's RF-DETR Small checkpoint and v1's per-part thresholds (resumable) | `rfdetr` (inferred) | `--records` and/or `--images`, `--checkpoint rfdetr_parts/output/checkpoint_best_total.pth`†, `--root .` → `--out` (required; docstrings: `evaluation/parts_stage2_trainval.json`, `evaluation/parts_report_test.json`) |
| [`build_stage2_prompts.py`](build_stage2_prompts.py) | gives the training records v1's inference prompt from `demo_v5.py`, with the detected part lists or `--no-parts`; also the module that provides `prompt()`, `PLANT_PARTS` and `CANON` | stdlib | `--records`, `--out` (both required), `--parts` (`detect_parts.py` output) → e.g. `plant_diagnostic/datasets/split_v1/stage2_train_prompted.json` |
| [`minigpt_val_loss.py`](minigpt_val_loss.py) | validation loss of every epoch checkpoint of a MiniGPT-v2 run; copies the best one to `checkpoint_best.pth` in the run folder | `minigptv` | `--name`, `--run`, `--records` (all required; `plant_diagnostic/datasets/split_v1/stage2_val_prompted.json`†) → `evaluation/minigpt_<name>_val_loss.json`, `<run>/checkpoint_best.pth` |
| [`minigpt_reports.py`](minigpt_reports.py) | v1-style reports from a MiniGPT-v2 checkpoint on the report test set, with and without the detector's part lists, including v1's consistency-check regeneration | `minigptv` | `--name`, `--ckpt`, `--parts` (all required; e.g. `evaluation/parts_report_test.json`†), `--test-set plant_diagnostic/datasets/split_v1/report_test_set.json`† → `evaluation/reports_<name>.json` (resumable) |
| [`score_reports.py`](score_reports.py) | scores report sets for the detector on/off ablation: part-hallucination rate, condition consistency, regeneration rate, bootstrap intervals, McNemar; scores the files on the command line as soon as it is imported | stdlib | positional `minigpt_reports.py` outputs → `<input>_scores.json` beside each input |
| [`score_reports_exploratory.py`](score_reports_exploratory.py) | post hoc breakdowns of the same report sets with `score_reports.py`'s rules (without soil, per part, the NOT VISIBLE list disobeyed) | stdlib | positional `minigpt_reports.py` outputs → stdout |

### v1 versus v2 (manuscript 3.9)

| File | Purpose | Environment | Inputs → outputs (defaults) |
|---|---|---|---|
| [`latency_v1.py`](latency_v1.py) | v1 as deployed in `demo_v5.py`, step by step on one GPU: MiniGPT-v2 encoding, ResNet-50 and the gate, RF-DETR through `grounding/detector.py`'s subprocess, the report, the consistency check and one regeneration; on the 49 seeded split v1 test photographs shared by the `latency_*.py` scripts | `minigptv` | `--resnet plant_diagnostic/models/resnet_strawberry.pth`†, `--split-json evaluation/split_v1.json`, the MiniGPT-v2 weights† named in `eval_configs/minigptv2_eval.yaml` (or `--writer-ckpt`; the detector checkpoint through `RFDETR_CHECKPOINT`) → `--out evaluation/latency_v1.json` |
| [`latency_student.py`](latency_student.py) | latency of the distilled Qwen3-VL-8B student under vLLM on the same 49 photographs, one request at a time and then batched; the label in each request is the classifier's prediction from `--labels` | `vllm` | `--model` (required; merged student), `--labels evaluation/latency_siglip2.json`†, `--split-json evaluation/split_v1.json` → `--out evaluation/latency_student.json` |
| [`latency_summary.py`](latency_summary.py) | the v1 versus v2 latency and GPU-memory table from the four `latency_*.py` outputs | `vision` (stdlib) | no options; `evaluation/latency_v1.json`†, `latency_rfdetr.json`†, `latency_siglip2.json`†, `latency_student.json`† → `evaluation/paper_data/latency_v1_v2.json`, Markdown on stdout |
| [`make_faithfulness_claims.py`](make_faithfulness_claims.py) | rebuilds the blind claim set of the v1 versus v2 faithfulness ratings (one claim per report sentence, shuffled per photograph with a fixed seed) and its key; `--check` compares with the committed set | stdlib | `--v1 evaluation/paper_data/latency_v1.json`, `--v2 evaluation/paper_data/latency_student.json`, `--claims evaluation/paper_data/faithfulness_v1v2_claims.json` → stdout; `--claims-out FILE`, `--key-out FILE`; `--seed-prefix` and `--v1` build the claim set of a further rating round (round 2: `faith-v1splitv1-v2`) |
| [`score_faithfulness.py`](score_faithfulness.py) | pre-registered scoring of the blind claim ratings: unsupported-claim rate per system with bootstrap intervals and a Wilcoxon test, McNemar on reports, a per-part breakdown, kappa when two raters rated | stdlib | positional `CLAIMS KEY RATINGS` (committed in `evaluation/paper_data/`), `--rater`, `--second-rater`, `--complete-only` → stdout (committed runs: `evaluation/paper_data/faithfulness_v1v2_scores*.txt`) |

## Running

The scripts open every path relative to the working directory, as in the cluster's working copy. Build that folder
with [`../infra/make_workdir.sh`](../infra/make_workdir.sh) (details in
[`../infra/README.md`](../infra/README.md#working-copy)):

```bash
bash v2/infra/make_workdir.sh /projects/$USER/agai-work    # once, from the root of a clone of this repository
source /projects/$USER/agai/agai_env.sh    # cluster variables, then a cd to /projects/$USER/agai/MiniGPT-4
cd /projects/$USER/agai-work
P=evaluation/paper_data
python3 cluster/vlm/make_faithfulness_claims.py --check
python3 cluster/vlm/score_faithfulness.py $P/faithfulness_v1v2_claims.json $P/faithfulness_v1v2_key.json \
    $P/faithfulness_v1v2_ratings.json --complete-only        # the primary analysis, faithfulness_v1v2_scores.txt
cp $P/reports_splitv1.json . && python3 cluster/vlm/score_reports.py reports_splitv1.json    # Table S16
/projects/$USER/agai/envs/vision/bin/python cluster/vlm/score_honesty.py \
    /projects/$USER/agai/annotations/v1_Qwen3-VL-32B-Instruct/test_both.jsonl
sbatch --dependency=afterok:<teacher job> cluster/slurm/train_vlm_reports.sbatch
sbatch -N 5 --export=ALL,VLLM_SCRIPT=cluster/vlm/judge_claims.py cluster/slurm/vllm_ray.sbatch \
    Qwen3-VL-235B-A22B-Instruct-AWQ 16,20,20,19,19 --modes zoom --max-model-len 8192 --gpu-mem 0.95 --enforce-eager
```

The first three commands need only committed files, and their output matches the committed results
(`score_reports.py` is run on a copy because it writes `<input>_scores.json` beside its input). A Slurm job runs in
`/projects/$USER/agai/MiniGPT-4` whatever folder it was submitted from; see
[`../slurm/README.md`](../slurm/README.md#where-jobs-run).

## Dependencies

- **Sibling modules** (each importer puts its own folder on `sys.path`, so they must stay together):
  `annotation_spec.py` is imported by `annotate_vllm.py`, `caption_bakeoff.py`, `consensus_dx.py`,
  `latency_student.py`, `resolve_dx.py` and `train_vlm_reports.py`; `score_reports.py` by
  `score_reports_exploratory.py`. `latency_v1.py` and `minigpt_reports.py` import `build_stage2_prompts.py` after
  putting `<working directory>/cluster` on `sys.path` (the flat layout, which the working copy provides); outside the
  working copy only the running script's own folder holds it, so it stays beside them.
- **v1 modules**, imported from the working directory: `latency_v1.py` imports `grounding.detector.run_rfdetr`,
  `grounding.inference.load_minigpt`, `knowledge_graph.rag_retriever.DiseaseRAG` and `resnet_classifier` (including
  its private `_DEVICE` and `_tfm`); `minigpt_reports.py` imports `load_minigpt` and `DiseaseRAG`;
  `minigpt_val_loss.py` imports `load_minigpt`. `load_minigpt` loads `minigpt4/` and reads
  `eval_configs/minigptv2_eval.yaml`; `run_rfdetr` starts `rfdetr_parts/detect_images.py` with the interpreter that
  `RFDETR_PYTHON` names.
- **v1 code run by a job.** [`../slurm/train_minigpt_splitv1.sbatch`](../slurm/train_minigpt_splitv1.sbatch) runs
  v1's `train.py` with `train_configs/minigptv2_strawberry_splitv1.yaml` on the records that `filter_stage2_records.py`,
  `detect_parts.py` and `build_stage2_prompts.py` prepare; `minigpt_val_loss.py`, `minigpt_reports.py` and
  `score_reports.py` follow.
- **Text copied from v1** (no import): `build_stage2_prompts.prompt()`, and the gate, prompts and regeneration prompt
  in `latency_v1.py` and `minigpt_reports.py`, come from `demo_v5.py`; `detect_parts.py`'s thresholds from
  `rfdetr_parts/detect_images.py`.
- **Files from and for other studies.** `latency_student.py` reads the output of `classifier/latency_siglip.py`;
  `latency_summary.py` reads those of `classifier/latency_siglip.py`, `detector/latency_rfdetr.py`, `latency_v1.py`
  and `latency_student.py`; `zeroshot_qwen3vl.py` reads the outputs of `benchmark/repro_resnet_v4.py` and
  `benchmark/clean_eval_sets.py`. The outputs of `eval_vlm_scores.py` feed `classifier/ensemble.py` and
  `benchmark/error_sheet.py`, those of `annotate_vllm.py` feed `benchmark/label_audit.py`, and those of
  `train_vlm_lora.py` and `zeroshot_qwen3vl.py` feed `benchmark/leaderboard.py` and `benchmark/compare_results.py`.

## Known problems

- **`latency_summary.py`** reads the four latency files from the top level of `evaluation/`; the committed copies are
  in `evaluation/paper_data/`, so it stops at the first file. It writes `evaluation/paper_data/latency_v1_v2.json`,
  which in the working copy overwrites the committed file through the link.
- **`latency_student.py`** finds no `evaluation/latency_siglip2.json` in this repository (the committed copy is
  `evaluation/paper_data/latency_siglip2.json`). Without it the student is given the dataset label, and the script
  only prints a note. To reproduce job 72394 from committed files, pass
  `--labels evaluation/paper_data/latency_siglip2.json`. On the cluster the file was at the default path (the same job
  wrote it), and the committed `latency_student.json` shows the classifier's prediction as the given label for all 49
  photographs.
- **Defaults and the repository.** `latency_v1.py`, `latency_student.py`, `minigpt_val_loss.py` and
  `minigpt_reports.py` write to the top level of `evaluation/`; the files the manuscript uses were copied by hand to
  [`evaluation/paper_data/`](../../evaluation/paper_data/). `score_reports.py` run on a committed report file there
  overwrites its committed `_scores.json`.
- **Inputs never committed.** The writer's split v1 records (`plant_diagnostic/datasets/split_v1/`, including
  `report_test_set.json`), the part lists of `detect_parts.py`, `plant_diagnostic/models/resnet_strawberry.pth`, v1's
  RF-DETR and MiniGPT-v2 checkpoints, `evaluation/clean_eval_sets.json` and
  `evaluation/holdout/repro_resnet_v4_hamilton.json`, and everything under `/projects/$USER/agai/` (annotations,
  bake-off and consensus runs, adapters, models).
- **Synthetic references.** `resolve_dx.py`'s `--synth-glob` default matches nothing in this repository: the lab's GPT
  images are not in `v1/plant_diagnostic/data/train_aug/`.
- **Photographs.** Split v1 and the bake-off set list photographs of which 292 and 11 are not in this repository (see
  [`../benchmark/README.md`](../benchmark/README.md#known-problems)).
- **Lab-server paths.** `grounding/config.py` (`PROJECT_ROOT`, where `run_rfdetr` runs the detector) and
  `eval_configs/minigptv2_eval.yaml` (the MiniGPT-v2 and Llama weights) name `/data/AGAI/MiniGPT-4/...`; see
  [Paths](../../v1/README.md#paths) in the v1 README. `latency_v1.py`, `minigpt_reports.py` and `minigpt_val_loss.py`
  go through them.
- **Writes into v1.** In the working copy `plant_diagnostic/` is a link to `v1/plant_diagnostic/`, so
  `filter_stage2_records.py` and `build_stage2_prompts.py` with the notebook's paths write into this repository's v1
  folder.
- **Stale text.** The docstrings of `build_stage2_prompts.py`, `detect_parts.py`, `latency_student.py`,
  `latency_summary.py`, `latency_v1.py`, `minigpt_reports.py`, `minigpt_val_loss.py`, `score_reports.py` and
  `score_reports_exploratory.py` show flat names (`python cluster/latency_v1.py`), which work in the working copy.
  `score_honesty.py`'s docstring names `annotate_vllm.py --labels both`; the option is `--tasks` with mode `both`.
  `judge_claims.py`'s docstring mentions scoring the judges against human ratings; that code is not among the v2
  scripts.
