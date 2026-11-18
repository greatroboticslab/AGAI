# `v1/plant_diagnostic/datasets/`: instruction data of the v1 report writer

The image-report records that v1's MiniGPT-v2 report writer was fine-tuned on (seven classes) and a small
question-answer set, first committed in October 2025 (v1, 2025 to April 2026). The v2 revival rebuilt its split v1
training records from the 2,159-record file.

## Contents

| File | Records | Contents |
|---|---:|---|
| [`stage2_train_7class_fixed.json`](stage2_train_7class_fixed.json) | 2,159 | JSON array (stored with Git LFS), one record per photograph of [`../data/train_aug/`](../data/train_aug/): `image` and `conversations`, a human turn `<Img><ImageHere></Img>` and the report, which starts with `Diagnosis: <label>.` |
| [`stage2_train_7class.fixed.jsonl`](stage2_train_7class.fixed.jsonl) | 2,159 | the same records, one per line |
| [`minigpt_vqa_7class.jsonl`](minigpt_vqa_7class.jsonl) | 99 | question-answer records on 99 `train_aug/` photographs (14 or 15 per class): `question`, `answer`, `label_id`, `label_text`, `conversations`; no code in the repository reads it |

Records per class in the 2,159: drought 300, frost 110, gray_mold 607, healthy 325, overwatered 298, root_rot 293,
white_mold 226, the same as the photographs in `train_aug/`. Photo paths are lab-server paths
(`/data/AGAI/MiniGPT-4/plant_diagnostic/data/train_aug/<class>/<file>`); every one of the 2,159 photographs is in
`../data/train_aug/` under the same class and file name.

## How the reports were made

The generator, [`generate_json_openai.py`](../scripts/gptscripts/json_generation/generate_json_openai.py), sends
each photograph with its class label (the name of its folder) to an OpenAI vision model (default `o3`) and appends one
record per photograph, in the format above, to a JSONL file;
[`convert_jsonl_to_json.py`](../scripts/convert_jsonl_to_json.py) turns that file into a JSON array. None of the
2,159 reports is the generator's templated fallback text. One difference: the committed generator starts every report
with the folder name (`Diagnosis: gray_mold.`), while the 2,159 reports start with a display name
(`Diagnosis: Gray mold.`, `Diagnosis: Root rot.`); the step that changed this is not in the repository.

## What uses it

- **v1.** `stage2_train_7class_fixed.json` is the training set in `v1/train_configs/minigptv2_strawberry.yaml` and
  `_v2.yaml`, the default in `v1/minigpt4/configs/datasets/strawberry_diagnostic.yaml` and
  `multitask_conversation/default.yaml`, and appears in `v1/eval_configs/minigptv2_eval.yaml`,
  `v1/hallucination_evaluation/` and `v1/tests/test_model_loading.py`, by its lab-server path (or, in
  `hallucination_evaluation/config.py`, relative to `v1/`).
- **v2.** `v2/vlm/filter_stage2_records.py` splits these records by split v1 (records of training photographs for
  training, of validation photographs for checkpoint selection, test photographs dropped) and writes them to
  `datasets/split_v1/`, which is not in the repository; `v2/benchmark/minigpt_train_overlap.py` reads the file to find
  which test photographs v1's writer was trained on. Both open it relative to the working directory (`v1/`, or the
  folder built by `v2/infra/make_workdir.sh`).

## Known problems

- **Lab-server paths.** Training on another machine needs the `/data/AGAI/MiniGPT-4` prefix replaced in the configs (or
  a link from that path to `v1/`). `LlavaConversationDataset` drops records whose photograph it cannot find and only
  prints the number, so a wrong image root trains on fewer records without an error
  ([`../../minigpt4/README.md`](../../minigpt4/README.md)).
