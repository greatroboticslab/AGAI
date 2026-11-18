# `v1/plant_diagnostic/jsondatasets/`: earlier five-class instruction data

Image-report records for five classes (drought, frost, healthy, overwatered, root rot), committed on 29 September 2025,
before the gray mold and white mold photographs were added on 4 October 2025 (v1, 2025 to April 2026). The
seven-class records that trained v1's report writer are in [`../datasets/`](../datasets/). Photo paths here are
`data/train/<class>/<file>`, relative to `v1/plant_diagnostic/`, and every listed photograph is in
[`../data/train/`](../data/train/).

## Contents

| File | Records | Contents |
|---|---:|---|
| [`strawberry_stage2.json`](strawberry_stage2.json) | 617 | training records (`image`, `conversations`): drought 57, frost 93, healthy 279, overwatered 70, root rot 118 |
| [`strawberry_stage2_val.json`](strawberry_stage2_val.json) | 68 | validation records; no photograph in common with the 617 |
| [`strawberry_stage2_train_v3.jsonl`](strawberry_stage2_train_v3.jsonl) | 685 | the 617 and the 68 records together, one per line |
| [`strawberry_v3_train.jsonl`](strawberry_v3_train.jsonl) | 617 | the records of `strawberry_stage2.json`, one per line |
| [`strawberry_v3_train_pairs.jsonl`](strawberry_v3_train_pairs.jsonl) | 617 | the same 617 photographs with the report as `text` (`image`, `text`) |
| [`holdout_val_list.json`](holdout_val_list.json) | 61 | 61 of the 617 training records |
| [`how_these_were_made`](how_these_were_made) | | one line: the pairs were made with OpenAI's o3 through the API, "check generate_stage2_jsonl.py"; that file is empty, and the only generator in the repository is [`../scripts/gptscripts/json_generation/generate_json_openai.py`](../scripts/gptscripts/json_generation/generate_json_openai.py) |

The `.json` files are stored with Git LFS.

## What uses it

- `v1/minigpt4/configs/datasets/strawberry_diagnostic.yaml` names `strawberry_stage2_val.json` (by its lab-server path)
  as the default validation records, with the image folder `plant_diagnostic/data/holdout`. v1's training configs set
  only the training records, so this default stays in their merged config.
- The smoke test of [`../src/strawberry_stage2_loader.py`](../src/strawberry_stage2_loader.py) is written for
  `strawberry_stage2.json` but names it `datasets/strawberry_stage2.json`.

## Known problems

- With the image folder `plant_diagnostic/data/holdout`, `LlavaConversationDataset` finds the `data/train/...`
  photographs of `strawberry_stage2_val.json` only when the working directory is `v1/plant_diagnostic/`, and drops the
  records it cannot find.
- `holdout_val_list.json` holds training records: its 61 photographs are in `data/train/`, and all 61 records are in
  `strawberry_stage2.json`.
