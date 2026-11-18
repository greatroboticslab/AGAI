# `v1/llama_weights/`: where the LLaMA-2-7B-chat weights go

Part of v1, the original system (2025 to April 2026). Expected location of `Llama-2-7b-chat-hf`, the language model
inside the MiniGPT-v2 report writer; the 2026 revival used the same weights to run v1 and to retrain the writer. The
weights are not in the repository: they are Meta's gated release on Hugging Face (`meta-llama/Llama-2-7b-chat-hf`),
and this folder is listed in [`.gitignore`](../../.gitignore) (`llama_weights/`).

## Files

| Entry | What it is |
|---|---|
| `Llama-2-7b-chat-hf` | a submodule entry (git mode 160000, added on 29 September 2025) that records commit `f5db02db724555f92da89c216ac04704f23d4590`. There is no `.gitmodules` file, so it names no repository: a clone gets an empty folder here, and `git submodule` commands cannot fetch it |

## Setup

Put the files of the Hugging Face repository (the config, tokenizer files including `tokenizer.model`, and the
`*.safetensors` weights) into `Llama-2-7b-chat-hf/`; access has to be requested on Hugging Face first. MiniGPT-v2
loads the folder with `LlamaTokenizer.from_pretrained` and `LlamaForCausalLM.from_pretrained`, in 8-bit when a config
sets `low_resource: true` ([`../minigpt4/models/base_model.py`](../minigpt4/models/base_model.py)).

On the cluster, [`../../v2/infra/hamilton_setup_stage1.sh`](../../v2/infra/hamilton_setup_stage1.sh) downloaded every
file of that repository except the `*.bin` shards (the same weights as the safetensors) and dotfiles into this folder
of the cluster checkout, with an access token, and checked the SHA-256 of the large files.

## Dependencies

The configs name the folder by absolute path ([Paths](../README.md#paths)):

- `/data/AGAI/MiniGPT-4/llama_weights/Llama-2-7b-chat-hf` (the lab server):
  [`../eval_configs/minigptv2_eval.yaml`](../eval_configs/minigptv2_eval.yaml),
  [`../train_configs/minigptv2_strawberry.yaml`](../train_configs/minigptv2_strawberry.yaml),
  [`../train_configs/minigptv2_strawberry_v2.yaml`](../train_configs/minigptv2_strawberry_v2.yaml), both files in
  [`../configs/models/`](../configs/models/),
  [`../minigpt4/configs/models/minigpt_v2.yaml`](../minigpt4/configs/models/minigpt_v2.yaml) and
  [`../tests/test_model_loading.py`](../tests/test_model_loading.py);
- `/projects/.../agai/MiniGPT-4/llama_weights/Llama-2-7b-chat-hf` (the cluster):
  [`../train_configs/minigptv2_strawberry_splitv1.yaml`](../train_configs/minigptv2_strawberry_splitv1.yaml).
