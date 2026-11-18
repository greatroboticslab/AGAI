# `v2/infra/`: cluster setup, job environment, model staging and the working-copy builder

Part of v2, the lab's 2026 revival of AGAI (October-November 2026). Scripts that set up the Hamilton cluster for the
revival (git-lfs, the working copy, public weights, Python environments), the environment every job sources, model
download and staging to node-local NVMe with SHA-256 checks, helpers for multi-node runs, and `make_workdir.sh`, which
rebuilds from this repository the single working folder that the v2 scripts expect.

Hardware, environments and multi-node serving: [Compute](../../docs/COMPUTE.md). Setup runs and the problems they met
are recorded in the lab notebook, [`../EXPERIMENT_LOG.md`](../EXPERIMENT_LOG.md).

## Files

| File | Purpose | Run with | Inputs → outputs |
|---|---|---|---|
| [`agai_env.sh`](agai_env.sh) | environment of every job: `AGAI_ROOT=/projects/$USER/agai`, `PATH` with the standalone git-lfs, pip and torch caches, `RFDETR_PYTHON` (the interpreter that v1's `grounding/detector.py` uses for RF-DETR), secrets from `~/.agai_secrets` when that file exists; its last line is `cd "$AGAI_ROOT/MiniGPT-4"` | `source`, as the copy `/projects/$USER/agai/agai_env.sh` | sets variables, changes the working directory |
| [`ddp_task.sh`](ddp_task.sh) | one rank of a multi-node LoRA run: picks the node's 10.0.0.x interface for NCCL and Gloo, stages the `--model` directory with `stage_model.sh`, then runs `cluster/vlm/train_vlm_lora.py` in `envs/${VLM_ENV:-vlm}` | `srun`, from [`../slurm/train_vlm_lora_multinode.sbatch`](../slurm/train_vlm_lora_multinode.sbatch) | the arguments of `train_vlm_lora.py` |
| [`hamilton_setup_stage1.sh`](hamilton_setup_stage1.sh) | setup stage 1, no lab data needed: standalone git-lfs 3.7.1, a clone of the repository named in the script into `/projects/$USER/agai/MiniGPT-4` with its LFS files (this repository added as remote `lab`), `envs/tools` (system Python 3.9, `gdown`), the MiniGPT-v2 stage-2 checkpoint, EVA-ViT-g and Llama-2-7b-chat-hf; idempotent | bash, by hand | `~/.agai_secrets` with `HF_TOKEN` (gated Llama weights) → `/projects/$USER/agai/{bin,envs,cache,incoming,logs}`, `MiniGPT-4/checkpoints/checkpoint_stage2.pth`, `MiniGPT-4/llama_weights/Llama-2-7b-chat-hf/`, `cache/torch/hub/checkpoints/eva_vit_g.pth` |
| [`hamilton_setup_envs.sh`](hamilton_setup_envs.sh) | setup stage 2: a portable Python 3.11 and the venvs `minigptv` (system Python 3.9; the pins for v1's MiniGPT-v2 code are explained in the header), `rfdetr`, `vlm` (transformers 4.57) and `vlm5` (transformers 5.20.2, for Qwen3.8); idempotent | bash, by hand | network → `/projects/$USER/agai/python/3.11/`, `envs/{minigptv,rfdetr,vlm,vlm5}/`, `envs/<env>.freeze.txt` |
| [`hf_fetch.sh`](hf_fetch.sh) | downloads a Hugging Face model repository with curl and checks LFS files by SHA-256 (no `huggingface_hub`, whose file locks hang on the NFS `/projects`); resumes partial files | bash | positional `REPO DEST [SKIP_REGEX]`, `HF_TOKEN` from `~/.agai_secrets` for gated models → `DEST/` |
| [`make_manifest.py`](make_manifest.py) | writes `SHA256SUMS` for the weight files of a model directory (reads the whole model once: a CPU job) | `python3` (stdlib) | positional `MODEL_DIR` → `MODEL_DIR/SHA256SUMS` |
| [`make_workdir.sh`](make_workdir.sh) | builds the single working folder the v2 scripts were written for, out of symbolic links into this repository; see [Working copy](#working-copy) | bash, from a clone | positional `DEST` (must not exist) → `DEST/` |
| [`ray_pp_order.py`](ray_pp_order.py) | prints the node IPs of a running Ray cluster in the order in which vLLM 0.32 gives them pipeline ranks | `envs/vllm`, from [`../slurm/vllm_ray.sbatch`](../slurm/vllm_ray.sbatch) | positional `DRIVER_IP`, `RAY_ADDRESS` → stdout |
| [`requirements/`](requirements/) | `pip freeze` of four environments on 30 October 2026 (see [`requirements/README.md`](requirements/README.md)) | | |
| [`setup_vision_env.sh`](setup_vision_env.sh) | creates `envs/vision` (torch 2.9.1, torchvision 0.24.1, transformers ≥ 4.57, diffusers, timm, open_clip, scikit-learn) for backbone probes, fine-tunes and image generation | bash, on the login node | needs the `agai_env.sh` copy and the Python 3.11 of stage 2 → `envs/vision/` |
| [`stage_model.sh`](stage_model.sh) | copies a model directory from `/projects` (NFS) to `/scratch/$USER/models/` on the node, skips files already there, checks the copies against `SHA256SUMS` with `verify_staged.py` (copies a mismatching file once more), prints the local path | bash (system `python3` for the check) | positional `SRC` → `/scratch/$USER/models/<name>/`, its path on stdout |
| [`stage_pp_shard.py`](stage_pp_shard.py) | stages only the weight files one pipeline-parallel rank needs (plus every non-layer tensor), checks them, and writes a per-node index | `envs/vllm`, from [`../slurm/vllm_ray.sbatch`](../slurm/vllm_ray.sbatch) | `--src`, `--partition`, `--rank` (all required), `--dst-root /scratch/$USER/models` → staged files, `model.safetensors.index.json`, `STAGED_FOR_RANK` |
| [`verify_staged.py`](verify_staged.py) | checks staged files against `SHA256SUMS`, reading each from disk after evicting it from the page cache (Linux only); `--fix` deletes mismatching copies | `python3` (stdlib) | positional `SRC DST`, `--files`, `--fix` → exit status 0 or 1 |

## Working copy

On the cluster every v2 script ran from the root of one working copy, `/projects/$USER/agai/MiniGPT-4`: the v1 tree,
with `evaluation/` inside it and the v2 scripts in `cluster/`. The scripts open `evaluation/...`,
`plant_diagnostic/...`, `rfdetr_parts/...`, v1 configs and v1 modules (`resnet_classifier`, `grounding`,
`knowledge_graph`) relative to the working directory.

- **`agai_env.sh`** is used as a copy, `/projects/$USER/agai/agai_env.sh`; no script in the repository makes the copy.
  All twelve files in [`../slurm/`](../slurm/) and `setup_vision_env.sh` source it, and its last line,
  `cd "$AGAI_ROOT/MiniGPT-4"`, makes `/projects/$USER/agai/MiniGPT-4` the working directory of every job (six job files
  also `cd` there themselves; see [`../slurm/README.md`](../slurm/README.md#where-jobs-run)).
- **`make_workdir.sh DEST`** rebuilds such a folder from this repository. `DEST` gets a link to every entry of `v1/`
  except `README.md` and `evaluation/`; a real `evaluation/` with links to the entries of `evaluation/` and
  `v1/evaluation/`; a real `cluster/` with `cluster/<study>/` links to the folders of `v2/`, a flat link
  `cluster/<file>` for every file in them (the lab notebook's names) and `cluster/EXPERIMENT_LOG.md`; and `docs/`. It
  finds the repository from its own location, refuses a `DEST` that exists, and needs a file system with symbolic
  links (Linux or macOS). The links are absolute paths, so a moved clone needs a new `DEST`.

```bash
bash v2/infra/make_workdir.sh /projects/$USER/agai-work    # from the root of a clone of this repository
source /projects/$USER/agai/agai_env.sh    # cluster variables, then a cd to /projects/$USER/agai/MiniGPT-4
cd /projects/$USER/agai-work
python3 cluster/vlm/score_faithfulness.py --help    # study path, as in these READMEs
python3 cluster/score_faithfulness.py --help        # flat path, as in the lab notebook
```

Writes in `DEST`: a new file that a script writes into `evaluation/` stays in `DEST/evaluation/` (copy the ones to keep
into `evaluation/paper_data/`). A write to a path that is a link changes this repository: anything under
`evaluation/paper_data/`, `evaluation/holdout/`, `plant_diagnostic/` or `rfdetr_parts/`, and the linked files
`evaluation/split_v1.json` and `evaluation/label_audit.csv`.

Slurm jobs submitted from `DEST` still run in `/projects/$USER/agai/MiniGPT-4`. For a job to use a folder built by
`make_workdir.sh`, the folder has to be at that path, or the last line of the `agai_env.sh` copy and the `cd` lines
of the six job files have to name it.

## Setting up an account

The order the scripts assume, all under `/projects/$USER/agai`:

1. `hamilton_setup_stage1.sh` (git-lfs, clone, `envs/tools`, public weights); needs `~/.agai_secrets` with `HF_TOKEN`.
2. `hamilton_setup_envs.sh` (Python 3.11, `envs/minigptv`, `envs/rfdetr`, `envs/vlm`, `envs/vlm5`).
3. A copy of `agai_env.sh` at `/projects/$USER/agai/agai_env.sh`.
4. `setup_vision_env.sh` (`envs/vision`).
5. `envs/vllm`: no script in the repository creates it;
   [`requirements/vllm.freeze.2026-10-30.txt`](requirements/vllm.freeze.2026-10-30.txt) lists its packages.
6. Models: `hf_fetch.sh REPO /projects/$USER/agai/models/<name>`, then `make_manifest.py` once per model (a CPU
   job), so `stage_model.sh` and `stage_pp_shard.py` can check every staged copy.

```bash
mkdir -p /projects/$USER/agai/logs
nohup setsid bash v2/infra/hamilton_setup_stage1.sh > /projects/$USER/agai/logs/setup_stage1.log 2>&1 < /dev/null &
bash v2/infra/hf_fetch.sh Qwen/Qwen3-VL-8B-Instruct /projects/$USER/agai/models/Qwen3-VL-8B-Instruct '\.bin$'
python3 v2/infra/make_manifest.py /projects/$USER/agai/models/Qwen3-VL-8B-Instruct
M=$(bash v2/infra/stage_model.sh /projects/$USER/agai/models/Qwen3-VL-8B-Instruct)    # on a compute node
```

## Dependencies

- **Sibling module.** `stage_pp_shard.py` imports `verify_staged.py` after putting its own folder on `sys.path`, and
  `stage_model.sh` runs `verify_staged.py` from its own folder: these three files must stay together.
- **Paths relative to the working directory.** `ddp_task.sh` calls `cluster/infra/stage_model.sh` and
  `cluster/vlm/train_vlm_lora.py`; the Slurm files call `cluster/infra/ray_pp_order.py`, `stage_pp_shard.py` and
  `stage_model.sh` (`latency_v1_v2.sbatch` by the flat name `cluster/stage_model.sh`).
- **v1.** `RFDETR_PYTHON` is read by v1's [`grounding/detector.py`](../../v1/grounding/detector.py);
  `hamilton_setup_envs.sh` pins `envs/minigptv` for v1's MiniGPT-v2 code (its header gives the reasons).

## Known problems

- **Where jobs run.** As above, every job runs in `/projects/$USER/agai/MiniGPT-4`. For the account that ran the
  revival, that folder is the original working copy, which the reorganization left unchanged (lab notebook, entry (ad)).
- **Stage 1 predates this layout.** It makes `/projects/$USER/agai/MiniGPT-4` a clone and puts `checkpoints/` and
  `llama_weights/` at the clone's root; `make_workdir.sh` links only the entries of `v1/`, and `v1/` has no
  `checkpoints/`.
- **Environments not reproducible from the repository alone.** `envs/vllm` has a freeze file and no setup script;
  `envs/vlm`, `envs/vlm5` and `envs/tools` have setup lines and no committed freeze. `envs/tools` gets only `gdown`,
  while `synthetic/import_synth.py` and `synthetic/build_synth_split.py` run in it with `imagehash` and Pillow, which no
  script installs; `imagehash` is in none of the four freeze files.
- **Unpinned versions.** `hamilton_setup_envs.sh` downloads the latest portable CPython 3.11 release and installs an
  unpinned torch in `envs/vlm5`; `setup_vision_env.sh` leaves transformers, diffusers, timm and open_clip unpinned. An
  environment built today can differ from the freezes in [`requirements/`](requirements/).
- **Gradio.** `hamilton_setup_envs.sh` installs Gradio 4.44.1 into `envs/minigptv` because, its header says,
  `demo_v5.py` uses `gr.Image(sources=...)`; the committed [`v1/demo_v5.py`](../../v1/demo_v5.py) uses Gradio 3's
  `gr.Image(source=...)` (see the v1 README).
- **Cluster only.** `verify_staged.py` needs Linux (`posix_fadvise`, `os.uname`); `stage_pp_shard.py` reads `USER`
  from the environment; secrets come from `~/.agai_secrets`, outside the repository (no secret is stored here).
