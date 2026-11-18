# `v2/infra/requirements/`: package lists of four cluster environments, 30 October 2026

Part of v2, the lab's 2026 revival of AGAI (October-November 2026). `pip freeze` output of four of the Python
environments under `/projects/$USER/agai/envs/` on the Hamilton cluster, taken on 30 October 2026 (lab notebook,
section 2026-10-30) and committed on 31 October 2026. They record the exact package versions behind the revival's
results.

## Files

| File | Environment | Built by | Key versions | Used by |
|---|---|---|---|---|
| [`minigptv.freeze.2026-10-30.txt`](minigptv.freeze.2026-10-30.txt) | `envs/minigptv`, system Python 3.9 (3.9.23 in the notebook) | [`../hamilton_setup_envs.sh`](../hamilton_setup_envs.sh) | torch 2.5.1+cu121, torchvision 0.20.1+cu121, transformers 4.45.2, peft 0.13.2, bitsandbytes 0.43.3, timm 0.6.13, gradio 4.44.1 | v1 code: MiniGPT-v2 (`vlm/latency_v1.py`, `vlm/minigpt_reports.py`, `vlm/minigpt_val_loss.py`, `benchmark/minigpt_diagnose.py`, v1's `train.py`) and the ResNet scripts |
| [`rfdetr.freeze.2026-10-30.txt`](rfdetr.freeze.2026-10-30.txt) | `envs/rfdetr`, portable Python 3.11 (3.11.17 in the notebook) | [`../hamilton_setup_envs.sh`](../hamilton_setup_envs.sh), then `pip install "rfdetr[train,loggers]==1.11.2"` by hand (lab notebook) | rfdetr 1.11.2, torch 2.14.1, torchvision 0.29.1, transformers 5.20.2, pycocotools 2.0.11 | [`../../detector/`](../../detector/), `vlm/detect_parts.py`, v1's detector subprocess through `RFDETR_PYTHON` |
| [`vision.freeze.2026-10-30.txt`](vision.freeze.2026-10-30.txt) | `envs/vision`, portable Python 3.11 | [`../setup_vision_env.sh`](../setup_vision_env.sh) | torch 2.9.1+cu128, torchvision 0.24.1+cu128, timm 1.0.30, open_clip_torch 3.3.0, diffusers 0.41.0, transformers 5.20.2, scikit-learn 1.9.1 | [`../../classifier/`](../../classifier/) probes, fine-tunes and evaluations; the scoring steps of some Slurm jobs |
| [`vllm.freeze.2026-10-30.txt`](vllm.freeze.2026-10-30.txt) | `envs/vllm`, Python version not recorded | no script in this repository | vllm 0.32.0, torch 2.13.0, transformers 5.19.0, ray 2.59.0 | the vLLM scripts in [`../../vlm/`](../../vlm/), `infra/ray_pp_order.py`, `infra/stage_pp_shard.py` |

There is no committed freeze of `envs/vlm`, `envs/vlm5` or `envs/tools`. `hamilton_setup_envs.sh` writes
`envs/<env>.freeze.txt` on the cluster for each environment it builds; those files are not in the repository.

## Using them

The setup scripts take the `+cu121` and `+cu128` builds of torch from the PyTorch index
(`https://download.pytorch.org/whl/cu121` for `minigptv`, `/whl/cu128` for `vision`), so installing these lists with
`pip install -r` needs the same index, for example:

```bash
/projects/$USER/agai/python/3.11/bin/python3.11 -m venv /projects/$USER/agai/envs/vision
/projects/$USER/agai/envs/vision/bin/pip install --extra-index-url https://download.pytorch.org/whl/cu128 \
    -r v2/infra/requirements/vision.freeze.2026-10-30.txt
```

The lists were taken from the environments as they stood on the cluster; the lab notebook records no rebuild from
them. Hardware and the environments' roles: [Compute](../../../docs/COMPUTE.md). Notebook:
[`../../EXPERIMENT_LOG.md`](../../EXPERIMENT_LOG.md).
