#!/usr/bin/env python3
"""
Train RF-DETR Small on the strawberry parts dataset.

Reads hyperparameters from a YAML config file, then launches training.
RF-DETR's built-in PyTorch Lightning stack handles per-epoch evaluation,
Rich-formatted per-class metrics, and TensorBoard logging.
"""

import argparse
from pathlib import Path

import yaml
from rfdetr import RFDETRSmall


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


TRAIN_PARAM_KEYS = {
    "dataset_dir", "output_dir", "epochs", "batch_size", "grad_accum_steps",
    "lr", "lr_encoder", "weight_decay", "resolution",
    "use_ema", "early_stopping", "early_stopping_patience", "early_stopping_min_delta",
    "checkpoint_interval", "tensorboard", "log_per_class_metrics", "progress_bar",
    "seed", "lr_scheduler", "warmup_epochs", "lr_min_factor", "drop_path",
}


def main():
    parser = argparse.ArgumentParser(description="Train RF-DETR Small")
    parser.add_argument("--config", default=str(Path(__file__).parent / "rfdetr_config.yaml"),
                        help="Path to YAML config (default: rfdetr_config.yaml)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    print("=" * 70)
    print("  RF-DETR Small — Strawberry Part Detection")
    print("=" * 70)
    for k, v in cfg.items():
        print(f"  {k}: {v}")
    print("=" * 70, flush=True)

    model = RFDETRSmall()

    train_kwargs = {k: v for k, v in cfg.items() if k in TRAIN_PARAM_KEYS}

    print("\nStarting training...\n", flush=True)
    model.train(**train_kwargs)

    best_ckpt = Path(cfg.get("output_dir", "output")) / "checkpoint_best_total.pth"
    if best_ckpt.exists():
        print(f"\nBest checkpoint saved to: {best_ckpt}")
    print("Done.")


if __name__ == "__main__":
    main()
