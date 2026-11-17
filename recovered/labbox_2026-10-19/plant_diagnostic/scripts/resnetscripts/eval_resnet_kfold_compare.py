#!/usr/bin/env python3
"""Stratified k-fold evaluation for fixed strawberry ResNet checkpoints.

Assumptions:
    * Folds are evaluation-only partitions of a labeled ImageFolder tree.
    * Checkpoints are already trained; this script does not update weights.
    * Accuracy, macro F1, and per-class metrics are computed from argmax labels.
    * Optional horizontal-flip TTA mirrors the wired app inference path.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms


def build_transform(img_size: int, mode: str) -> transforms.Compose:
    """Build the evaluation transform used by the selected inference mode."""
    if mode == "wired":
        steps = [
            transforms.Resize(int(img_size * 1.15)),
            transforms.CenterCrop(img_size),
        ]
    elif mode == "train":
        steps = [transforms.Resize((img_size, img_size))]
    else:
        raise ValueError(f"Unknown transform mode: {mode}")

    steps.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )
    return transforms.Compose(steps)


def stratified_folds(
    targets: Sequence[int],
    num_classes: int,
    k: int,
    seed: int,
) -> List[List[int]]:
    """Return k validation folds with per-class round-robin assignment."""
    if k < 2:
        raise ValueError("k must be >= 2")

    rng = random.Random(seed)
    per_class = [[] for _ in range(num_classes)]
    for idx, label in enumerate(targets):
        per_class[int(label)].append(idx)

    folds: List[List[int]] = [[] for _ in range(k)]
    for label, indices in enumerate(per_class):
        if len(indices) < k:
            raise ValueError(
                f"Class {label} has only {len(indices)} samples; "
                f"cannot build {k} non-empty stratified folds."
            )
        rng.shuffle(indices)
        for offset, sample_idx in enumerate(indices):
            folds[offset % k].append(sample_idx)

    for fold in folds:
        rng.shuffle(fold)
    return folds


def load_checkpoint(path: Path, device: torch.device) -> Tuple[nn.Module, List[str]]:
    """Load a ResNet-50 checkpoint and return model plus checkpoint classes."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    classes = ckpt.get("classes")
    if not isinstance(classes, (list, tuple)) or not classes:
        meta = ckpt.get("meta", {})
        classes = meta.get("classes") if isinstance(meta, dict) else None
    if not isinstance(classes, (list, tuple)) or not classes:
        raise ValueError(f"Checkpoint lacks a class list: {path}")

    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(classes))
    state = ckpt.get("model", ckpt)
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    return model, list(classes)


def dataset_to_checkpoint_map(
    dataset_classes: Sequence[str],
    checkpoint_classes: Sequence[str],
) -> torch.Tensor:
    """Map ImageFolder class indices to checkpoint class indices."""
    if set(dataset_classes) != set(checkpoint_classes):
        missing = sorted(set(checkpoint_classes) - set(dataset_classes))
        extra = sorted(set(dataset_classes) - set(checkpoint_classes))
        raise ValueError(
            "Class set mismatch: "
            f"missing_in_dataset={missing}, extra_in_dataset={extra}"
        )
    lookup = {name: idx for idx, name in enumerate(checkpoint_classes)}
    return torch.tensor([lookup[name] for name in dataset_classes], dtype=torch.long)


@torch.inference_mode()
def evaluate_fold(
    model: nn.Module,
    loader: DataLoader,
    label_map: torch.Tensor,
    num_classes: int,
    device: torch.device,
    tta: bool,
) -> np.ndarray:
    """Evaluate one fold and return a confusion matrix."""
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    label_map = label_map.to(device)

    for x, y_dataset in loader:
        x = x.to(device, non_blocking=True)
        y_true = label_map[y_dataset.to(device, non_blocking=True)]

        logits = model(x)
        if tta:
            logits = (logits + model(torch.flip(x, dims=[3]))) / 2.0
        y_pred = logits.argmax(dim=1)

        for truth, pred in zip(y_true.cpu().numpy(), y_pred.cpu().numpy()):
            cm[int(truth), int(pred)] += 1
    return cm


def metrics_from_cm(cm: np.ndarray) -> Dict[str, object]:
    """Compute scalar and per-class metrics from a confusion matrix."""
    total = int(cm.sum())
    correct = int(np.trace(cm))
    tp = np.diag(cm)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    precision = np.divide(
        tp,
        tp + fp,
        out=np.zeros_like(tp, dtype=float),
        where=(tp + fp) != 0,
    )
    recall = np.divide(
        tp,
        tp + fn,
        out=np.zeros_like(tp, dtype=float),
        where=(tp + fn) != 0,
    )
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(precision, dtype=float),
        where=(precision + recall) != 0,
    )
    return {
        "accuracy": float(correct / total) if total else 0.0,
        "macro_f1": float(np.mean(f1)) if len(f1) else 0.0,
        "precision": precision.tolist(),
        "recall": recall.tolist(),
        "f1": f1.tolist(),
        "support": cm.sum(axis=1).astype(int).tolist(),
        "confusion_matrix": cm.astype(int).tolist(),
    }


def parse_model_arg(model_arg: str) -> Tuple[str, Path]:
    """Parse a model spec in name=path form."""
    if "=" not in model_arg:
        path = Path(model_arg)
        return path.stem, path
    name, path = model_arg.split("=", 1)
    name = name.strip()
    if not name:
        raise ValueError(f"Empty model name in spec: {model_arg}")
    return name, Path(path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare fixed ResNet checkpoints across stratified folds."
    )
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--model", action="append", required=True, help="name=path")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--transform", choices=["wired", "train"], default="wired")
    parser.add_argument("--tta", action="store_true")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = datasets.ImageFolder(
        root=args.data_root,
        transform=build_transform(args.img_size, args.transform),
    )
    targets = getattr(dataset, "targets", [label for _, label in dataset.samples])
    folds = stratified_folds(targets, len(dataset.classes), args.k, args.seed)

    fold_counts = []
    for fold_indices in folds:
        counts = np.bincount(
            [targets[idx] for idx in fold_indices],
            minlength=len(dataset.classes),
        )
        fold_counts.append(
            {name: int(counts[i]) for i, name in enumerate(dataset.classes)}
        )

    summary: Dict[str, object] = {
        "data_root": str(Path(args.data_root).resolve()),
        "classes": dataset.classes,
        "k": args.k,
        "seed": args.seed,
        "transform": args.transform,
        "tta": bool(args.tta),
        "fold_counts": fold_counts,
        "models": {},
    }

    print(f"Dataset: {args.data_root}")
    print(f"Classes: {dataset.classes}")
    print(f"Device: {device}  transform={args.transform}  tta={args.tta}")

    for model_spec in args.model:
        name, ckpt_path = parse_model_arg(model_spec)
        model, checkpoint_classes = load_checkpoint(ckpt_path, device)
        label_map = dataset_to_checkpoint_map(dataset.classes, checkpoint_classes)

        fold_metrics = []
        aggregate_cm = np.zeros((len(checkpoint_classes), len(checkpoint_classes)), dtype=np.int64)
        for fold_id, indices in enumerate(folds):
            loader = DataLoader(
                Subset(dataset, indices),
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=(device.type == "cuda"),
            )
            cm = evaluate_fold(
                model=model,
                loader=loader,
                label_map=label_map,
                num_classes=len(checkpoint_classes),
                device=device,
                tta=args.tta,
            )
            aggregate_cm += cm
            fold_result = metrics_from_cm(cm)
            fold_result["fold"] = fold_id
            fold_metrics.append(fold_result)
            print(
                f"{name} fold {fold_id}: "
                f"acc={fold_result['accuracy']:.3f} "
                f"macro_f1={fold_result['macro_f1']:.3f} "
                f"n={int(cm.sum())}"
            )

        accs = [float(item["accuracy"]) for item in fold_metrics]
        macro_f1s = [float(item["macro_f1"]) for item in fold_metrics]
        aggregate_metrics = metrics_from_cm(aggregate_cm)
        model_summary = {
            "checkpoint": str(ckpt_path.resolve()),
            "classes": checkpoint_classes,
            "folds": fold_metrics,
            "mean_accuracy": float(np.mean(accs)),
            "std_accuracy": float(np.std(accs, ddof=1)) if len(accs) > 1 else 0.0,
            "mean_macro_f1": float(np.mean(macro_f1s)),
            "std_macro_f1": float(np.std(macro_f1s, ddof=1)) if len(macro_f1s) > 1 else 0.0,
            "aggregate": aggregate_metrics,
        }
        summary["models"][name] = model_summary
        print(
            f"{name} mean: acc={model_summary['mean_accuracy']:.3f} "
            f"+/- {model_summary['std_accuracy']:.3f}; "
            f"macro_f1={model_summary['mean_macro_f1']:.3f} "
            f"+/- {model_summary['std_macro_f1']:.3f}; "
            f"aggregate_acc={aggregate_metrics['accuracy']:.3f}"
        )

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(summary, indent=2))
        print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
