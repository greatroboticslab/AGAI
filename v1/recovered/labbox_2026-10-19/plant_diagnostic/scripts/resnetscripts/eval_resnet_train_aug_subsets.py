#!/usr/bin/env python3
"""Evaluate fixed ResNet checkpoints on train_aug real/synthetic subsets.

Assumptions:
    * This is evaluation-only. No images are copied, moved, deleted, or edited.
    * The dataset root is an ImageFolder-style tree with class-name subfolders.
    * Synthetic images are identified by filename/path markers produced by the
      current data import process: "synth_", "GPT", "ChatGPT", or "gpt".
    * Metrics are computed from argmax labels on fixed checkpoints.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms


VALID_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
SYNTH_MARKERS = ("synth_", "gpt", "chatgpt")


def is_synthetic_path(path: Path) -> bool:
    """Return True when a path matches current synthetic naming markers."""
    text = str(path).lower()
    return any(marker in text for marker in SYNTH_MARKERS)


def build_transform(img_size: int, mode: str) -> transforms.Compose:
    """Build evaluation transform."""
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


def collect_samples(root: Path) -> Tuple[List[str], List[Tuple[Path, int, bool]]]:
    """Collect ImageFolder-style samples without modifying the tree."""
    classes = sorted(
        path.name for path in root.iterdir() if path.is_dir() and not path.name.startswith(".")
    )
    if not classes:
        raise ValueError(f"No class directories under {root}")

    samples: List[Tuple[Path, int, bool]] = []
    for label_idx, class_name in enumerate(classes):
        class_dir = root / class_name
        for path in sorted(class_dir.iterdir()):
            if path.is_file() and path.suffix.lower() in VALID_EXTS:
                samples.append((path, label_idx, is_synthetic_path(path)))
    if not samples:
        raise ValueError(f"No images under {root}")
    return classes, samples


class PathDataset(Dataset):
    """Dataset over an explicit path list."""

    def __init__(
        self,
        samples: Sequence[Tuple[Path, int, bool]],
        transform: transforms.Compose,
    ) -> None:
        self.samples = list(samples)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        path, label, _is_synthetic = self.samples[index]
        with Image.open(path) as image:
            rgb = image.convert("RGB")
        return self.transform(rgb), label


def load_checkpoint(path: Path, device: torch.device) -> Tuple[nn.Module, List[str]]:
    """Load a ResNet-50 checkpoint and checkpoint class order."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    classes = ckpt.get("classes")
    if not isinstance(classes, (list, tuple)) or not classes:
        meta = ckpt.get("meta", {})
        classes = meta.get("classes") if isinstance(meta, dict) else None
    if not isinstance(classes, (list, tuple)) or not classes:
        raise ValueError(f"Checkpoint lacks class list: {path}")

    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(classes))
    state = ckpt.get("model", ckpt)
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    return model, list(classes)


def label_map(dataset_classes: Sequence[str], checkpoint_classes: Sequence[str]) -> torch.Tensor:
    """Map dataset class index to checkpoint class index."""
    if set(dataset_classes) != set(checkpoint_classes):
        missing = sorted(set(checkpoint_classes) - set(dataset_classes))
        extra = sorted(set(dataset_classes) - set(checkpoint_classes))
        raise ValueError(
            f"Class mismatch: missing_in_dataset={missing}, extra_in_dataset={extra}"
        )
    lookup = {name: idx for idx, name in enumerate(checkpoint_classes)}
    return torch.tensor([lookup[name] for name in dataset_classes], dtype=torch.long)


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    map_tensor: torch.Tensor,
    num_classes: int,
    device: torch.device,
    tta: bool,
) -> np.ndarray:
    """Return confusion matrix for one checkpoint/subset."""
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    map_tensor = map_tensor.to(device)
    for x, y_dataset in loader:
        x = x.to(device, non_blocking=True)
        y_true = map_tensor[y_dataset.to(device, non_blocking=True)]

        logits = model(x)
        if tta:
            logits = (logits + model(torch.flip(x, dims=[3]))) / 2.0
        y_pred = logits.argmax(dim=1)

        for truth, pred in zip(y_true.cpu().numpy(), y_pred.cpu().numpy()):
            cm[int(truth), int(pred)] += 1
    return cm


def metrics(cm: np.ndarray) -> Dict[str, object]:
    """Compute scalar and per-class metrics from a confusion matrix."""
    total = int(cm.sum())
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
        "accuracy": float(np.trace(cm) / total) if total else 0.0,
        "macro_f1": float(np.mean(f1)) if len(f1) else 0.0,
        "precision": precision.tolist(),
        "recall": recall.tolist(),
        "f1": f1.tolist(),
        "support": cm.sum(axis=1).astype(int).tolist(),
        "confusion_matrix": cm.astype(int).tolist(),
    }


def parse_model(spec: str) -> Tuple[str, Path]:
    """Parse name=path model argument."""
    if "=" not in spec:
        path = Path(spec)
        return path.stem, path
    name, path = spec.split("=", 1)
    if not name.strip():
        raise ValueError(f"Empty model name in spec: {spec}")
    return name.strip(), Path(path)


def subset_counts(
    classes: Sequence[str],
    samples: Sequence[Tuple[Path, int, bool]],
) -> Dict[str, Dict[str, int]]:
    """Count all/real/synthetic samples by class."""
    counts = {
        "all": {class_name: 0 for class_name in classes},
        "real": {class_name: 0 for class_name in classes},
        "synthetic": {class_name: 0 for class_name in classes},
    }
    for _path, label, synthetic in samples:
        class_name = classes[label]
        counts["all"][class_name] += 1
        counts["synthetic" if synthetic else "real"][class_name] += 1
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate fixed checkpoints on train_aug all/real/synthetic subsets."
    )
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--model", action="append", required=True, help="name=path")
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--transform", choices=["wired", "train"], default="wired")
    parser.add_argument("--tta", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    root = Path(args.data_root)
    classes, samples = collect_samples(root)
    transform = build_transform(args.img_size, args.transform)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    subsets = {
        "all": samples,
        "real": [sample for sample in samples if not sample[2]],
        "synthetic": [sample for sample in samples if sample[2]],
    }

    result: Dict[str, object] = {
        "data_root": str(root.resolve()),
        "classes": classes,
        "synthetic_markers": list(SYNTH_MARKERS),
        "transform": args.transform,
        "tta": bool(args.tta),
        "counts": subset_counts(classes, samples),
        "models": {},
    }

    print(f"Dataset: {root}")
    print(f"Classes: {classes}")
    print(f"Device: {device}  transform={args.transform}  tta={args.tta}")
    print("Counts:", json.dumps(result["counts"], sort_keys=True))

    for model_spec in args.model:
        name, ckpt_path = parse_model(model_spec)
        model, ckpt_classes = load_checkpoint(ckpt_path, device)
        mapping = label_map(classes, ckpt_classes)
        model_result: Dict[str, object] = {
            "checkpoint": str(ckpt_path.resolve()),
            "classes": ckpt_classes,
            "subsets": {},
        }
        for subset_name, subset_samples in subsets.items():
            dataset = PathDataset(subset_samples, transform)
            loader = DataLoader(
                dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=(device.type == "cuda"),
            )
            cm = evaluate(model, loader, mapping, len(ckpt_classes), device, args.tta)
            subset_metrics = metrics(cm)
            model_result["subsets"][subset_name] = subset_metrics
            print(
                f"{name} {subset_name}: "
                f"acc={subset_metrics['accuracy']:.3f} "
                f"macro_f1={subset_metrics['macro_f1']:.3f} "
                f"n={int(cm.sum())}"
            )
        result["models"][name] = model_result

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
