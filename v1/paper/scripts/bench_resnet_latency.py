#!/usr/bin/env python3
"""ResNet inference latency benchmark on a single GPU."""
from __future__ import annotations
import json, time
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from torchvision import transforms, models

REPO_ROOT = Path(__file__).resolve().parents[2]
HOLDOUT = REPO_ROOT / "plant_diagnostic" / "data" / "holdout"
CKPT = REPO_ROOT / "plant_diagnostic" / "models" / "resnet_strawberry_v4_release10holdout.pth"
OUT = REPO_ROOT / "paper" / "generated" / "latency_summary.json"

def main() -> None:
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    state = torch.load(str(CKPT), map_location="cpu")
    sd = state.get("state_dict", state) if isinstance(state, dict) else state
    if isinstance(sd, dict) and "model" in sd: sd = sd["model"]
    n_classes = 7
    m = models.resnet50(weights=None)
    m.fc = torch.nn.Linear(m.fc.in_features, n_classes)
    try:
        m.load_state_dict(sd, strict=False)
    except Exception:
        pass
    m = m.to(dev).eval()

    tfm = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(256),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225]),
    ])

    images = []
    for p in sorted(HOLDOUT.rglob("*")):
        if p.is_file() and p.suffix.lower() in {".jpg",".jpeg",".png",".webp"}:
            images.append(p)
    images = images[:70]

    # Warmup
    with torch.no_grad():
        x = tfm(Image.open(images[0]).convert("RGB")).unsqueeze(0).to(dev)
        for _ in range(3):
            _ = m(x); _ = m(torch.flip(x, dims=[3]))

    lat = []
    with torch.no_grad():
        for p in images:
            x = tfm(Image.open(p).convert("RGB")).unsqueeze(0).to(dev)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = m(x); _ = m(torch.flip(x, dims=[3]))
            torch.cuda.synchronize()
            lat.append((time.perf_counter()-t0)*1000.0)

    summary = {
        "n": len(lat),
        "median_ms": float(np.median(lat)),
        "mean_ms": float(np.mean(lat)),
        "p95_ms": float(np.percentile(lat, 95)),
        "device": str(dev),
        "tta": "horizontal flip (2 views)",
    }
    OUT.write_text(json.dumps(summary, indent=2))
    print(summary)

if __name__ == "__main__":
    main()
