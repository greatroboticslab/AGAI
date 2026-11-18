"""Robustness to low-quality phone photos: evaluate saved fine-tuned backbones (cluster/classifier/train_backbone.py checkpoints)
on the REAL split v1 test images degraded in fixed, deterministic ways (no synthetic images involved):
  clean       as is (should reproduce the saved test results)
  low_res     long side downscaled to 256 px, JPEG q=50 (a small re-sent phone photo)
  very_low    long side 128 px, JPEG q=35 (thumbnail-like; most fine detail gone)
  defocus     Gaussian blur, radius 0.6% of the long side (whole frame out of focus), JPEG q=70
  noisy_dark  brightness x0.55 + Gaussian sensor noise (sigma 12), JPEG q=60
Each model then resizes the degraded image to its own input size, with the same hflip TTA and temperature as
train_backbone.py. Writes evaluation/lowq_<model>.json.
  python cluster/classifier/eval_lowq.py --models ft_siglip2-so400m-512_s0,ft_siglip2-so400m-512_labsynth_s0
"""
import argparse
import hashlib
import io
import json
import os
import sys

os.environ.setdefault("HF_HUB_OFFLINE", "1")
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from PIL import Image, ImageFilter  # noqa: E402
from torch.utils.data import DataLoader, Dataset  # noqa: E402
from torchvision import transforms  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import backbones as B  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--models", required=True, help="comma-separated checkpoint names under /projects/$USER/agai/models")
ap.add_argument("--split", default="evaluation/split_v1.json")
ap.add_argument("--set", default="test")
ap.add_argument("--workers", type=int, default=8)
args = ap.parse_args()
LEVELS = ["clean", "low_res", "very_low", "defocus", "noisy_dark"]


def degrade(im, level, key):
    if level == "clean":
        return im
    W, H = im.size
    if level in ("low_res", "very_low"):
        s = (256 if level == "low_res" else 128) / max(W, H)
        if s < 1:
            im = im.resize((max(1, round(W * s)), max(1, round(H * s))), Image.BICUBIC)
        q = 50 if level == "low_res" else 35
    elif level == "defocus":
        im = im.filter(ImageFilter.GaussianBlur(radius=0.006 * max(W, H)))
        q = 70
    else:   # noisy_dark: same noise for an image whichever model is evaluated
        rng = np.random.default_rng(int(hashlib.md5(key.encode()).hexdigest()[:8], 16))
        a = np.asarray(im).astype(np.float32) * 0.55 + rng.normal(0, 12, (H, W, 3))
        im = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
        q = 60
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=q)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


class DS(Dataset):
    def __init__(self, its, level, tf):
        self.its, self.level, self.tf = its, level, tf

    def __len__(self):
        return len(self.its)

    def __getitem__(self, i):
        p, y = self.its[i]
        with Image.open(p) as im:
            return self.tf(degrade(im.convert("RGB"), self.level, p)), y


split = json.load(open(args.split))
for name in args.models.split(","):
    ck = torch.load(f"/projects/{os.environ['USER']}/agai/models/{name}.pt", map_location="cpu", weights_only=False)
    CLASSES = ck["classes"]
    its = [(p, CLASSES.index(c)) for c in CLASSES for p in split[args.set][c]]
    spec = dict(B.BACKBONES[ck["backbone"]])
    backbone, feats = B.load(spec, dtype=torch.float32)
    backbone.load_state_dict(ck["backbone_state"])
    backbone = backbone.cuda().eval()
    dim = ck["head"]["weight"].shape[1]
    head = nn.Linear(dim, len(CLASSES)).cuda()
    head.load_state_dict(ck["head"])
    tf = transforms.Compose([transforms.Resize((ck["res"], ck["res"])), transforms.ToTensor(),
                             transforms.Normalize(*ck["norm"])])
    out = {"model": name, "set": f"split_v1_{args.set}", "temperature": ck["temperature"], "levels": {}, "results": {}}
    y = np.array([c for _, c in its])
    for level in LEVELS:
        lg = []
        with torch.no_grad():
            for x, _ in DataLoader(DS(its, level, tf), batch_size=32, num_workers=args.workers):
                x = x.cuda(non_blocking=True)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    lg.append(((head(feats(x)).float() + head(feats(torch.flip(x, dims=[3]))).float()) / 2).cpu())
        pred = (torch.cat(lg) / ck["temperature"]).softmax(1).argmax(1).numpy()
        ok = int(np.sum(pred == y))
        out["levels"][level] = {
            "accuracy": round(ok / len(y), 4),
            "macro_accuracy": round(float(np.mean([np.mean(pred[y == i] == i) for i in range(len(CLASSES))])), 4),
            "per_class": {k: f"{int(np.sum((pred == i) & (y == i)))}/{int(np.sum(y == i))}" for i, k in enumerate(CLASSES)}}
        out["results"][level] = [{"image": p, "truth": CLASSES[c], "pred": CLASSES[int(q)]} for (p, c), q in zip(its, pred)]
        print(name, level, out["levels"][level]["accuracy"], out["levels"][level]["macro_accuracy"], flush=True)
    json.dump(out, open(f"evaluation/lowq_{name}.json", "w"), indent=1)
    del backbone, head
    torch.cuda.empty_cache()
