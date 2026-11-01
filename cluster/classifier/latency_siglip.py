"""Latency and GPU memory of the fine-tuned SigLIP 2 classifier (the v2 recipe) in-process (envs/vision), on the same
seeded split v1 test photographs as the other latency_*.py scripts: per photograph the application-style path (open,
resize, two views image + mirror, logits averaged, temperature-scaled softmax), batch 1, CUDA synchronised, after two
warm-up photographs. Also writes the predicted label and top-1 probability per photograph, which
cluster/latency_student.py hands to the v2 report writer as the proposed diagnosis. Writes
evaluation/latency_siglip2.json. Run from the repo root on a GPU node.
  python cluster/latency_siglip.py --model ft_siglip2-so400m-512_chatgpt_nof3_lq_s0
"""
import argparse
import json
import os
import random
import statistics
import subprocess
import sys
import time

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import backbones as B  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True, help="checkpoint name under /projects/$USER/agai/models (without .pt)")
ap.add_argument("--per-class", type=int, default=7)
ap.add_argument("--split-json", default="evaluation/split_v1.json")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", default="evaluation/latency_siglip2.json")
args = ap.parse_args()


def sample_items():
    """the same seeded sample in every latency_*.py script"""
    split = json.load(open(args.split_json))["test"]
    items = []
    for c in sorted(split):
        paths = sorted(split[c])
        items += [(p, c) for p in random.Random(f"{args.seed}:{c}").sample(paths, min(args.per_class, len(paths)))]
    return items


def smi_rows(query):
    """rows of nvidia-smi --query-<query> (empty when nvidia-smi is unavailable)"""
    try:
        out = subprocess.run(["nvidia-smi", f"--query-{query}", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=20).stdout
        return [[c.strip() for c in r.split(",")] for r in out.strip().splitlines() if r.strip()]
    except Exception:
        return []


def my_gpu():
    """(index, uuid) of this job's GPU: the one CUDA_VISIBLE_DEVICES names, else the first listed (nvidia-smi may list
    every GPU of the node, so row 0 is not necessarily ours)"""
    rows = smi_rows("gpu=index,uuid")
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")[0].strip()
    for idx, uuid in rows:
        if cvd and (cvd == idx or uuid.startswith(cvd)):
            return idx, uuid
    return tuple(rows[0]) if rows else (None, None)


def smi_device_used_mb():
    """memory in use on this job's GPU, all processes (nvidia-smi)"""
    idx, _ = my_gpu()
    return next((int(used) for i, used in smi_rows("gpu=index,memory.used") if i == idx and used.isdigit()), None)


def smi_process_mb(pid=os.getpid()):
    """memory of one process (pid) or of every process (pid=None) on this job's GPU, from nvidia-smi compute-apps
    (None when the driver reports no per-process figures)"""
    _, uuid = my_gpu()
    used = [int(m) for p, u, m in smi_rows("compute-apps=pid,gpu_uuid,used_memory")
            if u == uuid and (pid is None or p == str(pid)) and m.isdigit()]
    return sum(used) if used else None


def device_used_mb():
    """memory in use on the current CUDA device, all processes (torch, so the right device under any enumeration)"""
    free, total = torch.cuda.mem_get_info()
    return int((total - free) / 2 ** 20)


def stats(xs):
    xs = sorted(xs)
    return {"median_s": statistics.median(xs), "p90_s": xs[int(0.9 * (len(xs) - 1))], "mean_s": sum(xs) / len(xs),
            "n": len(xs)}


items = sample_items()
torch.cuda.reset_peak_memory_stats()
t0 = time.time()
ck = torch.load(f"/projects/{os.environ['USER']}/agai/models/{args.model}.pt", map_location="cpu", weights_only=False)
CLASSES = ck["classes"]
spec = dict(B.BACKBONES[ck["backbone"]])
backbone, feats = B.load(spec, dtype=torch.float32)
backbone.load_state_dict(ck["backbone_state"])
backbone = backbone.cuda().eval()
head = nn.Linear(ck["head"]["weight"].shape[1], len(CLASSES)).cuda()
head.load_state_dict(ck["head"])
head.eval()
tf = transforms.Compose([transforms.Resize((ck["res"], ck["res"])), transforms.ToTensor(),
                         transforms.Normalize(*ck["norm"])])
T = float(ck["temperature"])
torch.cuda.synchronize()
t_load = time.time() - t0


@torch.no_grad()
def classify(path):
    with Image.open(path) as im:
        x = tf(im.convert("RGB")).unsqueeze(0).cuda()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        lg = (head(feats(x)).float() + head(feats(torch.flip(x, dims=[3]))).float()) / 2
    p = (lg / T).softmax(1)[0]
    i = int(p.argmax())
    return CLASSES[i], float(p[i])


for p, c in items[:2]:                               # warm-up, not recorded
    classify(p)
torch.cuda.synchronize()
records = []
for p, c in items:
    torch.cuda.synchronize()
    t = time.time()
    pred, p1 = classify(p)
    torch.cuda.synchronize()
    records.append({"image": p, "label": c, "pred": pred, "p1": p1, "t_classify": time.time() - t})
out = {"model": args.model, "backbone": ck["backbone"], "res": ck["res"], "temperature": T,
       "gpu": torch.cuda.get_device_name(0), "n": len(records), "t_load_s": t_load,
       **{"t_classify": stats([r["t_classify"] for r in records])},
       "accuracy": sum(r["pred"] == r["label"] for r in records) / len(records),
       "peak_alloc_gb": torch.cuda.max_memory_allocated() / 1e9,
       "peak_reserved_gb": torch.cuda.max_memory_reserved() / 1e9, "smi_process_mb": smi_process_mb(),
       "device_used_mb": device_used_mb(),
       "predictions": {r["image"]: {"pred": r["pred"], "p1": r["p1"]} for r in records}, "records": records}
json.dump(out, open(args.out, "w"), indent=1)
print(f"{args.model}: load {t_load:.1f} s, median {out['t_classify']['median_s']:.3f} s, p90 {out['t_classify']['p90_s']:.3f} s, "
      f"accuracy on the sample {out['accuracy']:.3f}, process {out['smi_process_mb']} MB, "
      f"torch peak {out['peak_reserved_gb']:.2f} GB -> {args.out}")
