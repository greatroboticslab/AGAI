"""RF-DETR part-detector latency and GPU memory in-process (envs/rfdetr), for the v1 vs v2 pipeline comparison: the
lab's Small checkpoint (v1) and the Large checkpoint retrained on the detector split that agrees with split v1 (v2),
on the same seeded split v1 test photographs as the other latency_*.py scripts. Per photograph: open, predict at the
application's raw floor (0.30, the lowest per-part threshold of rfdetr_parts/detect_images.py) and the visible parts at
its per-part thresholds; CUDA synchronised, after two warm-up photographs. Model load time, median / p90 / mean seconds,
memory from torch (peaks) and nvidia-smi (this process, by PID on the job's GPU), one model at a time. Writes evaluation/latency_rfdetr.json. Run from the repo
root on a GPU node.
  python cluster/latency_rfdetr.py [--weights-large /projects/$USER/agai/rfdetr_runs/large_splitv1/checkpoint_best_total.pth]
"""
import argparse
import json
import os
import random
import statistics
import subprocess
import time

import torch
from PIL import Image
from rfdetr import RFDETRLarge, RFDETRSmall

ap = argparse.ArgumentParser()
ap.add_argument("--weights-small", default="rfdetr_parts/output/checkpoint_best_total.pth",
                help="the lab's v1 checkpoint")
ap.add_argument("--weights-large",
                default=f"/projects/{os.environ.get('USER', '')}/agai/rfdetr_runs/large_splitv1/checkpoint_best_total.pth")
ap.add_argument("--per-class", type=int, default=7)
ap.add_argument("--split-json", default="evaluation/split_v1.json")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", default="evaluation/latency_rfdetr.json")
args = ap.parse_args()

NAMES = {0: "flower", 1: "fruit", 2: "leaf", 3: "root", 4: "soil", 5: "stem"}   # predictions are 0-based
THRESH = {"flower": 0.50, "fruit": 0.50, "leaf": 0.40, "root": 0.30, "soil": 0.40, "stem": 0.40}
FLOOR = min(THRESH.values())


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


def detect(model, path):
    with Image.open(path) as im:
        det = model.predict(im.convert("RGB"), threshold=FLOOR)
    if det is None or det.class_id is None or det.confidence is None:
        return []
    ids, confs = det.class_id, det.confidence
    return sorted({NAMES[int(i)] for i, s in zip(ids, confs) if int(i) in NAMES and float(s) >= THRESH[NAMES[int(i)]]})


def measure(cls, weights, items):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    model = cls(pretrain_weights=weights)
    t_load = time.time() - t0
    t0 = time.time()
    detect(model, items[0][0])                      # first prediction: CUDA context, kernels
    torch.cuda.synchronize()
    t_first = time.time() - t0
    detect(model, items[1][0])                      # second warm-up
    per_image = []
    for p, c in items:
        torch.cuda.synchronize()
        t = time.time()
        visible = detect(model, p)
        torch.cuda.synchronize()
        per_image.append({"image": p, "label": c, "t_detect": time.time() - t, "visible": visible})
    out = {"weights": weights, "t_load_s": t_load, "t_first_predict_s": t_first, "floor": FLOOR, "thresholds": THRESH,
           **stats([r["t_detect"] for r in per_image]),
           "peak_alloc_gb": torch.cuda.max_memory_allocated() / 1e9,
           "peak_reserved_gb": torch.cuda.max_memory_reserved() / 1e9, "smi_process_mb": smi_process_mb(),
           "device_used_mb": device_used_mb(), "per_image": per_image}
    print(f"{cls.__name__} {weights}: load {t_load:.1f} s, first {t_first:.2f} s, median {out['median_s']:.3f} s, "
          f"p90 {out['p90_s']:.3f} s, process {out['smi_process_mb']} MB, torch peak {out['peak_reserved_gb']:.2f} GB",
          flush=True)
    del model
    torch.cuda.empty_cache()
    return out


items = sample_items()
result = {"gpu": torch.cuda.get_device_name(0), "n": len(items),
          "small_lab_v1": measure(RFDETRSmall, args.weights_small, items),
          "large_splitv1_v2": measure(RFDETRLarge, args.weights_large, items)}
json.dump(result, open(args.out, "w"), indent=1)
print("->", args.out)
