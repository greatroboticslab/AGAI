"""Validation loss of MiniGPT-v2 report-writer checkpoints, for checkpoint selection after training: the runner's
evaluation gate skips the validation split in training mode ("evaluate: false"), so the run saves one checkpoint per
epoch and no best one. For every checkpoint the mean token loss over the validation records (the same prompted
records and format as training) is computed; the best checkpoint is copied to checkpoint_best.pth in the run folder.
Writes evaluation/minigpt_<name>_val_loss.json. Run from the repo root with envs/minigptv on a GPU node.
  python cluster/minigpt_val_loss.py --name splitv1 --run /projects/$USER/agai/minigpt_runs/minigpt_strawberry_splitv1 \
      --records plant_diagnostic/datasets/split_v1/stage2_val_prompted.json
"""
import argparse
import glob
import json
import os
import re
import shutil
import sys
import time

sys.path.insert(0, os.getcwd())
import torch  # noqa: E402
from PIL import Image  # noqa: E402

from grounding.inference import load_minigpt  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--name", required=True)
ap.add_argument("--run", required=True, help="run folder with checkpoint_<epoch>.pth")
ap.add_argument("--records", required=True)
ap.add_argument("--batch-size", type=int, default=2)
ap.add_argument("--lora-alpha", type=float, default=128)
ap.add_argument("--limit", type=int, default=0)
args = ap.parse_args()

IMG = re.compile(r"<Img><ImageHere></Img>\s*")


def load_weights(model, path):
    sd = torch.load(path, map_location="cpu", weights_only=False)["model"]
    sd = {k: v for k, v in sd.items() if "lora_" in k or k.startswith("llama_proj")}
    sd = {re.sub(r"\.(lora_[AB])\.weight$", r".\1.default.weight", k): v for k, v in sd.items()}
    msg = model.load_state_dict(sd, strict=False)
    lora = [k for k in sd if "lora_" in k]
    loaded = [k for k in lora if k not in set(msg.unexpected_keys)]
    assert len(loaded) == len(lora) == 128, f"{path}: only {len(loaded)}/{len(lora)} LoRA tensors loaded"
    assert "llama_proj.weight" in sd and "llama_proj.weight" not in msg.unexpected_keys
    for m in model.modules():
        if isinstance(getattr(m, "scaling", None), dict) and "default" in m.scaling:
            m.scaling["default"] = args.lora_alpha / m.r["default"]


chat, _ = load_minigpt(0)
model, vis = chat.model, chat.vis_processor
model.eval()
records = json.load(open(args.records, encoding="utf-8"))
if args.limit:
    records = records[: args.limit]
# the dataset class's format: "<Img><ImageHere></Img> {human text} " and the assistant text as the answer
samples = [(r["image"], f"<Img><ImageHere></Img> {IMG.sub('', r['conversations'][0]['value']).strip()} ",
            r["conversations"][1]["value"]) for r in records]
ckpts = sorted(glob.glob(os.path.join(args.run, "checkpoint_[0-9]*.pth")),
               key=lambda p: int(re.search(r"checkpoint_(\d+)\.pth", p).group(1)))
assert ckpts, f"no epoch checkpoints in {args.run}"
out = {"name": args.name, "run": args.run, "n_records": len(samples), "epochs": {}}
for ck in ckpts:
    epoch = int(re.search(r"checkpoint_(\d+)\.pth", ck).group(1))
    load_weights(model, ck)
    total, n, t0 = 0.0, 0, time.time()
    with torch.no_grad():
        for b in range(0, len(samples), args.batch_size):
            batch = samples[b:b + args.batch_size]
            images = torch.stack([vis(Image.open(p).convert("RGB")) for p, _, _ in batch]).to(model.device)
            loss = model({"image": images, "instruction_input": [q for _, q, _ in batch],
                          "answer": [a for _, _, a in batch]})["loss"]
            total += float(loss) * len(batch)
            n += len(batch)
    out["epochs"][epoch] = {"val_loss": total / n, "seconds": round(time.time() - t0)}
    print(f"epoch {epoch}: val loss {total / n:.4f} ({time.time() - t0:.0f} s)", flush=True)
best = min(out["epochs"], key=lambda e: out["epochs"][e]["val_loss"])
out["best_epoch"] = best
shutil.copyfile(os.path.join(args.run, f"checkpoint_{best}.pth"), os.path.join(args.run, "checkpoint_best.pth"))
json.dump(out, open(f"evaluation/minigpt_{args.name}_val_loss.json", "w"), indent=1)
print(f"best epoch {best} -> {args.run}/checkpoint_best.pth")
