"""Shortcut probe: run saved fine-tuned backbones (cluster/classifier/train_backbone.py checkpoints) on synthetic images they did
NOT train on, and break the results down by the manifest's prompt fields (group, hand, quality, severity). Answers
e.g. "does a model call any dug-up roots root rot?" (healthy roots_ground -> root_rot) or "does a hand in frame
change the answer?". Synthetic images, so this is a diagnostic, not a benchmark.
  python cluster/classifier/eval_probe.py --models ft_siglip2-so400m-512_s0 --manifest evaluation/synth_chatgpt_v1_manifest.json \
      [--only evaluation/split_v1_chatgpt.json]   # restrict to that tree's held-out probe files
Writes evaluation/probe_<model>_<manifest name>.json."""
import argparse
import collections
import json
import os
import sys

os.environ.setdefault("HF_HUB_OFFLINE", "1")
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from PIL import Image  # noqa: E402
from torch.utils.data import DataLoader, Dataset  # noqa: E402
from torchvision import transforms  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import backbones as B  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--models", required=True)
ap.add_argument("--manifest", required=True)
ap.add_argument("--only", default="", help="split_<name>.json from build_train_tree.py: use its probe_files only")
ap.add_argument("--workers", type=int, default=8)
args = ap.parse_args()
man = json.load(open(args.manifest))
recs = man["kept"]
if args.only:
    keep = set(json.load(open(args.only))["probe_files"])
    recs = [r for r in recs if r["file"] in keep]
print(len(recs), "probe images", flush=True)


class DS(Dataset):
    def __init__(self, paths, tf):
        self.paths, self.tf = paths, tf

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        with Image.open(self.paths[i]) as im:
            return self.tf(im.convert("RGB")), i


for name in args.models.split(","):
    ck = torch.load(f"/projects/{os.environ['USER']}/agai/models/{name}.pt", map_location="cpu", weights_only=False)
    CLASSES = ck["classes"]
    spec = dict(B.BACKBONES[ck["backbone"]])
    backbone, feats = B.load(spec, dtype=torch.float32)
    backbone.load_state_dict(ck["backbone_state"])
    backbone = backbone.cuda().eval()
    head = nn.Linear(ck["head"]["weight"].shape[1], len(CLASSES)).cuda()
    head.load_state_dict(ck["head"])
    tf = transforms.Compose([transforms.Resize((ck["res"], ck["res"])), transforms.ToTensor(),
                             transforms.Normalize(*ck["norm"])])
    lg = []
    with torch.no_grad():
        for x, _ in DataLoader(DS([os.path.join(man["root"], r["file"]) for r in recs], tf), batch_size=32,
                               num_workers=args.workers):
            x = x.cuda(non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                lg.append(((head(feats(x)).float() + head(feats(torch.flip(x, dims=[3]))).float()) / 2).cpu())
    probs = (torch.cat(lg) / ck["temperature"]).softmax(1).numpy()
    rows = [dict(r, pred=CLASSES[int(p.argmax())], p_true=round(float(p[CLASSES.index(r["class"])]), 4))
            for r, p in zip(recs, probs)]
    out = {"model": name, "manifest": args.manifest, "only": args.only, "n": len(rows), "breakdown": {}}
    for key in ("class", "group", "hand", "quality", "severity"):
        tab = collections.defaultdict(lambda: collections.Counter())
        for r in rows:
            k = f"{r['class']}|{r.get(key)}" if key != "class" else r["class"]
            tab[k]["n"] += 1
            tab[k]["correct"] += r["pred"] == r["class"]
            tab[k]["pred_" + r["pred"]] += 1
        out["breakdown"][key] = {k: dict(v) for k, v in sorted(tab.items())}
    acc = np.mean([r["pred"] == r["class"] for r in rows]) if rows else float("nan")
    print(f"{name}: probe accuracy {acc:.3f} (n={len(rows)})", flush=True)
    for k, v in out["breakdown"]["group"].items():
        if k.split("|")[1] in ("roots_ground", "roots_bench", "roots_hand", "crown", "root_cut"):
            print(f"   {k:28s} {v['correct']}/{v['n']} correct, preds { {p[5:]: c for p, c in v.items() if p.startswith('pred_')} }")
    out["rows"] = [{"file": r["file"], "class": r["class"], "pred": r["pred"], "p_true": r["p_true"]} for r in rows]
    tag = os.path.basename(args.manifest).replace("_manifest.json", "")
    json.dump(out, open(f"evaluation/probe_{name}_{tag}{'_heldout' if args.only else ''}.json", "w"), indent=1)
    del backbone, head
    torch.cuda.empty_cache()
