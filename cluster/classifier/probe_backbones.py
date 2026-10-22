"""Frozen-feature linear probes of modern vision backbones on split v1 (the same images as every other run).

Per backbone (definitions in cluster/classifier/backbones.py): squash-resize to its native resolution (the lab's ResNet also
squashes), its own normalization, features averaged over the image and its horizontal flip; then class-balanced
logistic regression with C chosen on val macro accuracy, and the test split scored once. Ranks backbones cheaply
before any fine-tuning, and saves per-image class probabilities (val and test) for ensembles plus the features
themselves (features/<name>.npz).
  python cluster/classifier/probe_backbones.py [--only dinov2-giant,bioclip-2]
"""
import argparse
import json
import os
import sys
import time

os.environ.setdefault("HF_HUB_OFFLINE", "1")
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torchvision.transforms.functional as TF  # noqa: E402
from PIL import Image  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from torch.utils.data import DataLoader, Dataset  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from backbones import BACKBONES, load  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--only", default="dinov2-giant,dinov2-large,siglip2-so400m-512,convnextv2-large-384,eva02-large-448,bioclip-2")
ap.add_argument("--split", default="evaluation/split_v1.json")
ap.add_argument("--batch-size", type=int, default=32)
args = ap.parse_args()

split = json.load(open(args.split))
CLASSES = sorted(split["train"])
items = {s: [(p, CLASSES.index(c)) for c in CLASSES for p in split[s][c]] for s in ("train", "val", "test")}
feat_dir = f"/projects/{os.environ['USER']}/agai/features"
os.makedirs(feat_dir, exist_ok=True)


class DS(Dataset):
    def __init__(self, its, res, norm):
        self.its, self.res, self.norm = its, res, norm

    def __len__(self):
        return len(self.its)

    def __getitem__(self, i):
        img = Image.open(self.its[i][0]).convert("RGB").resize((self.res, self.res), Image.BICUBIC)
        return TF.normalize(TF.to_tensor(img), *self.norm)


@torch.no_grad()
def extract(name, spec):
    m, feats = load(spec)
    m = m.cuda().eval()
    out = {}
    for s, its in items.items():
        dl = DataLoader(DS(its, spec["res"], spec["norm"]), batch_size=args.batch_size, num_workers=8)
        chunks = []
        for x in dl:
            x = x.cuda().to(torch.bfloat16)
            chunks.append(((feats(x).float() + feats(torch.flip(x, dims=[3])).float()) / 2).cpu().numpy())
        out[s] = np.concatenate(chunks)
    del m
    torch.cuda.empty_cache()
    np.savez(f"{feat_dir}/{name}.npz", **{f"X_{s}": v for s, v in out.items()},
             **{f"paths_{s}": np.array([p for p, _ in items[s]]) for s in items})
    return out


def macro(pred, y):
    return float(np.mean([np.mean(pred[y == k] == k) for k in np.unique(y)]))


def write(name, s, probs, C):
    y = np.array([c for _, c in items[s]])
    pred = probs.argmax(1)
    results = [{"image": p, "truth": CLASSES[c], "pred": CLASSES[int(q)],
                "probs": {k: round(float(v), 5) for k, v in zip(CLASSES, pr)}}
               for (p, c), q, pr in zip(items[s], pred, probs)]
    per = {k: f"{int(np.sum((pred == i) & (y == i)))}/{int(np.sum(y == i))}" for i, k in enumerate(CLASSES)}
    ok = int(np.sum(pred == y))
    summary = {"model": f"probe_{name}", "set": f"split_v1_{s}", "n": len(y), "accuracy": f"{ok}/{len(y)} = {ok / len(y):.3f}",
               "macro_accuracy": round(macro(pred, y), 3), "per_class": per, "C": C}
    json.dump({"summary": summary, "results": results}, open(f"evaluation/probe_{name}_split_v1_{s}.json", "w"), indent=1)
    return summary


table = []
for name in args.only.split(","):
    spec = dict(BACKBONES[name])
    if not os.path.exists(spec["path"]):
        print(f"skip {name}: {spec['path']} missing", flush=True)
        continue
    t0 = time.time()
    try:
        X = extract(name, spec)
    except Exception as e:   # one broken backbone should not cost the rest of the sweep
        print(f"FAILED {name}: {type(e).__name__}: {str(e)[:300]}", flush=True)
        torch.cuda.empty_cache()
        continue
    t_feat = time.time() - t0
    y = {s: np.array([c for _, c in items[s]]) for s in items}
    sc = StandardScaler().fit(X["train"])
    Z = {s: sc.transform(v) for s, v in X.items()}
    best = None
    for C in (1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 0.1, 0.3, 1.0):
        clf = LogisticRegression(C=C, max_iter=5000, class_weight="balanced").fit(Z["train"], y["train"])
        pv = clf.predict(Z["val"])
        key = (macro(pv, y["val"]), float(np.mean(pv == y["val"])))
        if best is None or key > best[0]:
            best = (key, C, clf)
    (vmac, vacc), C, clf = best
    write(name, "val", clf.predict_proba(Z["val"]), C)
    te = write(name, "test", clf.predict_proba(Z["test"]), C)
    row = f"| {name} | {X['train'].shape[1]} | {C:g} | {vacc:.1%} / {vmac:.1%} | {te['accuracy']} | {te['macro_accuracy']:.1%} | " \
          + " | ".join(te["per_class"][k] for k in CLASSES) + f" | {t_feat:.0f}s |"
    print(row, flush=True)
    table.append(row)

print("\n| backbone | dim | C | val acc / macro | test acc | test macro | " + " | ".join(CLASSES) + " | feat time |")
print("|---|---|---|---|---|---|" + "---|" * len(CLASSES) + "---|")
print("\n".join(table))
