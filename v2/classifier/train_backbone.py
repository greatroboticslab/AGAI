"""Fine-tune a modern vision backbone + linear head on split v1 with the lab's ResNet recipe, so the backbone is the
only thing that changes vs plant_diagnostic/scripts/resnetscripts/train_resnet_straw5.py:
  same augmentations (squash resize, rotation 3 deg, hflip, perspective, color jitter, blur, autocontrast),
  class-weighted cross-entropy with label smoothing 0.05, AdamW weight decay 0.05, head-only warm-up epochs, then
  full fine-tuning (2-epoch warm-up + cosine), best epoch by val macro accuracy, temperature fitted on val, 2-view
  hflip TTA. Backbone layers get layer-wise learning-rate decay (the ViT counterpart of the lab's per-stage LRs).
--lowq-aug P (off by default, so the default recipe is unchanged) first degrades each training image with
probability P like a poor phone photo (tiny, JPEG-blocky, sometimes blurred or dark and noisy).
Writes evaluation/ft_<name>_split_v1_{val,test}.json (zero-shot format plus "probs") and the weights to
/projects/$USER/agai/models/ft_<name>.pt.
  python cluster/classifier/train_backbone.py --backbone siglip2-so400m-512 --tag s0
"""
import argparse
import copy
import io
import json
import math
import os
import random
import re
import sys
import time

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
ap.add_argument("--backbone", required=True, choices=list(B.BACKBONES))
ap.add_argument("--tag", default="s0")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--split", default="evaluation/split_v1.json")
ap.add_argument("--train-root", default="", help="optional class-folder tree replacing split train (e.g. +synthetic)")
ap.add_argument("--epochs-head", type=int, default=3)
ap.add_argument("--epochs", type=int, default=12)
ap.add_argument("--lr", type=float, default=2e-5, help="top backbone layer; lower layers decay by --layer-decay")
ap.add_argument("--lr-head", type=float, default=1e-3)
ap.add_argument("--layer-decay", type=float, default=0.85)
ap.add_argument("--batch-size", type=int, default=16)
ap.add_argument("--workers", type=int, default=8)
ap.add_argument("--grad-ckpt", action="store_true")
ap.add_argument("--lowq-aug", type=float, default=0.0,
                help="probability that a training image (real or synthetic, any class) is first degraded like a poor "
                     "phone photo: long side 96-320 px, JPEG q25-70, sometimes blur / darkening + noise")
args = ap.parse_args()
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
name = f"{args.backbone}_{args.tag}"

split = json.load(open(args.split))
CLASSES = sorted(split["train"])
items = {s: [(p, CLASSES.index(c)) for c in CLASSES for p in split[s][c]] for s in ("train", "val", "test")}
if args.train_root:
    items["train"] = [(os.path.join(args.train_root, c, f), CLASSES.index(c)) for c in CLASSES
                      for f in sorted(os.listdir(os.path.join(args.train_root, c)))]
spec = dict(B.BACKBONES[args.backbone])
backbone, feats = B.load(spec, dtype=torch.float32)
backbone = backbone.cuda()
if args.grad_ckpt:
    print("gradient checkpointing:", B.enable_grad_checkpointing(backbone), flush=True)
res, norm = spec["res"], spec["norm"]
with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
    dim = feats(torch.zeros(1, 3, res, res, device="cuda")).shape[1]
head = nn.Linear(dim, len(CLASSES)).cuda()
print(f"{name}: dim {dim}, res {res}, train {len(items['train'])}", flush=True)

class LowQ:
    """Random low-quality phone degradation (applied before the lab's transforms, so every class gets it equally)."""
    def __init__(self, p):
        self.p = p

    def __call__(self, im):
        if random.random() >= self.p:
            return im
        W, H = im.size
        s = random.choice([96, 128, 160, 224, 320]) / max(W, H)
        if s < 1:
            im = im.resize((max(1, round(W * s)), max(1, round(H * s))), Image.BICUBIC)
        if random.random() < 0.3:
            im = im.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.5)))
        if random.random() < 0.3:
            a = np.asarray(im).astype(np.float32) * random.uniform(0.5, 0.9)
            im = Image.fromarray(np.clip(a + np.random.normal(0, random.uniform(4, 14), a.shape), 0, 255).astype(np.uint8))
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=random.randint(25, 70))
        buf.seek(0)
        return Image.open(buf).convert("RGB")


tf_train = transforms.Compose(([LowQ(args.lowq_aug)] if args.lowq_aug > 0 else []) + [
    # identical to train_resnet_straw5.build_transforms, at this backbone's resolution
    transforms.Resize((res, res)), transforms.RandomRotation(degrees=3), transforms.RandomHorizontalFlip(),
    transforms.RandomPerspective(distortion_scale=0.10, p=0.15),
    transforms.ColorJitter(brightness=0.10, contrast=0.10, saturation=0.05, hue=0.02),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5)), transforms.RandomAutocontrast(p=0.20),
    transforms.ToTensor(), transforms.Normalize(*norm)])
tf_eval = transforms.Compose([transforms.Resize((res, res)), transforms.ToTensor(), transforms.Normalize(*norm)])


class DS(Dataset):
    def __init__(self, its, tf):
        self.its, self.tf = its, tf

    def __len__(self):
        return len(self.its)

    def __getitem__(self, i):
        p, y = self.its[i]
        with Image.open(p) as im:
            return self.tf(im.convert("RGB")), y


train_dl = DataLoader(DS(items["train"], tf_train), batch_size=args.batch_size, shuffle=True, drop_last=True,
                      num_workers=args.workers, pin_memory=True, persistent_workers=True)
counts = np.bincount([y for _, y in items["train"]], minlength=len(CLASSES)).astype(float)
w = counts.sum() / (counts + 1e-6)
loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(w / w.mean(), dtype=torch.float32, device="cuda"),
                              label_smoothing=0.05)


def logits_of(x):
    with torch.autocast("cuda", dtype=torch.bfloat16):
        return head(feats(x)).float()


@torch.no_grad()
def predict(its):
    backbone.eval()
    out = []
    for x, _ in DataLoader(DS(its, tf_eval), batch_size=32, num_workers=args.workers):
        x = x.cuda(non_blocking=True)
        out.append(((logits_of(x) + logits_of(torch.flip(x, dims=[3]))) / 2).cpu())
    return torch.cat(out)


def macro_acc(logits, its):
    y = torch.tensor([c for _, c in its])
    pred = logits.argmax(1)
    return float(np.mean([(pred[y == k] == k).float().mean().item() for k in range(len(CLASSES))])), \
        float((pred == y).float().mean())


def layer_id(n, n_layers):
    m = re.search(r"(?:layers?|blocks|resblocks)\.(\d+)\.", n)
    if m:
        return int(m.group(1)) + 1
    if re.search(r"embed|patch|pos|cls_token|reg_token|class_embedding|conv1|register|mask_token", n):
        return 0
    return n_layers + 1


def run_epochs(n_epochs, full):
    global best
    for p in backbone.parameters():
        p.requires_grad = full
    if full:
        names = [n for n, _ in backbone.named_parameters()]
        n_layers = max([layer_id(n, 0) for n in names if re.search(r"(?:layers?|blocks|resblocks)\.\d+\.", n)] + [0])
        groups = {}
        for n, p in backbone.named_parameters():
            lid = layer_id(n, n_layers)
            scale = args.layer_decay ** (n_layers + 1 - lid)
            nd = p.ndim == 1 or n.endswith(".bias")
            groups.setdefault((lid, nd), {"params": [], "lr": args.lr * scale, "weight_decay": 0.0 if nd else 0.05})
            groups[(lid, nd)]["params"].append(p)
        pg = list(groups.values()) + [{"params": head.parameters(), "lr": args.lr_head * 0.1, "weight_decay": 0.05}]
    else:
        pg = [{"params": head.parameters(), "lr": args.lr_head, "weight_decay": 0.05}]
    opt = torch.optim.AdamW(pg)
    base = [g["lr"] for g in opt.param_groups]
    steps, warm = n_epochs * len(train_dl), (2 * len(train_dl) if full else 0)
    step = 0
    for e in range(n_epochs):
        backbone.train(full)
        t0, tot = time.time(), 0.0
        for x, y in train_dl:
            x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
            f = min(1.0, (step + 1) / warm) if step < warm else 0.5 * (1 + math.cos(math.pi * (step - warm) / max(1, steps - warm)))
            for g, b in zip(opt.param_groups, base):
                g["lr"] = b * f
            if full:
                loss = loss_fn(logits_of(x), y)
            else:
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    fx = feats(x)
                loss = loss_fn(head(fx.float()), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for g in opt.param_groups for p in g["params"]], 1.0)
            opt.step()
            tot += loss.item()
            step += 1
        vm, va = macro_acc(predict(items["val"]), items["val"])
        tag = "full" if full else "head"
        print(f"[{tag}] {e + 1}/{n_epochs} loss {tot / len(train_dl):.4f} val acc {va:.3f} macro {vm:.3f} "
              f"({time.time() - t0:.0f}s, peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB)", flush=True)
        if (vm, va) > best[0]:   # snapshot straight to CPU (no second copy on the GPU)
            best = ((vm, va), f"{tag} {e + 1}", {n: v.detach().cpu().clone() for n, v in backbone.state_dict().items()},
                    copy.deepcopy(head.state_dict()))


best = ((-1.0, -1.0), None, None, None)
run_epochs(args.epochs_head, full=False)
run_epochs(args.epochs, full=True)
backbone.load_state_dict(best[2])
head.load_state_dict(best[3])
print(f"best epoch: {best[1]} (val macro {best[0][0]:.3f}, acc {best[0][1]:.3f})", flush=True)

val_logits = predict(items["val"])
T = torch.ones([], requires_grad=True)
yv = torch.tensor([c for _, c in items["val"]])
opt = torch.optim.LBFGS([T], lr=0.01, max_iter=100)


def closure():
    opt.zero_grad()
    loss = nn.functional.cross_entropy(val_logits / T.clamp_min(1e-3), yv)
    loss.backward()
    return loss


opt.step(closure)
T = float(T.detach().clamp_min(1e-3))
for s, lg in (("val", val_logits), ("test", predict(items["test"]))):
    probs = (lg / T).softmax(1).numpy()
    pred = probs.argmax(1)
    y = np.array([c for _, c in items[s]])
    results = [{"image": p, "truth": CLASSES[c], "pred": CLASSES[int(q)],
                "probs": {k: round(float(v), 5) for k, v in zip(CLASSES, pr)}} for (p, c), q, pr in zip(items[s], pred, probs)]
    per = {k: f"{int(np.sum((pred == i) & (y == i)))}/{int(np.sum(y == i))}" for i, k in enumerate(CLASSES)}
    ok = int(np.sum(pred == y))
    summary = {"model": f"ft_{name}", "set": f"split_v1_{s}", "n": len(y), "accuracy": f"{ok}/{len(y)} = {ok / len(y):.3f}",
               "macro_accuracy": round(float(np.mean([np.mean(pred[y == i] == i) for i in range(len(CLASSES))])), 3),
               "per_class": per, "best_epoch": best[1], "temperature": round(T, 3), "args": vars(args)}
    json.dump({"summary": summary, "results": results}, open(f"evaluation/ft_{name}_split_v1_{s}.json", "w"), indent=1)
    print(json.dumps({k: summary[k] for k in ("set", "accuracy", "macro_accuracy", "per_class", "best_epoch", "temperature")}), flush=True)
torch.save({"backbone": args.backbone, "head": head.state_dict(), "backbone_state": backbone.state_dict(), "classes": CLASSES,
            "temperature": T, "res": res, "norm": norm}, f"/projects/{os.environ['USER']}/agai/models/ft_{name}.pt")
