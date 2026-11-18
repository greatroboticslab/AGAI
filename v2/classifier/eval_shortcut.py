"""Resolution-shortcut test (2026-10-22): 78 % of the real healthy photos are small web thumbnails (median long side
275 px) while the disease photos are larger, so a classifier could learn "small / blurry -> healthy". Saved fine-tuned
backbones (cluster/classifier/train_backbone.py checkpoints) are run on the split v1 test images in three conditions, keeping the
full softmax probabilities (eval_lowq.py keeps only the argmax):
  clean          as is
  thumb275       long side downscaled to 275 px (the healthy median), JPEG q=75 (a typical web thumbnail)
  thumb275_raw   long side 275 px, no JPEG (resolution only, no compression artefacts)
Same preprocessing as training: resize to the model's input size, hflip TTA, the checkpoint's temperature. Images whose
long side is already <= 275 px are unchanged by the downscale. Writes evaluation/shortcut_<model>.json with the native
size of each image. Scored by the analysis that compares training recipes (the control: some rise in P(healthy) for
shrunk disease photos is legitimate because faint signs vanish; the shortcut is the difference between recipes).
  python cluster/classifier/eval_shortcut.py --models ft_siglip2-so400m-512_s0,ft_siglip2-so400m-512_lq_s0
"""
import argparse
import io
import json
import os
import sys

os.environ.setdefault("HF_HUB_OFFLINE", "1")
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from PIL import Image  # noqa: E402
from torch.utils.data import DataLoader, Dataset  # noqa: E402
from torchvision import transforms  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import backbones as B  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--models", required=True, help="comma-separated checkpoint names under /projects/$USER/agai/models")
ap.add_argument("--split", default="evaluation/split_v1.json")
ap.add_argument("--set", default="test")
ap.add_argument("--side", type=int, default=275)
ap.add_argument("--workers", type=int, default=8)
args = ap.parse_args()
LEVELS = ["clean", "thumb275", "thumb275_raw"]


def degrade(im, level):
    if level == "clean":
        return im
    W, H = im.size
    s = args.side / max(W, H)
    if s < 1:
        im = im.resize((max(1, round(W * s)), max(1, round(H * s))), Image.BICUBIC)
    if level == "thumb275_raw":
        return im
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=75)
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
            return self.tf(degrade(im.convert("RGB"), self.level)), y


split = json.load(open(args.split))
for name in args.models.split(","):
    ck = torch.load(f"/projects/{os.environ['USER']}/agai/models/{name}.pt", map_location="cpu", weights_only=False)
    CLASSES = ck["classes"]
    its = [(p, CLASSES.index(c)) for c in CLASSES for p in split[args.set][c]]
    sizes = {}
    for p, _ in its:
        with Image.open(p) as im:
            sizes[p] = list(im.size)
    spec = dict(B.BACKBONES[ck["backbone"]])
    backbone, feats = B.load(spec, dtype=torch.float32)
    backbone.load_state_dict(ck["backbone_state"])
    backbone = backbone.cuda().eval()
    dim = ck["head"]["weight"].shape[1]
    head = nn.Linear(dim, len(CLASSES)).cuda()
    head.load_state_dict(ck["head"])
    tf = transforms.Compose([transforms.Resize((ck["res"], ck["res"])), transforms.ToTensor(),
                             transforms.Normalize(*ck["norm"])])
    out = {"model": name, "classes": CLASSES, "temperature": ck["temperature"], "side": args.side, "results": []}
    probs = {}
    for level in LEVELS:
        lg = []
        with torch.no_grad():
            for x, _ in DataLoader(DS(its, level, tf), batch_size=32, num_workers=args.workers):
                x = x.cuda(non_blocking=True)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    lg.append(((head(feats(x)).float() + head(feats(torch.flip(x, dims=[3]))).float()) / 2).cpu())
        probs[level] = (torch.cat(lg) / ck["temperature"]).softmax(1).numpy()
        acc = float((probs[level].argmax(1) == [c for _, c in its]).mean())
        print(name, level, round(acc, 4), flush=True)
    for i, (p, c) in enumerate(its):
        out["results"].append({"image": p, "truth": CLASSES[c], "size": sizes[p],
                               **{lv: [round(float(v), 5) for v in probs[lv][i]] for lv in LEVELS}})
    json.dump(out, open(f"evaluation/shortcut_{name}.json", "w"))
    del backbone, head
    torch.cuda.empty_cache()
