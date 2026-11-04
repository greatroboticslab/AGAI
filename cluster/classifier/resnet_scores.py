"""Per-class probabilities from ResNet-50 checkpoints on split v1 val/test (the lab's resnet_classifier recipe:
2-view hflip TTA and the checkpoint's temperature), under two preprocessings:
  runtime: resnet_classifier._tfm (resize short side to 1.15x, center-crop 256), used by the demo and paper evals
  squash:  resize to 256x256 with no crop, which is what train_resnet_straw5.py trains and validates with
Writes evaluation/scores_<ckpt>_<variant>_split_v1_<split>.json (zero-shot result format plus "probs").
  python cluster/classifier/resnet_scores.py /projects/$USER/agai/models/resnet50_split_v1.pth [more.pth ...]
  --img-size N scores checkpoints trained at another input size with the same two preprocessings scaled to N
  (runtime: short side to 1.15 N, center crop N; squash: N x N), e.g. the 512-pixel resolution control.
"""
import json
import os
import sys

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

sys.path.insert(0, os.getcwd())
import resnet_classifier as rc  # noqa: E402

args = sys.argv[1:]
IMG = 256
if "--img-size" in args:
    i = args.index("--img-size")
    IMG = int(args[i + 1])
    del args[i:i + 2]
NORM = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
TFMS = {"runtime": rc._tfm(IMG),
        "squash": transforms.Compose([transforms.Resize((IMG, IMG)), transforms.ToTensor(), NORM])}
split = json.load(open("evaluation/split_v1.json"))

for ckpt in args:
    model = rc.load_resnet(ckpt)
    classes = [rc._normalize_label(c) for c in rc._CLASSES]
    temp = rc._TEMPERATURE
    name = os.path.basename(ckpt).replace(".pth", "")
    for variant, tfm in TFMS.items():
        for s in ("val", "test"):
            items = [(p, c) for c in sorted(split[s]) for p in split[s][c]]
            results = []
            with torch.no_grad():
                for p, c in items:
                    with Image.open(p) as im:
                        x = tfm(im.convert("RGB")).unsqueeze(0).to(rc._DEVICE)
                    logits = (model(x) + model(torch.flip(x, dims=[3]))) / 2
                    pr = F.softmax(logits / temp, dim=1).squeeze(0).cpu()
                    probs = {k: round(float(v), 5) for k, v in zip(classes, pr)}
                    results.append({"image": p, "truth": c, "pred": max(probs, key=probs.get), "probs": probs})
            per = {k: f"{sum(r['pred'] == k and r['truth'] == k for r in results)}/{sum(r['truth'] == k for r in results)}"
                   for k in sorted(split[s])}
            ok = sum(r["pred"] == r["truth"] for r in results)
            macro = sum(int(v.split("/")[0]) / int(v.split("/")[1]) for v in per.values()) / len(per)
            summary = {"model": name, "preprocess": variant, "set": f"split_v1_{s}", "n": len(results),
                       "accuracy": f"{ok}/{len(results)} = {ok / len(results):.3f}", "macro_accuracy": round(macro, 3),
                       "per_class": per, "temperature": temp}
            json.dump({"summary": summary, "results": results},
                      open(f"evaluation/scores_{name}_{variant}_split_v1_{s}.json", "w"), indent=1)
            print(json.dumps(summary), flush=True)
