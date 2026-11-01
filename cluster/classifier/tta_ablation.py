"""Mirrored-view (two-view test-time augmentation) ablation of ResNet-50 checkpoints on split v1 test: accuracy with
one view and with the image + its mirror (logits averaged, the lab's resnet_classifier recipe, runtime preprocessing),
and the images the mirror turns from right to wrong and from wrong to right (the counts Shanmugam et al. 2021 ask
for). Writes evaluation/tta_<ckpt>_split_v1_test.json.
  python cluster/classifier/tta_ablation.py /projects/$USER/agai/models/resnet50_split_v1*.pth
"""
import json
import os
import sys

import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, os.getcwd())
import resnet_classifier as rc  # noqa: E402

split = json.load(open("evaluation/split_v1.json"))
items = [(p, c) for c in sorted(split["test"]) for p in split["test"][c]]
tfm = rc._tfm(256)
for ckpt in sys.argv[1:]:
    model = rc.load_resnet(ckpt)
    classes = [rc._normalize_label(c) for c in rc._CLASSES]
    temp = rc._TEMPERATURE
    name = os.path.basename(ckpt).replace(".pth", "")
    results = []
    with torch.no_grad():
        for p, c in items:
            with Image.open(p) as im:
                x = tfm(im.convert("RGB")).unsqueeze(0).to(rc._DEVICE)
            z1 = model(x)
            z2 = (z1 + model(torch.flip(x, dims=[3]))) / 2
            p1, p2 = F.softmax(z1 / temp, 1).squeeze(0).cpu(), F.softmax(z2 / temp, 1).squeeze(0).cpu()
            results.append({"image": p, "truth": c, "pred_one_view": classes[int(p1.argmax())],
                            "pred_two_views": classes[int(p2.argmax())],
                            "p1_one_view": round(float(p1.max()), 5), "p1_two_views": round(float(p2.max()), 5)})
    one = sum(r["pred_one_view"] == r["truth"] for r in results)
    two = sum(r["pred_two_views"] == r["truth"] for r in results)
    fixed = sum(r["pred_one_view"] != r["truth"] and r["pred_two_views"] == r["truth"] for r in results)
    broken = sum(r["pred_one_view"] == r["truth"] and r["pred_two_views"] != r["truth"] for r in results)
    summary = {"model": name, "n": len(results), "accuracy_one_view": round(one / len(results), 4),
               "accuracy_two_views": round(two / len(results), 4), "wrong_to_right": fixed, "right_to_wrong": broken,
               "changed_predictions": sum(r["pred_one_view"] != r["pred_two_views"] for r in results)}
    json.dump({"summary": summary, "results": results}, open(f"evaluation/tta_{name}_split_v1_test.json", "w"), indent=1)
    print(json.dumps(summary), flush=True)
