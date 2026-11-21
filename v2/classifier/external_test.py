"""External test (lab notebook, entry (al)): v1's ResNet-50 and v2's SigLIP 2 classifier, each with both acceptance gates,
on a public strawberry disease test set they never saw.

    python cluster/external_test.py --images /projects/$USER/agai/external/mendeley_k2ptxmfjhj_v2/traditional_test \
        --out evaluation/external_mendeley_test.json

Images: the test split of the Mendeley Data release k2ptxmfjhj (version 2, "traditional augmentation"; CC BY-NC 4.0),
which extends Afzaal et al. 2021 (Sensors 21:6565) and Perez-Borrero et al. 2020; image-level labels are the set of
categories of the COCO instances in `_annotations.coco.json`. Groups: in scope, the images whose instances are all gray
mold and those whose instances are all "Asymptomatic Strawberry" (our healthy), the two conditions both label sets
share; out of scope, the images whose instances are all diseases outside our seven conditions; mixed, the rest.
Classifiers: v1's ResNet-50 retrained on split v1 (`models/resnet50_split_v1.pth`, the application's preprocessing,
two views, checkpoint temperature) and v2's SigLIP 2 recipe (`models/ft_siglip2-so400m-512_chatgpt_nof3_lq_s0.pt`, as
`latency_siglip.py`). Gates, as in `calibration_metrics.py` and Table S7: v1's hand-set class thresholds (healthy 0.40,
drought and root rot 0.65, frost 0.70, others 0.60; accept when p1 >= max(threshold, 0.5)), and the validation-chosen
thresholds of each classifier's seed-0 calibration (temperature-scaled).
"""
import argparse
import collections
import json
import os
import re
import sys

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())
import backbones as B  # noqa: E402
import resnet_classifier as RC  # noqa: E402

V1_GATE = {"healthy": 0.40, "drought": 0.65, "root_rot": 0.65, "frost": 0.70}
V1_DEFAULT = 0.60
GRAY = re.compile(r"gr[ae]y[\s_-]*mou?ld", re.I)
HEALTHY = re.compile(r"asymptomatic|healthy", re.I)  # "Asymptomatic Strawberry": our healthy


def accept(pred, p1, thresholds, default):
    return bool(p1 >= max(thresholds.get(pred, default), 0.5))


def val_thresholds(summary_path):
    s = json.load(open(summary_path))
    return s["seeds"][0]["with_temperature"]["gate_val"]["thresholds"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--resnet", default=f"/projects/{os.environ['USER']}/agai/models/resnet50_split_v1.pth")
    ap.add_argument("--siglip", default="ft_siglip2-so400m-512_chatgpt_nof3_lq_s0")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    coco = json.load(open(os.path.join(a.images, "_annotations.coco.json")))
    cat = {c["id"]: c["name"] for c in coco["categories"]}
    cats = collections.defaultdict(set)
    for an in coco["annotations"]:
        cats[an["image_id"]].add(cat[an["category_id"]])
    images = [(im["file_name"], sorted(cats.get(im["id"], set()))) for im in coco["images"]]

    # v1: the application's ResNet-50 path
    rmodel = RC.load_resnet(a.resnet)
    rclasses, rT = list(RC._CLASSES), float(RC._TEMPERATURE)
    rtf = RC._tfm(256)
    # v2: as latency_siglip.py
    ck = torch.load(f"/projects/{os.environ['USER']}/agai/models/{a.siglip}.pt", map_location="cpu", weights_only=False)
    sclasses = ck["classes"]
    backbone, feats = B.load(dict(B.BACKBONES[ck["backbone"]]), dtype=torch.float32)
    backbone.load_state_dict(ck["backbone_state"])
    backbone = backbone.cuda().eval()
    head = nn.Linear(ck["head"]["weight"].shape[1], len(sclasses)).cuda()
    head.load_state_dict(ck["head"])
    head.eval()
    stf = transforms.Compose([transforms.Resize((ck["res"], ck["res"])), transforms.ToTensor(),
                              transforms.Normalize(*ck["norm"])])
    sT = float(ck["temperature"])
    gates = {"resnet": {"v1": V1_GATE, "val": val_thresholds("evaluation/calib_resnet50_v1_summary.json")},
             "siglip": {"v1": V1_GATE, "val": val_thresholds("evaluation/calib_siglip2_recipe_summary.json")}}

    records = []
    with torch.no_grad():
        for name, cs in images:
            with Image.open(os.path.join(a.images, name)) as im:
                img = im.convert("RGB")
            x = rtf(img).unsqueeze(0).to(RC._DEVICE)
            lg = (rmodel(x) + rmodel(torch.flip(x, dims=[3]))) / 2
            pr = torch.softmax(lg / rT, 1)[0].float().cpu()
            y = stf(img).unsqueeze(0).cuda()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                sl = (head(feats(y)).float() + head(feats(torch.flip(y, dims=[3]))).float()) / 2
            ps = torch.softmax(sl / sT, 1)[0].float().cpu()
            rec = {"image": name, "categories": cs, "size": list(img.size)}
            for key, p, classes in (("resnet", pr, rclasses), ("siglip", ps, sclasses)):
                i = int(p.argmax())
                pred, p1 = classes[i], float(p[i])
                rec[key] = {"pred": pred, "p1": round(p1, 5), "probs": {c: round(float(v), 5) for c, v in zip(classes, p)},
                            "accept_v1_gate": accept(pred, p1, gates[key]["v1"], V1_DEFAULT),
                            "accept_val_gate": accept(pred, p1, gates[key]["val"], 0.5)}
            records.append(rec)

    def group(cs):
        if cs and all(GRAY.search(c) for c in cs):
            return "in scope: gray mold"
        if cs and all(HEALTHY.search(c) for c in cs):
            return "in scope: healthy"
        if cs and not any(GRAY.search(c) or HEALTHY.search(c) for c in cs):
            return "out of scope: other diseases"
        return "mixed or unannotated"

    summary = {"categories": sorted(set(cat.values())), "n_images": len(records), "groups": {}}
    by = collections.defaultdict(list)
    for r in records:
        by[group(r["categories"])].append(r)
    for g, rs in sorted(by.items()):
        out = {"n": len(rs)}
        for key in ("resnet", "siglip"):
            preds = collections.Counter(r[key]["pred"] for r in rs)
            m = {"predicted": dict(preds.most_common())}
            for gate in ("v1", "val"):
                acc = [r for r in rs if r[key][f"accept_{gate}_gate"]]
                m[f"accepted_{gate}_gate"] = round(len(acc) / len(rs), 4)
                if g == "in scope: gray mold":
                    m[f"accuracy_accepted_{gate}_gate"] = (round(sum(r[key]["pred"] == "gray_mold" for r in acc) / len(acc), 4)
                                                          if acc else None)
            if g == "in scope: gray mold":
                m["recall_ungated"] = round(preds.get("gray_mold", 0) / len(rs), 4)
            if g == "in scope: healthy":
                m["recall_ungated"] = round(preds.get("healthy", 0) / len(rs), 4)
                for gate in ("v1", "val"):
                    acc = [r for r in rs if r[key][f"accept_{gate}_gate"]]
                    m[f"accuracy_accepted_{gate}_gate"] = (round(sum(r[key]["pred"] == "healthy" for r in acc) / len(acc), 4) if acc else None)
            out[key] = m
        summary["groups"][g] = out
    json.dump({"summary": summary, "gates": gates, "records": records, "resnet_ckpt": a.resnet, "siglip": a.siglip,
               "resnet_temperature": rT, "siglip_temperature": sT}, open(a.out, "w"), indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
