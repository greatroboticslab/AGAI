"""Gray-mold vs white-mold toss-ups for a label review: images labeled one mold where the cross-validated frozen-
backbone ensemble (EVA-02 + PE-Core-G + SigLIP 2, out-of-fold, cluster/classifier/cv_probes.py) gives the other mold at least
--min-other probability, with the 32B annotator's verdict and caption alongside (evaluation/label_audit.csv).
Writes <out>/tossups.csv and contact sheets, most contested first. Run from the repo root.
  python cluster/benchmark/mold_tossups.py --out /projects/$USER/agai/logs/mold_tossups
"""
import argparse
import csv
import json
import os

import numpy as np
from PIL import Image, ImageDraw

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--min-other", type=float, default=0.15)
ap.add_argument("--backbones", default="eva02-large-448,pe-core-g14-448,siglip2-so400m-512")
args = ap.parse_args()
probs = {}
for b in args.backbones.split(","):
    for r in json.load(open(f"evaluation/cv5_probe_{b}.json"))["results"]:
        probs.setdefault(r["image"], {"truth": r["truth"], "p": []})["p"].append(r["probs"])
audit = {r["image"]: r for r in csv.DictReader(open("evaluation/label_audit.csv"))}
other = {"gray_mold": "white_mold", "white_mold": "gray_mold"}
rows = []
for img, v in probs.items():
    if v["truth"] not in other:
        continue
    p = {c: float(np.mean([q[c] for q in v["p"]])) for c in v["p"][0]}
    po = p[other[v["truth"]]]
    if po >= args.min_other:
        a = audit.get(img, {})
        rows.append({"image": img, "label": v["truth"], "p_label": round(p[v["truth"]], 3), "p_other_mold": round(po, 3),
                     "top": max(p, key=p.get), "teacher_verdict": a.get("teacher_verdict", ""),
                     "teacher_caption": a.get("teacher_caption", "")})
rows.sort(key=lambda r: -r["p_other_mold"])
os.makedirs(args.out, exist_ok=True)
with open(os.path.join(args.out, "tossups.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else ["image"])
    w.writeheader()
    w.writerows(rows)
W, H, cols, per = 300, 300, 6, 24
for s in range(0, len(rows), per):
    chunk = rows[s:s + per]
    sheet = Image.new("RGB", (cols * W, ((len(chunk) + cols - 1) // cols) * (H + 46)), "white")
    d = ImageDraw.Draw(sheet)
    for j, r in enumerate(chunk):
        im = Image.open(r["image"]).convert("RGB")
        im.thumbnail((W, H))
        x, y = (j % cols) * W, (j // cols) * (H + 46)
        sheet.paste(im, (x + (W - im.width) // 2, y))
        d.text((x + 3, y + H + 2), f"{s + j + 1}. label {r['label']}  p(label) {r['p_label']:.2f}", fill="black")
        d.text((x + 3, y + H + 16), f"p(other mold) {r['p_other_mold']:.2f}  annot: {r['teacher_verdict'] or '-'}", fill="gray")
        d.text((x + 3, y + H + 30), os.path.basename(r["image"])[:48], fill="gray")
    sheet.save(os.path.join(args.out, f"tossups_{s // per + 1:02d}.jpg"), quality=85)
print(f"{len(rows)} gray/white mold toss-ups (other mold >= {args.min_other}) -> {args.out}")
for r in rows:
    print(r["label"], "p_label", r["p_label"], "p_other", r["p_other_mold"], r["teacher_verdict"], os.path.basename(r["image"]))
