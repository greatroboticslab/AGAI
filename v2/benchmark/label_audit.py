"""Label audit over all 1,576 split v1 images: which labels are most likely wrong?

Two independent signals:
  ensemble: out-of-fold class probabilities from the 5-fold grouped CV of the top frozen backbones (EVA-02, PE-Core-G,
            SigLIP 2; evaluation/cv5_probe_*.json), so every image is judged by models that never saw it;
  teacher:  the 32B annotator's verdict on the image's own label (label_assessment.consistency in the true-label
            annotations under annotations/v1_Qwen3-VL-32B-Instruct/), when available.
Writes evaluation/label_audit.csv sorted by suspicion (both signals first, then by ensemble probability of the
label) and a contact sheet of the top candidates for a human (ideally a pathologist) to re-label.
  python cluster/benchmark/label_audit.py [--top 32]
"""
import argparse
import csv
import json
import os

import numpy as np
from PIL import Image, ImageDraw

ap = argparse.ArgumentParser()
ap.add_argument("--backbones", default="eva02-large-448,pe-core-g14-448,siglip2-so400m-512")
ap.add_argument("--teacher", default=f"/projects/{os.environ['USER']}/agai/annotations/v1_Qwen3-VL-32B-Instruct")
ap.add_argument("--top", type=int, default=32)
ap.add_argument("--sheet", default=f"/projects/{os.environ['USER']}/agai/logs/label_audit_top.jpg")
args = ap.parse_args()
ACCEPT = {"clearly_supports", "partially_supports"}

probs, truth = {}, {}
for b in args.backbones.split(","):
    for r in json.load(open(f"evaluation/cv5_probe_{b}.json"))["results"]:
        probs.setdefault(r["image"], []).append(r["probs"])
        truth[r["image"]] = r["truth"]
classes = sorted(next(iter(probs.values()))[0])
ens = {p: {c: float(np.mean([pr[c] for pr in v])) for c in classes} for p, v in probs.items()}

teacher = {}
for f in ("train_true.jsonl", "val_both.jsonl", "test_both.jsonl"):
    path = os.path.join(args.teacher, f)
    if not os.path.exists(path):
        continue
    for line in open(path):
        r = json.loads(line)
        if r.get("label_swapped") or "annotation" not in r:
            continue
        a = r["annotation"]
        teacher[r["image"]] = (a["label_assessment"]["consistency"], a["caption"],
                               [d["condition"] for d in a.get("differential", [])])

rows = []
for p, pr in ens.items():
    lab = truth[p]
    pred = max(pr, key=pr.get)
    t = teacher.get(p)
    t_rejects = t is not None and t[0] not in ACCEPT
    rows.append({"image": p, "label": lab, "ensemble_pred": pred, "p_label": round(pr[lab], 4),
                 "p_pred": round(pr[pred], 4), "teacher_verdict": t[0] if t else "",
                 "teacher_differential": " ".join(t[2]) if t else "", "teacher_caption": t[1] if t else "",
                 "both_disagree": bool(pred != lab and t_rejects)})
rows.sort(key=lambda r: (not r["both_disagree"], r["p_label"]))
with open("evaluation/label_audit.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
n_model = sum(r["ensemble_pred"] != r["label"] for r in rows)
n_low = sum(r["p_label"] < 0.1 for r in rows)
n_teacher = sum(r["teacher_verdict"] not in ("", *ACCEPT) for r in rows)
n_both = sum(r["both_disagree"] for r in rows)
print(f"{len(rows)} images | ensemble disagrees with label: {n_model} (p(label) < 0.1: {n_low}) | "
      f"teacher rejects label: {n_teacher} of {sum(r['teacher_verdict'] != '' for r in rows)} annotated | both: {n_both}")

top = rows[:args.top]
W, H, cols = 260, 260, 8
sheet = Image.new("RGB", (cols * W, ((len(top) + cols - 1) // cols) * (H + 44)), "white")
d = ImageDraw.Draw(sheet)
for i, r in enumerate(top):
    im = Image.open(r["image"]).convert("RGB")
    im.thumbnail((W, H))
    x, y = (i % cols) * W, (i // cols) * (H + 44)
    sheet.paste(im, (x + (W - im.width) // 2, y))
    d.text((x + 3, y + H + 2), f"{i + 1}. label {r['label']} -> {r['ensemble_pred']} ({r['p_pred']:.2f})", fill="black")
    d.text((x + 3, y + H + 16), f"teacher: {r['teacher_verdict'] or '-'}", fill="gray")
    d.text((x + 3, y + H + 30), os.path.basename(r["image"])[:40], fill="gray")
sheet.save(args.sheet, quality=85)
print("top candidates ->", args.sheet)
for i, r in enumerate(top[:12], 1):
    print(i, r["label"], "->", r["ensemble_pred"], r["p_label"], r["teacher_verdict"], r["image"])
