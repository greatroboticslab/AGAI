"""Import ChatGPT-generated strawberry photos (raw PNGs, Downloads/agai_synth/<class>/<class>_b<N>_<i>.png, uploaded
to /projects/$USER/agai/synth/chatgpt_raw/) as training-only images that do not stand out by format:
  - deterministic per-image random crop to a random aspect ratio (raw ChatGPT outputs come in a few fixed sizes and
    orientations per batch, which a classifier could learn as a shortcut),
  - phone-like downscale and JPEG compression (raw outputs are pristine PNGs; the real photos are compressed JPEGs),
  - pHash dedup within the pool and against every real split v1 val/test image (distance <= 6),
  - a manifest tying each image to its batch prompt, severity and whether a hand is in the prompt.
Writes /projects/$USER/agai/synth/<name>/<class>/*.jpg and evaluation/synth_<name>_manifest.json.
  python cluster/synthetic/import_synth.py --raw /projects/$USER/agai/synth/chatgpt_raw --index a.json,b.json --name chatgpt_v1
"""
import argparse
import hashlib
import json
import os
import random
import re

import imagehash
from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument("--raw", required=True)
ap.add_argument("--index", required=True, help="batch prompt index JSON files, comma-separated (later ones win)")
ap.add_argument("--name", default="chatgpt_v1")
ap.add_argument("--max-dist", type=int, default=6)
args = ap.parse_args()

prompts = {}   # (class, batch number) -> list of prompt records in order
for f in args.index.split(","):
    for b in json.load(open(f)):
        n = int(re.search(r"_(\d+)(?:\.txt)?$", b["batch"]).group(1))
        prompts[(b["class"], n)] = [p if isinstance(p, dict) else {"text": p, "severity": "unrated"} for p in b["prompts"]]

split = json.load(open("evaluation/split_v1.json"))
held = []
for s in ("val", "test"):
    for c, ps in split[s].items():
        for p in ps:
            with Image.open(p) as im:
                held.append(int(str(imagehash.phash(im.convert("RGB"))), 16))
out_root = f"/projects/{os.environ['USER']}/agai/synth/{args.name}"
ASPECTS = [(3, 4), (4, 3), (1, 1), (2, 3), (3, 2), (9, 16), (16, 9)]
review_path = os.path.join(args.raw, "review.json")
review = json.load(open(review_path)) if os.path.exists(review_path) else {}
excluded = review.get("exclude", {})
overrides = review.get("override", {})   # e.g. a hand ChatGPT added although the prompt asked for none
kept, dropped, seen_sha, kept_h = [], [], set(), []
for cls in sorted(os.listdir(args.raw)):
    d = os.path.join(args.raw, cls)
    if not os.path.isdir(d):
        continue
    os.makedirs(os.path.join(out_root, cls), exist_ok=True)
    uncertain = {int(m.group(1)) for m in (re.match(rf"{cls}_b(\d+)_MAPPING_UNCERTAIN\.txt$", f) for f in os.listdir(d)) if m}
    for f in sorted(os.listdir(d)):
        m = re.match(rf"{cls}_b(\d+)_(\d+)\.png$", f)
        if not m:
            continue
        b, i = int(m.group(1)), int(m.group(2))
        if f"{cls}/{f}" in excluded:   # failed the visual review (review.json)
            dropped.append({"file": f, "why": "review: " + excluded[f"{cls}/{f}"]})
            continue
        src = os.path.join(d, f)
        raw = open(src, "rb").read()
        sha = hashlib.sha256(raw).hexdigest()
        if sha in seen_sha:
            dropped.append({"file": f, "why": "exact duplicate"})
            continue
        seen_sha.add(sha)
        rng = random.Random(sha)
        img = Image.open(src).convert("RGB")
        aw, ah = rng.choice(ASPECTS)
        W, H = img.size
        scale = rng.uniform(0.75, 1.0)
        cw = min(W, int(H * aw / ah)) * scale
        ch = cw * ah / aw
        if ch > H * scale:
            ch = H * scale
            cw = ch * aw / ah
        x0 = rng.uniform(0, W - cw)
        y0 = rng.uniform(0, H - ch)
        img = img.crop((int(x0), int(y0), int(x0 + cw), int(y0 + ch)))
        long_side = rng.choice([800, 1024, 1280, 1600])
        if max(img.size) > long_side:
            s = long_side / max(img.size)
            img = img.resize((max(1, int(img.width * s)), max(1, int(img.height * s))), Image.BICUBIC)
        h = int(str(imagehash.phash(img)), 16)
        if any(bin(h ^ q).count("1") <= args.max_dist for q in held):
            dropped.append({"file": f, "why": "near-copy of a real val/test image"})
            continue
        if any(bin(h ^ q).count("1") <= args.max_dist for q in kept_h):
            dropped.append({"file": f, "why": "near-duplicate within the synthetic pool"})
            continue
        q = rng.randint(70, 92)
        name = f"gpt_{cls}_b{b}_{i:02d}.jpg"
        img.save(os.path.join(out_root, cls, name), "JPEG", quality=q)
        pr = prompts.get((cls, b), [])
        # ChatGPT sometimes returns 9 images for 10 prompts; then image i no longer matches prompt i
        rec = pr[i - 1] if 0 < i <= len(pr) and b not in uncertain else {"text": "", "severity": "unknown"}
        # v3 indexes say whether a hand was asked for; older ones only have the text ("no hands in the frame" is not one)
        hand = rec["hand"] if "hand" in rec else (bool(re.search(r"(?<!no )\bhands?\b|thumb", rec["text"])) if rec["text"] else None)
        hand = overrides.get(f"{cls}/{f}", {}).get("hand", hand)
        kept.append({"file": f"{cls}/{name}", "class": cls, "source": f, "batch": b, "index": i,
                     "prompt": rec["text"], "severity": rec.get("severity", "unrated"), "hand": hand,
                     "quality": rec.get("quality", "unrated"), "group": rec.get("group", ""), "aspect": f"{aw}:{ah}",
                     "size": list(img.size), "jpeg_quality": q, "sha256_raw": sha})
        kept_h.append(h)
summary = {}
for r in kept:
    s = summary.setdefault(r["class"], {"n": 0, "hand": 0, "hand_unknown": 0})
    s["n"] += 1
    s["hand"] += bool(r["hand"])
    s["hand_unknown"] += r["hand"] is None
json.dump({"name": args.name, "root": out_root, "summary": summary, "kept": kept, "dropped": dropped},
          open(f"evaluation/synth_{args.name}_manifest.json", "w"), indent=1)
print(json.dumps(summary), "| dropped", len(dropped), [d["why"] for d in dropped][:5])
