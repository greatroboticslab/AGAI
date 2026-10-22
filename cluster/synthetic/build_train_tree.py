"""Training tree = split v1 train (real photos) + imported synthetic images (manifest from cluster/synthetic/import_synth.py).
--hand-cap: per class, at most this share of the added synthetic images may have a hand/thumb in their prompt
(excess hand images are dropped deterministically), so a hand in frame cannot predict the label. Images whose prompt
is unknown (incomplete ChatGPT batches) count as no-hand. Val/test stay split v1 (real only).
--holdout: whole batches kept out of training; listed as probe_files for cluster/classifier/eval_probe.py.
Writes /projects/$USER/agai/splits/<name>/train/<class>/ (symlinks) and evaluation/split_<name>.json.
  python cluster/synthetic/build_train_tree.py --manifest evaluation/synth_chatgpt_v1_manifest.json --name v1_chatgpt \
      --hand-cap 0.33 --holdout healthy_5,root_rot_5
"""
import argparse
import collections
import json
import os
import random
import shutil

ap = argparse.ArgumentParser()
ap.add_argument("--manifest", required=True)
ap.add_argument("--name", required=True)
ap.add_argument("--classes", default="", help="only add synthetic images of these classes (default: all)")
ap.add_argument("--hand-cap", type=float, default=0.33)
ap.add_argument("--holdout", default="", help="batches kept out of training as a probe set, e.g. healthy_5,root_rot_5")
args = ap.parse_args()
m = json.load(open(args.manifest))
split = json.load(open("evaluation/split_v1.json"))
want = set(args.classes.split(",")) if args.classes else None
held = set(args.holdout.split(",")) if args.holdout else set()
by = collections.defaultdict(list)
probe = []
for r in m["kept"]:
    if f"{r['class']}_{r['batch']}" in held:
        probe.append(r["file"])
    elif want is None or r["class"] in want:
        by[r["class"]].append(r)
added = {}
for cls, rs in by.items():
    hands = [r for r in rs if r["hand"]]
    rest = [r for r in rs if not r["hand"]]
    max_hands = int(args.hand_cap * len(rest) / (1 - args.hand_cap))   # hands / (hands + rest) <= cap
    random.Random(f"handcap:{cls}").shuffle(hands)
    added[cls] = rest + hands[:max_hands]
out = f"/projects/{os.environ['USER']}/agai/splits/{args.name}/train"
if os.path.isdir(out):
    shutil.rmtree(out)   # symlinks only
for c, ps in split["train"].items():
    os.makedirs(os.path.join(out, c), exist_ok=True)
    for p in ps:
        src = p.split("plant_diagnostic/data/")[-1].split("/")[0]
        os.symlink(os.path.abspath(p), os.path.join(out, c, f"{src}__{os.path.basename(p)}"))
for cls, rs in added.items():
    os.makedirs(os.path.join(out, cls), exist_ok=True)
    for r in rs:
        os.symlink(os.path.join(m["root"], r["file"]), os.path.join(out, cls, f"synth__{os.path.basename(r['file'])}"))
counts = {c: {"real": len(split["train"][c]), "synthetic": len(added.get(c, [])),
              "synthetic_hand": sum(bool(r["hand"]) for r in added.get(c, []))} for c in sorted(split["train"])}
json.dump({"name": args.name, "manifest": args.manifest, "hand_cap": args.hand_cap, "train_root": out,
           "counts": counts, "synthetic_files": {c: [r["file"] for r in rs] for c, rs in added.items()},
           "holdout": sorted(held), "probe_files": probe},
          open(f"evaluation/split_{args.name}.json", "w"), indent=1)
print(json.dumps(counts), "| probe (held out):", len(probe))
