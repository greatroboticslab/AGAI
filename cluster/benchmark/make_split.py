"""Build one leak-free train/val/test split that every model (ResNet, fine-tuned VLMs) trains and tests on.

Pool: the 1580 deduplicated human-labeled originals (evaluation/clean_eval_sets.json, vlm_all_originals; no
synthetic or augmented images). Near-duplicate groups (evaluation/near_dupes.json) are assigned whole, so no
photo's copies straddle splits; groups whose copies carry different labels are excluded. Per class, groups are
shuffled (seed 0) and filled into test (>= 25% of the class), then val (>= 10%), then train.
Writes evaluation/split_v1.json and ImageFolder-style symlink trees in /projects/$USER/agai/splits/v1/{split}/{class}/.
"""
import collections
import json
import os
import random

POOL = json.load(open("evaluation/clean_eval_sets.json"))["vlm_all_originals"]
ND = json.load(open("evaluation/near_dupes.json"))
group_of = {p: gi for gi, g in enumerate(ND["groups"]) for p in g}
conflict = {p for g in ND["cross_label_groups"] for p in g}
out_root = f"/projects/{os.environ['USER']}/agai/splits/v1"

rng = random.Random(0)
split = {s: collections.defaultdict(list) for s in ("train", "val", "test")}
excluded = []
for cls, paths in sorted(POOL.items()):
    groups = collections.defaultdict(list)
    for p in paths:
        if p in conflict:
            excluded.append(p)
            continue
        groups[group_of.get(p, p)].append(p)
    order = list(groups.values())
    rng.shuffle(order)
    n, filled = sum(len(g) for g in order), {"test": 0, "val": 0}
    for g in order:
        s = "test" if filled["test"] < 0.25 * n else "val" if filled["val"] < 0.10 * n else "train"
        if s in filled:
            filled[s] += len(g)
        split[s][cls].extend(g)

for s, by_cls in split.items():
    for cls, paths in by_cls.items():
        d = os.path.join(out_root, s, cls)
        os.makedirs(d, exist_ok=True)
        for p in paths:
            link = os.path.join(d, f"{p.split('/')[-3]}__{os.path.basename(p)}")   # e.g. holdout__drought_1.jpg
            if not os.path.islink(link):
                os.symlink(os.path.abspath(p), link)

json.dump({"pool": "vlm_all_originals", "seed": 0, "excluded_label_conflicts": excluded,
           **{s: dict(v) for s, v in split.items()}}, open("evaluation/split_v1.json", "w"), indent=1)
classes = sorted(POOL)
print("split      " + " ".join(f"{c[:10]:>10s}" for c in classes) + "   total")
for s in ("train", "val", "test"):
    print(f"{s:10s} " + " ".join(f"{len(split[s][c]):10d}" for c in classes) + f"   {sum(len(v) for v in split[s].values())}")
print(f"excluded (label conflicts): {len(excluded)}; symlink trees in {out_root}")
