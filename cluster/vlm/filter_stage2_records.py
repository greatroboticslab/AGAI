"""Split the v1 report writer's instruction records by split v1, so the retrained writer never sees a validation or
test photograph. Each record points at a train_aug file: an original photograph or an offline augmented copy
"<stem>_augNNN.jpg" of one. A record follows its source photograph: split v1 train -> training records, val ->
validation records (checkpoint selection), test -> dropped. Records whose source cannot be matched to the split
(by class folder + file stem, then by SHA-256 of the file) are dropped and listed.
  python cluster/vlm/filter_stage2_records.py --records plant_diagnostic/datasets/stage2_train_7class_fixed.json \
      --image-root /projects/$USER/agai/MiniGPT-4 --out-dir plant_diagnostic/datasets/split_v1
Writes <out-dir>/stage2_train.json, stage2_val.json and stage2_filter_report.json. Image paths in the output are
relative to --image-root (the dataset config's image root).
"""
import argparse
import collections
import hashlib
import json
import os
import re

AUG = re.compile(r"^(.*)_aug\d+$")


def stem_of(path):
    s = os.path.splitext(os.path.basename(path))[0]
    m = AUG.match(s)
    return (m.group(1) if m else s), bool(m)


def sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", required=True)
    ap.add_argument("--split", default="evaluation/split_v1.json")
    ap.add_argument("--image-root", required=True, help="repo root the records' train_aug paths resolve under")
    ap.add_argument("--old-root", default="/data/AGAI/MiniGPT-4", help="root prefix stored in the records")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    split = json.load(open(args.split))
    where, by_sha = {}, {}
    for s in ("train", "val", "test"):
        for cls, paths in split[s].items():
            for p in paths:
                where[(cls, stem_of(p)[0])] = s
    recs = json.load(open(args.records, encoding="utf-8"))
    out = {"train": [], "val": []}
    dropped, unmatched = collections.Counter(), []
    for r in recs:
        old = r["image"]
        rel = old[len(args.old_root):].lstrip("/") if old.startswith(args.old_root) else old
        cls = rel.split("/")[-2]
        stem, is_aug = stem_of(rel)
        s = where.get((cls, stem))
        if s is None:  # exact-content fallback: the same photo under another name
            full = os.path.join(args.image_root, rel)
            if os.path.exists(full):
                if not by_sha:
                    for ss in ("train", "val", "test"):
                        for c, paths in split[ss].items():
                            for p in paths:
                                fp = os.path.join(args.image_root, p)
                                if os.path.exists(fp):
                                    by_sha[sha(fp)] = ss
                s = by_sha.get(sha(full))
        if s is None:
            unmatched.append(rel)
            dropped["unmatched"] += 1
            continue
        if s == "test":
            dropped["test"] += 1
            continue
        out[s].append({"image": rel, "conversations": r["conversations"], "source_split": s, "augmented": is_aug})
    os.makedirs(args.out_dir, exist_ok=True)
    for s in out:
        json.dump(out[s], open(os.path.join(args.out_dir, f"stage2_{s}.json"), "w"), indent=1)
    per = {s: collections.Counter(x["image"].split("/")[-2] for x in out[s]) for s in out}
    report = {"records": len(recs), "train": len(out["train"]), "val": len(out["val"]), "dropped": dict(dropped),
              "per_class": {s: dict(per[s]) for s in per}, "unmatched_examples": unmatched[:20]}
    json.dump(report, open(os.path.join(args.out_dir, "stage2_filter_report.json"), "w"), indent=1)
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
