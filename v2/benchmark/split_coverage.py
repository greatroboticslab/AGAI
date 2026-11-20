"""Which split v1 images are in the repository's photo copy, and which are not.

    python -P v2/benchmark/split_coverage.py --split evaluation/split_v1.json --root v1 --out evaluation/split_v1_coverage.json

The split lists paths relative to v1/ (plant_diagnostic/data/...). Writes, per split and condition, how many listed
images exist under --root, and the list of those that do not, so that a reader knows which part of the benchmark can
be rebuilt from the public repository alone.
"""
import argparse
import json
import os
from collections import Counter


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True)
    ap.add_argument("--root", required=True, help="folder the split paths are relative to (v1/)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    s = json.load(open(a.split, encoding="utf-8"))
    out = {"split_file": a.split, "root": a.root, "by_split": {}, "missing": []}
    tot, miss = 0, 0
    for part in ("train", "val", "test"):
        rows = {}
        for cond, paths in s[part].items():
            gone = [p for p in paths if not os.path.exists(os.path.join(a.root, p))]
            rows[cond] = {"listed": len(paths), "missing": len(gone)}
            out["missing"] += [{"split": part, "condition": cond, "path": p} for p in gone]
            tot += len(paths)
            miss += len(gone)
        out["by_split"][part] = rows
    out["total_listed"], out["total_missing"] = tot, miss
    folders = Counter(os.path.dirname(m["path"]).split("/")[2] if m["path"].count("/") >= 3 else "?" for m in out["missing"])
    out["missing_by_source_folder"] = dict(folders)
    with open(a.out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(out, f, indent=1)
    print(f"{miss} of {tot} listed images are not under {a.root}")
    for part, rows in out["by_split"].items():
        print(part, {c: f"{r['missing']}/{r['listed']}" for c, r in rows.items()})
    print("missing by source folder:", out["missing_by_source_folder"])


if __name__ == "__main__":
    main()
