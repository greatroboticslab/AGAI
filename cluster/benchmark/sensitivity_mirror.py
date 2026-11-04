"""Sensitivity of the split v1 test results to the mirror-aware near-duplicate audit: accuracy and macro accuracy of
each per-image result file with and without the test images that have a near-duplicate (image or mirror) in training
or validation (cluster/benchmark/mirror_dupes.py -> evaluation/mirror_dupes_v1.json).
  python cluster/benchmark/sensitivity_mirror.py evaluation/mirror_dupes_v1.json evaluation/ft_*_split_v1_test.json ...
Result files hold {"results": [{"image", "truth", "pred", ...}]} (or a top-level list)."""
import json
import os
import sys
from collections import defaultdict

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
audit = json.load(open(sys.argv[1]))
drop = set(audit["test_images_affected"])
print(f"{len(drop)} test images dropped (near-duplicate, image or mirror, in training or validation)")


def load(path):
    d = json.load(open(path))
    rows = d.get("results") if isinstance(d, dict) else d
    out = []
    for r in rows:
        img, truth, pred = r.get("image"), r.get("truth", r.get("label")), r.get("pred", r.get("prediction"))
        if img is None or truth is None or pred is None:
            continue
        out.append((img, truth, pred))
    return out


def score(rows):
    n = len(rows)
    acc = sum(t == p for _, t, p in rows) / n if n else float("nan")
    per = defaultdict(lambda: [0, 0])
    for _, t, p in rows:
        per[t][1] += 1
        per[t][0] += (t == p)
    macro = sum(c / m for c, m in per.values()) / len(per) if per else float("nan")
    return n, acc, macro


print(f"{'file':64s} {'n':>4s} {'acc':>6s} {'acc-10':>7s} {'delta':>6s} {'macro':>6s} {'macro-10':>8s}")
for path in sys.argv[2:]:
    rows = load(path)
    if not rows:
        print(f"{os.path.basename(path):64s} (no image/truth/pred records)")
        continue
    n, a, m = score(rows)
    kept = [r for r in rows if r[0] not in drop]
    n2, a2, m2 = score(kept)
    dropped = n - n2
    print(f"{os.path.basename(path)[:64]:64s} {n:4d} {100*a:6.1f} {100*a2:7.1f} {100*(a2-a):+6.1f} {100*m:6.1f} {100*m2:8.1f}   (dropped {dropped})")
