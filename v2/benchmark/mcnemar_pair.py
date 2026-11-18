"""Exact McNemar test between two per-image result files on the images they share.
  python cluster/benchmark/mcnemar_pair.py A.json B.json [more pairs as A B ...]
Prints images only A gets right / only B gets right and the two-sided exact P (binomial)."""
import json
import math
import sys


def load(path):
    d = json.load(open(path))
    rows = d.get("results") if isinstance(d, dict) else d
    return {r["image"]: r.get("pred", r.get("prediction")) == r.get("truth", r.get("label")) for r in rows}


args = sys.argv[1:]
for a, b in zip(args[::2], args[1::2]):
    A, B = load(a), load(b)
    common = sorted(set(A) & set(B))
    only_a = sum(1 for i in common if A[i] and not B[i])
    only_b = sum(1 for i in common if B[i] and not A[i])
    n = only_a + only_b
    p = min(1.0, 2 * sum(math.comb(n, k) for k in range(0, min(only_a, only_b) + 1)) / 2 ** n) if n else 1.0
    acc_a = sum(A[i] for i in common) / len(common)
    acc_b = sum(B[i] for i in common) / len(common)
    print(f"{a.split('/')[-1]} ({100*acc_a:.1f} %) vs {b.split('/')[-1]} ({100*acc_b:.1f} %): n={len(common)}, only A {only_a} / only B {only_b}, P = {p:.4g}")
