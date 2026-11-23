"""Tables and prediction checks for the external test of lab notebook (al), read from the result file (no number typed
by hand).

    python -P v2/classifier/external_test_summary.py evaluation/paper_data/external_mendeley_test.json
"""
import collections
import json
import sys

d = json.load(open(sys.argv[1], encoding="utf-8"))
recs, summ = d["records"], d["summary"]
NAMES = {"resnet": "v1 ResNet-50 (split v1)", "siglip": "v2 SigLIP 2 (recipe)"}
SPLIT_V1 = {"resnet": {"gray_mold": 98.8, "healthy": 86.4}, "siglip": {"gray_mold": 96.3, "healthy": 95.1}}  # entry (al)
print("photographs:", summ["n_images"], "| categories:", summ["categories"])
for g, v in summ["groups"].items():
    print(f"\n== {g}: n = {v['n']}")
    for key in ("resnet", "siglip"):
        m = v[key]
        line = f"  {NAMES[key]:26s}"
        if "recall_ungated" in m:
            line += f" recall {100 * m['recall_ungated']:.1f} %;"
        for gate in ("v1", "val"):
            line += f" {gate} gate accepts {100 * m[f'accepted_{gate}_gate']:.1f} %"
            acc = m.get(f"accuracy_accepted_{gate}_gate")
            if acc is not None:
                line += f" (right among accepted {100 * acc:.1f} %)"
            line += ";"
        print(line)
        print("      labels given:", m["predicted"])

# per out-of-scope disease
by = collections.defaultdict(list)
for r in recs:
    by[tuple(r["categories"])].append(r)
print("\n== per category")
for cs, rs in sorted(by.items()):
    for key in ("resnet", "siglip"):
        preds = collections.Counter(r[key]["pred"] for r in rs)
        acc = {g: sum(r[key][f"accept_{g}_gate"] for r in rs) for g in ("v1", "val")}
        print(f"  {', '.join(cs):26s} {key:7s} n={len(rs)}  accepted v1 gate {acc['v1']}, val gate {acc['val']}  labels {dict(preds.most_common(4))}")

# predictions of (al)
g = summ["groups"]
gm, he, oo = g.get("in scope: gray mold"), g.get("in scope: healthy"), g.get("out of scope: other diseases")
print("\n== predictions of (al)")
for key in ("resnet", "siglip"):
    print(f"  P1 {NAMES[key]}: gray mold recall {100 * gm[key]['recall_ungated']:.1f} % here against {SPLIT_V1[key]['gray_mold']} % on split v1; "
          f"healthy {100 * he[key]['recall_ungated']:.1f} % against {SPLIT_V1[key]['healthy']} %")
print(f"  P2 v2 at least v1: gray mold {100 * gm['siglip']['recall_ungated']:.1f} vs {100 * gm['resnet']['recall_ungated']:.1f}; "
      f"healthy {100 * he['siglip']['recall_ungated']:.1f} vs {100 * he['resnet']['recall_ungated']:.1f}")
print("  P3 each gate accepts more than half of the out-of-scope photographs:",
      {f"{key}/{gate}": round(100 * oo[key][f"accepted_{gate}_gate"], 1) for key in ("resnet", "siglip") for gate in ("v1", "val")})
fruit = [r for r in recs if any(c in ("Anthracnose Fruit Rot", "Blossom Blight", "Powderly Mildew Fruit") for c in r["categories"])]
for key in ("resnet", "siglip"):
    mold = sum(r[key]["pred"] in ("gray_mold", "white_mold") for r in fruit)
    print(f"  P4 {NAMES[key]}: out-of-scope fruit diseases called gray or white mold: {mold} of {len(fruit)} ({100 * mold / len(fruit):.1f} %)")
print("\ngates:", json.dumps(d["gates"]))
