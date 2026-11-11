"""Turn William's per-image calibration files (evaluation/paper_data/calib_<model>_test_seed<k>.json, commit ee395a0)
into the evaluation-JSON schema that fig_classifier_results.py reads, pooling the seeds.

usage: python calib_to_eval.py <out.json> <calib_seed0.json> [<calib_seed1.json> ...]
Each prediction keeps the temperature-scaled top-1 probability (p1) and the deployed v1 gate decision.
"""
import json
import sys

out, files = sys.argv[1], sys.argv[2:]
classes = None
preds = []
for f in files:
    d = json.load(open(f, encoding="utf-8"))
    classes = classes or d["classes"]
    assert d["classes"] == classes, f
    for r in d["results"]:
        preds.append({"truth": r["truth"], "pred_top1": r["pred"], "p1": r["p1"],
                      "gate_v1": bool(r["accepted_v1_gate"]), "seed_file": f.rsplit("/", 1)[-1]})
K = len(classes)
cm = [[0] * K for _ in range(K)]
for p in preds:
    cm[classes.index(p["truth"])][classes.index(p["pred_top1"])] += 1
json.dump({"class_order": classes, "confusion_matrix": cm, "predictions": preds, "n_files": len(files)},
          open(out, "w", encoding="utf-8"))
print(len(files), "files,", len(preds), "predictions ->", out)
