"""Compare two report runs of the same writer on the same photographs, paired per photograph and condition (notebook
entry (ae): the training prompt format against the application's). Uses the measures of score_reports.py.
    python cluster/vlm/compare_report_runs.py evaluation/paper_data/reports_splitv1.json \
        evaluation/paper_data/reports_splitv1_humantpl.json
"""
import json
import os
import sys

_argv, sys.argv = sys.argv, sys.argv[:1]          # import score_reports without running its command line
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from score_reports import PART_RE, diagnosis_ok, mcnemar, mentions  # noqa: E402
sys.argv = _argv


def flags(r):
    m = mentions(r["final_report"])
    absent = [p for p in PART_RE if p not in r["annotated_parts"]]
    not_visible = [p for p in PART_RE if p not in r["detected_parts"]] if r["condition"] == "parts" else []
    return {"h_part": any(m[p][1] for p in absent),
            "h_part_no_soil": any(m[p][1] for p in absent if p != "soil"),
            "names_not_visible": any(m[p][1] for p in not_visible),
            "consistent": diagnosis_ok(r["label"], r["final_report"]),
            "regenerated": len(r["attempts"]) > 1, "length": len(r["final_report"].split())}


a_path, b_path = sys.argv[1], sys.argv[2]
A = {(r["image"], r["condition"]): flags(r) for r in json.load(open(a_path))}
B = {(r["image"], r["condition"]): flags(r) for r in json.load(open(b_path))}
assert A.keys() == B.keys(), "the two runs cover different photographs or conditions"
print(f"A = {a_path}\nB = {b_path}")
for cond in ("parts", "noparts"):
    keys = sorted(k for k in A if k[1] == cond)
    print(f"\n[{cond}] {len(keys)} photographs")
    for m in ("h_part", "h_part_no_soil", "names_not_visible", "consistent", "regenerated"):
        if m == "names_not_visible" and cond == "noparts":
            continue
        a = sum(A[k][m] for k in keys); b = sum(B[k][m] for k in keys)
        only_a = sum(A[k][m] and not B[k][m] for k in keys); only_b = sum(B[k][m] and not A[k][m] for k in keys)
        print(f"  {m:18s} A {100 * a / len(keys):5.1f} %  B {100 * b / len(keys):5.1f} %  "
              f"(A only {only_a} / B only {only_b}, exact McNemar P = {mcnemar(only_a, only_b):.3g})")
    print(f"  mean length        A {sum(A[k]['length'] for k in keys) / len(keys):.0f}  "
          f"B {sum(B[k]['length'] for k in keys) / len(keys):.0f} words")
