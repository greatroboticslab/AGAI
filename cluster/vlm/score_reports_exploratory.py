"""Exploratory (post hoc, not pre-registered) breakdowns of the Table S16 report sets, with score_reports.py's own
mention and negation rules: H_part without soil, false assertions per part, how often the prompt's NOT VISIBLE list was
disobeyed, the detector lists against the annotation, and paired tests of each variant. Written after the pre-registered
scoring showed that soil (which the rule counts together with mulch and substrate) dominates the false mentions.
  python cluster/score_reports_exploratory.py evaluation/reports_splitv1.json [more.json ...]
"""
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_argv, sys.argv = sys.argv, sys.argv[:1]     # score_reports.py scores sys.argv[1:] at import
import score_reports as SR  # noqa: E402

sys.argv = _argv
PARTS = list(SR.PART_RE)


def pct(v):
    return f"{100 * sum(v) / len(v):.1f} %"


def summarise(rows, path):
    print(f"\n### {path}: {len(rows)} reports")
    by = {c: [r for r in rows if r["condition"] == c] for c in ("noparts", "parts")}
    variants = {}
    for cond, rs in by.items():
        if not rs:
            continue
        ment = [SR.mentions(r["final_report"]) for r in rs]
        absent = [[p for p in PARTS if p not in r["annotated_parts"]] for r in rs]
        v = {"h_part": [any(m[p][1] for p in a) for m, a in zip(ment, absent)],
             "h_part_without_soil": [any(m[p][1] for p in a if p != "soil") for m, a in zip(ment, absent)]}
        per_part = {p: f"{sum(m[p][1] for m, a in zip(ment, absent) if p in a)}/{sum(p in a for a in absent)}" for p in PARTS}
        low = [r["final_report"].lower() for r in rs]
        words = {w: sum(bool(re.search(rf"\b{w}\b", t)) for t in low) for w in ("soil", "mulch", "substrate", "ground")}
        print(f"[{cond}] n={len(rs)} H_part {pct(v['h_part'])} | without soil {pct(v['h_part_without_soil'])} | "
              f"false assertions / photographs without the part {per_part} | reports using the words {words}")
        if cond == "parts":
            told = [[p for p in PARTS if p not in r["detected_parts"]] for r in rs]
            v["names_not_visible_part"] = [any(m[p][1] for p in t) for m, t in zip(ment, told)]
            v["names_not_visible_part_without_soil"] = [any(m[p][1] for p in t if p != "soil") for m, t in zip(ment, told)]
            missed = sum(any(p in r["annotated_parts"] and p not in r["detected_parts"] for p in PARTS) for r in rs)
            extra = sum(any(p not in r["annotated_parts"] and p in r["detected_parts"] for p in PARTS) for r in rs)
            print(f"   names a part the prompt called NOT VISIBLE: {pct(v['names_not_visible_part'])}, without soil "
                  f"{pct(v['names_not_visible_part_without_soil'])}; the detector's list misses an annotated part in {missed} "
                  f"and adds an unannotated one in {extra} of {len(rs)} photographs")
        variants[cond] = v
    if "parts" in variants and "noparts" in variants:
        ia = {r["image"]: i for i, r in enumerate(by["parts"])}
        ib = {r["image"]: i for i, r in enumerate(by["noparts"])}
        common = sorted(set(ia) & set(ib))
        for key in ("h_part", "h_part_without_soil"):
            pa, pb = variants["parts"][key], variants["noparts"][key]
            only_a = sum(pa[ia[i]] and not pb[ib[i]] for i in common)
            only_b = sum(pb[ib[i]] and not pa[ia[i]] for i in common)
            print(f"   paired {key}: with-parts only {only_a} / without-parts only {only_b} of {len(common)}, "
                  f"exact McNemar P = {SR.mcnemar(only_a, only_b):.4f}")


if __name__ == "__main__":
    for path in sys.argv[1:]:
        summarise(json.load(open(path)), path)
