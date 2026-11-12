"""Blind claim set for the v1 vs v2 report-faithfulness ratings (lab notebook entry (q), 2026-11-04).

One claim = one sentence of a report. The v1 reports are the `report` fields of evaluation/paper_data/latency_v1.json
(retrained MiniGPT-v2 writer, as deployed), the v2 reports those of latency_student.json (distilled 8B student), for
the same 49 seeded split v1 test photographs. Per photograph the claims of both reports are pooled and shuffled with
a seeded generator (`faith-v1v2:<photo index>`), so the rating page shows them without their system.

  python cluster/vlm/make_faithfulness_claims.py --check
      rebuild the claim set and compare it with evaluation/paper_data/faithfulness_v1v2_claims.json
  python cluster/vlm/make_faithfulness_claims.py --key-out PATH
      also write the key (claim id -> system and sentence index; photograph id -> labels and the student's verdict).
      The key un-blinds the ratings: do not open it while rating. cluster/vlm/score_faithfulness.py needs it.
  python cluster/vlm/make_faithfulness_claims.py --claims-out PATH
      write the blind claim set (no system information).

Sentence rule: split at sentence punctuation and line breaks; list numbering and the section labels "Visible cues:" /
"Recommendation:" are removed; v1's "Diagnosis: <label>" line is dropped because it restates the label; fragments
shorter than 20 characters are dropped.
"""
import argparse
import json
import os
import random
import re
import sys
from collections import Counter

ap = argparse.ArgumentParser()
ap.add_argument("--v1", default="evaluation/paper_data/latency_v1.json")
ap.add_argument("--v2", default="evaluation/paper_data/latency_student.json")
ap.add_argument("--claims", default="evaluation/paper_data/faithfulness_v1v2_claims.json", help="existing claim set for --check")
ap.add_argument("--check", action="store_true")
ap.add_argument("--claims-out", default="")
ap.add_argument("--key-out", default="")
args = ap.parse_args()

v1 = json.load(open(args.v1, encoding="utf-8"))["records"]
v2 = json.load(open(args.v2, encoding="utf-8"))["records"]
assert [r["image"] for r in v1] == [r["image"] for r in v2], "the two systems must cover the same photographs in the same order"


def sentences(text, system):
    t = text.replace("\r", "")
    if system == "v1":
        t = re.sub(r"^\s*Diagnosis:\s*[^\n]*\n", "", t.strip() + "\n", count=1, flags=re.I)   # the label line is not a claim
    t = re.sub(r"\n\s*\d+[.)]\s*(Visible cues|Recommendations?|Diagnosis|Treatment)\s*:\s*", "\n", t, flags=re.I)
    t = re.sub(r"^\s*(Visible cues|Recommendations?)\s*:\s*", "", t, flags=re.I | re.M)
    t = re.sub(r"\n\s*[-*•]\s*", "\n", t)
    parts = re.split(r"(?<=[.!?])\s+|\n+", t)
    out = []
    for p in parts:
        p = re.sub(r"^\s*\d+[.)]\s*", "", p).strip()
        if len(p) >= 20:
            out.append(p)
    return out


photos, key, stats = [], {}, Counter()
for i, (a, b) in enumerate(zip(v1, v2)):
    rng = random.Random(f"faith-v1v2:{i}")
    pool = [("v1", k, s) for k, s in enumerate(sentences(a["report"], "v1"))] + \
           [("v2", k, s) for k, s in enumerate(sentences(b["report"], "v2"))]
    rng.shuffle(pool)
    pid = f"p{i:02d}"
    claims = []
    for j, (system, k, s) in enumerate(pool):
        cid = f"{pid}c{j:02d}"
        claims.append({"id": cid, "text": s})
        key[cid] = {"system": system, "order": k}
        stats[system] += 1
    photos.append({"id": pid, "image": a["image"], "label": a["label"], "file": os.path.basename(a["image"]), "claims": claims})
    key[pid] = {"v1_label": a["final_label"], "v1_p1": a["p1"], "v2_given_label": b["given_label"],
                "v2_verdict": b["verdict"], "dataset_label": a["label"]}

blind = {"name": "faithfulness_v1v2", "seed": "faith-v1v2:<photo index>", "photos": photos}
print(f"{len(photos)} photographs, {sum(stats.values())} claims: v1 {stats['v1']}, v2 {stats['v2']}")

if args.check:
    old = json.load(open(args.claims, encoding="utf-8"))
    same = [(p["id"], [(c["id"], c["text"]) for c in p["claims"]]) for p in old["photos"]] == \
           [(p["id"], [(c["id"], c["text"]) for c in p["claims"]]) for p in photos]
    print("identical to", args.claims if same else "DIFFERENT from " + args.claims)
    if not same:
        sys.exit(1)
if args.claims_out:
    json.dump(blind, open(args.claims_out, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print("blind claim set ->", args.claims_out)
if args.key_out:
    json.dump(key, open(args.key_out, "w", encoding="utf-8"), indent=1)
    print("key (un-blinds the ratings; do not open while rating) ->", args.key_out)
