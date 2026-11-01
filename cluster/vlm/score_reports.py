"""Score generated reports (cluster/minigpt_reports.py output) for the detector on/off ablation (manuscript eq. 11,
Table S16). Per prompt condition ("parts": the application's prompt with the detector's part lists, "noparts": the
same prompt without them):
  part-hallucination rate H_part: share of reports that assert at least one plant part absent from the photograph's
      annotation (asserted = mentioned and not negated within the sentence, same negation rules as the consistency
      check); also reported with every mention counted, and against the detector's list ("not visible" parts named)
  condition consistency: share of reports whose "Diagnosis:" line names the given condition
  regeneration rate: share of reports the consistency check flagged at least once
  mean report length in words; per-part false mention counts
Bootstrap 95 % intervals over photographs; the paired difference between conditions with an exact McNemar test.
  python cluster/score_reports.py evaluation/reports_splitv1.json [more.json ...]
"""
import json
import math
import random
import re
import sys

PART_RE = {"leaf": r"\b(?:leaf|leaves|leaflets?|foliage|canopy)\b",
           "fruit": r"\b(?:fruits?|berry|berries|calyx|calyces|sepals?)\b",
           "stem": r"\b(?:stems?|petioles?|runners?|stolons?)\b",
           "flower": r"\b(?:flowers?|blossoms?|blooms?|petals?|flowering)\b",
           "soil": r"\b(?:soil|substrate|ground|potting mix|mulch|dirt)\b",
           "root": r"\b(?:roots?|rootball|root zone)\b"}
NEG = re.compile(r"\b(?:no|not|non|never|neither|nor|without|absent|absence of|free of|lack of|lacks|lacking|unlike|"
                 r"rather than|instead of|cannot|can't|unable to|not visible|not seen|not shown|no visible)\b")
LABEL_WORDS = {"healthy": ["healthy"], "drought": ["drought"], "frost": ["frost"], "overwatered": ["overwater"],
               "root_rot": ["root rot"], "gray_mold": ["gray mold", "grey mold", "botrytis"],
               "white_mold": ["white mold", "sclerotinia"]}


def mentions(text):
    """{part: (any mention, asserted mention)}"""
    out = {}
    sentences = [s for s in re.split(r"(?<=[.!?;:])\s+|\n+", text.lower()) if s.strip()]
    for part, rx in PART_RE.items():
        any_m = asserted = False
        for s in sentences:
            for m in re.finditer(rx, s):
                any_m = True
                before = " ".join(s[:m.start()].split()[-6:])
                after = s[m.end():m.end() + 40]
                negated = bool(NEG.search(before)) or bool(re.match(r"^\W*(?:\w+\W+){0,3}(?:(?:is|are|was|were)\s+)?(?:not|n't|absent|missing)\b", after))
                asserted |= not negated
        out[part] = (any_m, asserted)
    return out


def diagnosis_ok(label, text):
    m = re.search(r"diagnosis:?\s*\**\s*([a-z _-]+)", text.lower())
    head = m.group(1) if m else text.lower()[:80]
    return any(w in head for w in LABEL_WORDS[label])


def boot_ci(vals, reps=5000, seed=0):
    rng = random.Random(seed)
    n = len(vals)
    bs = sorted(sum(vals[rng.randrange(n)] for _ in range(n)) / n for _ in range(reps))
    return bs[int(0.025 * reps)], bs[int(0.975 * reps)]


def mcnemar(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(0, k + 1)) / 2 ** n * 2
    return min(1.0, p)


for path in sys.argv[1:]:
    rows = json.load(open(path))
    print(f"\n### {path}: {len(rows)} reports")
    per = {}
    for cond in sorted({r["condition"] for r in rows}):
        rs = [r for r in rows if r["condition"] == cond]
        h_assert, h_any, h_notvis, cons, regen, lengths = [], [], [], [], [], []
        false_parts = {p: 0 for p in PART_RE}
        for r in rs:
            m = mentions(r["final_report"])
            absent = [p for p in PART_RE if p not in r["annotated_parts"]]
            not_visible = [p for p in PART_RE if p not in r["detected_parts"]] if r["condition"] == "parts" else []
            h_assert.append(any(m[p][1] for p in absent))
            h_any.append(any(m[p][0] for p in absent))
            h_notvis.append(any(m[p][1] for p in not_visible))
            for p in absent:
                false_parts[p] += m[p][1]
            cons.append(diagnosis_ok(r["label"], r["final_report"]))
            regen.append(len(r["attempts"]) > 1)
            lengths.append(len(r["final_report"].split()))
        per[cond] = {"n": len(rs), "h_part_asserted": h_assert, "h_part_any": h_any, "names_not_visible_part": h_notvis,
                     "consistency": cons, "regenerated": regen, "length": lengths, "false_parts": false_parts}
        pct = lambda v: f"{100 * sum(v) / len(v):.1f} % ({100 * boot_ci([float(x) for x in v])[0]:.1f}-{100 * boot_ci([float(x) for x in v])[1]:.1f})"
        print(f"[{cond}] n={len(rs)} H_part asserted {pct(h_assert)} | any mention {pct(h_any)} | "
              f"names a not-visible part {pct(h_notvis) if cond == 'parts' else 'n/a'} | diagnosis consistent {pct(cons)} | "
              f"regenerated {pct(regen)} | still flagged {sum(r['still_flagged'] for r in rs)} | "
              f"mean length {sum(lengths) / len(lengths):.0f} words | false parts {false_parts}")
    if "parts" in per and "noparts" in per:
        a = {r["image"]: r for r in rows if r["condition"] == "parts"}
        b = {r["image"]: r for r in rows if r["condition"] == "noparts"}
        common = sorted(set(a) & set(b))
        for key in ("h_part_asserted", "consistency"):
            idx_a = {r["image"]: i for i, r in enumerate([r for r in rows if r["condition"] == "parts"])}
            idx_b = {r["image"]: i for i, r in enumerate([r for r in rows if r["condition"] == "noparts"])}
            only_a = sum(per["parts"][key][idx_a[im]] and not per["noparts"][key][idx_b[im]] for im in common)
            only_b = sum(per["noparts"][key][idx_b[im]] and not per["parts"][key][idx_a[im]] for im in common)
            print(f"paired {key}: with-parts only {only_a} / without-parts only {only_b} of {len(common)} photographs, "
                  f"exact McNemar P = {mcnemar(only_a, only_b):.4f}")
    json.dump({c: {k: (v if k == "false_parts" else (sum(v) / len(v) if v and k != "n" else v)) for k, v in d.items()}
               for c, d in per.items()}, open(path.replace(".json", "_scores.json"), "w"), indent=1)
