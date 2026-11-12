"""Score the blind v1 vs v2 report-claim ratings (pre-registered in lab notebook entry (q), 2026-11-04).
  python -I score_faithfulness.py claims.json key.json ratings.jsonl [--rater NAME] [--second-rater NAME]
ratings.jsonl: one JSON object per line as db_list returns them ({"id":..., "data": {rater, name, claim, photo, v, t}, ...})
or a flat list of rating dicts. Verdicts: supported, not_supported, unsure, not_visual.
Metrics: M1 unsupported-claim rate per system (not_supported / (supported + not_supported)) with bootstrap 95 %
intervals over photographs and a paired Wilcoxon signed-rank test on per-photograph differences; M2 share of reports
with at least one unsupported claim, exact McNemar; M3 visual and non-visual claims per report; M4 (exploratory)
unsupported claims by plant part. Cohen's kappa between two raters on supported / not_supported when both rated.
Written before any rating was read."""
import argparse
import json
import math
import random
import re
import statistics as st
from collections import Counter, defaultdict

ap = argparse.ArgumentParser()
ap.add_argument("claims"); ap.add_argument("key"); ap.add_argument("ratings")
ap.add_argument("--rater", default=None, help="rater name (or uid) whose ratings are primary; default: the most complete")
ap.add_argument("--second-rater", default=None)
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()

claims = json.load(open(args.claims, encoding="utf-8"))
key = json.load(open(args.key, encoding="utf-8"))
text_of, photo_of = {}, {}
for p in claims["photos"]:
    for c in p["claims"]:
        text_of[c["id"]] = c["text"]; photo_of[c["id"]] = p["id"]

rows = []
raw = open(args.ratings, encoding="utf-8").read().strip()
if raw.startswith("["):
    rows = json.loads(raw)
else:
    for line in raw.splitlines():
        line = line.strip()
        if not line or not line.startswith("{"):
            continue
        d = json.loads(line)
        rows.append(d.get("data", d))
VERDICTS = ("supported", "not_supported", "unsure", "not_visual", "advice_not_visible")
rows = [r for r in rows if r.get("claim") in key and r.get("v") in VERDICTS]
by_rater = defaultdict(dict)
for r in rows:
    who = r.get("name") or r.get("rater")
    prev = by_rater[who].get(r["claim"])
    if not prev or (r.get("t") or "") >= (prev.get("t") or ""):
        by_rater[who][r["claim"]] = r
print("raters:", {k: len(v) for k, v in by_rater.items()}, "of", len(text_of), "claims")
primary = args.rater or max(by_rater, key=lambda k: len(by_rater[k]))
R = by_rater[primary]
print(f"primary rater: {primary} ({len(R)} ratings)")

SYS = ("v1", "v2")
PART_RE = {"leaf": r"\bleaf|leaves|foliage\b", "fruit": r"\bfruit|berr(y|ies)\b", "stem": r"\bstem|petiole|runner", "flower": r"\bflower|blossom|bloom",
           "root": r"\broot|crown\b", "soil": r"\bsoil|substrate|ground|potting|mulch|dirt"}


def per_photo():
    """per photograph and system: counts of supported / not_supported / unsure / not_visual among rated claims"""
    out = defaultdict(lambda: {s: Counter() for s in SYS})
    for cid, r in R.items():
        out[photo_of[cid]][key[cid]["system"]][r["v"]] += 1
    return out


pp = per_photo()
photos = sorted(pp)
print(f"photographs with ratings: {len(photos)} of {len(claims['photos'])}")


def rate(cnt):
    vis = cnt["supported"] + cnt["not_supported"]
    return (cnt["not_supported"] / vis) if vis else None


def pooled(system, which):
    tot = Counter()
    for p in photos:
        tot.update({k: v for k, v in pp[p][system].items()})
    vis = tot["supported"] + tot["not_supported"]
    return tot, (tot["not_supported"] / vis if vis else float("nan"))


def boot_ci(values_by_photo, n=10000):
    """bootstrap over photographs of a pooled rate: values_by_photo = list of (not_supported, visual) pairs"""
    rng = random.Random(args.seed); m = len(values_by_photo); ests = []
    for _ in range(n):
        s = [values_by_photo[rng.randrange(m)] for _ in range(m)]
        ns, vis = sum(a for a, b in s), sum(b for a, b in s)
        ests.append(ns / vis if vis else float("nan"))
    ests = sorted(e for e in ests if not math.isnan(e))
    return ests[int(0.025 * len(ests))], ests[int(0.975 * len(ests)) - 1]


def wilcoxon(diffs):
    """signed-rank test, normal approximation with tie correction; zeros dropped"""
    d = [x for x in diffs if x != 0]
    n = len(d)
    if n < 6:
        return n, float("nan")
    order = sorted(range(n), key=lambda k: abs(d[k]))      # average ranks of |d|, ties share a rank
    rk = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and abs(d[order[j + 1]]) == abs(d[order[i]]):
            j += 1
        for k in range(i, j + 1):
            rk[order[k]] = (i + j) / 2 + 1
        i = j + 1
    w_plus = sum(rk[k] for k in range(n) if d[k] > 0)
    mean = n * (n + 1) / 4
    ties = Counter(abs(x) for x in d)
    var = n * (n + 1) * (2 * n + 1) / 24 - sum(t ** 3 - t for t in ties.values()) / 48
    z = (w_plus - mean) / math.sqrt(var) if var > 0 else 0.0
    p = math.erfc(abs(z) / math.sqrt(2))
    return n, p


def mcnemar(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(0, k + 1)) / 2 ** n * 2
    return min(1.0, p)


print("\n== M1 unsupported-claim rate (not supported / visual claims) ==")
pairs = {}
for s in SYS:
    tot, r = pooled(s, "rate")
    vals = [(pp[p][s]["not_supported"], pp[p][s]["supported"] + pp[p][s]["not_supported"]) for p in photos]
    lo, hi = boot_ci(vals) if photos else (float("nan"), float("nan"))
    pairs[s] = r
    print(f"{s}: {100 * r:.1f} % ({100 * lo:.1f} to {100 * hi:.1f}); claims rated {sum(tot.values())}: "
          f"supported {tot['supported']}, not supported {tot['not_supported']}, unsure {tot['unsure']}, not visual {tot['not_visual']}")
diffs = []
for p in photos:
    a, b = rate(pp[p]["v1"]), rate(pp[p]["v2"])
    if a is not None and b is not None:
        diffs.append(a - b)
n_eff, pw = wilcoxon(diffs)
print(f"paired difference v1 - v2 over {len(diffs)} photographs: median {100 * st.median(diffs) if diffs else float('nan'):.1f} points, "
      f"Wilcoxon signed-rank P = {pw:.4f} (n nonzero = {n_eff})")
if not math.isnan(pairs['v1']) and pairs['v1'] > 0:
    print(f"P1 (v2 at most half of v1): v2 / v1 = {pairs['v2'] / pairs['v1']:.2f} -> {'supported' if pairs['v2'] <= 0.5 * pairs['v1'] else 'not supported'}")

print("\n== M2 reports with at least one unsupported claim ==")
flags = {s: {p: pp[p][s]["not_supported"] > 0 for p in photos} for s in SYS}
for s in SYS:
    k = sum(flags[s].values())
    print(f"{s}: {k} of {len(photos)} ({100 * k / max(1, len(photos)):.1f} %)")
b = sum(1 for p in photos if flags["v1"][p] and not flags["v2"][p])
c = sum(1 for p in photos if flags["v2"][p] and not flags["v1"][p])
print(f"discordant: v1 only {b}, v2 only {c}; exact McNemar P = {mcnemar(b, c):.4f}")

print("\n== M3 claims per report (rated claims) ==")
for s in SYS:
    vis = [pp[p][s]["supported"] + pp[p][s]["not_supported"] + pp[p][s]["unsure"] for p in photos]
    nonvis = [pp[p][s]["not_visual"] + pp[p][s]["advice_not_visible"] for p in photos]
    print(f"{s}: visual (incl. unsure) {st.mean(vis):.1f} per report, non-visual {st.mean(nonvis):.1f} per report")

print("\n== M5 (secondary, added 2026-11-04 before the first photograph was complete): advice assuming something not visible ==")
for s in SYS:
    adv = sum(pp[p][s]["advice_not_visible"] for p in photos)
    nonvis = sum(pp[p][s]["not_visual"] + pp[p][s]["advice_not_visible"] for p in photos)
    reports = sum(1 for p in photos if pp[p][s]["advice_not_visible"] > 0)
    print(f"{s}: {adv} sentences ({adv / max(1, len(photos)):.2f} per report; {100 * adv / nonvis if nonvis else float('nan'):.1f} % of non-visual sentences); "
          f"reports with at least one: {reports} of {len(photos)}")
a1 = sum(pp[p]['v1']['advice_not_visible'] for p in photos); a2 = sum(pp[p]['v2']['advice_not_visible'] for p in photos)
print(f"P5 (v1 has more such sentences per report than v2): v1 {a1} vs v2 {a2} -> {'supported' if a1 > a2 else 'not supported'}")

print("\n== M4 (exploratory) unsupported claims by plant part named in the claim ==")
for s in SYS:
    cnt = Counter()
    for cid, r in R.items():
        if key[cid]["system"] == s and r["v"] == "not_supported":
            t = text_of[cid].lower()
            hit = [part for part, rx in PART_RE.items() if re.search(rx, t)]
            for part in hit or ["(no part named)"]:
                cnt[part] += 1
    print(f"{s}: {dict(cnt.most_common())}")

print("\n== P4 (exploratory): v2 unsupported rate by the student's verdict ==")
byv = defaultdict(Counter)
for cid, r in R.items():
    if key[cid]["system"] == "v2":
        verdict = key[photo_of[cid]].get("v2_verdict", "?")
        byv[verdict][r["v"]] += 1
for verdict, cnt in byv.items():
    vis = cnt["supported"] + cnt["not_supported"]
    print(f"{verdict}: {cnt['not_supported']} of {vis} visual claims unsupported ({100 * cnt['not_supported'] / vis if vis else float('nan'):.1f} %)")

if args.second_rater and args.second_rater in by_rater:
    S = by_rater[args.second_rater]
    both = [cid for cid in R if cid in S and R[cid]["v"] in ("supported", "not_supported") and S[cid]["v"] in ("supported", "not_supported")]
    if both:
        a = [R[c]["v"] for c in both]; b2 = [S[c]["v"] for c in both]
        po = sum(x == y for x, y in zip(a, b2)) / len(both)
        pa, pb = Counter(a), Counter(b2)
        pe = sum(pa[v] * pb[v] for v in ("supported", "not_supported")) / len(both) ** 2
        kappa = (po - pe) / (1 - pe) if pe < 1 else float("nan")
        print(f"\n== agreement with {args.second_rater} on {len(both)} claims rated supported / not supported by both: "
              f"observed {100 * po:.1f} %, Cohen's kappa {kappa:.2f} ==")
