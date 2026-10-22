"""Table of evaluation/lowq_<model>.json (cluster/classifier/eval_lowq.py) over seeds: mean +- sd accuracy per degradation,
and per-class accuracy at a chosen level.
  python cluster/classifier/lowq_table.py ft_siglip2-so400m-512 _s0,_s1,_s2 _labsynth_s0,_labsynth_s1,_labsynth_s2"""
import json
import statistics as st
import sys

L = ["clean", "low_res", "very_low", "defocus", "noisy_dark"]
base = sys.argv[1]
for arm in sys.argv[2:]:
    rows = [json.load(open(f"evaluation/lowq_{base}{s}.json")) for s in arm.split(",")]
    acc = {l: [100 * r["levels"][l]["accuracy"] for r in rows] for l in L}
    mac = {l: [100 * r["levels"][l]["macro_accuracy"] for r in rows] for l in L}
    sd = lambda v: st.stdev(v) if len(v) > 1 else 0.0
    print(f"{arm} (n={len(rows)})")
    print("   acc   " + "  ".join(f"{l} {st.mean(acc[l]):.1f}+-{sd(acc[l]):.1f}" for l in L))
    print("   macro " + "  ".join(f"{l} {st.mean(mac[l]):.1f}" for l in L))
    for lvl in ("very_low", "defocus"):
        pc = {}
        for r in rows:
            for c, v in r["levels"][lvl]["per_class"].items():
                a, b = map(int, v.split("/"))
                pc.setdefault(c, []).append(100 * a / b)
        print(f"   {lvl} per class: " + ", ".join(f"{c} {st.mean(v):.0f}" for c, v in pc.items()))
