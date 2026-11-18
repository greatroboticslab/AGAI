"""Contact sheet of the images a result file gets wrong, with each listed member's prediction, for eyeballing
errors and possible label noise. Run from the repo root:
  python cluster/benchmark/error_sheet.py evaluation/ensemble_E2_8b_eva_siglip_equal_split_v1_test.json out.jpg \
      8B=evaluation/scores_Qwen3-VL-8B-Instruct_lora_split_v1_test.json EVA=evaluation/probe_eva02-large-448_split_v1_test.json
"""
import json
import os
import sys

from PIL import Image, ImageDraw

result_file, out = sys.argv[1], sys.argv[2]
members = {kv.split("=", 1)[0]: {r["image"]: r["pred"] for r in json.load(open(kv.split("=", 1)[1]))["results"]}
           for kv in sys.argv[3:]}
rows = json.load(open(result_file))["results"]
errs = sorted([r for r in rows if r["pred"] != r["truth"]], key=lambda r: (r["truth"], r["pred"]))
W, H, cols = 300, 300, 4
n_rows = (len(errs) + cols - 1) // cols
sheet = Image.new("RGB", (cols * W, max(1, n_rows) * (H + 48)), "white")
d = ImageDraw.Draw(sheet)
for i, r in enumerate(errs):
    im = Image.open(r["image"]).convert("RGB")
    im.thumbnail((W, H))
    x, y = (i % cols) * W, (i // cols) * (H + 48)
    sheet.paste(im, (x + (W - im.width) // 2, y))
    d.text((x + 4, y + H + 2), f"{i + 1}. label {r['truth']} -> {r['pred']}", fill="black")
    d.text((x + 4, y + H + 16), " ".join(f"{n}:{m.get(r['image'], '?')[:9]}" for n, m in members.items()), fill="gray")
    d.text((x + 4, y + H + 30), os.path.basename(r["image"])[:46], fill="gray")
    print(i + 1, r["truth"], "->", r["pred"], {n: m.get(r["image"]) for n, m in members.items()}, r["image"])
sheet.save(out, quality=88)
print(len(errs), "errors ->", out, sheet.size)
