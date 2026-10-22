"""Zoom-in claim judge: a VLM re-checks every grounded claim of the captioner bake-off at the place the examiner
pointed to, WITHOUT the dataset label, and says whether the claim is correct as stated there.
  zoom   - image 1: the whole photo with a red box around the claimed point; image 2: the inside of the box, enlarged
  nozoom - the whole photo with the red box only (ablation: does the enlarged crop help?)
Decoys measure how often a judge accepts claims that are not there: a symptom claim taken from a photo with another
label, boxed on the same plant part of this photo (the part's location comes from this photo's own claims).
Scored against the human ratings of the review page (scratchpad consistency/judge scoring).
Output: /projects/$USER/agai/bakeoff/<run>/judge/<judge>.<mode>.jsonl, one record per (claim or decoy). Resumable.
  python cluster/vlm/judge_claims.py --model /scratch/$USER/models/Qwen3-VL-32B-Instruct --modes zoom,nozoom
  python cluster/vlm/judge_claims.py --model x --dry-run /tmp/judge_preview      # renders a few inputs, no model
  (multi-node: --pp N with a running Ray cluster, see cluster/slurm/vllm_ray.sbatch)
"""
import argparse
import base64
import collections
import glob
import io
import json
import os
import random
import re
import time

from PIL import Image, ImageDraw

SYSTEM = (
    "You are a careful plant pathologist checking another examiner's observations on photos of strawberry plants, "
    "fruit or roots. Each observation comes with the place the examiner pointed to, marked by a red box on the photo. "
    "First describe what is actually visible inside the red box, then decide whether the observation is correct as "
    "stated for that place. Judge only what you can see: do not accept a detail because it would be typical, and do "
    "not reject a detail only because it is small or faint if you can see it. Answer 'correct' if the observation "
    "matches what is visible, 'incorrect' if what is visible contradicts it or the described feature is not there, "
    "and 'cannot_tell' if the image does not allow a decision (blurred, too small, hidden).")
USER = {
    "zoom": "Image 1: the whole photo with the red box. Image 2: the inside of the red box, enlarged.\n"
            "Observation to check ({part}): \"{finding}\"\nReply in JSON.",
    "nozoom": "The red box marks the place the examiner pointed to.\n"
              "Observation to check ({part}): \"{finding}\"\nReply in JSON.",
}
SCHEMA = {"type": "object", "additionalProperties": False, "required": ["visible_in_box", "verdict"],
          "properties": {"visible_in_box": {"type": "string", "maxLength": 300},
                         "verdict": {"type": "string", "enum": ["correct", "incorrect", "cannot_tell"]}}}
PARTS = [("calyx", r"calyx|sepal"), ("fruit", r"fruit|berr"), ("flower", r"flower|petal|blossom|\bbuds?\b"),
         ("root", r"root|crown"), ("leaf", r"leaf|leaves|leaflet|foliage"),
         ("stem", r"stem|petiole|runner|stolon|peduncle|stalk")]
# decoys: only definite disease-sign claims (no hedges, negations or healthy descriptions, which may be true anywhere)
SYMPTOM = re.compile(r"mold|mould|fuzz|powder|cotton|mycel|sporul|\brot\b|rotting|rotted|decay|soak|necro|blacken|"
                     r"lesion", re.I)
HEDGE = re.compile(r"\bno\b|\bnot\b|n't|without|absent|healthy|turgid|normal|shadow|possibl|may\b|might", re.I)
# a decoy comes from a photo of another condition GROUP (a white-mold sign on a gray-mold photo may well be true)
GROUP = {"gray_mold": "mold", "white_mold": "mold", "root_rot": "wet", "overwatered": "wet", "drought": "dry",
         "frost": "frost", "healthy": "healthy"}


def norm_part(text):
    for name, rx in PARTS:
        if re.search(rx, text, re.I):
            return name
    return None


def build_items(set_items, bake_dir, n_decoys):
    """Every pass-1 claim of every examiner, plus up to n_decoys decoys per photo (deterministic)."""
    label = {it["image"]: it["label"] for it in set_items}
    claims = []
    for p in sorted(glob.glob(os.path.join(bake_dir, "*.jsonl"))):
        if ".samples." in p:
            continue
        for line in open(p):
            r = json.loads(line)
            if r["image"] not in label or "pass1" not in r:
                continue
            for i, o in enumerate(r["pass1"]["observations"], 1):
                claims.append({"kind": "claim", "image": r["image"], "claimant": r["model"], "idx": i,
                               "part": o["part"], "finding": o["finding"], "point": o["point"],
                               "part_n": norm_part(o["part"] + " " + o["finding"])})
    decoys = []
    for it in set_items:
        anchors = {}
        for c in sorted(claims, key=lambda c: (c["claimant"], c["idx"])):
            if c["image"] == it["image"] and c["part_n"] and c["part_n"] not in anchors:
                anchors[c["part_n"]] = c["point"]
        pool = [c for c in claims if GROUP[label[c["image"]]] != GROUP[it["label"]] and c["part_n"] in anchors
                and SYMPTOM.search(c["finding"]) and not HEDGE.search(c["finding"])]
        pool.sort(key=lambda c: (c["image"], c["claimant"], c["idx"]))
        random.Random(f"decoy:{it['image']}").shuffle(pool)
        seen = set()
        for c in pool:
            if len(seen) >= n_decoys:
                break
            if c["finding"] in seen:
                continue
            seen.add(c["finding"])
            decoys.append({"kind": "decoy", "image": it["image"], "claimant": "", "idx": 0, "part": c["part"],
                           "finding": c["finding"], "point": anchors[c["part_n"]], "part_n": c["part_n"],
                           "src_image": c["image"], "src_claimant": c["claimant"], "src_idx": c["idx"]})
    return claims + decoys


def item_key(x):
    return (x["kind"], x["image"], x["claimant"], x["idx"], x.get("src_image", ""), x.get("src_claimant", ""),
            x.get("src_idx", 0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--name", default="", help="judge name in the output (default: basename of --model)")
    ap.add_argument("--set", default="evaluation/bakeoff_set_v1.json")
    ap.add_argument("--run", default="bakeoff_v1")
    ap.add_argument("--modes", default="zoom,nozoom")
    ap.add_argument("--decoys", type=int, default=4, help="decoy claims per photo")
    ap.add_argument("--full-side", type=int, default=1792, help="long side of the whole photo (as in pass 1)")
    ap.add_argument("--crop-frac", type=float, default=0.35, help="box side as a share of the photo's short side")
    ap.add_argument("--crop-side", type=int, default=896, help="side of the enlarged crop")
    ap.add_argument("--desc-key", default="visible_in_box",
                    help="JSON key of the description field; GLM-4.6V's grammar decoding fails on 'visible_in_box' "
                         "(vLLM: 'grammar rejected tokens [0]' right after 'visible'), use 'description' for it")
    ap.add_argument("--limit", type=int, default=0, help="smoke test: first N items per mode")
    ap.add_argument("--dry-run", default="", help="render the first --limit items into this dir, no model")
    ap.add_argument("--chunk", type=int, default=128)
    ap.add_argument("--pp", type=int, default=1)
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--gpu-mem", type=float, default=0.90)
    ap.add_argument("--dtype", default="auto")
    ap.add_argument("--quantization", default=None)
    ap.add_argument("--enforce-eager", action="store_true")
    args = ap.parse_args()

    set_items = json.load(open(args.set))["images"]
    name = args.name or os.path.basename(args.model.rstrip("/"))
    bake_dir = f"/projects/{os.environ['USER']}/agai/bakeoff/{args.run}"
    out_dir = os.path.join(bake_dir, "judge")
    os.makedirs(out_dir, exist_ok=True)
    items = build_items(set_items, bake_dir, args.decoys)
    n_kind = collections.Counter(x["kind"] for x in items)
    print(f"{name}: {n_kind['claim']} claims + {n_kind['decoy']} decoys on {len(set_items)} photos", flush=True)

    def render(x, mode):
        img = Image.open(x["image"]).convert("RGB")
        w, h = img.size
        s = max(min(w, h) * args.crop_frac, min(160, w, h))
        cx, cy = x["point"][0] / 1000 * w, x["point"][1] / 1000 * h
        x0, y0 = min(max(cx - s / 2, 0), w - s), min(max(cy - s / 2, 0), h - s)
        box = (int(x0), int(y0), int(x0 + s), int(y0 + s))
        full = img.copy()
        ImageDraw.Draw(full).rectangle(box, outline=(255, 0, 0), width=max(3, int(max(w, h) / 250)))
        full.thumbnail((args.full_side, args.full_side))
        out = [full]
        if mode == "zoom":
            out.append(img.crop(box).resize((args.crop_side, args.crop_side), Image.LANCZOS))
        return out

    def url(im):
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=92)
        return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()

    def text(x, mode):
        return USER[mode].format(part=x["part"], finding=x["finding"])

    if args.dry_run:
        os.makedirs(args.dry_run, exist_ok=True)
        for j, x in enumerate(items[:args.limit or 4] + [d for d in items if d["kind"] == "decoy"][:args.limit or 4]):
            for k, im in enumerate(render(x, "zoom")):
                im.save(os.path.join(args.dry_run, f"{j:02d}_{x['kind']}_{k}.jpg"), quality=90)
            print(j, x["kind"], os.path.basename(x["image"]), x.get("src_image", "")[-30:], "|", text(x, "zoom"))
        return

    from vllm import LLM, SamplingParams
    from vllm.sampling_params import StructuredOutputsParams

    kw = dict(model=args.model, max_model_len=args.max_model_len, gpu_memory_utilization=args.gpu_mem,
              limit_mm_per_prompt={"image": 2}, dtype=args.dtype, seed=0, trust_remote_code=True,
              tensor_parallel_size=args.tp, pipeline_parallel_size=args.pp)
    if args.pp * args.tp > 1 and os.environ.get("RAY_ADDRESS"):
        kw["distributed_executor_backend"] = "ray"
    if args.quantization:
        kw["quantization"] = args.quantization
    if args.enforce_eager:
        kw["enforce_eager"] = True
    t0 = time.time()
    llm = LLM(**kw)
    print(f"loaded in {time.time() - t0:.0f}s", flush=True)
    schema = SCHEMA
    if args.desc_key != "visible_in_box":
        schema = {**SCHEMA, "required": [args.desc_key, "verdict"],
                  "properties": {args.desc_key: SCHEMA["properties"]["visible_in_box"],
                                 "verdict": SCHEMA["properties"]["verdict"]}}
    sp = SamplingParams(temperature=0.0, max_tokens=400, repetition_penalty=1.05,
                        structured_outputs=StructuredOutputsParams(json=schema))
    for mode in args.modes.split(","):
        out_path = os.path.join(out_dir, f"{name}.{mode}.jsonl")
        done = set()
        if os.path.exists(out_path):   # failed records (no verdict) are retried
            done = {item_key(r) for r in map(json.loads, open(out_path)) if "verdict" in r}
        todo = [x for x in items if item_key(x) not in done]
        if args.limit:
            todo = todo[:args.limit]
        print(f"[{mode}] {len(todo)} to judge -> {out_path}", flush=True)
        for b in range(0, len(todo), args.chunk):
            chunk = todo[b:b + args.chunk]
            msgs = [[{"role": "system", "content": SYSTEM},
                     {"role": "user", "content": [{"type": "image_url", "image_url": {"url": url(im)}}
                                                  for im in render(x, mode)] + [{"type": "text", "text": text(x, mode)}]}]
                    for x in chunk]
            t0 = time.time()
            outs = llm.chat(msgs, sp, use_tqdm=False, chat_template_kwargs={"enable_thinking": False})
            with open(out_path, "a") as f:
                for x, o in zip(chunk, outs):
                    rec = {k: v for k, v in x.items() if k != "part_n"}
                    rec.update({"judge": name, "mode": mode, "finish_reason": o.outputs[0].finish_reason})
                    try:
                        ans = json.loads(o.outputs[0].text)
                        ans["visible_in_box"] = ans.pop(args.desc_key)   # same field name in every output file
                        rec.update(ans)
                    except Exception as e:
                        rec.update({"error": f"{type(e).__name__}: {e}", "raw": o.outputs[0].text[:1000]})
                    f.write(json.dumps(rec) + "\n")
            print(f"[{mode}] {b + len(chunk)}/{len(todo)} in {time.time() - t0:.0f}s", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
