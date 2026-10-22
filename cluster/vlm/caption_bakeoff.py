"""Captioner bake-off: the same two-pass captioning with every candidate VLM, on the same hard real photos.
  pass 1 (image, NO label): describe only what is visible - each finding with a point [x, y] (0-1000) and how clearly
          it is seen, plus what was checked and is absent. No disease names, so the label cannot bias perception.
  pass 2 (text only: the dataset label + the pass-1 findings): which findings support or contradict the label,
          which expected signs are not visible, and a caption. It cannot add visual details: it never sees the image.
Pass 1 is what gets rated (each finding present / absent, plus missed details) to compare perception and
hallucination between models. JSON schemas are enforced during decoding (vLLM structured outputs).
Output: /projects/$USER/agai/bakeoff/<run>/<model>.jsonl (one record per image). Resumable.
--samples N adds N sampled repeats of pass 1 (temperature --temperature, seed = repeat number) in
<model>.samples.jsonl, for the self-consistency test (does a claim recur when the same model looks again?).
  python cluster/vlm/caption_bakeoff.py --model /scratch/$USER/models/Qwen3-VL-32B-Instruct
  (multi-node: --pp N with a running Ray cluster, see cluster/slurm/vllm_ray.sbatch)
"""
import argparse
import base64
import io
import json
import os
import sys
import time

from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from annotation_spec import CLASSES, PARTS  # noqa: E402

PASS1_SYSTEM = (
    "You are a careful plant pathologist examining one photo of a strawberry plant, fruit or roots. Describe ONLY "
    "what is visible. Do not name any disease, disorder or cause, and do not guess one. Look closely for small or "
    "faint details: fuzzy, powdery, dusty or cottony growth (and its color: gray, white, brown), soft or "
    "water-soaked patches, spots and lesions, discolored or bronzed tissue, wilting or drooping leaves and stems "
    "(and whether neighbouring stems are upright), dry, crispy or curled leaf edges, the color of flower centers "
    "(yellow, brown or black), fruit color and texture, roots and crown if visible (color, firmness), and soil "
    "moisture. For every finding give a point [x, y] on it, with x and y from 0 to 1000 (0,0 = top-left corner), and "
    "say how clearly you can see it. List what you checked and found absent. If the image is blurry, tiny, a "
    "composite of several photos, a screenshot, or has text overlays, say so.")
PASS1_USER = "Describe what is visible in this photo, following the instructions. Reply in JSON."
S = lambda n: {"type": "string", "maxLength": n}   # noqa: E731  bounded strings: greedy decoding otherwise loops
PASS1_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "required": ["image_quality", "plant_parts_visible", "observations", "checked_absent"],
    "properties": {
        "image_quality": {"type": "object", "additionalProperties": False, "required": ["usable", "issues"],
                          "properties": {"usable": {"type": "boolean"},
                                         "issues": {"type": "array", "maxItems": 5, "items": S(120)}}},
        "plant_parts_visible": {"type": "array", "maxItems": len(PARTS), "items": {"type": "string", "enum": PARTS}},
        "observations": {"type": "array", "minItems": 1, "maxItems": 10, "items": {
            "type": "object", "additionalProperties": False, "required": ["part", "finding", "point", "visibility"],
            "properties": {"part": S(60), "finding": S(240),
                           "point": {"type": "array", "minItems": 2, "maxItems": 2,
                                     "items": {"type": "integer", "minimum": 0, "maximum": 1000}},
                           "visibility": {"type": "string", "enum": ["clear", "faint", "uncertain"]}}}},
        "checked_absent": {"type": "array", "maxItems": 8, "items": S(160)}}}
PASS2_SYSTEM = (
    "You write training captions for a strawberry diagnosis model. You get the dataset label (usually right, "
    "sometimes wrong) and numbered observations that an examiner made from the photo WITHOUT knowing the label. Use "
    "only those observations as visual evidence; never add a visual detail that is not in them. Decide which "
    "observations support the label, which contradict it, which expected signs of the label were not observed, "
    "and whether the photo supports the label. Then write a 2-4 sentence caption whose confidence matches the "
    "evidence ('consistent with', 'likely', 'cannot be confirmed from this photo').")
PASS2_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "required": ["consistency", "supporting", "contradicting", "expected_not_observed", "caption"],
    "properties": {
        "consistency": {"type": "string", "enum": ["clearly_supports", "partially_supports", "does_not_support",
                                                   "cannot_tell"]},
        "supporting": {"type": "array", "maxItems": 10, "items": {"type": "integer", "minimum": 1, "maximum": 10}},
        "contradicting": {"type": "array", "maxItems": 10, "items": {"type": "integer", "minimum": 1, "maximum": 10}},
        "expected_not_observed": {"type": "array", "maxItems": 6, "items": S(160)},
        "caption": S(700)}}


def pass2_text(label, ann):
    obs = "\n".join(f"{i + 1}. [{o['visibility']}] {o['part']}: {o['finding']}" for i, o in enumerate(ann["observations"]))
    absent = "; ".join(ann.get("checked_absent", [])) or "none listed"
    issues = "; ".join(ann["image_quality"].get("issues", [])) or "none"
    return (f"Dataset label: {label} ({CLASSES[label]}).\nImage quality issues: {issues}.\n"
            f"Observations:\n{obs}\nChecked and absent: {absent}\nReply in JSON.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--name", default="", help="model name in the output (default: basename of --model)")
    ap.add_argument("--set", default="evaluation/bakeoff_set_v1.json")
    ap.add_argument("--run", default="bakeoff_v1")
    ap.add_argument("--pp", type=int, default=1, help="pipeline-parallel size (multi-node needs a Ray cluster)")
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--max-side", type=int, default=1792, help="long side sent to the model (fine detail)")
    ap.add_argument("--max-model-len", type=int, default=16384)
    ap.add_argument("--gpu-mem", type=float, default=0.90)
    ap.add_argument("--dtype", default="auto")
    ap.add_argument("--quantization", default=None)
    ap.add_argument("--enforce-eager", action="store_true", help="no CUDA graphs (saves ~1 GB per GPU)")
    ap.add_argument("--samples", type=int, default=0, help="sampled pass-1 repeats per image (self-consistency)")
    ap.add_argument("--temperature", type=float, default=0.7, help="sampling temperature of the repeats")
    args = ap.parse_args()

    items = json.load(open(args.set))["images"]
    name = args.name or os.path.basename(args.model.rstrip("/"))
    out_dir = f"/projects/{os.environ['USER']}/agai/bakeoff/{args.run}"
    os.makedirs(out_dir, exist_ok=True)
    out_path = f"{out_dir}/{name}.jsonl"
    done = {json.loads(line)["image"] for line in open(out_path)} if os.path.exists(out_path) else set()
    todo = [it for it in items if it["image"] not in done]
    print(f"{name}: {len(todo)} of {len(items)} images to caption -> {out_path}", flush=True)
    samp_path = f"{out_dir}/{name}.samples.jsonl"
    samp_done = set()
    if args.samples and os.path.exists(samp_path):
        samp_done = {(r["image"], r["sample"]) for r in map(json.loads, open(samp_path))}
    samp_todo = {k: [it for it in items if (it["image"], k) not in samp_done] for k in range(1, args.samples + 1)}
    n_samp = sum(len(v) for v in samp_todo.values())
    if args.samples:
        print(f"{name}: {n_samp} sampled pass-1 repeats (T={args.temperature}) to do -> {samp_path}", flush=True)
    if not todo and not n_samp:
        return

    from vllm import LLM, SamplingParams
    from vllm.sampling_params import StructuredOutputsParams

    def sp(schema, n, temperature=0.0, seed=None):
        return SamplingParams(temperature=temperature, max_tokens=n, repetition_penalty=1.05, seed=seed,
                              structured_outputs=StructuredOutputsParams(json=schema))

    kw = dict(model=args.model, max_model_len=args.max_model_len, gpu_memory_utilization=args.gpu_mem,
              limit_mm_per_prompt={"image": 1}, dtype=args.dtype, seed=0, trust_remote_code=True,
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
    ctk = {"enable_thinking": False}

    def data_url(path):
        img = Image.open(path).convert("RGB")
        img.thumbnail((args.max_side, args.max_side))
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=95)
        return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()

    def pass1_msgs(its):
        return [[{"role": "system", "content": PASS1_SYSTEM},
                 {"role": "user", "content": [{"type": "image_url", "image_url": {"url": data_url(it["image"])}},
                                              {"type": "text", "text": PASS1_USER}]}] for it in its]

    for k, its in samp_todo.items():
        if not its:
            continue
        t0 = time.time()
        outs = llm.chat(pass1_msgs(its), sp(PASS1_SCHEMA, 1800, args.temperature, k), use_tqdm=False,
                        chat_template_kwargs=ctk)
        with open(samp_path, "a") as f:
            for it, o in zip(its, outs):
                rec = {"image": it["image"], "label": it["label"], "model": name, "sample": k,
                       "temperature": args.temperature}
                try:
                    rec["pass1"] = json.loads(o.outputs[0].text)
                except Exception as e:
                    rec["pass1_error"] = f"{type(e).__name__}: {e}"
                    rec["pass1_raw"] = o.outputs[0].text[:3000]
                f.write(json.dumps(rec) + "\n")
        print(f"{name}: repeat {k}: {len(its)} images in {time.time() - t0:.0f}s", flush=True)
    if not todo:
        return

    t0 = time.time()
    msgs = pass1_msgs(todo)
    outs1 = llm.chat(msgs, sp(PASS1_SCHEMA, 1800), use_tqdm=False, chat_template_kwargs=ctk)
    t1 = time.time() - t0
    recs = []
    for it, o in zip(todo, outs1):
        rec = {"image": it["image"], "label": it["label"], "model": name, "pass1_tokens": len(o.outputs[0].token_ids)}
        try:
            rec["pass1"] = json.loads(o.outputs[0].text)
        except Exception as e:
            rec["pass1_error"] = f"{type(e).__name__}: {e}"
            rec["pass1_raw"] = o.outputs[0].text[:3000]
        recs.append(rec)
    ok = [r for r in recs if "pass1" in r]
    t0 = time.time()
    outs2 = llm.chat([[{"role": "system", "content": PASS2_SYSTEM},
                       {"role": "user", "content": pass2_text(r["label"], r["pass1"])}] for r in ok],
                     sp(PASS2_SCHEMA, 900), use_tqdm=False, chat_template_kwargs=ctk)
    t2 = time.time() - t0
    for r, o in zip(ok, outs2):
        try:
            r["pass2"] = json.loads(o.outputs[0].text)
        except Exception as e:
            r["pass2_error"] = f"{type(e).__name__}: {e}"
    with open(out_path, "a") as f:
        for r in recs:
            r["timing"] = {"pass1_s_per_image": round(t1 / len(todo), 2), "pass2_s_per_image": round(t2 / max(1, len(ok)), 2)}
            f.write(json.dumps(r) + "\n")
    print(f"{name}: pass 1 {t1:.0f}s ({t1 / len(todo):.1f} s/img), pass 2 {t2:.0f}s; "
          f"{len(ok)}/{len(recs)} parsed", flush=True)


if __name__ == "__main__":
    main()
