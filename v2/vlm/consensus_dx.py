"""Zero-shot diagnosis consensus (Hongbo: "caption many times, then the most likely disease is more statistically
correct"; Gus: "when two models disagree, say both"). Each model diagnoses every photo of a split once greedily
(sample 0) and N more times with sampling (seed = sample number). The votes give a majority diagnosis, a vote
share (confidence) and a ranked differential; two-model voting, agreement and selective prediction are computed
from these files offline. The prompt is the zero-shot prompt of cluster/vlm/zeroshot_qwen3vl.py, asking first for the
key visible evidence in one sentence (JSON, schema-enforced) so every sample parses.
Output: /projects/$USER/agai/consensus/<run>/<model>.jsonl, one record per (image, sample). Resumable.
  python cluster/vlm/consensus_dx.py --model /scratch/$USER/models/Qwen3-VL-32B-Instruct --samples 20
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
from annotation_spec import CLASSES  # noqa: E402

HEAD = ("You are a plant pathologist. Classify the main problem with the strawberry plant in this image into "
        "exactly one of these categories:\n" + "\n".join(f"- {k}: {v}" for k, v in CLASSES.items()))
FIELDS = {"evidence": {"type": "string", "maxLength": 200}, "diagnosis": {"type": "string", "enum": list(CLASSES)}}


def schema(order):
    return {"type": "object", "additionalProperties": False, "required": list(order),
            "properties": {k: FIELDS[k] for k in order}}


# --prompt-style (format ablation): the answer format only; same categories, same image, greedy for sample 0
STYLES = {
    "evidence": (HEAD + "\nFirst name the key visible evidence in one short sentence, then the category. Reply in JSON.",
                 schema(["evidence", "diagnosis"])),
    "after": (HEAD + "\nName the category, then the key visible evidence in one short sentence. Reply in JSON.",
              schema(["diagnosis", "evidence"])),
    "bare": (HEAD + "\nReply in JSON with only the category.", schema(["diagnosis"])),
    "text": (HEAD + "\nReply with only the category name.", None),   # = the prompt of cluster/vlm/zeroshot_qwen3vl.py
}
PROMPT, SCHEMA = STYLES["evidence"]


def parse_text(text):
    """First category named in a free-text answer ("gray mold" or "gray_mold"), or None."""
    t = text.lower()
    hits = [(t.find(n), c) for c in CLASSES for n in (c, c.replace("_", " ")) if n in t]
    return min(hits)[1] if hits else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--name", default="", help="model name in the output (default: basename of --model)")
    ap.add_argument("--split-json", default="evaluation/split_v1.json")
    ap.add_argument("--split", default="test")
    ap.add_argument("--run", default="v1")
    ap.add_argument("--samples", type=int, default=20, help="sampled diagnoses per photo, besides the greedy one")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--prompt-style", choices=list(STYLES), default="evidence", help="answer format (ablation)")
    ap.add_argument("--max-side", type=int, default=1024, help="as in cluster/vlm/zeroshot_qwen3vl.py")
    ap.add_argument("--chunk", type=int, default=16, help="photos per vLLM call (x (samples + 1) requests)")
    ap.add_argument("--pp", type=int, default=1)
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--gpu-mem", type=float, default=0.90)
    ap.add_argument("--dtype", default="auto")
    ap.add_argument("--quantization", default=None)
    ap.add_argument("--enforce-eager", action="store_true")
    args = ap.parse_args()

    split = json.load(open(args.split_json))[args.split]
    items = [(p, c) for c, ps in sorted(split.items()) for p in ps]
    name = args.name or os.path.basename(args.model.rstrip("/"))
    out_dir = f"/projects/{os.environ['USER']}/agai/consensus/{args.run}"
    os.makedirs(out_dir, exist_ok=True)
    out_path = f"{out_dir}/{name}.jsonl"
    done = set()
    if os.path.exists(out_path):
        done = {(r["image"], r["sample"]) for r in map(json.loads, open(out_path)) if "diagnosis" in r}
    todo = [(p, c, k) for p, c in items for k in range(args.samples + 1) if (p, k) not in done]
    print(f"{name}: {len(items)} photos x {args.samples + 1} -> {len(todo)} diagnoses to do -> {out_path}", flush=True)
    if not todo:
        return

    from vllm import LLM, SamplingParams
    from vllm.sampling_params import StructuredOutputsParams

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
    prompt, sch = STYLES[args.prompt_style]
    extra = {"structured_outputs": StructuredOutputsParams(json=sch)} if sch else {}
    n_tok = 120 if sch else 16
    params = [SamplingParams(temperature=0.0, max_tokens=n_tok, **extra)] + [
        SamplingParams(temperature=args.temperature, max_tokens=n_tok, seed=k, **extra)
        for k in range(1, args.samples + 1)]

    def data_url(path):
        img = Image.open(path).convert("RGB")
        img.thumbnail((args.max_side, args.max_side))
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=92)
        return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()

    # requests of one photo stay adjacent so the image prefix is computed once and reused from the prefix cache
    photos = list(dict.fromkeys(p for p, _, _ in todo))
    by_photo = {}
    for p, c, k in todo:
        by_photo.setdefault(p, []).append((c, k))
    for b in range(0, len(photos), args.chunk):
        reqs = [(p, c, k) for p in photos[b:b + args.chunk] for c, k in by_photo[p]]
        urls = {p: data_url(p) for p in photos[b:b + args.chunk]}
        msgs = [[{"role": "user", "content": [{"type": "image_url", "image_url": {"url": urls[p]}},
                                              {"type": "text", "text": prompt}]}] for p, _, _ in reqs]
        t0 = time.time()
        outs = llm.chat(msgs, [params[k] for _, _, k in reqs], use_tqdm=False,
                        chat_template_kwargs={"enable_thinking": False})
        with open(out_path, "a") as f:
            for (p, c, k), o in zip(reqs, outs):
                rec = {"image": p, "label": c, "model": name, "sample": k, "prompt_style": args.prompt_style,
                       "temperature": 0.0 if k == 0 else args.temperature}
                text = o.outputs[0].text
                if sch is None:
                    d = parse_text(text)
                    rec.update({"diagnosis": d, "raw": text[:200]} if d else {"error": "no category", "raw": text[:200]})
                else:
                    try:
                        rec.update(json.loads(text))
                    except Exception as e:
                        rec.update({"error": f"{type(e).__name__}: {e}", "raw": text[:500]})
                f.write(json.dumps(rec) + "\n")
        print(f"{min(b + args.chunk, len(photos))}/{len(photos)} photos ({len(reqs)} diagnoses) in "
              f"{time.time() - t0:.0f}s", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
