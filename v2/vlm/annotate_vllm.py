"""Structured photo annotations (spec: cluster/vlm/annotation_spec.py) with an open VLM served by vLLM, with the JSON
schema enforced during decoding (every reply parses and validates). One model load serves several tasks.

Each task is SPLIT:MODE with MODE
  true  - condition on the dataset label (training annotations)
  swap  - condition on a deliberately WRONG label (seeded per image); an honest annotator answers
          label_assessment.consistency = does_not_support / cannot_tell, a sycophantic one agrees anyway.
          On the train split only a --swap-frac subset is used (wrong-label training examples).
  both  - true + swap for every image (the label-swap honesty test).
Output: /projects/$USER/agai/annotations/<run>_<model>/<split>_<mode>.jsonl, one record per image and given label
with the annotation (schema: cluster/vlm/annotation_spec.py), "given_label" and "label_swapped". Resumable.
(vLLM starts its engine with multiprocessing "spawn", which re-imports this file: all work stays under main().)
  python cluster/vlm/annotate_vllm.py --model /scratch/$USER/models/Qwen3-VL-32B-Instruct --tasks val:both,test:both,train:true,train:swap
  (multi-node: --pp N with a running Ray cluster, see cluster/slurm/vllm_ray.sbatch; quantized checkpoints: --dtype auto)
"""
import argparse
import base64
import collections
import io
import json
import os
import random
import sys
import time

from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from annotation_spec import CLASSES, SYSTEM, bounded_schema, user_text, validate  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--name", default="", help="model name in the output dir (default: basename of --model)")
    ap.add_argument("--tasks", default="val:both")
    ap.add_argument("--split-json", default="evaluation/split_v1.json")
    ap.add_argument("--swap-frac", type=float, default=0.35, help="share of train images that also get a wrong label")
    ap.add_argument("--per-class", type=int, default=0, help="pilot: N random images per class")
    ap.add_argument("--run", default="v1")
    ap.add_argument("--max-side", type=int, default=1280)
    ap.add_argument("--max-tokens", type=int, default=2000)
    ap.add_argument("--max-model-len", type=int, default=12288)
    ap.add_argument("--gpu-mem", type=float, default=0.92)
    ap.add_argument("--quantization", default=None, help="e.g. bitsandbytes for a pre-quantized bnb4 checkpoint")
    ap.add_argument("--dtype", default="bfloat16", help="'auto' for AWQ / FP8 checkpoints")
    ap.add_argument("--pp", type=int, default=1, help="pipeline-parallel size (multi-node needs a Ray cluster)")
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--enforce-eager", action="store_true", help="no CUDA graphs (saves ~1 GB per GPU)")
    ap.add_argument("--chunk", type=int, default=96, help="requests per vLLM batch (results are flushed after each)")
    ap.add_argument("--order", choices=["observations-first", "verdict-first"], default="observations-first",
                    help="verdict-first moves label_assessment (consistency first) ahead of the observations "
                         "(schema-order ablation; nothing else changes)")
    args = ap.parse_args()

    split = json.load(open(args.split_json))
    mname = args.name or os.path.basename(args.model.rstrip("/"))
    out_dir = f"/projects/{os.environ['USER']}/agai/annotations/{args.run}_{mname}"
    os.makedirs(out_dir, exist_ok=True)

    def task_jobs(s, mode):
        items = [(p, c) for c, ps in sorted(split[s].items()) for p in ps]
        if args.per_class:
            by = collections.defaultdict(list)
            for p, c in items:
                by[c].append((p, c))
            items = [x for c in sorted(by) for x in random.Random(0).sample(by[c], min(args.per_class, len(by[c])))]
        jobs = []
        for p, c in items:
            if mode in ("true", "both"):
                jobs.append((p, c, c))
            if mode in ("swap", "both"):
                rng = random.Random(f"swap:{p}")
                if s == "train" and mode == "swap" and rng.random() > args.swap_frac:
                    continue
                jobs.append((p, c, rng.choice([k for k in CLASSES if k != c])))
        return jobs

    from vllm import LLM, SamplingParams

    schema = bounded_schema()
    if args.order == "verdict-first":   # JSON-grammar decoding writes properties in schema order
        props = schema["properties"]
        schema["properties"] = {"label_assessment": props["label_assessment"],
                                **{k: v for k, v in props.items() if k != "label_assessment"}}
        schema["required"] = ["label_assessment"] + [k for k in schema["required"] if k != "label_assessment"]
    common = dict(temperature=0.0, max_tokens=args.max_tokens, repetition_penalty=1.05)   # mild: breaks greedy loops
    try:   # vLLM renamed guided decoding to structured outputs; support both
        from vllm.sampling_params import StructuredOutputsParams
        sp = SamplingParams(**common, structured_outputs=StructuredOutputsParams(json=schema))
    except ImportError:
        from vllm.sampling_params import GuidedDecodingParams
        sp = SamplingParams(**common, guided_decoding=GuidedDecodingParams(json=schema))
    llm_kw = dict(model=args.model, max_model_len=args.max_model_len, gpu_memory_utilization=args.gpu_mem,
                  limit_mm_per_prompt={"image": 1}, dtype=args.dtype, seed=0, trust_remote_code=True,
                  tensor_parallel_size=args.tp, pipeline_parallel_size=args.pp)
    if args.pp * args.tp > 1 and os.environ.get("RAY_ADDRESS"):
        llm_kw["distributed_executor_backend"] = "ray"
    if args.quantization:
        llm_kw["quantization"] = args.quantization
    if args.enforce_eager:
        llm_kw["enforce_eager"] = True
    llm = LLM(**llm_kw)

    def data_url(path):
        img = Image.open(path).convert("RGB")
        img.thumbnail((args.max_side, args.max_side))
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=92)
        return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()

    for task in args.tasks.split(","):
        s, mode = task.split(":")
        out_path = f"{out_dir}/{s}_{mode}.jsonl"
        done = set()
        if os.path.exists(out_path):
            for line in open(out_path):
                r = json.loads(line)
                if "error" not in r:
                    done.add((r["image"], r["given_label"]))
        jobs = [j for j in task_jobs(s, mode) if (j[0], j[2]) not in done]
        print(f"[{task}] {len(jobs)} annotations to do -> {out_path}", flush=True)
        with open(out_path, "a") as f:
            for b in range(0, len(jobs), args.chunk):
                chunk = jobs[b:b + args.chunk]
                msgs = [[{"role": "system", "content": SYSTEM},
                         {"role": "user", "content": [{"type": "image_url", "image_url": {"url": data_url(p)}},
                                                      {"type": "text", "text": user_text(given)}]}]
                        for p, _, given in chunk]
                t0 = time.time()
                outs = llm.chat(msgs, sp, use_tqdm=False, chat_template_kwargs={"enable_thinking": False})
                dt = time.time() - t0
                n_tok = 0
                for (p, truth, given), o in zip(chunk, outs):
                    text = o.outputs[0].text
                    n_tok += len(o.outputs[0].token_ids)
                    rec = {"image": p, "label": truth, "given_label": given, "label_swapped": given != truth,
                           "split": s, "model": mname, "finish_reason": o.outputs[0].finish_reason}
                    try:
                        ann = json.loads(text)
                        rec.update({"annotation": ann, "schema_problems": validate(ann)})
                    except Exception as e:
                        rec.update({"error": f"{type(e).__name__}: {e}", "raw": text[:3000]})
                    f.write(json.dumps(rec) + "\n")
                f.flush()
                print(f"[{task}] {b + len(chunk)}/{len(jobs)}: {len(chunk)} in {dt:.0f}s ({n_tok / dt:.0f} out tok/s)",
                      flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
