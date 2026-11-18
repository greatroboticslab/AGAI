"""Latency and GPU memory of the v2 report writer, the distilled Qwen3-VL-8B student served by vLLM (envs/vllm), on
the same seeded split v1 test photographs as the other latency_*.py scripts. Each photograph is one request with the
annotator prompt of cluster/vlm/annotation_spec.py (system instructions, image, "Dataset label: X", X being the v2
classifier's prediction from evaluation/latency_siglip2.json, or the dataset label when that file is absent), greedy
decoding under the JSON schema exactly as in cluster/vlm/annotate_vllm.py. Reports per-request latency one request at
a time (what a user waits for), generated tokens and the verdict parsed from the JSON, then one batched call over all
photographs for throughput. GPU memory: vLLM pre-allocates --gpu-mem of the card, so the figures to report are the
weights on disk and the engine's own log lines (model weights, KV cache); nvidia-smi figures for the job's GPU (memory
in use, and the engine processes' total) are recorded for completeness.
Writes evaluation/latency_student.json. Run from the repo root on a GPU node.
  python cluster/latency_student.py --model /scratch/$USER/models/<merged student> [--labels evaluation/latency_siglip2.json]
"""
import argparse
import base64
import glob
import io
import json
import os
import random
import statistics
import subprocess
import sys
import time

from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from annotation_spec import SYSTEM, bounded_schema, user_text, validate  # noqa: E402

ACCEPT = {"clearly_supports", "partially_supports"}


def smi_rows(query):
    """rows of nvidia-smi --query-<query> (empty when nvidia-smi is unavailable)"""
    try:
        out = subprocess.run(["nvidia-smi", f"--query-{query}", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=20).stdout
        return [[c.strip() for c in r.split(",")] for r in out.strip().splitlines() if r.strip()]
    except Exception:
        return []


def my_gpu():
    """(index, uuid) of this job's GPU: the one CUDA_VISIBLE_DEVICES names, else the first listed (nvidia-smi may list
    every GPU of the node, so row 0 is not necessarily ours)"""
    rows = smi_rows("gpu=index,uuid")
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")[0].strip()
    for idx, uuid in rows:
        if cvd and (cvd == idx or uuid.startswith(cvd)):
            return idx, uuid
    return tuple(rows[0]) if rows else (None, None)


def smi_device_used_mb():
    """memory in use on this job's GPU, all processes (nvidia-smi)"""
    idx, _ = my_gpu()
    return next((int(used) for i, used in smi_rows("gpu=index,memory.used") if i == idx and used.isdigit()), None)


def smi_process_mb(pid=os.getpid()):
    """memory of one process (pid) or of every process (pid=None) on this job's GPU, from nvidia-smi compute-apps
    (None when the driver reports no per-process figures)"""
    _, uuid = my_gpu()
    used = [int(m) for p, u, m in smi_rows("compute-apps=pid,gpu_uuid,used_memory")
            if u == uuid and (pid is None or p == str(pid)) and m.isdigit()]
    return sum(used) if used else None


def stats(xs):
    xs = sorted(xs)
    return {"median_s": statistics.median(xs), "p90_s": xs[int(0.9 * (len(xs) - 1))], "mean_s": sum(xs) / len(xs),
            "n": len(xs)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="merged student checkpoint directory")
    ap.add_argument("--labels", default="evaluation/latency_siglip2.json",
                    help="cluster/latency_siglip.py output: the proposed diagnosis per photograph")
    ap.add_argument("--per-class", type=int, default=7)
    ap.add_argument("--split-json", default="evaluation/split_v1.json")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-side", type=int, default=1280)
    ap.add_argument("--max-tokens", type=int, default=2000)
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    ap.add_argument("--out", default="evaluation/latency_student.json")
    args = ap.parse_args()

    split = json.load(open(args.split_json))["test"]
    items = []
    for c in sorted(split):                          # the same seeded sample in every latency_*.py script
        paths = sorted(split[c])
        items += [(p, c) for p in random.Random(f"{args.seed}:{c}").sample(paths, min(args.per_class, len(paths)))]
    labels = {}
    if os.path.exists(args.labels):
        labels = {k: v["pred"] for k, v in json.load(open(args.labels))["predictions"].items()}
        print(f"proposed diagnoses from {args.labels} for {len(labels)} photographs", flush=True)
    else:
        print(f"{args.labels} not found: the dataset label is given instead", flush=True)

    from vllm import LLM, SamplingParams
    schema = bounded_schema()
    common = dict(temperature=0.0, max_tokens=args.max_tokens, repetition_penalty=1.05)
    try:
        from vllm.sampling_params import StructuredOutputsParams
        sp = SamplingParams(**common, structured_outputs=StructuredOutputsParams(json=schema))
    except ImportError:
        from vllm.sampling_params import GuidedDecodingParams
        sp = SamplingParams(**common, guided_decoding=GuidedDecodingParams(json=schema))
    t0 = time.time()
    llm = LLM(model=args.model, max_model_len=args.max_model_len, gpu_memory_utilization=args.gpu_mem,
              limit_mm_per_prompt={"image": 1}, dtype="bfloat16", seed=0, trust_remote_code=True)
    t_load = time.time() - t0

    def data_url(path):
        img = Image.open(path).convert("RGB")
        img.thumbnail((args.max_side, args.max_side))
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=92)
        return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()

    def msg(path, given):
        return [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": [{"type": "image_url", "image_url": {"url": data_url(path)}},
                                             {"type": "text", "text": user_text(given)}]}]

    def parse(o):
        text = o.outputs[0].text
        try:
            ann = json.loads(text)
            errs = validate(ann)
            verdict = (ann.get("label_assessment") or {}).get("consistency")
            report = ann.get("report")
        except Exception as e:
            ann, errs, verdict, report = None, [str(e)], None, text[:500]
        return {"tokens": len(o.outputs[0].token_ids), "finish_reason": o.outputs[0].finish_reason, "verdict": verdict,
                "valid": not errs, "errors": errs, "report": report}

    def one(path, given):
        t = time.time()
        o = llm.chat([msg(path, given)], sp, use_tqdm=False, chat_template_kwargs={"enable_thinking": False})[0]
        dt = time.time() - t
        return {"t_report": dt, **parse(o)}

    def given(p, c):
        return labels.get(p, c)

    for p, c in items[:2]:                            # warm-up (CUDA graphs, grammar compilation), not recorded
        one(p, given(p, c))
    records = []
    for n, (p, c) in enumerate(items):
        r = one(p, given(p, c))
        r.update(image=p, label=c, given_label=given(p, c))
        records.append(r)
        print(f"{n + 1}/{len(items)} {os.path.basename(p)}: given {r['given_label']} -> {r['verdict']} "
              f"{r['t_report']:.1f} s, {r['tokens']} tok, valid={r['valid']}", flush=True)
    smi_device, smi_procs = smi_device_used_mb(), smi_process_mb(None)   # the engine runs in child processes
    t = time.time()
    outs = llm.chat([msg(p, given(p, c)) for p, c in items], sp, use_tqdm=False,
                    chat_template_kwargs={"enable_thinking": False})
    t_batch = time.time() - t
    weights_gb = sum(os.path.getsize(f) for f in glob.glob(os.path.join(args.model, "*.safetensors"))) / 1e9
    out = {"model": args.model, "n": len(records), "t_load_s": t_load, "weights_on_disk_gb": weights_gb,
           "gpu_memory_utilization": args.gpu_mem, "max_model_len": args.max_model_len, "device_used_mb": smi_device,
           "smi_gpu_processes_mb": smi_procs,
           "t_report": stats([r["t_report"] for r in records]), "tokens": stats([r["tokens"] for r in records]),
           "n_valid": sum(r["valid"] for r in records),
           "n_accept_given": sum(r["verdict"] in ACCEPT for r in records),
           "n_given_equals_label": sum(r["given_label"] == r["label"] for r in records),
           "batch": {"n": len(items), "seconds": t_batch, "images_per_s": len(items) / t_batch,
                     "tokens": sum(len(o.outputs[0].token_ids) for o in outs)},
           "settings": {"temperature": 0.0, "repetition_penalty": 1.05, "max_tokens": args.max_tokens,
                        "max_side": args.max_side, "schema": "annotation_spec.bounded_schema", "per_class": args.per_class,
                        "seed": args.seed},
           "records": records}
    json.dump(out, open(args.out, "w"), indent=1)
    print(f"student: load {t_load:.0f} s, one request median {out['t_report']['median_s']:.1f} s "
          f"(p90 {out['t_report']['p90_s']:.1f}), batched {out['batch']['images_per_s']:.2f} photographs/s, "
          f"weights {weights_gb:.1f} GB on disk, GPU in use {smi_device} MB at utilisation {args.gpu_mem} -> {args.out}")


if __name__ == "__main__":
    main()
