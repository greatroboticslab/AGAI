"""Assemble the v1 vs v2 latency and GPU-memory table from the four latency_*.py outputs (same photographs): per
stage and per pipeline the median, p90 and mean seconds per photograph, per-process GPU memory, and the co-resident
total each pipeline needs on one card. v1 as deployed = ResNet-50 + RF-DETR Small through the application's subprocess
+ MiniGPT-v2 report with its consistency check; "v1, detector in-process" swaps the subprocess time for the in-process
RF-DETR Small time. v2 = SigLIP 2 classifier + RF-DETR Large + the distilled Qwen3-VL-8B student under vLLM, measured in
separate processes and summed per photograph. Writes evaluation/paper_data/latency_v1_v2.json and prints Markdown.
  python cluster/latency_summary.py
"""
import json
import os
import statistics

E = "evaluation"
v1 = json.load(open(f"{E}/latency_v1.json"))
det = json.load(open(f"{E}/latency_rfdetr.json"))
sig = json.load(open(f"{E}/latency_siglip2.json"))
stu = json.load(open(f"{E}/latency_student.json"))


def by_image(records, key):
    return {r["image"]: r[key] for r in records}


def summ(xs):
    xs = sorted(xs)
    if not xs:
        return None
    return {"median_s": round(statistics.median(xs), 3), "p90_s": round(xs[int(0.9 * (len(xs) - 1))], 3),
            "mean_s": round(sum(xs) / len(xs), 3), "n": len(xs)}


imgs = [r["image"] for r in v1["records"]]
small = by_image(det["small_lab_v1"]["per_image"], "t_detect")
large = by_image(det["large_splitv1_v2"]["per_image"], "t_detect")
sig_t = by_image(sig["records"], "t_classify")
stu_t = by_image(stu["records"], "t_report")
missing = [i for i in imgs if i not in small or i not in large or i not in sig_t or i not in stu_t]
assert not missing, f"photographs missing from a stage file: {missing[:3]}"

v1_dep = [r["t_total"] for r in v1["records"]]
v1_inproc = [r["t_total"] - r["t_detect"] + (small[r["image"]] if r["final_label"] != "unknown" else 0.0)
             for r in v1["records"]]
v2 = [sig_t[i] + large[i] + stu_t[i] for i in imgs]
flagged = [r for r in v1["records"] if r["flagged"]]
stages = {
    "v1 MiniGPT-v2 image encoding": summ([r["t_encode"] for r in v1["records"]]),
    "v1 ResNet-50, two views + gate": summ([r["t_classify"] for r in v1["records"]]),
    "v1 RF-DETR Small, application subprocess": summ([r["t_detect"] for r in v1["records"] if r["final_label"] != "unknown"]),
    "v1 RF-DETR Small, in-process": summ(list(small.values())),
    "v1 MiniGPT-v2 report, first attempt": summ([r["t_generate"] for r in v1["records"]]),
    "v1 regeneration (flagged photographs only)": summ([r["t_retry"] for r in flagged]),
    "v2 SigLIP 2 classifier, two views": summ(list(sig_t.values())),
    "v2 RF-DETR Large, in-process": summ(list(large.values())),
    "v2 Qwen3-VL-8B student, one request under vLLM": summ(list(stu_t.values())),
}
pipelines = {"v1 as deployed": summ(v1_dep), "v1, detector in-process": summ(v1_inproc), "v2": summ(v2)}
v1_proc, small_proc = v1.get("smi_process_mb"), det["small_lab_v1"].get("smi_process_mb")
memory = {   # per process from nvidia-smi compute-apps (None when the driver reports none); torch peaks as a check
    "v1 MiniGPT-v2 + ResNet-50 process, nvidia-smi MB": v1_proc,
    "v1 MiniGPT-v2 + ResNet-50 process, torch peak reserved GB": round(v1["peak_reserved_gb"], 2),
    "v1 RF-DETR Small process, nvidia-smi MB": small_proc,
    "v1 RF-DETR Small process, torch peak reserved GB": round(det["small_lab_v1"]["peak_reserved_gb"], 2),
    "v1 co-resident total, MB": (v1_proc + small_proc) if v1_proc and small_proc else None,
    "v2 SigLIP 2 process, nvidia-smi MB": sig.get("smi_process_mb"),
    "v2 SigLIP 2 process, torch peak reserved GB": round(sig["peak_reserved_gb"], 2),
    "v2 RF-DETR Large process, nvidia-smi MB": det["large_splitv1_v2"].get("smi_process_mb"),
    "v2 RF-DETR Large process, torch peak reserved GB": round(det["large_splitv1_v2"]["peak_reserved_gb"], 2),
    "v2 student weights on disk, GB": round(stu["weights_on_disk_gb"], 1),
    "v2 student (vLLM engine) processes, nvidia-smi MB": stu.get("smi_gpu_processes_mb"),
    "v2 student GPU in use at gpu_memory_utilization, MB": stu.get("device_used_mb"),
    "v2 student gpu_memory_utilization": stu["gpu_memory_utilization"],
}
counts = {
    "n_photographs": len(imgs),
    "v1 report tokens, median": statistics.median([r["tokens"] for r in v1["records"]]),
    "v1 gate: unknown": v1["n_unknown"], "v1 consistency check: flagged": v1["n_flagged"],
    "v1 still flagged after regeneration": v1["n_still_flagged"],
    "v2 classifier accuracy on the sample": round(sig["accuracy"], 3),
    "v2 tokens, median": statistics.median([r["tokens"] for r in stu["records"]]),
    "v2 valid JSON": stu["n_valid"], "v2 accepts the proposed label": stu["n_accept_given"],
    "v2 proposed label equals dataset label": stu["n_given_equals_label"],
    "v2 batched throughput, photographs per s": round(stu["batch"]["images_per_s"], 3),
    "gpu": v1.get("gpu"),
}
out = {"stages": stages, "pipelines": pipelines, "memory": memory, "counts": counts,
       "load_times_s": {"v1 MiniGPT-v2 + ResNet-50": v1["t_load_s"], "RF-DETR Small": det["small_lab_v1"]["t_load_s"],
                        "RF-DETR Large": det["large_splitv1_v2"]["t_load_s"], "SigLIP 2": sig["t_load_s"],
                        "student (vLLM engine)": stu["t_load_s"]}}
os.makedirs(f"{E}/paper_data", exist_ok=True)
json.dump(out, open(f"{E}/paper_data/latency_v1_v2.json", "w"), indent=1)

print(f"\n| Stage or pipeline ({len(imgs)} photographs, {counts['gpu']}) | median s | p90 s | mean s |")
print("|---|---|---|---|")
for k, s in {**stages, **pipelines}.items():
    if s:
        print(f"| {k} | {s['median_s']:.2f} | {s['p90_s']:.2f} | {s['mean_s']:.2f} |")
print("\n| Memory | value |\n|---|---|")
for k, v in memory.items():
    print(f"| {k} | {v} |")
print("\n| Count | value |\n|---|---|")
for k, v in counts.items():
    print(f"| {k} | {v} |")
print(f"\n-> {E}/paper_data/latency_v1_v2.json")
