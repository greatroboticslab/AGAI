#!/usr/bin/env python3
"""Run the deployed MiniGPT-v2 on the 70 disease-holdout images and
freeze the outputs. Uses the v4 ResNet predictions (already computed)
as the diagnosis anchor and the RF-DETR holdout summary as the
visibility constraint.

Run inside the minigptv conda env. Outputs:
  paper/generated/minigpt_holdout_outputs.jsonl
  paper/generated/minigpt_holdout_summary.json (latency, basic counts)
"""

from __future__ import annotations
import json, os, sys, time
from pathlib import Path

# Disable verbose logging early
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from grounding.inference import load_minigpt, run_minigpt  # noqa: E402

HOLDOUT_V4 = REPO_ROOT / "evaluation" / "holdout" / "resnet_v4_release10holdout_full_holdout.json"
RFDETR_SUMMARY = REPO_ROOT / "paper" / "generated" / "rfdetr_holdout_summary.json"
OUT_JSONL = REPO_ROOT / "paper" / "generated" / "minigpt_holdout_outputs.jsonl"
OUT_SUMMARY = REPO_ROOT / "paper" / "generated" / "minigpt_holdout_summary.json"


def main() -> None:
    v4 = json.loads(HOLDOUT_V4.read_text())["predictions"]
    rfdetr = json.loads(RFDETR_SUMMARY.read_text())
    rfdetr_per_image = {row["image"]: row for row in rfdetr["per_image"]}

    print(f"Loading MiniGPT (this takes ~30 s)...", flush=True)
    chat, conv_template = load_minigpt(gpu_id=0)
    print("Loaded. Running inference on 70 holdout images...", flush=True)

    # Truncate output file
    OUT_JSONL.write_text("")

    latencies: list[float] = []
    by_class: dict[str, int] = {}
    for i, p in enumerate(v4):
        path = p["path"]
        truth = p["truth"]
        # Use the ResNet top-1 as the diagnosis anchor (this is the deployed flow).
        diagnosis = p["pred_top1"]
        rfd = rfdetr_per_image.get(path, {})
        detected_parts = {part: 1.0 for part in rfd.get("visible_parts", [])} or None

        t0 = time.perf_counter()
        try:
            response = run_minigpt(
                chat, conv_template, path, diagnosis,
                detected_parts=detected_parts,
            )
            err = None
        except Exception as e:  # report and continue
            response = ""
            err = repr(e)
        dt = time.perf_counter() - t0
        latencies.append(dt)
        by_class[truth] = by_class.get(truth, 0) + 1

        record = {
            "image": path,
            "truth": truth,
            "resnet_top1": p["pred_top1"],
            "resnet_p1": p["p1"],
            "resnet_top2": p["top2"],
            "rfdetr_visible_parts": rfd.get("visible_parts", []),
            "rfdetr_num_total_detections": rfd.get("num_total_detections"),
            "diagnosis_used": diagnosis,
            "response": response,
            "latency_s": dt,
            "error": err,
        }
        with OUT_JSONL.open("a") as f:
            f.write(json.dumps(record) + "\n")
        print(f"  [{i+1:>2}/70] {Path(path).name} ({truth} -> {diagnosis}) "
              f"latency={dt:.1f}s len={len(response)}", flush=True)

    import numpy as np
    summary = {
        "n": len(latencies),
        "median_latency_s": float(np.median(latencies)),
        "mean_latency_s": float(np.mean(latencies)),
        "p95_latency_s": float(np.percentile(latencies, 95)),
        "min_latency_s": float(np.min(latencies)),
        "max_latency_s": float(np.max(latencies)),
        "checkpoint": "/data/AGAI/MiniGPT-4/output/minigpt_strawberry/checkpoint_best.pth",
    }
    OUT_SUMMARY.write_text(json.dumps(summary, indent=2))
    print(f"\nMedian: {summary['median_latency_s']:.1f}s  P95: {summary['p95_latency_s']:.1f}s")
    print(f"Wrote {OUT_JSONL}")


if __name__ == "__main__":
    main()
