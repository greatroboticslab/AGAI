"""Generate v1-style diagnostic reports with a MiniGPT-v2 report-writer checkpoint for the report test set
(plant_diagnostic/datasets/split_v1/report_test_set.json: split v1 test photographs with plant-part masks), under two
prompts: the application's prompt with the v1 detector's part lists (cluster/detect_parts.py output) and the same
prompt without part lists (the detector on/off ablation, manuscript Table S16). Decoding as in demo_v5.py
(temperature 0.2, top-p 0.9, repetition penalty 1.01); the consistency check regenerates once with the stricter
prompt when a hallmark phrase of another condition is asserted, as the application does. Everything is saved for
cluster/score_reports.py. Run from the repo root with envs/minigptv on a GPU node.
  python cluster/minigpt_reports.py --name splitv1 --ckpt /projects/$USER/agai/minigpt_runs/.../checkpoint_best.pth \
      --parts evaluation/parts_report_test.json
"""
import argparse
import json
import os
import re
import sys
import time

sys.path.insert(0, os.getcwd())
import torch  # noqa: E402
from PIL import Image  # noqa: E402

from grounding.inference import load_minigpt  # noqa: E402
from knowledge_graph.rag_retriever import DiseaseRAG  # noqa: E402

sys.path.insert(0, os.path.join(os.getcwd(), "cluster"))
from build_stage2_prompts import CANON, PLANT_PARTS, prompt  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--name", required=True)
ap.add_argument("--ckpt", required=True, help="LoRA + projection checkpoint (checkpoint_best.pth)")
ap.add_argument("--parts", required=True, help="detect_parts.py output for the test photographs")
ap.add_argument("--test-set", default="plant_diagnostic/datasets/split_v1/report_test_set.json")
ap.add_argument("--lora-alpha", type=float, default=128)
ap.add_argument("--max-new-tokens", type=int, default=600)
ap.add_argument("--conditions", default="parts,noparts")
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()

TEMPLATE = "[INST] <Img><ImageHere></Img> {} [/INST]"  # the training format of minigpt_v2 (prompt_template)


def retry_prompt(label, visible):
    """demo_v5.py's stricter regeneration prompt"""
    title = label.title()
    part_lines = ""
    if visible:
        part_lines = (f"VISIBLE (describe these): {', '.join(sorted(visible))}.\n"
                      f"NOT VISIBLE (do NOT mention these at all): {', '.join(sorted(PLANT_PARTS - set(visible)))}.\n")
    system = f"""<<SYS>>You are a plant diagnostician. The diagnosis has already been determined: {title}
Your task is to examine the image and explain why this diagnosis is correct.
You MUST use this exact diagnosis: {title}
{part_lines}Provide a detailed medical report in this format:
1) Diagnosis: {title}
2) Visible cues: Describe the visual symptoms you observe that support this diagnosis.
3) Recommendation: Provide specific, actionable treatment steps.
Be detailed and thorough. Complete all recommendations fully.
IMPORTANT: Only describe symptoms of {title}. Do not mention symptoms of other diseases.
<</SYS>>"""
    return system


def load_weights(model, path):
    sd = torch.load(path, map_location="cpu", weights_only=False)["model"]
    sd = {k: v for k, v in sd.items() if "lora_" in k or k.startswith("llama_proj")}
    sd = {re.sub(r"\.(lora_[AB])\.weight$", r".\1.default.weight", k): v for k, v in sd.items()}
    msg = model.load_state_dict(sd, strict=False)
    lora = [k for k in sd if "lora_" in k]
    loaded = [k for k in lora if k not in set(msg.unexpected_keys)]
    assert len(loaded) == len(lora) == 128, f"{path}: only {len(loaded)}/{len(lora)} LoRA tensors loaded"
    assert "llama_proj.weight" in sd and "llama_proj.weight" not in msg.unexpected_keys
    for m in model.modules():
        if isinstance(getattr(m, "scaling", None), dict) and "default" in m.scaling:
            m.scaling["default"] = args.lora_alpha / m.r["default"]


def generate(model, vis, image, text):
    torch.manual_seed(args.seed)
    with torch.no_grad():
        out = model.generate(torch.stack([vis(image)]), [text], max_new_tokens=args.max_new_tokens, do_sample=True,
                             temperature=0.2, top_p=0.9, repetition_penalty=1.01)
    return out[0].strip()


chat, _ = load_minigpt(0)
model, vis = chat.model, chat.vis_processor
load_weights(model, args.ckpt)
rag = DiseaseRAG()
parts_det = json.load(open(args.parts))
items = json.load(open(args.test_set))
out_path = f"evaluation/reports_{args.name}.json"
results = json.load(open(out_path)) if os.path.exists(out_path) else []
done = {(r["image"], r["condition"]) for r in results}
t0 = time.time()
for n, it in enumerate(items):
    label = CANON[it["label"]]
    detected = parts_det.get(it["image"], {}).get("visible", [])
    with Image.open(it["image"]) as im:
        image = im.convert("RGB")
    for cond in args.conditions.split(","):
        if (it["image"], cond) in done:
            continue
        visible = detected if cond == "parts" else []
        system, ask = prompt(label, visible)
        text = TEMPLATE.format(f"{system}\n{ask} ")
        attempts = []
        report = generate(model, vis, image, text)
        check = rag.check_definite_hallucination(it["label"], report)
        attempts.append({"report": report, "flagged": check["has_definite_hallucination"],
                         "hallucinations": check["hallucinations"], "negated": check.get("negated_mentions", [])})
        if check["has_definite_hallucination"]:
            report = generate(model, vis, image, TEMPLATE.format(f"{retry_prompt(label, visible)}\n{ask} "))
            check = rag.check_definite_hallucination(it["label"], report)
            attempts.append({"report": report, "flagged": check["has_definite_hallucination"],
                             "hallucinations": check["hallucinations"], "negated": check.get("negated_mentions", [])})
        results.append({"image": it["image"], "label": it["label"], "condition": cond, "detected_parts": detected,
                        "annotated_parts": it["annotated_parts"], "attempts": attempts, "final_report": report,
                        "still_flagged": check["has_definite_hallucination"]})
    if n % 10 == 0:
        json.dump(results, open(out_path, "w"), indent=1)
        print(f"{n + 1}/{len(items)} {time.time() - t0:.0f}s {it['image'].split('/')[-1]}: "
              f"{results[-1]['final_report'][:90]!r}", flush=True)
json.dump(results, open(out_path, "w"), indent=1)
print(f"done: {len(results)} reports in {(time.time() - t0) / 60:.1f} min -> {out_path}")
