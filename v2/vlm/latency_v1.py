"""v1 pipeline latency and GPU memory as deployed in demo_v5.py, on one GPU, for the v1 vs v2 comparison (manuscript
3.9, abstract). For each photograph the application's steps in the application's order and settings: MiniGPT-v2 image
encoding, ResNet-50 classification (the application's debug forward pass, then diagnose_or_none: two views each) with
the acceptance gate (class thresholds, p1 >= 0.5), RF-DETR part detection through the application's subprocess
(grounding/detector.py: a fresh interpreter and model load per call, as deployed; RFDETR_PYTHON names the interpreter
on the cluster, and a failed call is an error here, not the application's silent fall-back to the no-parts prompt),
the report with the application's prompt and decoding (temperature 0.2, top-p 0.9, repetition penalty 1.01, up to 2000
new tokens), its _postprocess_caption clean-up, then the consistency check and one regeneration with the stricter
prompt when a hallmark phrase of another condition is asserted. Sample: the same seeded 7 split v1 test photographs per
condition as the other latency_*.py scripts. Wall time per step with CUDA synchronised, after two warm-up photographs;
memory of this process from torch (peaks) and nvidia-smi (by PID on the job's GPU), and the GPU's memory in use (the
detector subprocess's own memory is measured by cluster/latency_rfdetr.py). Writes evaluation/latency_v1.json. Run
from the repo root with envs/minigptv on a GPU node, after `source /projects/$USER/agai/agai_env.sh`.
  python cluster/latency_v1.py [--per-class 7]
"""
import argparse
import json
import os
import random
import re
import statistics
import subprocess
import sys
import time

sys.path.insert(0, os.getcwd())
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from PIL import Image  # noqa: E402

from grounding.detector import run_rfdetr  # noqa: E402
from grounding.inference import load_minigpt  # noqa: E402
from knowledge_graph.rag_retriever import DiseaseRAG  # noqa: E402
from resnet_classifier import _DEVICE, _tfm, diagnose_or_none, load_resnet  # noqa: E402

sys.path.insert(0, os.path.join(os.getcwd(), "cluster"))
from build_stage2_prompts import PLANT_PARTS, prompt  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--per-class", type=int, default=7)
ap.add_argument("--split-json", default="evaluation/split_v1.json")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--resnet", default="plant_diagnostic/models/resnet_strawberry.pth")
ap.add_argument("--out", default="evaluation/latency_v1.json")
args = ap.parse_args()

# demo_v5.py's acceptance gate and label names
CLASS_THRESH = {"healthy": 0.40, "overwatered": 0.60, "root_rot": 0.65, "drought": 0.65, "frost": 0.70,
                "gray_mold": 0.60, "white_mold": 0.60}
CANON = {"healthy": "healthy", "overwatered": "overwatering", "root_rot": "root rot", "drought": "drought",
         "frost": "frost injury", "gray_mold": "gray mold", "white_mold": "white mold"}
# demo_v5.py's prompt when the classifier's label is not accepted (verbatim)
UNKNOWN_SYSTEM = """<<SYS>>You are a plant diagnostician. Confidence is too low to choose a diagnosis.

RESPONSE STRUCTURE (follow exactly):
1. State: "Based on the image provided, I cannot confidently diagnose the plant with a specific disease or condition."

2. List exactly 3 possible causes:
   - Lack of close-up images
   - Limited view of the plant
   - The image doesn't reveal any obvious signs of pests or diseases.
   
3. List exactly 3 recommended fixes:
   - Close-up images of the leaves, stems, and roots to examine for any signs of pests, diseases, or abnormalities.
   - Images of the plant's overall growth, including its size, shape, and any unusual features.
   - Images of the soil and surrounding environment to assess the plant's root health and potential stressors.

4. End with: "Based on these additional images, we can begin to identify potential causes for the plant's decline or health. If you have any further questions or concerns, please feel free to ask."

FORMATTING REQUIREMENTS:
- Use compact numbered lists: "1. explanation" (not "1.\nexplanation")
- No extra line breaks between list items
- Keep explanations on the same line as numbers
- Use dashes (-) for bullet points, not asterisks (*)
- Follow the exact structure above - do not deviate

Do not provide differential possibilities or lengthy explanations.<</SYS>>""".strip()
UNKNOWN_ASK = ("Examine this image. The diagnosis is Unknown. Describe the visible symptoms and provide treatment "
               "recommendations.")


def sample_items():
    """the same seeded sample in every latency_*.py script: --per-class split v1 test photographs per condition"""
    split = json.load(open(args.split_json))["test"]
    items = []
    for c in sorted(split):
        paths = sorted(split[c])
        items += [(p, c) for p in random.Random(f"{args.seed}:{c}").sample(paths, min(args.per_class, len(paths)))]
    return items


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


def device_used_mb():
    """memory in use on the current CUDA device, all processes (torch, so the right device under any enumeration)"""
    free, total = torch.cuda.mem_get_info()
    return int((total - free) / 2 ** 20)


def stats(xs):
    xs = sorted(xs)
    return {"median_s": statistics.median(xs), "p90_s": xs[int(0.9 * (len(xs) - 1))], "mean_s": sum(xs) / len(xs),
            "n": len(xs)} if xs else None


def accept(pred):
    """demo_v5.py's _accept_label plus its p1 < 0.50 rule: (canonical label or 'unknown', p1)"""
    if not pred:
        return "unknown", 0.0
    lbl = str(pred.get("label", "")).strip().lower()
    p1 = float(pred.get("p1", 0.0))
    p2 = float(pred.get("p2", 0.0)) if "p2" in pred else 0.0  # diagnose_or_none returns no p2: margin rule inactive
    if p1 < CLASS_THRESH.get(lbl, 0.55) or (p2 and p1 - p2 < 0.08) or p1 < 0.50:
        return "unknown", p1
    return CANON.get(lbl, "unknown"), p1


def retry_system(label, visible):
    """demo_v5.py's stricter regeneration prompt"""
    title = label.title()
    part_lines = ""
    if visible:
        part_lines = (f"VISIBLE (describe these): {', '.join(sorted(visible))}.\n"
                      f"NOT VISIBLE (do NOT mention these at all): {', '.join(sorted(PLANT_PARTS - set(visible)))}.\n")
    return f"""<<SYS>>You are a plant diagnostician. The diagnosis has already been determined: {title}
Your task is to examine the image and explain why this diagnosis is correct.
You MUST use this exact diagnosis: {title}
{part_lines}Provide a detailed medical report in this format:
1) Diagnosis: {title}
2) Visible cues: Describe the visual symptoms you observe that support this diagnosis.
3) Recommendation: Provide specific, actionable treatment steps.
Be detailed and thorough. Complete all recommendations fully.
IMPORTANT: Only describe symptoms of {title}. Do not mention symptoms of other diseases.
<</SYS>>"""


def _postprocess_caption(text):
    """demo_v5.py's clean-up of the writer's text, applied before the consistency check (copied verbatim)"""
    if not text:
        return ""
    t = text.strip()
    t = (t.replace("\u201c", '"').replace("\u201d", '"')
         .replace("\u2018", "'").replace("\u2019", "'"))
    t = re.sub(r"</?[^>\s]{1,32}>?", "", t)
    t = t.replace("\u200b", "").replace("\\n", "\n")
    t = t.replace("<", "").replace(">", "")
    t = re.sub(r"\*\s*", "• ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    t = re.sub(r"\n\n+", "\n\n", t)
    return t


def sync():
    torch.cuda.synchronize()


def encode(pil):
    conv = CONV.copy()
    img_list = []
    chat.upload_img(pil, conv, img_list)
    chat.encode_img(img_list)
    return conv, img_list


def generate(conv, img_list, system, ask):
    conv.system = system
    chat.ask(ask, conv)
    text, tokens = chat.answer(conv=conv, img_list=img_list, temperature=0.2, max_new_tokens=2000, max_length=4000,
                               num_beams=1, repetition_penalty=1.01)
    return _postprocess_caption(text), int(len(tokens))   # the application cleans the text before checking it


def run_one(path, truth):
    rec = {"image": path, "label": truth}
    with Image.open(path) as im:
        pil = im.convert("RGB")
    sync(); t = time.time()
    conv, img_list = encode(pil)                       # the application encodes the image first
    sync(); rec["t_encode"] = time.time() - t
    t = time.time()
    with Image.open(path) as im:                        # the application's debug pass (printed only): two views,
        x = _tfm(256)(im.convert("RGB")).unsqueeze(0).to(_DEVICE)   # softmax at T = 0.78, then the classification
    F.softmax(((resnet(x) + resnet(torch.flip(x, dims=[3]))) / 2) / 0.78, dim=1)
    pred = diagnose_or_none(resnet, path, img_size=256)  # two views, temperature-scaled softmax
    label, p1 = accept(pred)
    sync(); rec["t_classify"] = time.time() - t
    rec.update(pred_label=pred.get("label") if pred else None, p1=p1, final_label=label)
    visible, rec["t_detect"] = [], 0.0
    if label != "unknown":                              # the application runs the detector only for an accepted label
        t = time.time()
        det = run_rfdetr([path])
        rec["t_detect"] = time.time() - t
        if not det or "detected_parts" not in det.get(path, {}):   # the application would fall back to the no-parts
            raise RuntimeError(f"detector subprocess failed for {path}: {det!r}; is RFDETR_PYTHON set and honoured "
                               "by grounding/detector.py?")        # prompt; a measurement must not pass that off
        visible = sorted((det[path]["detected_parts"] or {}).keys())
    rec["visible"] = visible
    system, ask = (UNKNOWN_SYSTEM, UNKNOWN_ASK) if label == "unknown" else prompt(label, visible)
    sync(); t = time.time()
    report, n_tok = generate(conv, img_list, system, ask)
    sync(); rec["t_generate"] = time.time() - t
    rec.update(tokens=n_tok, flagged=False, t_retry=0.0, retry_tokens=0, still_flagged=False)
    if label != "unknown":
        check = rag.check_definite_hallucination(label, report)
        if check.get("has_definite_hallucination"):
            rec["flagged"] = True
            sync(); t = time.time()
            conv, img_list = encode(pil)               # the application re-encodes the image for the retry
            report, n_tok = generate(conv, img_list, retry_system(label, visible), ask)
            sync(); rec["t_retry"] = time.time() - t
            rec["retry_tokens"] = n_tok
            rec["still_flagged"] = bool(rag.check_definite_hallucination(label, report).get("has_definite_hallucination"))
    rec["t_total"] = rec["t_encode"] + rec["t_classify"] + rec["t_detect"] + rec["t_generate"] + rec["t_retry"]
    rec["report"] = report
    return rec


items = sample_items()
torch.cuda.reset_peak_memory_stats()
t0 = time.time()
chat, CONV = load_minigpt(0)        # the deployed writer (eval_configs/minigptv2_eval.yaml) and the demo's conversation
chat.model.eval()
resnet = load_resnet(args.resnet)
rag = DiseaseRAG()
sync()
t_load = time.time() - t0
smi_after_load = smi_process_mb()
print(f"loaded in {t_load:.0f} s; this process {smi_after_load} MB (nvidia-smi); {len(items)} photographs", flush=True)
for p, c in items[:2]:              # warm-up, not recorded
    run_one(p, c)
records = []
for n, (p, c) in enumerate(items):
    r = run_one(p, c)
    records.append(r)
    print(f"{n + 1}/{len(items)} {os.path.basename(p)}: {r['final_label']} p1={r['p1']:.2f} parts={r['visible']} "
          f"encode {r['t_encode']:.2f} classify {r['t_classify']:.2f} detect {r['t_detect']:.1f} "
          f"generate {r['t_generate']:.1f} ({r['tokens']} tok) retry {r['t_retry']:.1f} total {r['t_total']:.1f} s",
          flush=True)
accepted = [r for r in records if r["final_label"] != "unknown"]
out = {"pipeline": "v1 as deployed (demo_v5.py)", "gpu": torch.cuda.get_device_name(0), "n": len(records),
       "t_load_s": t_load, "smi_process_after_load_mb": smi_after_load, "smi_process_mb": smi_process_mb(),
       "device_used_mb": device_used_mb(),
       "peak_alloc_gb": torch.cuda.max_memory_allocated() / 1e9, "peak_reserved_gb": torch.cuda.max_memory_reserved() / 1e9,
       "t_encode": stats([r["t_encode"] for r in records]), "t_classify": stats([r["t_classify"] for r in records]),
       "t_detect_subprocess": stats([r["t_detect"] for r in accepted]),
       "t_generate": stats([r["t_generate"] for r in records]),
       "t_retry": stats([r["t_retry"] for r in records if r["flagged"]]),
       "t_total": stats([r["t_total"] for r in records]),
       "tokens": stats([r["tokens"] for r in records]), "n_unknown": len(records) - len(accepted),
       "n_flagged": sum(r["flagged"] for r in records), "n_still_flagged": sum(r["still_flagged"] for r in records),
       "settings": {"temperature": 0.2, "top_p": 0.9, "repetition_penalty": 1.01, "max_new_tokens": 2000,
                    "resnet": args.resnet, "per_class": args.per_class, "seed": args.seed},
       "records": records}
json.dump(out, open(args.out, "w"), indent=1)
print(f"v1 total per photograph: median {out['t_total']['median_s']:.1f} s, p90 {out['t_total']['p90_s']:.1f} s; "
      f"process memory {out['smi_process_mb']} MB (torch peak reserved {out['peak_reserved_gb']:.1f} GB) -> {args.out}")
