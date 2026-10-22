"""Score MiniGPT-v2 checkpoints as 7-class diagnosers on the same images as the Qwen zero-shot runs.

  official stage-2 / stage-3 (zero-shot, comparable to the Qwen runs): the same class-list prompt as
      cluster/vlm/zeroshot_qwen3vl.py, official '[INST] ... [/INST]' format, official LoRA alpha 16
  lab fine-tuned (output/minigpt_strawberry/checkpoint_best.pth): its training format, image only,
      'Human: <img> Assistant:' -> 'Diagnosis: X. ...', LoRA alpha 128. Also reported without the test images
      it trained on (evaluation/minigpt_lab_train_overlap.json from cluster/benchmark/minigpt_train_overlap.py).
The base model (Llama-2-7B 8-bit + EVA-ViT-g) loads once; checkpoints swap the LoRA + projection weights.
Run from the repo root with envs/minigptv on a GPU node.
"""
import collections
import json
import os
import re
import sys
import time

sys.path.insert(0, os.getcwd())
import torch  # noqa: E402
from PIL import Image  # noqa: E402

from grounding.inference import load_minigpt  # noqa: E402

CLASSES = {  # identical to cluster/vlm/zeroshot_qwen3vl.py
    "healthy": "no stress or disease; uniformly green, turgid foliage and normal fruit/flowers",
    "drought": "water deficit; wilting, dry, curled or crispy leaf edges, dry cracked soil",
    "overwatered": "excess water; limp or yellowing leaves, waterlogged or puddled soil, no fungal growth",
    "root_rot": "Phytophthora/Pythium root and crown rot; dark, decayed roots or crown, collapsing plant",
    "frost": "cold injury; blackened flower centers, bronzed or water-soaked damaged tissue",
    "gray_mold": "Botrytis cinerea; gray, fuzzy, dusty mold on fruit or flowers",
    "white_mold": "Sclerotinia; white, cottony mycelium, sometimes black sclerotia, on fruit, stems or crown",
}
PROMPT = (
    "You are a plant pathologist. Classify the main problem with the strawberry plant in this image into "
    "exactly one of these categories:\n"
    + "\n".join(f"- {k}: {v}" for k, v in CLASSES.items())
    + "\nReply with only the category name."
)
# label phrases -> class keys (most specific first)
LABELS = [("white mold", "white_mold"), ("gray mold", "gray_mold"), ("grey mold", "gray_mold"),
          ("root rot", "root_rot"), ("overwater", "overwatered"), ("frost", "frost"),
          ("drought", "drought"), ("healthy", "healthy")]
RUNS = [  # name, checkpoint, prompt format, LoRA alpha
    ("official_stage2", "checkpoints/checkpoint_stage2.pth", "[INST] <Img><ImageHere></Img> {} [/INST]", 16),
    ("official_stage3", "checkpoints/minigptv2_stage3.pth", "[INST] <Img><ImageHere></Img> {} [/INST]", 16),
    ("official_stage3_vqa", "checkpoints/minigptv2_stage3.pth", "[INST] <Img><ImageHere></Img> [vqa] {} [/INST]", 16),
    ("lab_finetuned", "output/minigpt_strawberry/checkpoint_best.pth", "Human: <Img><ImageHere></Img>  Assistant: ", 128),
]


def parse(text):
    t = text.lower().replace("_", " ")
    m = re.search(r"diagnosis:\s*([a-z ]+)", t)
    t = m.group(1) if m else t
    hits = [(t.find(p), k) for p, k in LABELS if p in t]
    return min(hits)[1] if hits else None


def load_weights(model, path):
    sd = torch.load(path, map_location="cpu", weights_only=False)["model"]
    # only the trained weights: the lab checkpoint also carries bitsandbytes 8-bit state (SCB, weight_format)
    # for every frozen layer, copies of the base model that initialized 8-bit layers refuse to reload
    sd = {k: v for k, v in sd.items() if "lora_" in k or k.startswith("llama_proj")}
    # official checkpoints predate peft's adapter names: lora_A.weight -> lora_A.default.weight
    sd = {re.sub(r"\.(lora_[AB])\.weight$", r".\1.default.weight", k): v for k, v in sd.items()}
    msg = model.load_state_dict(sd, strict=False)
    lora = [k for k in sd if "lora_" in k]
    loaded = [k for k in lora if k not in set(msg.unexpected_keys)]
    assert len(loaded) == len(lora) == 128, f"{path}: only {len(loaded)}/{len(lora)} LoRA tensors loaded"
    assert "llama_proj.weight" in sd and "llama_proj.weight" not in msg.unexpected_keys
    return len(sd)


def set_lora_alpha(model, alpha):
    n = 0
    for m in model.modules():
        if isinstance(getattr(m, "scaling", None), dict) and "default" in m.scaling:
            m.scaling["default"] = alpha / m.r["default"]
            n += 1
    return n


only = set(sys.argv[1:])  # optional run names to restrict to
chat, _ = load_minigpt(0)
model, vis = chat.model, chat.vis_processor
sets = {
    "paper68": [(r["found_at"], r["truth"]) for r in json.load(open("evaluation/holdout/repro_resnet_v4_hamilton.json"))["rows"]],
    "resnet161": [(p, c) for c, ps in sorted(json.load(open("evaluation/clean_eval_sets.json"))["resnet_v4_clean"].items()) for p in ps],
}
overlap = {k: set(v) for k, v in json.load(open("evaluation/minigpt_lab_train_overlap.json")).items()}

for name, ckpt, fmt, alpha in RUNS:
    if only and name not in only:
        continue
    if not os.path.isfile(ckpt):
        print(f"skip {name}: {ckpt} missing")
        continue
    nkeys = load_weights(model, ckpt)
    nlora = set_lora_alpha(model, alpha)
    text = fmt.format(PROMPT) if "{}" in fmt else fmt
    print(f"\n=== {name}: {ckpt} ({nkeys} tensors, LoRA alpha {alpha} on {nlora} layers)", flush=True)
    for set_name, items in sets.items():
        results, t0 = [], time.time()
        for b in range(0, len(items), 8):
            batch = items[b:b + 8]
            images = torch.stack([vis(Image.open(p).convert("RGB")) for p, _ in batch])
            with torch.no_grad():
                answers = model.generate(images, [text] * len(batch), max_new_tokens=24, do_sample=False)
            for (p, truth), a in zip(batch, answers):
                results.append({"image": p, "truth": truth, "pred": parse(a), "raw": a.strip()[:120]})
        per = collections.defaultdict(lambda: [0, 0])
        for r in results:
            per[r["truth"]][1] += 1
            per[r["truth"]][0] += r["pred"] == r["truth"]
        n, ok = len(results), sum(r["pred"] == r["truth"] for r in results)
        summary = {"run": name, "set": set_name, "accuracy": f"{ok}/{n} = {ok / n:.3f}",
                   "macro_accuracy": round(sum(c / t for c, t in per.values()) / len(per), 3),
                   "per_class": {k: f"{v[0]}/{v[1]}" for k, v in sorted(per.items())},
                   "unparsed": sum(r["pred"] is None for r in results),
                   "sec_per_image": round((time.time() - t0) / n, 2)}
        if name == "lab_finetuned":
            clean = [r for r in results if r["image"] not in overlap.get(set_name, set())]
            summary["accuracy_excluding_its_training_images"] = (
                f"{sum(r['pred'] == r['truth'] for r in clean)}/{len(clean)}")
        json.dump({"summary": summary, "prompt": text, "results": results},
                  open(f"evaluation/minigpt_{name}_{set_name}.json", "w"), indent=2)
        print(json.dumps(summary, indent=1), flush=True)
        print("sample outputs:", [r["raw"][:60] for r in results[:3]], flush=True)
