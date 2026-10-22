"""Per-class probabilities from a (LoRA fine-tuned) VLM on split v1 val/test, for ensembles and calibration.

One forward pass per image with the training/zero-shot prompt; the next-token logits at the answer position are
read for the first token of each class name (asserted distinct) and softmaxed over the 7 classes. "mass" records
how much of the full next-token distribution those 7 tokens hold (near 1 for a fine-tuned model).
Writes evaluation/scores_<model>_<tag>_split_v1_<split>.json (zero-shot result format plus "probs").
  python cluster/vlm/eval_vlm_scores.py --model /scratch/.../Qwen3-VL-8B-Instruct --adapter /projects/$USER/agai/adapters/Qwen3-VL-8B-Instruct_lora
"""
import argparse
import json
import os

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

CLASSES = {  # identical to cluster/vlm/zeroshot_qwen3vl.py and cluster/vlm/train_vlm_lora.py
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
NAMES = sorted(CLASSES)

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--adapter", default="", help="LoRA adapter dir; empty = zero-shot base model")
ap.add_argument("--tag", default="lora")
ap.add_argument("--split-json", default="evaluation/split_v1.json")
ap.add_argument("--splits", default="val,test")
ap.add_argument("--max-side", type=int, default=1024)
ap.add_argument("--batch-size", type=int, default=8)
args = ap.parse_args()

model = AutoModelForImageTextToText.from_pretrained(args.model, dtype=torch.bfloat16, device_map="cuda:0",
                                                    attn_implementation="sdpa")
processor = AutoProcessor.from_pretrained(args.model)
processor.tokenizer.padding_side = "left"
for mod in model.modules():   # same exact speed fix as training / zero-shot: Conv3d(kernel == stride) -> linear
    proj = getattr(mod, "proj", None)
    if isinstance(proj, torch.nn.Conv3d) and tuple(proj.kernel_size) == tuple(proj.stride) \
            and all(p == 0 for p in proj.padding):
        def fast(hidden_states, _w=proj.weight, _b=proj.bias):
            x = hidden_states.to(_w.dtype).reshape(-1, _w[0].numel())
            return torch.nn.functional.linear(x, _w.reshape(_w.shape[0], -1), _b).view(-1, _w.shape[0])
        mod.forward = fast
if args.adapter:
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, args.adapter)
model.eval()

first = [processor.tokenizer(n, add_special_tokens=False).input_ids[0] for n in NAMES]
assert len(set(first)) == len(first), f"class names share a first token: {dict(zip(NAMES, first))}"
first_t = torch.tensor(first, device="cuda:0")
split = json.load(open(args.split_json))
base = os.path.basename(args.model.rstrip("/"))


def load_image(p):
    img = Image.open(p).convert("RGB")
    img.thumbnail((args.max_side, args.max_side))
    return img


for s in args.splits.split(","):
    items = [(p, c) for c in NAMES for p in split[s][c]]
    results = []
    with torch.no_grad():
        for b in range(0, len(items), args.batch_size):
            batch = items[b:b + args.batch_size]
            convs = [[{"role": "user", "content": [{"type": "image", "image": load_image(p)},
                                                   {"type": "text", "text": PROMPT}]}] for p, _ in batch]
            enc = processor.apply_chat_template(convs, tokenize=True, add_generation_prompt=True, return_dict=True,
                                                return_tensors="pt", padding=True,
                                                enable_thinking=False).to(model.device)
            logits = model(**enc).logits[:, -1].float()
            full = logits.log_softmax(-1)
            sel = logits[:, first_t]
            probs = sel.softmax(-1).cpu()
            mass = full[:, first_t].exp().sum(-1).cpu()
            for (p, c), pr, ms in zip(batch, probs, mass):
                results.append({"image": p, "truth": c, "pred": NAMES[int(pr.argmax())],
                                "probs": {k: round(float(v), 5) for k, v in zip(NAMES, pr)}, "mass": round(float(ms), 4)})
    ok = sum(r["pred"] == r["truth"] for r in results)
    per = {k: f"{sum(r['pred'] == k and r['truth'] == k for r in results)}/{sum(r['truth'] == k for r in results)}"
           for k in NAMES}
    macro = sum(int(v.split("/")[0]) / int(v.split("/")[1]) for v in per.values()) / len(per)
    summary = {"model": base, "tag": args.tag, "adapter": args.adapter, "set": f"split_v1_{s}", "n": len(results),
               "accuracy": f"{ok}/{len(results)} = {ok / len(results):.3f}", "macro_accuracy": round(macro, 3),
               "per_class": per, "mean_mass": round(sum(r["mass"] for r in results) / len(results), 4)}
    out = f"evaluation/scores_{base}_{args.tag}_split_v1_{s}.json"
    json.dump({"summary": summary, "prompt": PROMPT, "results": results}, open(out, "w"), indent=1)
    print(json.dumps(summary), flush=True)
