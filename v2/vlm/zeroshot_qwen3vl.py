"""Zero-shot diagnosis with a vision-language model (Qwen3-VL, Qwen3.8, ...) on a strawberry holdout,
for comparison with ResNet v4. Same prompt and parsing for every model.

  --set paper68   the paper's holdout as reproduced by cluster/benchmark/repro_resnet_v4.py (same 68 images ResNet v4 saw)
  --set current   the lab's current plant_diagnostic/data/holdout/<class>/ folders (re-carved after the paper)
  --set resnet161 images ResNet v4 never trained on (cluster/benchmark/clean_eval_sets.py): fair for every model
  --set all1580   every human-labeled original, deduplicated: fair for zero-shot VLMs only
Run from the repo root with envs/vlm on a GPU node, e.g.
  python cluster/vlm/zeroshot_qwen3vl.py --model /projects/$USER/agai/models/Qwen3-VL-32B-Instruct --quant 4bit
"""
import argparse
import collections
import json
import os
import re
import time

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

CLASSES = {
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

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--quant", choices=["none", "8bit", "4bit"], default="none")
ap.add_argument("--set", nargs="+", choices=["paper68", "current", "resnet161", "all1580"], default=["paper68"],
                help="one or more sets; the model loads once (loading from /projects is slow)")
ap.add_argument("--max-side", type=int, default=1024, help="downscale images so the longest side is at most this")
ap.add_argument("--batch-size", type=int, default=8, help="images per generate() call (left-padded)")
ap.add_argument("--thinking", choices=["off", "on"], default="off",
                help="hybrid-thinking models (Qwen3.8): off answers directly, like the Instruct models; "
                     "on lets it reason (up to --think-budget tokens) and reads the answer after </think>")
ap.add_argument("--think-budget", type=int, default=1024,
                help="thinking tokens before the thought is closed and an answer is forced")
ap.add_argument("--save-quantized", default=None,
                help="after a --quant load, save the quantized model here (much faster to load next time)")
args = ap.parse_args()


def load_items(name):
    if name == "paper68":
        rows = json.load(open("evaluation/holdout/repro_resnet_v4_hamilton.json"))["rows"]
        return [(r["found_at"], r["truth"], r["repro_pred"]) for r in rows]
    if name in ("resnet161", "all1580"):
        key = "resnet_v4_clean" if name == "resnet161" else "vlm_all_originals"
        sets = json.load(open("evaluation/clean_eval_sets.json"))[key]
        return [(p, c, None) for c, paths in sorted(sets.items()) for p in paths]
    root = "plant_diagnostic/data/holdout"
    return [(os.path.join(root, c, f), c, None) for c in sorted(os.listdir(root)) if c in CLASSES
            for f in sorted(os.listdir(os.path.join(root, c)))]


quant = None
if args.quant == "4bit":
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                               bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
elif args.quant == "8bit":
    quant = BitsAndBytesConfig(load_in_8bit=True)
t0 = time.time()
model = AutoModelForImageTextToText.from_pretrained(
    args.model, dtype=torch.bfloat16, device_map="cuda:0", attn_implementation="sdpa", quantization_config=quant)
processor = AutoProcessor.from_pretrained(args.model)


def linearize_patch_embeds(model):
    """The vision patch embedding is a Conv3d whose kernel equals its stride, i.e. a matmul over each flattened
    patch, but cuDNN runs it pathologically slowly in bf16 (GPU ~30% busy, one CPU core pinned). Swap in the
    identical F.linear on reshaped weights, after checking both agree on a random batch."""
    n = 0
    for m in model.modules():
        proj = getattr(m, "proj", None)
        if not (isinstance(proj, torch.nn.Conv3d) and tuple(proj.kernel_size) == tuple(proj.stride)
                and all(p == 0 for p in proj.padding)):
            continue
        w = proj.weight
        def fast(hidden_states, _w=w, _b=proj.bias):
            x = hidden_states.to(_w.dtype).reshape(-1, _w[0].numel())
            return torch.nn.functional.linear(x, _w.reshape(_w.shape[0], -1), _b).view(-1, _w.shape[0])
        with torch.no_grad():
            probe = torch.randn(64, w[0].numel(), device=w.device, dtype=w.dtype)
            ref = m.forward(probe)
            err = (fast(probe).float() - ref.float()).abs().max().item() / (ref.float().abs().max().item() + 1e-6)
        assert err < 2e-2, f"linearized patch embed disagrees with Conv3d (rel err {err:.3g})"
        m.forward = fast
        n += 1
    return n


print(f"linearized {linearize_patch_embeds(model)} vision patch-embed Conv3d layer(s)", flush=True)
processor.tokenizer.padding_side = "left"   # batched generation with a decoder-only model
print(f"loaded {os.path.basename(args.model)} ({args.quant}) in {time.time() - t0:.0f}s, "
      f"GPU {torch.cuda.memory_allocated() / 2**30:.1f} GiB", flush=True)
if args.save_quantized and args.quant != "none" and not os.path.exists(args.save_quantized):
    model.save_pretrained(args.save_quantized)
    processor.save_pretrained(args.save_quantized)
    print(f"saved quantized copy to {args.save_quantized}", flush=True)

label_re = re.compile("|".join(sorted(CLASSES, key=len, reverse=True)))
special_re = re.compile(r"<\|[^|]*\|>")
pad_id = processor.tokenizer.pad_token_id
thinking = args.thinking == "on"
torch.manual_seed(0)


def force_answer(conversation, gen_ids):
    """Continue one conversation after an exhausted thinking budget: prompt + its thoughts + '</think>' -> answer."""
    single = processor.apply_chat_template([conversation], tokenize=True, add_generation_prompt=True,
                                           return_dict=True, return_tensors="pt", enable_thinking=True).to(model.device)
    close = processor.tokenizer("\n</think>\n\n", add_special_tokens=False, return_tensors="pt").input_ids.to(model.device)
    n_prompt = single["input_ids"].shape[1]
    ids = torch.cat([single["input_ids"], gen_ids[gen_ids != pad_id][None], close], dim=1)
    extra = {}
    for k, v in single.items():
        if k in ("input_ids", "attention_mask"):
            continue
        if "token_type" in k and torch.is_tensor(v) and v.shape[-1] == n_prompt:   # per-token side input: pad as text
            v = torch.cat([v, v.new_zeros(*v.shape[:-1], ids.shape[1] - n_prompt)], dim=-1)
        extra[k] = v
    with torch.no_grad():
        out = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), **extra, max_new_tokens=12, do_sample=False)
    return processor.batch_decode(out[:, ids.shape[1]:], skip_special_tokens=True)[0]


for set_name in args.set:
    items = load_items(set_name)
    results, t0 = [], time.time()
    for b in range(0, len(items), args.batch_size):
        batch = items[b:b + args.batch_size]
        conversations = []
        for path, _, _ in batch:
            img = Image.open(path).convert("RGB")
            img.thumbnail((args.max_side, args.max_side))
            conversations.append([{"role": "user", "content": [{"type": "image", "image": img},
                                                               {"type": "text", "text": PROMPT}]}])
        inputs = processor.apply_chat_template(conversations, tokenize=True, add_generation_prompt=True,
                                               return_dict=True, return_tensors="pt", padding=True,
                                               enable_thinking=thinking).to(model.device)
        # thinking: Qwen's recommended sampling (greedy decoding loops in thinking mode); otherwise greedy
        gen = (dict(max_new_tokens=args.think_budget, do_sample=True, temperature=0.6, top_p=0.95, top_k=20)
               if thinking else dict(max_new_tokens=12, do_sample=False))
        with torch.no_grad():
            out = model.generate(**inputs, **gen)
        new = out[:, inputs["input_ids"].shape[1]:]
        texts = processor.batch_decode(new, skip_special_tokens=False)
        for i, ((path, truth, resnet), text) in enumerate(zip(batch, texts)):
            forced = thinking and "</think>" not in text
            if forced:   # budget used up mid-thought: close the thought and make it answer
                text = text + "\n</think>\n\n" + force_answer(conversations[i], new[i])
            text = special_re.sub("", text)
            answer = text.rsplit("</think>", 1)[-1]   # thinking mode: only the final answer counts
            m = label_re.search(answer.lower().replace(" ", "_"))
            results.append({"image": path, "truth": truth, "pred": m.group(0) if m else None, "raw": text.strip(),
                            "resnet_v4": resnet, "gen_tokens": int((new[i] != pad_id).sum()), "forced_answer": forced})

    n = len(results)
    correct = sum(r["pred"] == r["truth"] for r in results)
    per_class = collections.defaultdict(lambda: [0, 0])
    confusion = collections.Counter()
    for r in results:
        per_class[r["truth"]][1] += 1
        per_class[r["truth"]][0] += r["pred"] == r["truth"]
        confusion[(r["truth"], r["pred"])] += 1
    summary = {
        "model": os.path.basename(args.model), "quant": args.quant, "thinking": args.thinking, "set": set_name, "n": n,
        "accuracy": f"{correct}/{n} = {correct / n:.3f}",
        "per_class": {k: f"{v[0]}/{v[1]}" for k, v in sorted(per_class.items())},
        "macro_accuracy": round(sum(v[0] / v[1] for v in per_class.values()) / len(per_class), 3),
        "unparsed": sum(r["pred"] is None for r in results),
        "mean_gen_tokens": round(sum(r["gen_tokens"] for r in results) / n, 1),
        "forced_answers": sum(r["forced_answer"] for r in results),
        "sec_per_image": round((time.time() - t0) / n, 2),
    }
    if set_name == "paper68":
        summary["resnet_v4_same_images"] = f"{sum(r['resnet_v4'] == r['truth'] for r in results)}/{n}"
    tag = f"_thinking{args.think_budget}" if thinking else ""
    out = f"evaluation/zeroshot_{os.path.basename(args.model)}_{args.quant}{tag}_{set_name}.json"
    json.dump({"summary": summary, "prompt": PROMPT, "results": results,
               "confusion": [[t, p, c] for (t, p), c in sorted(confusion.items(), key=str)]}, open(out, "w"), indent=2)
    print(json.dumps(summary, indent=2))
    print("top confusions:", [f"{t}->{p}:{c}" for (t, p), c in confusion.most_common() if t != p][:8], flush=True)
