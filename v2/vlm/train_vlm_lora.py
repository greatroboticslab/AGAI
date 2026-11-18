"""LoRA / QLoRA fine-tune of a vision-language model (Qwen3-VL, ...) as a 7-class strawberry diagnoser on the
leak-free split (evaluation/split_v1.json), with the SAME prompt and answer parsing as the zero-shot runs, so
before/after fine-tuning compare directly.

- Each example: user [image, class-list prompt] -> assistant "<class name>"; only the answer tokens are supervised.
- LoRA on all linear layers (vision encoder included: the zero-shot errors were perceptual). 4-bit NF4 base with
  --quant 4bit, or a pre-quantized model dir (e.g. models/Qwen3-VL-32B-Instruct-bnb4); bf16 otherwise.
- Batch size 1 per GPU with gradient accumulation. Multi-node (srun, one rank per node, see
  cluster/infra/ddp_task.sh): each rank trains on its shard and LoRA gradients are averaged with one bf16 all-reduce
  per optimizer step (Hamilton's nodes only talk over 1 GbE, so accumulate more per exchange).
  Effective batch = --grad-accum x number of nodes.
- Validation accuracy after every epoch picks the adapter to keep; the test split is scored once, at the end,
  and written in the zero-shot result format (evaluation/finetuned_<model>_<tag>_test.json).
"""
import argparse
import collections
import datetime
import json
import math
import os
import random
import re
import time

import torch
import torch.distributed as dist
from PIL import Image
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

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

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--quant", choices=["none", "4bit"], default="none")
ap.add_argument("--split", default="evaluation/split_v1.json")
ap.add_argument("--epochs", type=int, default=3)
ap.add_argument("--lr", type=float, default=1e-4)
ap.add_argument("--lora-r", type=int, default=16)
ap.add_argument("--lora-alpha", type=int, default=32)
ap.add_argument("--grad-accum", type=int, default=8, help="micro-batches per optimizer step, per GPU")
ap.add_argument("--max-side", type=int, default=1024)
ap.add_argument("--limit", type=int, default=0, help="debug: use only N train/val/test images")
ap.add_argument("--tag", default="lora")
ap.add_argument("--seed", type=int, default=0, help="LoRA init and data order (0 = the original runs)")
ap.add_argument("--out-dir", default=f"/projects/{os.environ.get('USER', 'user')}/agai/adapters")
args = ap.parse_args()

world, rank = int(os.environ.get("WORLD_SIZE", "1")), int(os.environ.get("RANK", "0"))
if world > 1:
    dist.init_process_group("nccl", timeout=datetime.timedelta(minutes=60))
main = rank == 0


def log(*a):
    if main:
        print(*a, flush=True)


torch.manual_seed(args.seed)
random.seed(args.seed)

split = json.load(open(args.split))
items = {s: [(p, c) for c, ps in sorted(split[s].items()) for p in ps] for s in ("train", "val", "test")}
if args.limit:
    items = {s: random.Random(1).sample(v, min(args.limit, len(v))) for s, v in items.items()}
log({s: len(v) for s, v in items.items()}, f"| {world} GPU(s), effective batch {args.grad_accum * world}")

prequantized = "quantization_config" in json.load(open(os.path.join(args.model, "config.json")))
quant = (BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16,
                            bnb_4bit_use_double_quant=True) if args.quant == "4bit" and not prequantized else None)
# pass quantization_config only when quantizing here: an explicit None overwrites a pre-quantized model's own
model = AutoModelForImageTextToText.from_pretrained(args.model, dtype=torch.bfloat16, device_map="cuda:0",
                                                    attn_implementation="sdpa",
                                                    **({"quantization_config": quant} if quant else {}))
processor = AutoProcessor.from_pretrained(args.model)
processor.tokenizer.padding_side = "left"


def linearize_patch_embeds(m):
    """Same speed fix as the zero-shot script: Conv3d(kernel == stride) -> identical F.linear (frozen layer)."""
    n = 0
    for mod in m.modules():
        proj = getattr(mod, "proj", None)
        if isinstance(proj, torch.nn.Conv3d) and tuple(proj.kernel_size) == tuple(proj.stride) \
                and all(p == 0 for p in proj.padding):
            w = proj.weight

            def fast(hidden_states, _w=w, _b=proj.bias):
                x = hidden_states.to(_w.dtype).reshape(-1, _w[0].numel())
                return torch.nn.functional.linear(x, _w.reshape(_w.shape[0], -1), _b).view(-1, _w.shape[0])
            mod.forward = fast
            n += 1
    return n


log(f"linearized {linearize_patch_embeds(model)} patch-embed layer(s); pre-quantized: {prequantized}")
# 4-bit or not, the frozen non-quantized weights stay bf16: prepare_model_for_kbit_training would upcast the
# embeddings and lm_head (~1.6B params on the 32B) to fp32, ~3 GB more on a 32 GB card, a guard meant for fp16
model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
model.enable_input_require_grads()
model = get_peft_model(model, LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.05,
                                         target_modules="all-linear", task_type="CAUSAL_LM"))
if main:
    model.print_trainable_parameters()
params = [p for p in model.parameters() if p.requires_grad]
if world > 1:   # identical starting adapters on every rank
    for p in params:
        dist.broadcast(p.data, src=0)

think_kw = {"enable_thinking": False}   # non-thinking answers for hybrid models; ignored by Instruct templates


def load_image(path):
    img = Image.open(path).convert("RGB")
    img.thumbnail((args.max_side, args.max_side))
    return img


def encode(path, answer):
    """Prompt exactly as at inference, then the answer + end-of-turn; the prompt is masked from the loss."""
    conv = [{"role": "user", "content": [{"type": "image", "image": load_image(path)}, {"type": "text", "text": PROMPT}]}]
    enc = processor.apply_chat_template([conv], tokenize=True, add_generation_prompt=True, return_dict=True,
                                        return_tensors="pt", **think_kw)
    ans = processor.tokenizer(answer + "<|im_end|>\n", add_special_tokens=False, return_tensors="pt").input_ids
    n = enc["input_ids"].shape[1]
    enc["input_ids"] = torch.cat([enc["input_ids"], ans], dim=1)
    enc["attention_mask"] = torch.ones_like(enc["input_ids"])
    for k in list(enc.keys()):   # per-token side inputs (e.g. mm_token_type_ids): pad the answer as text
        if "token_type" in k:
            enc[k] = torch.cat([enc[k], enc[k].new_zeros(*enc[k].shape[:-1], ans.shape[1])], dim=-1)
    enc["labels"] = torch.cat([torch.full((1, n), -100), ans], dim=1)
    return enc


comm_s = 0.0   # wall time in the gradient all-reduce this epoch: what the 1 GbE link costs


def sync_grads():
    """Average LoRA gradients across ranks with one bf16 all-reduce (1 GbE between nodes)."""
    global comm_s
    if world == 1:
        return
    t = time.time()
    grads = [p.grad if p.grad is not None else torch.zeros_like(p) for p in params]
    flat = torch._utils._flatten_dense_tensors([g.to(torch.bfloat16) for g in grads])
    dist.all_reduce(flat)
    flat.div_(world)
    for p, g in zip(params, torch._utils._unflatten_dense_tensors(flat, grads)):
        if p.grad is None:
            p.grad = g.to(p.dtype)
        else:
            p.grad.copy_(g)
    torch.cuda.synchronize()
    comm_s += time.time() - t


label_re = re.compile("|".join(sorted(CLASSES, key=len, reverse=True)))


@torch.no_grad()
def evaluate(split_items, batch_size=8):
    """Each rank scores a strided shard; results are gathered so every rank sees the full list."""
    model.eval()
    mine = list(enumerate(split_items))[rank::world]
    results = []
    for b in range(0, len(mine), batch_size):
        batch = mine[b:b + batch_size]
        convs = [[{"role": "user", "content": [{"type": "image", "image": load_image(p)},
                                               {"type": "text", "text": PROMPT}]}] for _, (p, _) in batch]
        enc = processor.apply_chat_template(convs, tokenize=True, add_generation_prompt=True, return_dict=True,
                                            return_tensors="pt", padding=True, **think_kw).to(model.device)
        out = model.generate(**enc, max_new_tokens=12, do_sample=False)
        texts = processor.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        for (idx, (p, truth)), t in zip(batch, texts):
            m = label_re.search(t.lower().replace(" ", "_"))
            results.append((idx, {"image": p, "truth": truth, "pred": m.group(0) if m else None, "raw": t.strip()}))
    if world > 1:
        gathered = [None] * world
        dist.all_gather_object(gathered, results)
        results = [r for part in gathered for r in part]
    model.train()
    return [r for _, r in sorted(results, key=lambda x: x[0])]


def summarize(results, name):
    per = collections.defaultdict(lambda: [0, 0])
    for r in results:
        per[r["truth"]][1] += 1
        per[r["truth"]][0] += r["pred"] == r["truth"]
    n, ok = len(results), sum(r["pred"] == r["truth"] for r in results)
    return {"model": os.path.basename(args.model), "tag": args.tag, "set": name, "n": n,
            "accuracy": f"{ok}/{n} = {ok / n:.3f}", "macro_accuracy": round(sum(c / t for c, t in per.values()) / len(per), 3),
            "per_class": {k: f"{v[0]}/{v[1]}" for k, v in sorted(per.items())}}


per_rank = len(items["train"]) // world          # equal shards, so every rank takes the same number of steps
steps = math.ceil(per_rank * args.epochs / args.grad_accum)
opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / max(1, steps // 20))
                                          * 0.5 * (1 + math.cos(math.pi * min(s, steps) / steps)))
adapter_dir = os.path.join(args.out_dir, f"{os.path.basename(args.model)}_{args.tag}")
best = (-1.0, None)
model.train()
for epoch in range(args.epochs):
    order = items["train"][:]
    random.Random(epoch + 1000 * args.seed).shuffle(order)   # same permutation on every rank, then strided shards
    mine = order[rank::world][:per_rank]
    t0, running, comm_s = time.time(), 0.0, 0.0
    for i, (path, cls) in enumerate(mine):
        enc = encode(path, cls).to(model.device)
        loss = model(**enc).loss / args.grad_accum
        loss.backward()
        running += loss.item() * args.grad_accum
        if (i + 1) % args.grad_accum == 0 or i + 1 == len(mine):
            sync_grads()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
        if (i + 1) % 50 == 0:
            log(f"epoch {epoch + 1} step {i + 1}/{len(mine)} (x{world} GPUs) loss {running / 50:.4f} "
                f"({(time.time() - t0) / (i + 1):.2f} s/img/GPU, GPU peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB)")
            running = 0.0
    train_s = time.time() - t0
    log(f"epoch {epoch + 1} trained {len(mine)} img/GPU in {train_s / 60:.1f} min: {train_s / len(mine):.2f} s/img/GPU, "
        f"of which gradient sync {comm_s:.0f} s ({comm_s / train_s:.0%}), GPU peak "
        f"{torch.cuda.max_memory_allocated() / 2**30:.1f} GiB")
    val = summarize(evaluate(items["val"]), "val")
    log(f"epoch {epoch + 1} val: {val['accuracy']} macro {val['macro_accuracy']} {val['per_class']} "
        f"({(time.time() - t0) / 60:.1f} min)")
    if val["macro_accuracy"] > best[0]:
        best = (val["macro_accuracy"], epoch + 1)
        if main:
            model.save_pretrained(adapter_dir)
            log(f"  saved adapter (best so far) to {adapter_dir}")
    if world > 1:
        dist.barrier()

# score the test split once, with the best-on-validation adapter
if best[1] != args.epochs:
    from peft import set_peft_model_state_dict
    from safetensors.torch import load_file
    set_peft_model_state_dict(model, load_file(os.path.join(adapter_dir, "adapter_model.safetensors")))
test_results = evaluate(items["test"])
if main:
    summary = summarize(test_results, "test")
    summary.update({"best_epoch": best[1], "val_macro": best[0], "quant": "4bit" if (quant or prequantized) else "none",
                    "lora_r": args.lora_r, "lr": args.lr, "epochs": args.epochs, "max_side": args.max_side,
                    "train_n": len(items["train"]), "gpus": world, "effective_batch": args.grad_accum * world})
    out = f"evaluation/finetuned_{os.path.basename(args.model)}_{args.tag}_test.json"
    json.dump({"summary": summary, "prompt": PROMPT, "results": test_results}, open(out, "w"), indent=2)
    print(json.dumps(summary, indent=1), flush=True)
if world > 1:
    dist.destroy_process_group()
