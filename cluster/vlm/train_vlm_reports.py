"""LoRA fine-tune of a VLM (Qwen3-VL-8B) to write the structured, honest photo annotations of
cluster/vlm/annotation_spec.py, distilled from a teacher's annotations (cluster/vlm/annotate_vllm.py JSONL files), including
the wrong-label examples, so the student learns to say when a given label is not supported by the photo.

Prompt = exactly the teacher's (system instructions + image + "Dataset label: X"); target = the teacher's JSON,
compact. Answer-only loss, LoRA r=16 on all linear layers, bf16, one GPU. Val loss each epoch picks the adapter.
Then cluster/vlm/merge_lora.py merges it for vLLM, and cluster/vlm/score_honesty.py runs the label-swap test.
  python cluster/vlm/train_vlm_reports.py --model /scratch/$USER/models/Qwen3-VL-8B-Instruct \
      --train /projects/$USER/agai/annotations/v1_Qwen3-VL-32B-Instruct/train_true.jsonl,.../train_swap.jsonl \
      --val /projects/$USER/agai/annotations/v1_Qwen3-VL-32B-Instruct/val_both.jsonl
"""
import argparse
import json
import math
import os
import random
import sys
import time

import torch
from PIL import Image
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForImageTextToText, AutoProcessor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from annotation_spec import SYSTEM, user_text  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--train", required=True, help="comma-separated JSONL files")
ap.add_argument("--val", required=True)
ap.add_argument("--epochs", type=int, default=2)
ap.add_argument("--lr", type=float, default=1e-4)
ap.add_argument("--lora-r", type=int, default=16)
ap.add_argument("--grad-accum", type=int, default=8)
ap.add_argument("--max-side", type=int, default=1024)
ap.add_argument("--val-limit", type=int, default=120)
ap.add_argument("--tag", default="reports_v1")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--keep-teacher-mistakes", action="store_true",
                help="by default only teacher annotations whose verdict matches the truth are kept (see load)")
args = ap.parse_args()
torch.manual_seed(args.seed)
random.seed(args.seed)
ACCEPT = {"clearly_supports", "partially_supports"}


def load(files):
    """Valid teacher annotations. Unless --keep-teacher-mistakes: keep only those whose label verdict is right,
    i.e. accepts its true label or rejects a swapped one (we know which, the teacher does not). The 32B teacher
    gets ~73% of these verdicts right, so plain distillation would teach its sycophancy too."""
    rows, wrong = [], 0
    for f in files.split(","):
        for line in open(f):
            r = json.loads(line)
            if "annotation" not in r or r.get("schema_problems") or r.get("finish_reason", "stop") != "stop":
                continue
            accepts = r["annotation"]["label_assessment"]["consistency"] in ACCEPT
            if not args.keep_teacher_mistakes and accepts == r["label_swapped"]:
                wrong += 1
                continue
            rows.append(r)
    print(f"{files.split('/')[-1]}...: kept {len(rows)}, dropped {wrong} with a wrong verdict", flush=True)
    return rows


train, val = load(args.train), load(args.val)
val = random.Random(1).sample(val, min(args.val_limit, len(val)))
print(f"train {len(train)} ({sum(r['label_swapped'] for r in train)} wrong-label), val {len(val)}", flush=True)

model = AutoModelForImageTextToText.from_pretrained(args.model, dtype=torch.bfloat16, device_map="cuda:0",
                                                    attn_implementation="sdpa")
processor = AutoProcessor.from_pretrained(args.model)
for mod in model.modules():   # same exact Conv3d -> linear speed fix as the other Qwen-VL scripts
    proj = getattr(mod, "proj", None)
    if isinstance(proj, torch.nn.Conv3d) and tuple(proj.kernel_size) == tuple(proj.stride) \
            and all(p == 0 for p in proj.padding):
        def fast(hidden_states, _w=proj.weight, _b=proj.bias):
            x = hidden_states.to(_w.dtype).reshape(-1, _w[0].numel())
            return torch.nn.functional.linear(x, _w.reshape(_w.shape[0], -1), _b).view(-1, _w.shape[0])
        mod.forward = fast
model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
model.enable_input_require_grads()
model = get_peft_model(model, LoraConfig(r=args.lora_r, lora_alpha=2 * args.lora_r, lora_dropout=0.05,
                                         target_modules="all-linear", task_type="CAUSAL_LM"))
model.print_trainable_parameters()
params = [p for p in model.parameters() if p.requires_grad]


def encode(r):
    img = Image.open(r["image"]).convert("RGB")
    img.thumbnail((args.max_side, args.max_side))
    conv = [{"role": "system", "content": [{"type": "text", "text": SYSTEM}]},
            {"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": user_text(r["given_label"])}]}]
    enc = processor.apply_chat_template([conv], tokenize=True, add_generation_prompt=True, return_dict=True,
                                        return_tensors="pt", enable_thinking=False)
    target = json.dumps(r["annotation"], ensure_ascii=False, separators=(",", ":")) + "<|im_end|>\n"
    ans = processor.tokenizer(target, add_special_tokens=False, return_tensors="pt").input_ids
    n = enc["input_ids"].shape[1]
    enc["input_ids"] = torch.cat([enc["input_ids"], ans], dim=1)
    enc["attention_mask"] = torch.ones_like(enc["input_ids"])
    for k in list(enc.keys()):
        if "token_type" in k:
            enc[k] = torch.cat([enc[k], enc[k].new_zeros(*enc[k].shape[:-1], ans.shape[1])], dim=-1)
    enc["labels"] = torch.cat([torch.full((1, n), -100), ans], dim=1)
    return enc


@torch.no_grad()
def val_loss():
    model.eval()
    tot = 0.0
    for r in val:
        tot += model(**encode(r).to(model.device)).loss.item()
    model.train()
    return tot / len(val)


steps = math.ceil(len(train) * args.epochs / args.grad_accum)
opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / max(1, steps // 20))
                                          * 0.5 * (1 + math.cos(math.pi * min(s, steps) / steps)))
out_dir = os.path.join(f"/projects/{os.environ['USER']}/agai/adapters", f"{os.path.basename(args.model.rstrip('/'))}_{args.tag}")
best = (val_loss(), 0)
print(f"val loss before training {best[0]:.4f}", flush=True)
model.train()
for epoch in range(args.epochs):
    order = train[:]
    random.Random(epoch + 1000 * args.seed).shuffle(order)
    t0, running = time.time(), 0.0
    for i, r in enumerate(order):
        loss = model(**encode(r).to(model.device)).loss / args.grad_accum
        loss.backward()
        running += loss.item() * args.grad_accum
        if (i + 1) % args.grad_accum == 0 or i + 1 == len(order):
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
        if (i + 1) % 100 == 0:
            print(f"epoch {epoch + 1} step {i + 1}/{len(order)} loss {running / 100:.4f} "
                  f"({(time.time() - t0) / (i + 1):.2f} s/ex, peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB)", flush=True)
            running = 0.0
    vl = val_loss()
    print(f"epoch {epoch + 1} val loss {vl:.4f} ({(time.time() - t0) / 60:.1f} min)", flush=True)
    if vl < best[0]:
        best = (vl, epoch + 1)
        model.save_pretrained(out_dir)
        processor.save_pretrained(out_dir)
        print(f"  saved adapter to {out_dir}", flush=True)
print(f"best epoch {best[1]} (val loss {best[0]:.4f})", flush=True)
