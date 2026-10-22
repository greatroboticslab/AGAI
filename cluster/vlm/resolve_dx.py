"""Resolving disagreements between two zero-shot diagnosers (photos where their greedy answers in
cluster/vlm/consensus_dx.py differ). Methods, each run by this model on the photos where it still disagrees with --peer:
  choose      - "two pathologists disagree: one says X, the other Y; look again and decide" (control for narrowing
                the choice to two candidates)
  refs_real   - choose, plus reference photos of both candidates from the TRAIN split (real)
  refs_synth  - choose, plus reference images of both candidates from the lab's GPT-generated synthetic set
                (Hongbo: "feed each model a synthetic image of its predicted disease to check against")
  debateR     - round R of a debate (Hongbo + Gus: feed the other model's diagnosis and evidence back, force a
                re-think): this model sees its own and the peer's round R-1 answer with evidence and answers again;
                round 0 = the greedy consensus answers. Photos where the two agreed in round R-1 are settled.
Output: /projects/$USER/agai/consensus/<run>/resolve/<model>.<method>.jsonl. Resumable.
  python cluster/vlm/resolve_dx.py --model /scratch/$USER/models/Qwen3-VL-32B-Instruct --name Qwen3-VL-32B-Instruct \
      --peer Qwen3.8-27B --methods choose,refs_real,refs_synth,debate1
"""
import argparse
import base64
import glob
import io
import json
import os
import random
import sys
import time

from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from annotation_spec import CLASSES  # noqa: E402

CATS = "\n".join(f"- {k}: {v}" for k, v in CLASSES.items())
SCHEMA = {"type": "object", "additionalProperties": False, "required": ["evidence", "diagnosis"],
          "properties": {"evidence": {"type": "string", "maxLength": 200},
                         "diagnosis": {"type": "string", "enum": list(CLASSES)}}}
INTRO = "You are a plant pathologist. Categories:\n" + CATS + "\n\n"
CHOOSE = ("Two plant pathologists disagree about the strawberry photo below: one diagnosed {a}, the other {b}. Look at "
          "the photo carefully and decide which diagnosis is right (or another category if neither fits). First name "
          "the key visible evidence in one short sentence, then the category. Reply in JSON.")
REFS = ("Two plant pathologists disagree about the strawberry photo below (the LAST image): one diagnosed {a}, the "
        "other {b}. The images before it are reference examples: {refs}. Compare the photo with the references, "
        "then decide which diagnosis is right (or another category if neither fits). First name the key visible "
        "evidence in one short sentence, then the category. Reply in JSON.")
DEBATE = ("Look at the strawberry photo below. Earlier you diagnosed {own} (your evidence: \"{own_ev}\"). Another "
          "plant pathologist diagnosed {peer} (their evidence: \"{peer_ev}\"). Look at the photo again carefully; "
          "keep your diagnosis or change it, whichever the photo supports. First name the key visible evidence in "
          "one short sentence, then the category. Reply in JSON.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--name", default="")
    ap.add_argument("--peer", required=True, help="the other model's name in consensus/<run>/")
    ap.add_argument("--methods", default="choose,refs_real,refs_synth,debate1")
    ap.add_argument("--run", default="v1")
    ap.add_argument("--split-json", default="evaluation/split_v1.json")
    ap.add_argument("--synth-glob", default="plant_diagnostic/data/train_aug/{c}/*synth*GPT*")
    ap.add_argument("--refs-per-class", type=int, default=2)
    ap.add_argument("--max-side", type=int, default=1024)
    ap.add_argument("--ref-side", type=int, default=512)
    ap.add_argument("--pp", type=int, default=1)
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--gpu-mem", type=float, default=0.90)
    ap.add_argument("--dtype", default="auto")
    ap.add_argument("--enforce-eager", action="store_true")
    args = ap.parse_args()

    name = args.name or os.path.basename(args.model.rstrip("/"))
    base = f"/projects/{os.environ['USER']}/agai/consensus/{args.run}"
    out_dir = os.path.join(base, "resolve")
    os.makedirs(out_dir, exist_ok=True)

    def greedy(model):
        return {r["image"]: r for r in map(json.loads, open(f"{base}/{model}.jsonl"))
                if r.get("sample") == 0 and "diagnosis" in r}

    def answers(model, rnd):
        if rnd == 0:
            return greedy(model)
        p = f"{out_dir}/{model}.debate{rnd}.jsonl"
        prev = answers(model, rnd - 1)   # settled photos keep their earlier answer
        prev.update({r["image"]: r for r in map(json.loads, open(p)) if "diagnosis" in r} if os.path.exists(p) else {})
        return prev

    split = json.load(open(args.split_json))
    train = split["train"]
    synth = {c: sorted(glob.glob(args.synth_glob.format(c=c))) for c in CLASSES}
    g_own, g_peer = greedy(name), greedy(args.peer)
    disagree = sorted(p for p in g_own if p in g_peer and g_own[p]["diagnosis"] != g_peer[p]["diagnosis"])
    print(f"{name} vs {args.peer}: {len(disagree)} photos with different greedy answers; synthetic refs per class "
          f"{ {c: len(v) for c, v in synth.items()} }", flush=True)

    def refs(photo, c, kind):
        pool = sorted(train[c]) if kind == "real" else synth[c]
        return random.Random(f"ref:{photo}:{c}:{kind}").sample(pool, min(args.refs_per_class, len(pool)))

    def jobs(method):
        if method.startswith("debate"):
            rnd = int(method[6:])
            own, peer = answers(name, rnd - 1), answers(args.peer, rnd - 1)
            todo = [p for p in disagree if own[p]["diagnosis"] != peer[p]["diagnosis"]]
            return [(p, DEBATE.format(own=own[p]["diagnosis"], own_ev=own[p]["evidence"], peer=peer[p]["diagnosis"],
                                      peer_ev=peer[p]["evidence"]), [], {"own_prev": own[p]["diagnosis"],
                                                                         "peer_prev": peer[p]["diagnosis"]})
                    for p in todo]
        out = []
        for p in disagree:
            a, b = sorted([g_own[p]["diagnosis"], g_peer[p]["diagnosis"]],
                          key=lambda c: random.Random(f"order:{p}:{c}").random())   # random order per photo
            if method == "choose":
                out.append((p, CHOOSE.format(a=a, b=b), [], {}))
                continue
            kind = method.split("_")[1]
            ra, rb = refs(p, a, kind), refs(p, b, kind)
            if not ra or not rb:
                continue
            desc = f"the first {len(ra)} show {a}, the next {len(rb)} show {b}"
            out.append((p, REFS.format(a=a, b=b, refs=desc), ra + rb, {"refs": ra + rb}))
        return out

    from vllm import LLM, SamplingParams
    from vllm.sampling_params import StructuredOutputsParams

    kw = dict(model=args.model, max_model_len=args.max_model_len, gpu_memory_utilization=args.gpu_mem,
              limit_mm_per_prompt={"image": 1 + 2 * args.refs_per_class}, dtype=args.dtype, seed=0,
              trust_remote_code=True, tensor_parallel_size=args.tp, pipeline_parallel_size=args.pp)
    if args.pp * args.tp > 1 and os.environ.get("RAY_ADDRESS"):
        kw["distributed_executor_backend"] = "ray"
    if args.enforce_eager:
        kw["enforce_eager"] = True
    t0 = time.time()
    llm = LLM(**kw)
    print(f"loaded in {time.time() - t0:.0f}s", flush=True)
    sp = SamplingParams(temperature=0.0, max_tokens=120, structured_outputs=StructuredOutputsParams(json=SCHEMA))

    def url(path, side):
        img = Image.open(path).convert("RGB")
        img.thumbnail((side, side))
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=92)
        return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()

    label = {p: c for c, ps in split["test"].items() for p in ps}
    label.update({p: c for c, ps in split["val"].items() for p in ps})
    for method in args.methods.split(","):
        out_path = f"{out_dir}/{name}.{method}.jsonl"
        done = {r["image"] for r in map(json.loads, open(out_path)) if "diagnosis" in r} if os.path.exists(out_path) else set()
        todo = [j for j in jobs(method) if j[0] not in done]
        print(f"[{method}] {len(todo)} photos -> {out_path}", flush=True)
        if not todo:
            continue
        msgs = [[{"role": "user", "content": [{"type": "image_url", "image_url": {"url": url(r, args.ref_side)}} for r in rr]
                  + [{"type": "image_url", "image_url": {"url": url(p, args.max_side)}},
                     {"type": "text", "text": INTRO + text}]}] for p, text, rr, _ in todo]
        t0 = time.time()
        outs = llm.chat(msgs, sp, use_tqdm=False, chat_template_kwargs={"enable_thinking": False})
        with open(out_path, "a") as f:
            for (p, _, _, extra), o in zip(todo, outs):
                rec = {"image": p, "label": label.get(p), "model": name, "peer": args.peer, "method": method, **extra}
                try:
                    rec.update(json.loads(o.outputs[0].text))
                except Exception as e:
                    rec.update({"error": f"{type(e).__name__}: {e}", "raw": o.outputs[0].text[:500]})
                f.write(json.dumps(rec) + "\n")
        print(f"[{method}] {len(todo)} in {time.time() - t0:.0f}s", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
