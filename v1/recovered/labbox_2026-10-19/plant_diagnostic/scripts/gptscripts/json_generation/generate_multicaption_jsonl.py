#!/usr/bin/env python3
"""
Multi-caption JSONL generator for MiniGPT-4 stage-2 fine-tuning.

For each image, emits N variants sampled from a pool of conversation styles
(clinical, quick triage, treatment-focused, differential, severity, layperson).
Each variant is a distinct JSONL record so the VLM trainer treats them as
independent examples — teaching the model to respond to varied user phrasings
rather than always producing the same rigid paragraph.

Resumable: keys on (image_path, variant_id), so reruns skip already-generated
variants instead of regenerating duplicates.

Usage:
  python generate_multicaption_jsonl.py \
    --img-root data/train_aug \
    --out      datasets/strawberry_v4_multicaption.jsonl \
    --variants 3 \
    --model    o3 \
    --sleep    0.6
"""
from __future__ import annotations
import argparse, base64, json, logging, os, random, sys, time
from pathlib import Path
from typing import Optional, Set, Tuple

VALID_LABELS = {"drought", "frost", "healthy", "overwatered",
                "root_rot", "gray_mold", "white_mold"}

# Pairs of (label, plausible-confusable) for differential prompting.
DIFFERENTIALS = {
    "drought":     ["overwatered", "frost", "root_rot"],
    "overwatered": ["drought", "root_rot", "healthy"],
    "frost":       ["drought", "gray_mold"],
    "root_rot":    ["overwatered", "drought"],
    "gray_mold":   ["white_mold", "frost"],
    "white_mold":  ["gray_mold", "overwatered"],
    "healthy":     ["drought", "overwatered"],
}

# ---------------------------------------------------------------------------
# Variant pool. Each variant defines:
#   id:         stable string used for dedup keying
#   user_msg:   list of phrasings — one is sampled per call (linguistic variety)
#   system:     style guidance for that variant
#   require_label_prefix: if True, response must start with 'Diagnosis: <label>.'
# ---------------------------------------------------------------------------

VARIANT_POOL = [
    # ---- 1. Clinical (legacy style — keep so old + new captions stay compatible) ----
    {
        "id": "clinical_full",
        "user_msgs": [
            "<Img><ImageHere></Img>",
            "<Img><ImageHere></Img> Please analyze this strawberry plant.",
            "<Img><ImageHere></Img> What is the diagnosis?",
        ],
        "system": (
            "You are a strawberry plant diagnostician.\n"
            "Rules:\n"
            "1) The label is PRE-SET. Use it exactly.\n"
            "2) Base everything on what is VISIBLE.\n"
            "3) Single paragraph, ~6–9 sentences. Start with 'Diagnosis: <label>.' then "
            "(a) 3–6 visual clues, (b) likely cause, (c) 2–3 actionable steps with "
            "measurements, (d) 2–3 monitoring checks for next 7 days.\n"
            "4) Return STRICT JSON: {\"report\": \"<text>\"}"
        ),
        "require_label_prefix": True,
    },

    # ---- 2. Quick triage — short, conversational, label-first but no rigid prefix ----
    {
        "id": "quick_triage",
        "user_msgs": [
            "<Img><ImageHere></Img> What's wrong with my strawberry plant?",
            "<Img><ImageHere></Img> Quick — what am I looking at here?",
            "<Img><ImageHere></Img> My plant looks weird. Any idea?",
            "<Img><ImageHere></Img> Help me identify this real quick.",
        ],
        "system": (
            "You are a strawberry plant diagnostician giving a quick, conversational triage.\n"
            "The label is PRE-SET — your answer must reflect it accurately, but do NOT "
            "use a rigid 'Diagnosis:' prefix. Write 2–3 plain sentences: (a) what it is "
            "(call the condition by name, e.g., 'looks like gray mold'), (b) one or two "
            "visible cues that gave it away, (c) the single most useful next action.\n"
            "Tone: natural, like texting a knowledgeable friend.\n"
            "Return STRICT JSON: {\"report\": \"<text>\"}"
        ),
        "require_label_prefix": False,
    },

    # ---- 3. Treatment-focused — assumes the diagnosis, jumps to action ----
    {
        "id": "treatment_focus",
        "user_msgs": [
            "<Img><ImageHere></Img> What should I do?",
            "<Img><ImageHere></Img> What's the fix?",
            "<Img><ImageHere></Img> How do I save this plant?",
            "<Img><ImageHere></Img> Walk me through what to do this week.",
        ],
        "system": (
            "The label is PRE-SET. The user wants action steps, not a description.\n"
            "Open with one sentence naming the condition (e.g., 'This is overwatered.'). "
            "Then give a numbered or dash-prefixed action list of 3–5 steps with concrete "
            "measurements (cm/inches, ml/cups, hours/days, frequencies). Close with one "
            "sentence on what to watch for over the next 5–7 days.\n"
            "Prefer cultural controls (irrigation depth/timing, airflow, mulch, drainage, "
            "sanitation) over chemicals.\n"
            "Return STRICT JSON: {\"report\": \"<text>\"}"
        ),
        "require_label_prefix": False,
    },

    # ---- 4. Differential — contrasts label with a plausible confusable ----
    {
        "id": "differential",
        "user_msgs_template": [
            "<Img><ImageHere></Img> Is this {label} or {alt}?",
            "<Img><ImageHere></Img> Could this be {alt} instead of {label}?",
            "<Img><ImageHere></Img> Help me tell {label} from {alt} here.",
        ],
        "system": (
            "The TRUE label is PRE-SET. The user asks whether this could be a confusable "
            "alternative. Confirm the true label and explain how to distinguish it from "
            "the alternative using ONLY visible features. Cover: (a) which features in "
            "the image support the true label, (b) which features would have indicated "
            "the alternative if present, (c) the single most discriminative cue. "
            "3–5 sentences, plain prose.\n"
            "Return STRICT JSON: {\"report\": \"<text>\"}"
        ),
        "require_label_prefix": False,
        "is_differential": True,
    },

    # ---- 5. Severity / urgency — assessment + how soon to act ----
    {
        "id": "severity",
        "user_msgs": [
            "<Img><ImageHere></Img> How bad is this?",
            "<Img><ImageHere></Img> Is this serious? Should I worry?",
            "<Img><ImageHere></Img> How urgent is intervention?",
        ],
        "system": (
            "The label is PRE-SET. Assess severity and urgency from VISIBLE cues only.\n"
            "Open with one sentence naming the condition. Then assign one of: "
            "EARLY / MODERATE / ADVANCED, justified by 2–3 visible features. State the "
            "urgency window (e.g., 'act within 24h', 'within a week', 'monitor only'). "
            "End with the single highest-priority action.\n"
            "4–6 sentences. Return STRICT JSON: {\"report\": \"<text>\"}"
        ),
        "require_label_prefix": False,
    },

    # ---- 6. Layperson — casual, no jargon, encouraging ----
    {
        "id": "layperson",
        "user_msgs": [
            "<Img><ImageHere></Img> I'm new to strawberries. What's going on with this one?",
            "<Img><ImageHere></Img> Total beginner — can you explain in simple terms?",
            "<Img><ImageHere></Img> No idea what I'm doing. Plain English please?",
        ],
        "system": (
            "The label is PRE-SET. The user is a beginner. Avoid jargon (no 'turgor', "
            "'mycelium', 'sporulation', 'crown') — use 'fluffy fungus', 'base of the "
            "plant', 'firmness' etc. 3–5 sentences: name the condition in plain words, "
            "describe what's visible without scary terminology, give one simple thing "
            "to do this week and one to watch for. Encouraging tone, not alarmist.\n"
            "Return STRICT JSON: {\"report\": \"<text>\"}"
        ),
        "require_label_prefix": False,
    },
]

DEFAULT_MODEL = os.getenv("OPENAI_VISION_MODEL", "o3")
MAX_RETRIES = 2
BACKOFF = 1.25


def setup_logger(log_file: Path):
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.FileHandler(log_file, encoding="utf-8"),
                  logging.StreamHandler(sys.stdout)],
    )


def to_data_url(p: Path) -> str:
    mime = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png",
            ".bmp": "image/bmp", ".webp": "image/webp", ".gif": "image/gif",
            ".tif": "image/tiff", ".tiff": "image/tiff"}.get(p.suffix.lower(), "image/jpeg")
    b64 = base64.b64encode(p.read_bytes()).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def supports_temperature(model: str) -> bool:
    return not (model or "").lower().startswith("o3")


def safe_parse_report(txt: str) -> Optional[str]:
    txt = (txt or "").strip()
    if not txt:
        return None
    try:
        obj = json.loads(txt)
    except Exception:
        s, e = txt.find("{"), txt.rfind("}")
        if s >= 0 and e > s:
            try:
                obj = json.loads(txt[s:e + 1])
            except Exception:
                return None
        else:
            return None
    rep = obj.get("report") if isinstance(obj, dict) else None
    return rep.strip() if isinstance(rep, str) and rep.strip() else None


def load_done_keys(jsonl_path: Path) -> Set[Tuple[str, str]]:
    """A 'done' key is (image_path, variant_id)."""
    done: Set[Tuple[str, str]] = set()
    if not jsonl_path.exists():
        return done
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
                img = obj.get("image")
                vid = obj.get("variant_id", "clinical_full")
                if isinstance(img, str):
                    done.add((img, vid))
            except Exception:
                continue
    return done


def iter_images(root: Path):
    for cls_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        label = cls_dir.name
        if label not in VALID_LABELS:
            continue
        for img in sorted(cls_dir.rglob("*")):
            if img.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".gif", ".tif", ".tiff"):
                yield img, label


def render_user_msg(variant: dict, label: str, rng: random.Random) -> str:
    if variant.get("is_differential"):
        alt = rng.choice(DIFFERENTIALS.get(label, ["healthy"]))
        return rng.choice(variant["user_msgs_template"]).format(label=label, alt=alt)
    return rng.choice(variant["user_msgs"])


def call_openai(client, model: str, image: Path, system: str, user_msg: str,
                label: str, temperature: Optional[float]) -> Optional[str]:
    data_url = to_data_url(image)
    label_hint = json.dumps({"label": label,
                             "note": "Use this fixed label for accuracy; visual-only."},
                            ensure_ascii=False)
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            kwargs = {
                "model": model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": [
                        {"type": "text", "text": label_hint + "\n\n" + user_msg},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ]},
                ],
            }
            if temperature is not None and supports_temperature(model):
                kwargs["temperature"] = temperature
            resp = client.chat.completions.create(**kwargs)
            rep = safe_parse_report(resp.choices[0].message.content)
            if rep:
                return rep
            raise ValueError("Model returned no JSON 'report'")
        except Exception as e:
            msg = str(e)
            if "invalid_request_error" in msg or "unsupported_value" in msg or "400" in msg:
                logging.error("Non-retryable for %s: %s", image.name, msg)
                break
            logging.warning("OpenAI call failed (%d/%d) for %s: %s",
                            attempt, MAX_RETRIES, image.name, msg)
            if attempt == MAX_RETRIES:
                break
            time.sleep(BACKOFF ** (attempt - 1))
    return None


def enforce_clinical_prefix(report: str, label: str) -> str:
    prefix = f"Diagnosis: {label}"
    t = report.strip()
    if not t.startswith(prefix):
        report = f"{prefix}. " + t.split("Diagnosis:", 1)[-1].strip()
    return report


def main():
    ap = argparse.ArgumentParser("Multi-caption stage-2 generator for MiniGPT-4")
    ap.add_argument("--img-root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--variants", type=int, default=3,
                    help="N variants per image, sampled without replacement from the pool.")
    ap.add_argument("--model", type=str, default=DEFAULT_MODEL)
    ap.add_argument("--temperature", type=float, default=None,
                    help="Ignored for o3 family.")
    ap.add_argument("--sleep", type=float, default=0.6)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--always-include-clinical", action="store_true",
                    help="Always include the clinical_full variant in every image's variant list.")
    ap.add_argument("--log-file", type=Path,
                    default=Path("logs/generate_multicaption.log"))
    args = ap.parse_args()

    if args.variants < 1 or args.variants > len(VARIANT_POOL):
        sys.exit(f"--variants must be 1..{len(VARIANT_POOL)}")
    if not os.getenv("OPENAI_API_KEY"):
        sys.exit("OPENAI_API_KEY is not set.")
    try:
        from openai import OpenAI
    except Exception:
        sys.exit("pip install openai")

    client = OpenAI()
    setup_logger(args.log_file)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    done = load_done_keys(args.out)
    logging.info("Resuming: %d (image, variant) pairs already in %s", len(done), args.out)

    rng = random.Random(args.seed)
    written = fb = 0

    with args.out.open("a", encoding="utf-8") as out_f:
        for img, label in iter_images(args.img_root):
            img_str = str(img)
            # Per-image deterministic variant selection so reruns are stable.
            img_rng = random.Random(f"{args.seed}|{img_str}")
            pool = list(VARIANT_POOL)
            img_rng.shuffle(pool)
            picked = []
            if args.always_include_clinical:
                clin = next((v for v in pool if v["id"] == "clinical_full"), None)
                if clin:
                    picked.append(clin)
                    pool = [v for v in pool if v["id"] != "clinical_full"]
            for v in pool:
                if len(picked) >= args.variants:
                    break
                picked.append(v)

            for variant in picked:
                vid = variant["id"]
                key = (img_str, vid)
                if key in done:
                    continue
                user_msg = render_user_msg(variant, label, img_rng)
                report = call_openai(client, args.model, img, variant["system"],
                                     user_msg, label, args.temperature)
                if not report:
                    fb += 1
                    logging.warning("FAIL %s [%s] — skipping (no fallback for variants)",
                                    img.name, vid)
                    continue
                if variant.get("require_label_prefix"):
                    report = enforce_clinical_prefix(report, label)
                rec = {
                    "image": img_str,
                    "label": label,
                    "variant_id": vid,
                    "conversations": [
                        {"from": "human", "value": user_msg},
                        {"from": "assistant", "value": report},
                    ],
                }
                out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                out_f.flush()
                written += 1
                logging.info("[OK] %s [%s]", img.relative_to(args.img_root), vid)
                time.sleep(args.sleep)

    logging.info("DONE. Wrote %d records (%d failures) -> %s", written, fb, args.out)


if __name__ == "__main__":
    main()
