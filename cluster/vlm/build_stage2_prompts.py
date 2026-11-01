"""Give the report writer's training records the inference prompt (manuscript 2.7: the April training instruction was
empty while inference supplies a long system prompt). For every record the human turn becomes the v1 application's
system prompt and user text (demo_v5.py), built from the record's condition and the parts the v1 detector found
(cluster/detect_parts.py), with the no-parts variant when nothing was detected. With --no-parts the prompt without
part lists is used for every record (the detector on/off ablation).
  python cluster/build_stage2_prompts.py --parts evaluation/parts_stage2_trainval.json \
      --records plant_diagnostic/datasets/split_v1/stage2_train.json --out plant_diagnostic/datasets/split_v1/stage2_train_prompted.json
"""
import argparse
import json

PLANT_PARTS = {"flower", "fruit", "leaf", "root", "soil", "stem"}
CANON = {"healthy": "healthy", "overwatered": "overwatering", "root_rot": "root rot", "drought": "drought",
         "frost": "frost injury", "gray_mold": "gray mold", "white_mold": "white mold"}


def prompt(label, visible):
    """(system block, user text) exactly as demo_v5.py builds them for an uploaded photograph"""
    title = label.title()
    if visible:
        visible_csv = ", ".join(sorted(visible))
        hidden_csv = ", ".join(sorted(PLANT_PARTS - set(visible)))
        system = f"""<<SYS>>You are a plant diagnostician. The diagnosis has already been determined: {title}
A vision-based detector has confirmed which plant parts are visible in this image.
VISIBLE (describe these): {visible_csv}.
NOT VISIBLE (do NOT mention these at all): {hidden_csv}.
You must ONLY describe symptoms on the VISIBLE parts listed above.
Do NOT name, reference, or explain anything involving {hidden_csv} — not even as causes, locations, or in advice.
The green sepals on top of a strawberry are the calyx, part of the fruit, not leaves.
Provide a detailed medical report:
1) Diagnosis: {title}
2) Visible cues: Describe the visual symptoms on the detected parts.
3) Recommendation: Provide specific, actionable treatment steps.
Be detailed and thorough.<</SYS>>"""
        ask = (f"Analyze this strawberry plant image. The diagnosis is {label}. "
               f"VISIBLE parts: {visible_csv}. NOT VISIBLE (do not mention): {hidden_csv}. "
               f"Describe symptoms ONLY on the visible parts.")
    else:
        system = f"""<<SYS>>You are a plant diagnostician. The diagnosis has already been determined: {title}
Your task is to examine the image and explain why this diagnosis is correct.
You MUST use this exact diagnosis: {title}
Provide a detailed medical report in this format:
1) Diagnosis: {title}
2) Visible cues: Describe the visual symptoms you observe that support this diagnosis.
3) Recommendation: Provide specific, actionable treatment steps.
Be detailed and thorough. Complete all recommendations fully.
<</SYS>>"""
        ask = f"Examine this image. The diagnosis is {title}. Describe the visible symptoms and provide treatment recommendations."
    return system, ask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", required=True)
    ap.add_argument("--parts", default="", help="detect_parts.py output; required unless --no-parts")
    ap.add_argument("--no-parts", action="store_true")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    parts = json.load(open(args.parts)) if args.parts else {}
    out, missing, with_parts = [], 0, 0
    for r in json.load(open(args.records, encoding="utf-8")):
        cls = r["image"].split("/")[-2]
        label = CANON[cls]
        visible = [] if args.no_parts else parts.get(r["image"], {}).get("visible")
        if visible is None:
            missing += 1
            visible = []
        with_parts += bool(visible)
        system, ask = prompt(label, visible)
        conv = [{"from": "human", "value": f"<Img><ImageHere></Img> {system}\n{ask}"}] + r["conversations"][1:]
        out.append(dict(r, conversations=conv, visible_parts=visible, label=label))
    json.dump(out, open(args.out, "w", encoding="utf-8"), indent=1)
    print(f"{len(out)} records -> {args.out}; with part lists {with_parts}; records without a detection entry {missing}")


if __name__ == "__main__":
    main()
