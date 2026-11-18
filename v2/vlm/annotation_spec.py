"""Shared spec for the photo annotations (written by cluster/vlm/annotate_vllm.py, learned by
cluster/vlm/train_vlm_reports.py): the 7 conditions, the annotator instructions, and the JSON schema. Label-conditioned
on purpose: the annotator sees the dataset label but must keep what is visible apart from what the label predicts,
and say when a photo does not show the labeled condition."""
CLASSES = {  # identical to cluster/vlm/zeroshot_qwen3vl.py
    "healthy": "no stress or disease; uniformly green, turgid foliage and normal fruit/flowers",
    "drought": "water deficit; wilting, dry, curled or crispy leaf edges, dry cracked soil",
    "overwatered": "excess water; limp or yellowing leaves, waterlogged or puddled soil, no fungal growth",
    "root_rot": "Phytophthora/Pythium root and crown rot; dark, decayed roots or crown, collapsing plant",
    "frost": "cold injury; blackened flower centers, bronzed or water-soaked damaged tissue",
    "gray_mold": "Botrytis cinerea; gray, fuzzy, dusty mold on fruit or flowers",
    "white_mold": "Sclerotinia; white, cottony mycelium, sometimes black sclerotia, on fruit, stems or crown",
}
ZS_PROMPT = (  # identical to cluster/vlm/zeroshot_qwen3vl.py
    "You are a plant pathologist. Classify the main problem with the strawberry plant in this image into "
    "exactly one of these categories:\n"
    + "\n".join(f"- {k}: {v}" for k, v in CLASSES.items())
    + "\nReply with only the category name."
)
SYSTEM = (
    "You are an expert strawberry plant pathologist writing training annotations for an AI that diagnoses "
    "strawberry plant problems from photos. Accuracy and honesty matter more than completeness: the model trained "
    "on your annotations must learn to state only what a photo shows and to say clearly when something cannot be "
    "seen or confirmed.\n\n"
    "Each photo comes with its dataset label. Labels were assigned by researchers and are usually right, but some "
    "are wrong, and some photos show the condition only weakly.\n\n"
    "Rules:\n"
    "1. Describe only what is actually visible. Never state a symptom because the label implies it.\n"
    "2. Expected signs of the labeled condition that are not visible go in missing_expected_signs, not in the "
    "findings.\n"
    "3. If the photo looks more like another condition, or like a mix, say so in label_assessment and "
    "differential.\n"
    "4. Use plain, precise language a grower understands: colors, textures, where on the plant.\n"
    "5. plant_parts_visible lists only parts you can actually see.\n"
    "6. observations: concrete visible findings, one per item, with the part and how certain you are.\n"
    "7. caption: 2-4 factual sentences describing the photo and the main problem, with confidence matching the "
    "evidence ('consistent with', 'likely', 'cannot be confirmed from this photo').\n"
    "8. report: 3-6 sentences for a grower: the diagnosis, the key visible evidence, what is uncertain or not "
    "visible, and one practical next check to confirm it.\n"
    "9. qa_pairs: 4 short question/answer pairs a grower might ask about this photo, all true for this photo. One "
    "asks about the main visible problem; at least one asks about a sign that is NOT present or not visible (the "
    "correct answer says so).\n"
    "10. If the image is not a real photo (illustration, collage, heavy text overlay, screenshot), say so in "
    "image_quality.\n\n"
    "Conditions:\n" + "\n".join(f"- {k}: {v}" for k, v in CLASSES.items()) + "\n"
    "Key distinctions: gray mold (Botrytis) is gray-brown, dusty or fuzzy sporulation, mostly on fruit and "
    "flowers, with soft brown rotting fruit. White mold (Sclerotinia) is bright white cottony mycelium, often with "
    "hard black sclerotia, usually at the crown, on stems or on fruit touching the soil. Drought gives dry, crispy, "
    "curled leaves and dry soil; overwatering gives limp or yellowing leaves with wet or waterlogged soil and no "
    "fungal growth; root rot gives dark decayed roots or crown and a collapsing plant; frost gives blackened flower "
    "centers and bronzed or water-soaked tissue."
)
SETTINGS = ["field", "greenhouse_or_tunnel", "pot_or_container", "indoor", "lab_or_cut_sample",
            "illustration_or_composite", "unknown"]
PARTS = ["leaf", "stem_or_petiole", "crown", "root", "flower", "unripe_fruit", "ripe_fruit", "runner",
         "soil_or_substrate"]
CONSISTENCY = ["clearly_supports", "partially_supports", "does_not_support", "cannot_tell"]
COND = list(CLASSES) + ["other"]
SCHEMA = {   # array sizes are bounded: greedy decoding otherwise loops ("leaf", "leaf", ...) until max_tokens
    "type": "object", "additionalProperties": False,
    "required": ["image_quality", "setting", "plant_parts_visible", "observations", "label_assessment",
                 "differential", "caption", "report", "qa_pairs", "visual_confidence"],
    "properties": {
        "image_quality": {"type": "object", "additionalProperties": False, "required": ["usable", "issues"],
                          "properties": {"usable": {"type": "boolean"},
                                         "issues": {"type": "array", "maxItems": 5, "items": {"type": "string"}}}},
        "setting": {"type": "string", "enum": SETTINGS},
        "plant_parts_visible": {"type": "array", "maxItems": len(PARTS), "items": {"type": "string", "enum": PARTS}},
        "observations": {"type": "array", "minItems": 1, "maxItems": 8, "items": {
            "type": "object", "additionalProperties": False, "required": ["part", "finding", "certainty"],
            "properties": {"part": {"type": "string"}, "finding": {"type": "string"},
                           "certainty": {"type": "string", "enum": ["clear", "probable", "uncertain"]}}}},
        "label_assessment": {
            "type": "object", "additionalProperties": False,
            "required": ["consistency", "supporting_evidence", "missing_expected_signs", "conflicting_evidence"],
            "properties": {
                "consistency": {"type": "string", "enum": CONSISTENCY},
                "supporting_evidence": {"type": "array", "maxItems": 6, "items": {"type": "string"}},
                "missing_expected_signs": {"type": "array", "maxItems": 6, "items": {"type": "string"}},
                "conflicting_evidence": {"type": "array", "maxItems": 6, "items": {"type": "string"}}}},
        "differential": {"type": "array", "minItems": 1, "maxItems": 4, "items": {
            "type": "object", "additionalProperties": False, "required": ["condition", "reasoning"],
            "properties": {"condition": {"type": "string", "enum": COND}, "reasoning": {"type": "string"}}}},
        "caption": {"type": "string"},
        "report": {"type": "string"},
        "qa_pairs": {"type": "array", "minItems": 3, "maxItems": 6, "items": {
            "type": "object", "additionalProperties": False, "required": ["question", "answer"],
            "properties": {"question": {"type": "string"}, "answer": {"type": "string"}}}},
        "visual_confidence": {"type": "string", "enum": ["high", "medium", "low"]},
    },
}


def bounded_schema():
    """SCHEMA plus maxLength on every free-text string, for open models decoding greedily under a grammar (a
    runaway string otherwise eats the token budget). Kept out of SCHEMA so the OpenAI strict schema stays plain."""
    import copy
    s = copy.deepcopy(SCHEMA)
    limits = {"caption": 700, "report": 1400}

    def walk(node, key=None):
        if node.get("type") == "string" and "enum" not in node:
            node["maxLength"] = limits.get(key, 400)
        for k, v in node.get("properties", {}).items():
            walk(v, k)
        if isinstance(node.get("items"), dict):
            walk(node["items"], key)
    walk(s)
    return s
# For models without schema-constrained decoding: the same structure spelled out as a template.
TEMPLATE = """{
  "image_quality": {"usable": true, "issues": []},
  "setting": "one of: %s",
  "plant_parts_visible": ["any of: %s"],
  "observations": [{"part": "...", "finding": "...", "certainty": "clear | probable | uncertain"}],
  "label_assessment": {"consistency": "one of: %s", "supporting_evidence": ["..."],
                       "missing_expected_signs": ["..."], "conflicting_evidence": ["..."]},
  "differential": [{"condition": "one of: %s", "reasoning": "..."}],
  "caption": "...",
  "report": "...",
  "qa_pairs": [{"question": "...", "answer": "..."}],
  "visual_confidence": "high | medium | low"
}""" % (", ".join(SETTINGS), ", ".join(PARTS), ", ".join(CONSISTENCY), ", ".join(COND))


def user_text(label):
    return f"Annotate this photo. Dataset label: {label} ({CLASSES[label]})."


def validate(ann):
    """Problems with an annotation dict (empty list = valid)."""
    errs = [f"missing {k}" for k in SCHEMA["required"] if k not in ann]
    if ann.get("setting") not in SETTINGS:
        errs.append(f"bad setting {ann.get('setting')!r}")
    errs += [f"bad part {p!r}" for p in ann.get("plant_parts_visible", []) if p not in PARTS]
    if (ann.get("label_assessment") or {}).get("consistency") not in CONSISTENCY:
        errs.append("bad label_assessment.consistency")
    if ann.get("visual_confidence") not in ("high", "medium", "low"):
        errs.append("bad visual_confidence")
    if len(ann.get("qa_pairs", [])) < 3:
        errs.append("fewer than 3 qa_pairs")
    return errs
