"""
Hallucination analysis engine.

No ML dependencies — pure regex + heuristic text analysis.
Takes RF-DETR detected parts and MiniGPT response text, returns a rich
result dict with raw mentions, per-sentence filter decisions, and final
grounding verdicts.
"""

import re

from .config import PLANT_PARTS, PART_ALIASES, CALYX_DISEASES

# ── Precompiled word-boundary patterns per part ──────────────────────────────
# Substring matching ("stem" in "system") causes false positives.
# Use \b word boundaries to match whole words only.

_PART_PATTERNS = {}
for _part, _aliases in PART_ALIASES.items():
    _PART_PATTERNS[_part] = re.compile(
        r'\b(?:' + '|'.join(re.escape(a) for a in _aliases) + r')\b',
        re.IGNORECASE,
    )

# ── Negation filter ──────────────────────────────────────────────────────────
# Sentences that deny a part's presence ("no visible leaf damage",
# "roots are not visible") should not count as hallucination triggers.

_NEGATION_RE = re.compile(
    r"("
    r"no\s+visible|not\s+visible|"
    r"no\s.{0,60}(visible|present)|"
    r"not\s+detected|cannot\s+see|"
    r"(is|are)\s+not\s+(visible|present|shown|seen)|"
    r"isn't\s+visible|"
    r"without\s.{0,20}(visible|symptom)|"
    r"absence\s+of\b|lack\s+of\b"
    r")",
    re.IGNORECASE,
)

# ── Advice / treatment filter ────────────────────────────────────────────────
# Sentences giving care recommendations ("irrigate early morning",
# "trim off petioles") mention parts incidentally, not as observations.

_ADVICE_RE = re.compile(
    r"("
    r"to\s+(address|manage|remedy|reduce|prevent|improve|alleviate)|"
    r"immediately\s|over\s+the\s+next\s+\d|"
    r"apply\s+a|irrigat[ei]|watering\s|"
    r"remove\s+and\s+discard|discard\s+all|"
    r"trim\s.{0,12}(off|away|back)|prune\b|"
    r"sanitize|ensure\s.{0,20}(dry|drainage)|"
    r"keep\s.{0,20}(off|dry|moist)|"
    r"improve\s+(airflow|drainage)|"
    r"replant|"
    r"adjust\s+(irrigation|watering|drainage)|"
    r"over\s+time,\s+check|check\s+daily|inspect\s+daily|"
    r"look\s+for\s|monitor\b|watch\s+for"
    r")",
    re.IGNORECASE,
)

# ── Fruit-context pattern (for calyx remap) ─────────────────────────────────

_FRUIT_CONTEXT_RE = re.compile(
    r"(fruit|berry|berries|calyx|receptacle)", re.IGNORECASE,
)

# ── Compound-noun filter ─────────────────────────────────────────────────────
# Only strip part words that are unambiguously NOT visual observations:
# disease names ("root rot") and compound adjectives ("above-ground").

_NON_OBSERVATIONAL_RE = re.compile(
    r"("
    r"root[\s-]*rot"
    r"|above[\s-]*ground"
    r")",
    re.IGNORECASE,
)


def _has_standalone_mention(sentence, part):
    """True if the part word appears outside of known compound nouns."""
    cleaned = _NON_OBSERVATIONAL_RE.sub("___", sentence)
    return bool(_PART_PATTERNS[part].search(cleaned))


# ── Sentence splitting ───────────────────────────────────────────────────────

_DECIMAL_PLACEHOLDER = "\u2024"  # one-dot leader, never in natural text


def _split_sentences(text):
    """Split text on sentence boundaries, preserving decimal numbers."""
    protected = re.sub(r'(\d)\.(\d)', r'\1' + _DECIMAL_PLACEHOLDER + r'\2', text)
    parts = re.split(r'[.!?\n]', protected)
    return [s.strip().replace(_DECIMAL_PLACEHOLDER, '.') for s in parts if s.strip()]


# ── Per-sentence classification ──────────────────────────────────────────────

def classify_mention(sentence, part, disease=""):
    """Decide how a plant-part mention should be treated.

    Returns:
        (category, detail) where category is one of:
          "assertive"   – claims the part is visible (counts)
          "negation"    – denies the part's presence (filtered)
          "advice"      – treatment / recommendation context (filtered)
          "calyx_remap" – leaf reference that is actually the calyx → fruit
          "compound"    – part word only appears in a compound noun (filtered)
    """
    low = sentence.lower()

    if _NEGATION_RE.search(low):
        return "negation", "sentence negates or denies part presence"

    if _ADVICE_RE.search(low):
        return "advice", "sentence is treatment / care advice"

    if (part == "leaf"
            and disease in CALYX_DISEASES
            and _FRUIT_CONTEXT_RE.search(low)):
        return "calyx_remap", "leaf + fruit context in mold disease → calyx (fruit)"

    if not _has_standalone_mention(sentence, part):
        return "compound", "part only appears in compound noun (not a direct observation)"

    return "assertive", "direct observation claim"


# ── Main analysis ────────────────────────────────────────────────────────────

def analyze(detected_parts, text, disease=""):
    """Run the full hallucination-analysis pipeline.

    Args:
        detected_parts: dict  {part_name: confidence} from RF-DETR.
        text:           str   MiniGPT response text.
        disease:        str   disease class label (enables disease-specific
                              filters such as the calyx remap for mold).

    Returns:
        dict with keys:
          detected_parts        – sorted list of detected part names
          raw_mentions          – {part: [sentences]} before any filtering
          filter_log            – list of filter decisions (part, sentence,
                                  filter name, action, detail)
          post_filter_mentions  – {part: [sentences]} after filtering
          mentioned_parts       – sorted list from post_filter_mentions
          grounded              – parts both detected and mentioned
          hallucinated          – parts mentioned but NOT detected
          detected_not_mentioned – parts detected but not mentioned
          flags                 – list of hallucination flag dicts
    """
    sentences = _split_sentences(text)

    # ── Step 1: raw keyword matching (no filtering) ──────────────────────
    raw_mentions = {}
    for part in PLANT_PARTS:
        pat = _PART_PATTERNS[part]
        matching = [s for s in sentences if pat.search(s)]
        if matching:
            raw_mentions[part] = matching

    # ── Step 2: classify each mention, build filter log ──────────────────
    filter_log = []
    filtered_mentions = {}

    for part, sents in sorted(raw_mentions.items()):
        for sent in sents:
            category, detail = classify_mention(sent, part, disease)

            if category in ("negation", "advice", "compound"):
                filter_log.append({
                    "part": part, "sentence": sent,
                    "filter": category, "action": "removed",
                    "detail": detail,
                })
            elif category == "calyx_remap":
                filter_log.append({
                    "part": "leaf", "sentence": sent,
                    "filter": "calyx_remap", "action": "remapped → fruit",
                    "detail": detail,
                })
                filtered_mentions.setdefault("fruit", []).append(sent)
            else:
                filtered_mentions.setdefault(part, []).append(sent)

    # ── Step 3: grounding comparison ─────────────────────────────────────
    detected_set = set(detected_parts.keys())
    mentioned_set = set(filtered_mentions.keys())

    grounded = sorted(mentioned_set & detected_set)
    hallucinated = sorted(mentioned_set - detected_set)
    missed = sorted(detected_set - mentioned_set)

    flags = []
    for part in hallucinated:
        flags.append({
            "part": part,
            "reason": f"MiniGPT mentions '{part}' but RF-DETR did not detect it",
            "sentences": filtered_mentions[part],
        })

    return {
        "detected_parts": sorted(detected_set),
        "raw_mentions": dict(sorted(raw_mentions.items())),
        "filter_log": filter_log,
        "post_filter_mentions": dict(sorted(filtered_mentions.items())),
        "mentioned_parts": sorted(mentioned_set),
        "grounded": grounded,
        "hallucinated": hallucinated,
        "detected_not_mentioned": missed,
        "flags": flags,
    }
