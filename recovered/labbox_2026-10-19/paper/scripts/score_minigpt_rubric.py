#!/usr/bin/env python3
"""Automated rubric scoring of the deployed MiniGPT-v2 outputs on the
70-image holdout. No human raters; all signals are derived from the
output text plus the structured artifacts already in the repo.

Metrics
-------
* Diagnosis fidelity: does the report mention the diagnosis it was
  conditioned on (the ResNet top-1 label)?
* Part-grounding violations: number of NOT-VISIBLE parts (parts not in
  the RF-DETR visible set for the image) that are nonetheless
  mentioned as observations. Sentences negating presence ("no visible
  X") are excluded via the existing analyzer's negation filter.
* Symptom coverage: count of disease-knowledge-graph symptom keywords
  hit, normalized by the number of available keywords.
* Off-class disease mentions: count of *other* disease keywords
  mentioned (drift toward incorrect diagnoses).
* Length: token-length proxy.

Inputs:
  paper/generated/minigpt_holdout_outputs.jsonl
  knowledge_graph/disease_knowledge_base.json
Outputs:
  paper/generated/minigpt_rubric_table.tex
  paper/generated/minigpt_rubric_summary.json
"""

from __future__ import annotations
import json, re
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
INPUT_JSONL = REPO_ROOT / "paper" / "generated" / "minigpt_holdout_outputs.jsonl"
KB_PATH = REPO_ROOT / "knowledge_graph" / "disease_knowledge_base.json"
TEX_OUT = REPO_ROOT / "paper" / "generated" / "minigpt_rubric_table.tex"
JSON_OUT = REPO_ROOT / "paper" / "generated" / "minigpt_rubric_summary.json"

PART_PATTERNS = {
    "leaf":   re.compile(r"\b(leaf|leaves|foliage|leaflets?)\b", re.I),
    "fruit":  re.compile(r"\b(fruit|fruits|berry|berries)\b", re.I),
    "flower": re.compile(r"\b(flower|flowers|blossom|blossoms|petals?)\b", re.I),
    "stem":   re.compile(r"\b(stem|stems|runner|runners|petioles?)\b", re.I),
    "root":   re.compile(r"\b(roots?)\b", re.I),
    "soil":   re.compile(r"\b(soil|mulch|growing\s+media|substrate)\b", re.I),
}

NEG_RE = re.compile(
    r"\b("
    r"no\s+visible|not\s+visible|"
    r"no\s.{0,40}(visible|present|shown)|"
    r"not\s+detected|cannot\s+see|"
    r"(is|are)\s+not\s+(visible|present|shown|seen)|"
    r"isn't\s+visible|"
    r"absence\s+of|lack\s+of"
    r")\b",
    re.I,
)

CANON = {
    "drought": "drought",
    "frost": "frost",
    "frost injury": "frost",
    "gray mold": "gray_mold",
    "gray_mold": "gray_mold",
    "white mold": "white_mold",
    "white_mold": "white_mold",
    "healthy": "healthy",
    "overwatered": "overwatered",
    "overwatering": "overwatered",
    "root rot": "root_rot",
    "root_rot": "root_rot",
}

DISEASE_PATTERNS = {
    "drought":     re.compile(r"\bdrought\b", re.I),
    "frost":       re.compile(r"\bfrost(\s+injur(y|ies))?\b", re.I),
    "gray_mold":   re.compile(r"\b(gr[ae]y\s+mold|gr[ae]ymold|botrytis)\b", re.I),
    "white_mold":  re.compile(r"\b(white\s+mold|whitemold|sclerotinia)\b", re.I),
    "healthy":     re.compile(r"\bhealthy\b", re.I),
    "overwatered": re.compile(r"\b(overwater(ed|ing)|over[\-\s]water)\b", re.I),
    "root_rot":    re.compile(r"\broot[\s\-]rot\b", re.I),
}


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def kb_symptom_keywords(kb: dict) -> dict[str, list[re.Pattern]]:
    out: dict[str, list[re.Pattern]] = {}
    for key, entry in kb["diseases"].items():
        canon = CANON.get(key, key)
        kws: set[str] = set()
        for line in entry.get("symptoms", []) + entry.get("visual_indicators", []):
            for tok in re.findall(r"[a-zA-Z][a-zA-Z\-]{4,}", line):
                kws.add(tok.lower())
        # Take a focused subset of distinctive words per disease.
        out[canon] = [re.compile(rf"\b{re.escape(k)}\b", re.I) for k in sorted(kws)]
    return out


def analyze_record(rec: dict, kb_kws: dict) -> dict:
    text = rec.get("response", "") or ""
    truth = rec["truth"]
    diagnosis = rec["diagnosis_used"]
    visible = set(rec.get("rfdetr_visible_parts", []))
    not_visible = set(PART_PATTERNS.keys()) - visible

    sentences = split_sentences(text)

    # Diagnosis-fidelity: does the report mention the diagnosis label?
    diag_pat = DISEASE_PATTERNS.get(diagnosis)
    diagnosis_mentioned = bool(diag_pat and diag_pat.search(text))

    # Off-class mentions: other diseases (besides diagnosis) referenced.
    other_disease_mentions = 0
    for cls, pat in DISEASE_PATTERNS.items():
        if cls == diagnosis:
            continue
        if pat.search(text):
            other_disease_mentions += 1

    # Part-grounding violations: NOT-VISIBLE parts mentioned in
    # non-negated, non-advice sentences.
    violations = 0
    grounding_hits = 0
    for sent in sentences:
        is_neg = bool(NEG_RE.search(sent))
        for part in not_visible:
            if PART_PATTERNS[part].search(sent) and not is_neg:
                violations += 1
                break  # at most one violation per sentence
        for part in visible:
            if PART_PATTERNS[part].search(sent):
                grounding_hits += 1
                break

    # Symptom-coverage: ratio of truth-class keywords hit.
    truth_kws = kb_kws.get(truth, [])
    if truth_kws:
        hits = sum(1 for p in truth_kws if p.search(text))
        symptom_coverage = hits / len(truth_kws)
    else:
        symptom_coverage = float("nan")

    return {
        "image": rec["image"],
        "truth": truth,
        "diagnosis_used": diagnosis,
        "diagnosis_correct": diagnosis == truth,
        "n_sentences": len(sentences),
        "n_chars": len(text),
        "diagnosis_mentioned": diagnosis_mentioned,
        "off_class_mentions": other_disease_mentions,
        "grounding_violations": violations,
        "grounding_hits": grounding_hits,
        "symptom_coverage": symptom_coverage,
        "visible_parts": sorted(visible),
        "latency_s": rec.get("latency_s"),
    }


def main() -> None:
    if not INPUT_JSONL.exists():
        raise SystemExit(f"Missing {INPUT_JSONL}; run run_minigpt_on_holdout.py first")
    kb = json.loads(KB_PATH.read_text())
    kb_kws = kb_symptom_keywords(kb)

    records = [json.loads(line) for line in INPUT_JSONL.read_text().splitlines() if line.strip()]
    scored = [analyze_record(r, kb_kws) for r in records]

    # Aggregate
    n = len(scored)
    diag_correct = sum(1 for s in scored if s["diagnosis_correct"])
    diag_mentioned = sum(1 for s in scored if s["diagnosis_mentioned"])
    avg_violations = sum(s["grounding_violations"] for s in scored) / max(n, 1)
    avg_offclass = sum(s["off_class_mentions"] for s in scored) / max(n, 1)
    avg_coverage = sum(s["symptom_coverage"] for s in scored) / max(n, 1)
    median_chars = sorted(s["n_chars"] for s in scored)[n // 2]
    median_lat = sorted(s["latency_s"] for s in scored)[n // 2] if scored else float("nan")

    # Per-class diagnosis fidelity and grounding rate
    per_class: dict[str, dict] = {}
    for s in scored:
        c = s["truth"]
        d = per_class.setdefault(c, {"n": 0, "diag": 0, "ment": 0, "viol": 0, "cov": 0.0})
        d["n"] += 1
        d["diag"] += int(s["diagnosis_correct"])
        d["ment"] += int(s["diagnosis_mentioned"])
        d["viol"] += s["grounding_violations"]
        d["cov"] += s["symptom_coverage"]

    DISPLAY = {
        "drought": "Drought", "frost": "Frost", "gray_mold": "Gray mold",
        "healthy": "Healthy", "overwatered": "Overwatered",
        "root_rot": "Root rot", "white_mold": "White mold",
    }
    rows = []
    for c in ["drought", "frost", "gray_mold", "healthy", "overwatered", "root_rot", "white_mold"]:
        d = per_class.get(c, {"n": 0, "diag": 0, "ment": 0, "viol": 0, "cov": 0.0})
        if d["n"] == 0: continue
        rows.append(
            f"{DISPLAY[c]} & {d['n']} & "
            f"{100*d['diag']/d['n']:.1f}\\% & "
            f"{100*d['ment']/d['n']:.1f}\\% & "
            f"{d['viol']/d['n']:.2f} & "
            f"{100*d['cov']/d['n']:.1f}\\% \\\\"
        )
    table = (
        "\\begin{tabular}{lrrrrr}\n"
        "\\toprule\n"
        "Class & $n$ & Diagnosis correct (ResNet) & "
        "Report mentions diagnosis & Mean grounding violations & "
        "Symptom coverage \\\\\n"
        "\\midrule\n"
        + "\n".join(rows) + "\n"
        f"\\textbf{{All}} & \\textbf{{{n}}} & \\textbf{{{100*diag_correct/n:.1f}\\%}} & "
        f"\\textbf{{{100*diag_mentioned/n:.1f}\\%}} & "
        f"\\textbf{{{avg_violations:.2f}}} & "
        f"\\textbf{{{100*avg_coverage:.1f}\\%}} \\\\\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
    )
    TEX_OUT.write_text(table)

    summary = {
        "n": n,
        "diagnosis_correct_rate": diag_correct / n,
        "diagnosis_mentioned_rate": diag_mentioned / n,
        "mean_grounding_violations": avg_violations,
        "mean_off_class_mentions": avg_offclass,
        "mean_symptom_coverage": avg_coverage,
        "median_n_chars": median_chars,
        "median_latency_s": median_lat,
    }
    JSON_OUT.write_text(json.dumps(summary, indent=2))
    for k, v in summary.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
