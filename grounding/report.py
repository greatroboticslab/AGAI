"""
Terminal reporting for grounding test results.
"""

from pathlib import Path

from .config import PLANT_PARTS

_W = 72

_FILTER_SYMBOLS = {
    "negation":    "NEG",
    "advice":      "ADV",
    "calyx_remap": "CAL",
    "compound":    "CMP",
}


def _trunc(text, maxlen=90):
    return (text[:maxlen] + "...") if len(text) > maxlen else text


def print_header(grounding_on, classes):
    mode = "GROUNDED (RF-DETR parts injected)" if grounding_on else "UNGROUNDED (free-form)"
    print("=" * _W)
    print(f"  RF-DETR + MiniGPT  —  {mode}")
    print(f"  Testing classes: {', '.join(classes)}")
    print("=" * _W)


def print_footer():
    print("=" * _W)
    print("  Test complete.")
    print("=" * _W)


def print_detection(cls, image_path, detected_parts, num_dets, grounding_on):
    print()
    print("=" * _W)
    print(f"  {cls.upper()}  |  {Path(image_path).name}")
    print("=" * _W)

    print(f"\n  [RF-DETR Detection]")
    print(f"    Detections: {num_dets}")
    if detected_parts:
        max_name = max(len(n) for n in detected_parts)
        for name in sorted(detected_parts, key=detected_parts.get, reverse=True):
            bar = "#" * int(detected_parts[name] * 20)
            print(f"    {name:<{max_name}}  {detected_parts[name]:.3f}  {bar}")
    else:
        print("    (none)")

    if grounding_on:
        visible = sorted(detected_parts.keys()) if detected_parts else []
        hidden = sorted(PLANT_PARTS - set(visible))
        print(f"    Injected VISIBLE:     {', '.join(visible) if visible else '(none)'}")
        print(f"    Injected NOT VISIBLE: {', '.join(hidden) if hidden else '(none)'}")


def print_response(text):
    print(f"\n  [MiniGPT Response]")
    print(f"  {'-' * (_W - 4)}")
    for line in text.split("\n"):
        print(f"  {line}")
    print(f"  {'-' * (_W - 4)}")


def print_analysis(result):
    print(f"\n  [Response Analysis]")

    raw = result["raw_mentions"]
    if raw:
        parts_str = ", ".join(f"{p} ({len(s)})" for p, s in sorted(raw.items()))
        print(f"    Raw mentions:  {parts_str}")
    else:
        print(f"    Raw mentions:  (none)")

    log = result["filter_log"]
    if log:
        print(f"\n    Filters applied:")
        for entry in log:
            sym = _FILTER_SYMBOLS.get(entry["filter"], "???")
            snip = _trunc(entry["sentence"])
            if entry["action"] == "removed":
                print(f"      [{sym}] -{entry['part']:<7}  \"{snip}\"")
            else:
                print(f"      [{sym}] {entry['part']:<7} -> fruit   \"{snip}\"")

    post = result["post_filter_mentions"]
    if post:
        parts_str = ", ".join(f"{p} ({len(s)})" for p, s in sorted(post.items()))
        print(f"\n    Observed parts: {parts_str}")
    else:
        print(f"\n    Observed parts: (none)")

    detected = set(result["detected_parts"])
    mentioned = set(result["mentioned_parts"])
    all_parts = sorted(detected | mentioned)

    if all_parts:
        print(f"\n    Grounding:")
        for part in all_parts:
            in_det = part in detected
            in_men = part in mentioned
            if in_det and in_men:
                print(f"      + {part:<8} detected & mentioned")
            elif in_det and not in_men:
                print(f"      . {part:<8} detected, not mentioned")
            elif in_men and not in_det:
                print(f"      ~ {part:<8} mentioned, not detected")


def print_summary(all_results):
    if not all_results:
        return

    print(f"\n{'=' * _W}")
    print(f"  SUMMARY")
    print(f"{'=' * _W}")

    for cls, result in all_results.items():
        detected = sorted(result["detected_parts"])
        mentioned = sorted(result["mentioned_parts"])
        det_str = ", ".join(detected) if detected else "none"
        men_str = ", ".join(mentioned) if mentioned else "none"
        print(f"  {cls:<14} detected: [{det_str}]  mentioned: [{men_str}]")

    print(f"  {'─' * (_W - 4)}")
