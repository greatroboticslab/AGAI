"""
Central configuration for the grounding pipeline.

Edit GROUND_PARTS to toggle prompt injection on/off,
or pass --no-ground at the CLI for a one-off override.
"""

from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path("/data/AGAI/MiniGPT-4")
TRAIN_DIR = PROJECT_ROOT / "plant_diagnostic" / "data" / "train"
RFDETR_DETECT_SCRIPT = PROJECT_ROOT / "rfdetr_parts" / "detect_images.py"
RFDETR_CHECKPOINT = PROJECT_ROOT / "rfdetr_parts" / "output" / "checkpoint_best_total.pth"
MINIGPT_EVAL_CONFIG = "eval_configs/minigptv2_eval.yaml"

# ── Conda environments ───────────────────────────────────────────────────────

RFDETR_CONDA_ENV = "rfdetr"

# ── Disease and part definitions ─────────────────────────────────────────────

DISEASE_CLASSES = [
    "drought", "frost", "gray_mold", "healthy",
    "overwatered", "root_rot", "white_mold",
]

PLANT_PARTS = {"flower", "fruit", "leaf", "root", "soil", "stem"}

PART_ALIASES = {
    "leaf":   ["leaf", "leaves", "foliage"],
    "fruit":  ["fruit", "berry", "berries"],
    "flower": ["flower", "flowers", "blossom", "blossoms", "petal", "petals"],
    "stem":   ["stem", "stems", "runner", "runners", "petiole", "petioles"],
    "root":   ["root", "roots"],
    "soil":   ["soil", "mulch", "ground"],
}

# ── Grounding toggle ─────────────────────────────────────────────────────────
# When True, RF-DETR detected parts are injected into MiniGPT's prompt so it
# only describes plant parts that are actually visible.
# Set to False or pass --no-ground to let MiniGPT describe any parts freely.

GROUND_PARTS = True

# ── Calyx remap ──────────────────────────────────────────────────────────────
# For mold diseases the calyx (green sepals on the fruit) is often described
# as "leaf." Sentences that mention leaf + fruit context in these diseases
# are remapped to the fruit category instead.

CALYX_DISEASES = {"white_mold", "gray_mold"}

# ── Reproducibility ──────────────────────────────────────────────────────────

SEED = 42
