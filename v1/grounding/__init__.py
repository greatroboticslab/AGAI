"""
Grounding pipeline for MiniGPT plant diagnostics.

Uses RF-DETR part detections to constrain MiniGPT's prompt —
detected parts are injected as VISIBLE, undetected as NOT VISIBLE.
Post-response analysis filters negations, advice, calyx remaps,
and compound nouns to understand what MiniGPT actually observed.

Modules:
    config     – all tuneable constants (paths, toggles, thresholds)
    detector   – RF-DETR subprocess wrapper
    analyzer   – text analysis with filter tracking
    inference  – MiniGPT model loading, prompt building, generation
    report     – terminal output formatting
"""

from .analyzer import analyze
from .config import GROUND_PARTS, PLANT_PARTS, DISEASE_CLASSES
