"""
Hallucination Evaluation Framework

A comprehensive framework for measuring hallucination rates in the 
MiniGPT-v2 plant diagnostic system.

Usage:
    from hallucination_evaluation import HallucinationChecker, compute_summary
    
    checker = HallucinationChecker()
    result = checker.check_response(image_path, predicted_label, generated_text)
    
See README.md for full documentation.
"""

from .hallucination_checker import (
    HallucinationChecker,
    HallucinationResult,
    EvaluationSummary,
    compute_summary
)
from .config import (
    UNIQUE_INDICATORS,
    DISEASE_CLASSES,
    PLANT_REGIONS
)

__all__ = [
    "HallucinationChecker",
    "HallucinationResult", 
    "EvaluationSummary",
    "compute_summary",
    "UNIQUE_INDICATORS",
    "DISEASE_CLASSES",
    "PLANT_REGIONS"
]

__version__ = "1.0.0"

