"""
Hallucination Checker Module

Detects two types of hallucinations:
1. Wrong-Disease Hallucination: Mentions symptoms unique to a different disease
2. Visibility Hallucination: Describes plant parts not visible in the image

Uses the knowledge base and training data for ground truth.
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from collections import defaultdict

from config import (
    KNOWLEDGE_BASE_PATH, TRAINING_DATA_PATH, UNIQUE_INDICATORS,
    PLANT_REGIONS, CLAIM_KEYWORDS, LABEL_ALIASES, DISEASE_CLASSES
)


@dataclass
class HallucinationResult:
    """Result of hallucination check for a single response."""
    image_path: str
    ground_truth_label: str
    predicted_label: str
    generated_text: str
    
    # Wrong-disease hallucination
    wrong_disease_mentions: List[str] = field(default_factory=list)
    diseases_confused_with: List[str] = field(default_factory=list)
    has_wrong_disease_hallucination: bool = False
    
    # Visibility hallucination
    invisible_region_claims: List[str] = field(default_factory=list)
    visible_regions_in_image: List[str] = field(default_factory=list)
    has_visibility_hallucination: bool = False
    
    # Grounding metrics
    correct_symptoms_mentioned: List[str] = field(default_factory=list)
    grounding_score: float = 0.0
    
    # Overall
    total_hallucination_score: float = 0.0
    
    def to_dict(self) -> Dict:
        return {
            "image_path": self.image_path,
            "ground_truth_label": self.ground_truth_label,
            "predicted_label": self.predicted_label,
            "generated_text": self.generated_text,
            "wrong_disease_mentions": self.wrong_disease_mentions,
            "diseases_confused_with": self.diseases_confused_with,
            "has_wrong_disease_hallucination": self.has_wrong_disease_hallucination,
            "invisible_region_claims": self.invisible_region_claims,
            "visible_regions_in_image": self.visible_regions_in_image,
            "has_visibility_hallucination": self.has_visibility_hallucination,
            "correct_symptoms_mentioned": self.correct_symptoms_mentioned,
            "grounding_score": self.grounding_score,
            "total_hallucination_score": self.total_hallucination_score
        }


class HallucinationChecker:
    """
    Comprehensive hallucination detection for plant diagnostic responses.
    """
    
    def __init__(self):
        self.knowledge_base = self._load_knowledge_base()
        self.training_data = self._load_training_data()
        self.image_to_description = self._build_image_description_map()
        
    def _load_knowledge_base(self) -> Dict:
        """Load the disease knowledge base."""
        try:
            with open(KNOWLEDGE_BASE_PATH, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Warning: Could not load knowledge base: {e}")
            return {"diseases": {}}
    
    def _load_training_data(self) -> List[Dict]:
        """Load training data for ground truth descriptions."""
        try:
            with open(TRAINING_DATA_PATH, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Warning: Could not load training data: {e}")
            return []
    
    def _build_image_description_map(self) -> Dict[str, str]:
        """
        Build a mapping from image filename to expected description.
        This uses the training data as ground truth for what's in each image.
        """
        mapping = {}
        for entry in self.training_data:
            img_path = entry.get("image", "")
            # Extract filename
            filename = Path(img_path).name
            # Get the assistant response (ground truth)
            for conv in entry.get("conversations", []):
                if conv.get("from") == "assistant":
                    mapping[filename] = conv.get("value", "")
                    break
        return mapping
    
    def _normalize_label(self, label: str) -> str:
        """Normalize disease label to standard form."""
        label = label.lower().strip().replace("-", "_").replace(" ", "_")
        return LABEL_ALIASES.get(label.replace("_", " "), label)
    
    def _extract_disease_from_path(self, image_path: str) -> str:
        """Extract ground truth disease label from image path."""
        path = Path(image_path)
        # Disease is usually in parent directory name
        parent = path.parent.name.lower()
        return self._normalize_label(parent)
    
    def check_wrong_disease_hallucination(
        self, 
        generated_text: str, 
        correct_disease: str
    ) -> Tuple[List[str], List[str], bool]:
        """
        Check if the generated text mentions symptoms unique to other diseases.
        
        Returns:
            - List of wrong-disease mentions found
            - List of diseases confused with
            - Boolean indicating if hallucination detected
        """
        text_lower = generated_text.lower()
        correct_disease = self._normalize_label(correct_disease)
        
        # Negation patterns to skip (avoid flagging "not waterlogged", "no signs of", etc.)
        negation_patterns = [
            "not ", "no ", "without ", "absence of ", "lacks ", "don't ", "doesn't ",
            "isn't ", "aren't ", "never ", "avoid ", "prevent ", "unlike "
        ]
        
        wrong_mentions = []
        confused_diseases = set()
        
        for disease, indicators in UNIQUE_INDICATORS.items():
            if disease == correct_disease:
                continue  # Skip the correct disease
            
            for indicator in indicators:
                indicator_lower = indicator.lower()
                if indicator_lower in text_lower:
                    # Check for negation before the indicator
                    idx = text_lower.find(indicator_lower)
                    # Look at 30 chars before the indicator for negation
                    prefix = text_lower[max(0, idx-30):idx]
                    
                    # Skip if negated
                    is_negated = any(neg in prefix for neg in negation_patterns)
                    if is_negated:
                        continue
                    
                    wrong_mentions.append(f"{indicator} (→{disease})")
                    confused_diseases.add(disease)
        
        return wrong_mentions, list(confused_diseases), len(wrong_mentions) > 0
    
    def check_visibility_hallucination(
        self,
        generated_text: str,
        image_path: str
    ) -> Tuple[List[str], List[str], bool]:
        """
        Check if the generated text describes plant parts not visible in the image.
        
        Uses training data descriptions as ground truth for what's visible.
        
        Returns:
            - List of invisible region claims
            - List of visible regions in the image
            - Boolean indicating if hallucination detected
        """
        filename = Path(image_path).name
        # Also try without augmentation suffix
        base_filename = re.sub(r'_aug\d+', '', filename)
        
        # Get ground truth description
        ground_truth = self.image_to_description.get(
            filename, 
            self.image_to_description.get(base_filename, "")
        )
        
        if not ground_truth:
            # No ground truth available, can't check visibility
            return [], [], False
        
        text_lower = generated_text.lower()
        ground_truth_lower = ground_truth.lower()
        
        # Determine which regions are mentioned in ground truth (visible)
        visible_regions = []
        for region, keywords in PLANT_REGIONS.items():
            for keyword in keywords:
                if keyword in ground_truth_lower:
                    visible_regions.append(region)
                    break
        
        # Check for claims about regions NOT in ground truth
        invisible_claims = []
        for region, keywords in PLANT_REGIONS.items():
            if region in visible_regions:
                continue  # Region is visible, skip
            
            for keyword in keywords:
                # Check if generated text makes a claim about this region
                for claim_word in CLAIM_KEYWORDS:
                    pattern = rf"{claim_word}\s+.{{0,30}}{keyword}"
                    if re.search(pattern, text_lower):
                        claim_context = self._extract_claim_context(text_lower, keyword)
                        invisible_claims.append(f"{region}: '{claim_context}'")
                        break
        
        return invisible_claims, visible_regions, len(invisible_claims) > 0
    
    def _extract_claim_context(self, text: str, keyword: str, window: int = 50) -> str:
        """Extract context around a keyword mention."""
        idx = text.find(keyword)
        if idx == -1:
            return keyword
        start = max(0, idx - window)
        end = min(len(text), idx + len(keyword) + window)
        context = text[start:end]
        return f"...{context}..."
    
    def check_correct_symptoms(
        self,
        generated_text: str,
        correct_disease: str
    ) -> Tuple[List[str], float]:
        """
        Check how many correct symptoms for the disease are mentioned.
        
        Returns:
            - List of correct symptoms mentioned
            - Grounding score (0-1)
        """
        text_lower = generated_text.lower()
        correct_disease = self._normalize_label(correct_disease)
        
        diseases = self.knowledge_base.get("diseases", {})
        disease_info = diseases.get(correct_disease, {})
        
        expected_symptoms = disease_info.get("visual_indicators", [])
        expected_symptoms.extend(disease_info.get("symptoms", [])[:5])
        
        correct_mentions = []
        for symptom in expected_symptoms:
            # Check for key words from the symptom
            words = [w.lower() for w in symptom.split() if len(w) > 4][:3]
            if any(word in text_lower for word in words):
                correct_mentions.append(symptom[:50])
        
        # Calculate grounding score
        if not expected_symptoms:
            grounding_score = 0.5  # Neutral if no expected symptoms
        else:
            grounding_score = len(correct_mentions) / min(len(expected_symptoms), 10)
            grounding_score = min(1.0, grounding_score)
        
        return correct_mentions, grounding_score
    
    def check_response(
        self,
        image_path: str,
        predicted_label: str,
        generated_text: str,
        ground_truth_label: Optional[str] = None
    ) -> HallucinationResult:
        """
        Perform comprehensive hallucination check on a model response.
        
        Args:
            image_path: Path to the image
            predicted_label: The disease label predicted/used by the model
            generated_text: The model's generated explanation
            ground_truth_label: Optional override for ground truth (else extracted from path)
            
        Returns:
            HallucinationResult with all metrics
        """
        # Determine ground truth
        if ground_truth_label:
            gt_label = self._normalize_label(ground_truth_label)
        else:
            gt_label = self._extract_disease_from_path(image_path)
        
        pred_label = self._normalize_label(predicted_label)
        
        # Check wrong-disease hallucination
        wrong_mentions, confused_with, has_wrong_disease = self.check_wrong_disease_hallucination(
            generated_text, pred_label
        )
        
        # Check visibility hallucination
        invisible_claims, visible_regions, has_visibility = self.check_visibility_hallucination(
            generated_text, image_path
        )
        
        # Check correct symptoms
        correct_symptoms, grounding_score = self.check_correct_symptoms(
            generated_text, pred_label
        )
        
        # Calculate total hallucination score
        # 0 = no hallucination, 1 = severe hallucination
        wrong_disease_penalty = min(1.0, len(wrong_mentions) * 0.3)
        visibility_penalty = min(1.0, len(invisible_claims) * 0.2)
        grounding_bonus = (1 - grounding_score) * 0.3  # Lower grounding = higher penalty
        
        total_score = min(1.0, wrong_disease_penalty + visibility_penalty + grounding_bonus)
        
        return HallucinationResult(
            image_path=str(image_path),
            ground_truth_label=gt_label,
            predicted_label=pred_label,
            generated_text=generated_text,
            wrong_disease_mentions=wrong_mentions,
            diseases_confused_with=confused_with,
            has_wrong_disease_hallucination=has_wrong_disease,
            invisible_region_claims=invisible_claims,
            visible_regions_in_image=visible_regions,
            has_visibility_hallucination=has_visibility,
            correct_symptoms_mentioned=correct_symptoms,
            grounding_score=round(grounding_score, 3),
            total_hallucination_score=round(total_score, 3)
        )


@dataclass 
class EvaluationSummary:
    """Summary statistics for hallucination evaluation."""
    total_samples: int = 0
    
    # Wrong-disease hallucination
    wrong_disease_count: int = 0
    wrong_disease_rate: float = 0.0
    most_confused_pairs: List[Tuple[str, str, int]] = field(default_factory=list)
    
    # Visibility hallucination
    visibility_hallucination_count: int = 0
    visibility_hallucination_rate: float = 0.0
    most_common_invisible_claims: List[Tuple[str, int]] = field(default_factory=list)
    
    # Grounding
    avg_grounding_score: float = 0.0
    low_grounding_count: int = 0  # Grounding < 0.3
    
    # Overall
    any_hallucination_count: int = 0
    any_hallucination_rate: float = 0.0
    avg_hallucination_score: float = 0.0
    
    # Per-class breakdown
    per_class_stats: Dict[str, Dict] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            "total_samples": self.total_samples,
            "wrong_disease_hallucination": {
                "count": self.wrong_disease_count,
                "rate": self.wrong_disease_rate,
                "most_confused_pairs": self.most_confused_pairs
            },
            "visibility_hallucination": {
                "count": self.visibility_hallucination_count,
                "rate": self.visibility_hallucination_rate,
                "most_common_claims": self.most_common_invisible_claims
            },
            "grounding": {
                "avg_score": self.avg_grounding_score,
                "low_grounding_count": self.low_grounding_count
            },
            "overall": {
                "any_hallucination_count": self.any_hallucination_count,
                "any_hallucination_rate": self.any_hallucination_rate,
                "avg_hallucination_score": self.avg_hallucination_score
            },
            "per_class": self.per_class_stats
        }


def compute_summary(results: List[HallucinationResult]) -> EvaluationSummary:
    """Compute summary statistics from a list of hallucination results."""
    if not results:
        return EvaluationSummary()
    
    summary = EvaluationSummary()
    summary.total_samples = len(results)
    
    confusion_counts = defaultdict(int)
    invisible_claim_counts = defaultdict(int)
    grounding_scores = []
    hallucination_scores = []
    
    per_class = defaultdict(lambda: {
        "count": 0,
        "wrong_disease": 0,
        "visibility": 0,
        "grounding_sum": 0.0
    })
    
    for result in results:
        # Wrong-disease stats
        if result.has_wrong_disease_hallucination:
            summary.wrong_disease_count += 1
            for confused in result.diseases_confused_with:
                pair = (result.predicted_label, confused)
                confusion_counts[pair] += 1
        
        # Visibility stats
        if result.has_visibility_hallucination:
            summary.visibility_hallucination_count += 1
            for claim in result.invisible_region_claims:
                region = claim.split(":")[0] if ":" in claim else claim
                invisible_claim_counts[region] += 1
        
        # Grounding stats
        grounding_scores.append(result.grounding_score)
        if result.grounding_score < 0.3:
            summary.low_grounding_count += 1
        
        # Overall stats
        if result.has_wrong_disease_hallucination or result.has_visibility_hallucination:
            summary.any_hallucination_count += 1
        hallucination_scores.append(result.total_hallucination_score)
        
        # Per-class stats
        cls = result.ground_truth_label
        per_class[cls]["count"] += 1
        per_class[cls]["grounding_sum"] += result.grounding_score
        if result.has_wrong_disease_hallucination:
            per_class[cls]["wrong_disease"] += 1
        if result.has_visibility_hallucination:
            per_class[cls]["visibility"] += 1
    
    # Compute rates
    n = summary.total_samples
    summary.wrong_disease_rate = round(summary.wrong_disease_count / n, 3)
    summary.visibility_hallucination_rate = round(summary.visibility_hallucination_count / n, 3)
    summary.any_hallucination_rate = round(summary.any_hallucination_count / n, 3)
    summary.avg_grounding_score = round(sum(grounding_scores) / n, 3)
    summary.avg_hallucination_score = round(sum(hallucination_scores) / n, 3)
    
    # Top confusion pairs
    summary.most_confused_pairs = sorted(
        [(p[0], p[1], c) for p, c in confusion_counts.items()],
        key=lambda x: x[2],
        reverse=True
    )[:10]
    
    # Top invisible claims
    summary.most_common_invisible_claims = sorted(
        list(invisible_claim_counts.items()),
        key=lambda x: x[1],
        reverse=True
    )[:10]
    
    # Per-class summary
    for cls, stats in per_class.items():
        cnt = stats["count"]
        summary.per_class_stats[cls] = {
            "count": cnt,
            "wrong_disease_rate": round(stats["wrong_disease"] / cnt, 3) if cnt > 0 else 0,
            "visibility_rate": round(stats["visibility"] / cnt, 3) if cnt > 0 else 0,
            "avg_grounding": round(stats["grounding_sum"] / cnt, 3) if cnt > 0 else 0
        }
    
    return summary


# Quick test
if __name__ == "__main__":
    checker = HallucinationChecker()
    
    print("=== Hallucination Checker Test ===\n")
    
    # Test wrong-disease check
    test_text = "The plant shows curled leaves and cracked soil typical of drought. There is also gray fuzz on the berries."
    wrong, confused, has_wrong = checker.check_wrong_disease_hallucination(test_text, "drought")
    print(f"Wrong-disease test:")
    print(f"  Mentions: {wrong}")
    print(f"  Confused with: {confused}")
    print(f"  Has hallucination: {has_wrong}")
    
    # Test full check
    print("\n--- Full check ---")
    result = checker.check_response(
        image_path="/data/AGAI/MiniGPT-4/plant_diagnostic/data/train_aug/drought/test.jpg",
        predicted_label="drought",
        generated_text=test_text
    )
    print(f"Ground truth: {result.ground_truth_label}")
    print(f"Wrong-disease hallucination: {result.has_wrong_disease_hallucination}")
    print(f"Visibility hallucination: {result.has_visibility_hallucination}")
    print(f"Grounding score: {result.grounding_score}")
    print(f"Total hallucination score: {result.total_hallucination_score}")

