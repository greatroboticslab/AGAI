#!/usr/bin/env python3
"""
Analyze Training Responses for Hallucination Patterns

This script analyzes the training data responses to understand:
1. What symptoms are described for each class
2. Cross-class symptom overlap (potential confusion sources)
3. Image-to-description consistency

No model loading required - purely text analysis.

Usage:
    python analyze_training_responses.py
    python analyze_training_responses.py --output analysis_report.json
"""

import json
import re
import argparse
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Set, Tuple

import sys
sys.path.insert(0, str(Path(__file__).parent))

from config import TRAINING_DATA_PATH, UNIQUE_INDICATORS, DISEASE_CLASSES


def load_training_data() -> List[Dict]:
    """Load the training data."""
    with open(TRAINING_DATA_PATH, 'r') as f:
        return json.load(f)


def extract_disease_from_path(image_path: str) -> str:
    """Extract disease class from image path."""
    path = Path(image_path)
    parent = path.parent.name.lower()
    
    # Handle special cases
    if "gray" in parent or "grey" in parent:
        return "gray_mold"
    if "white" in parent:
        return "white_mold"
    if "root" in parent:
        return "root_rot"
    if "frost" in parent:
        return "frost_injury"
    if "over" in parent or "water" in parent:
        return "overwatering"
    if "drought" in parent:
        return "drought"
    if "health" in parent:
        return "healthy"
    
    return parent


def extract_response_text(entry: Dict) -> str:
    """Extract assistant response from conversation entry."""
    for conv in entry.get("conversations", []):
        if conv.get("from") == "assistant":
            return conv.get("value", "")
    return ""


def extract_symptoms_from_text(text: str) -> List[str]:
    """Extract symptom phrases from response text."""
    text_lower = text.lower()
    
    # Look for phrases that describe visual symptoms
    symptoms = []
    
    # Pattern: adjective + noun (e.g., "curled leaves", "gray fuzz")
    patterns = [
        r'\b(curled?|wilted?|droopy?|limp|dull|glossy|yellow\w*|brown\w*|black\w*|white\w*|gray\w*|grey\w*|red\w*|pale|dark|mushy|soft|dry|wet|soggy|crispy?) (leaves?|foliage|petioles?|stems?|roots?|crown|berries?|fruits?|soil|margins?|edges?|tips?)\b',
        r'\b(leaves?|foliage|petioles?|stems?|roots?|crown|berries?|fruits?)\s+(are|is|appear|show|have|display)\s+(\w+)\b',
        r'\b(fuzz|mold|mould|spots?|patches?|lesions?|discoloration|necrosis|wilting|drooping|curling)\b',
        r'\b(water.?logged|saturated|dry|cracked|mushy|rotting|decaying)\b'
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, text_lower)
        for match in matches:
            if isinstance(match, tuple):
                symptom = " ".join(match).strip()
            else:
                symptom = match.strip()
            if symptom and len(symptom) > 3:
                symptoms.append(symptom)
    
    # Also extract any unique indicators mentioned
    for disease, indicators in UNIQUE_INDICATORS.items():
        for indicator in indicators:
            if indicator.lower() in text_lower:
                symptoms.append(f"[{disease}] {indicator}")
    
    return list(set(symptoms))


def analyze_class_symptoms(training_data: List[Dict]) -> Dict[str, Counter]:
    """Analyze symptom frequency per disease class."""
    class_symptoms = defaultdict(Counter)
    
    for entry in training_data:
        disease = extract_disease_from_path(entry.get("image", ""))
        response = extract_response_text(entry)
        symptoms = extract_symptoms_from_text(response)
        
        for symptom in symptoms:
            class_symptoms[disease][symptom] += 1
    
    return dict(class_symptoms)


def find_cross_class_overlaps(class_symptoms: Dict[str, Counter]) -> Dict[str, List[Tuple[str, List[str]]]]:
    """Find symptoms that appear across multiple classes (confusion risks)."""
    overlaps = defaultdict(list)
    
    # Build reverse index: symptom -> classes
    symptom_to_classes = defaultdict(set)
    for disease, symptoms in class_symptoms.items():
        for symptom in symptoms.keys():
            symptom_to_classes[symptom].add(disease)
    
    # Find overlapping symptoms
    for symptom, classes in symptom_to_classes.items():
        if len(classes) > 1:
            for disease in classes:
                other_classes = [c for c in classes if c != disease]
                overlaps[disease].append((symptom, other_classes))
    
    return dict(overlaps)


def check_unique_indicator_presence(training_data: List[Dict]) -> Dict[str, Dict]:
    """Check how often unique indicators appear in correct vs wrong classes."""
    results = {}
    
    for disease, indicators in UNIQUE_INDICATORS.items():
        correct_mentions = 0
        wrong_mentions = defaultdict(int)
        total_class_samples = 0
        
        for entry in training_data:
            entry_disease = extract_disease_from_path(entry.get("image", ""))
            response = extract_response_text(entry).lower()
            
            if entry_disease == disease:
                total_class_samples += 1
            
            for indicator in indicators:
                if indicator.lower() in response:
                    if entry_disease == disease:
                        correct_mentions += 1
                    else:
                        wrong_mentions[entry_disease] += 1
        
        results[disease] = {
            "indicators": indicators,
            "correct_mentions": correct_mentions,
            "total_class_samples": total_class_samples,
            "wrong_class_mentions": dict(wrong_mentions)
        }
    
    return results


def analyze_response_consistency(training_data: List[Dict]) -> Dict:
    """Check consistency of responses for same base images (different augmentations)."""
    # Group by base image (without augmentation suffix)
    base_image_responses = defaultdict(list)
    
    for entry in training_data:
        img_path = entry.get("image", "")
        filename = Path(img_path).name
        
        # Remove augmentation suffix
        base_name = re.sub(r'_aug\d+', '', filename)
        base_name = re.sub(r'\.(jpg|jpeg|png|webp)$', '', base_name, flags=re.IGNORECASE)
        
        response = extract_response_text(entry)
        base_image_responses[base_name].append({
            "full_path": img_path,
            "response_length": len(response),
            "response_preview": response[:200]
        })
    
    # Find images with multiple responses
    multi_response = {k: v for k, v in base_image_responses.items() if len(v) > 1}
    
    return {
        "unique_base_images": len(base_image_responses),
        "images_with_multiple_responses": len(multi_response),
        "avg_responses_per_image": sum(len(v) for v in base_image_responses.values()) / len(base_image_responses) if base_image_responses else 0,
        "sample_multi_response": dict(list(multi_response.items())[:3])
    }


def run_analysis(output_path: str = None) -> Dict:
    """Run full analysis and generate report."""
    
    print("=" * 60)
    print("TRAINING DATA HALLUCINATION PATTERN ANALYSIS")
    print("=" * 60)
    
    # Load data
    print("\n[1/5] Loading training data...")
    training_data = load_training_data()
    print(f"  Loaded {len(training_data)} entries")
    
    # Class distribution
    print("\n[2/5] Analyzing class distribution...")
    class_counts = Counter(
        extract_disease_from_path(e.get("image", "")) 
        for e in training_data
    )
    print("  Class distribution:")
    for cls, count in sorted(class_counts.items()):
        print(f"    {cls}: {count}")
    
    # Symptom analysis
    print("\n[3/5] Analyzing symptoms per class...")
    class_symptoms = analyze_class_symptoms(training_data)
    
    # Cross-class overlaps
    print("\n[4/5] Finding cross-class symptom overlaps...")
    overlaps = find_cross_class_overlaps(class_symptoms)
    
    # Unique indicator presence
    print("\n[5/5] Checking unique indicator patterns...")
    indicator_analysis = check_unique_indicator_presence(training_data)
    
    # Consistency check
    print("\n[6/6] Checking response consistency...")
    consistency = analyze_response_consistency(training_data)
    
    # Compile results
    results = {
        "total_entries": len(training_data),
        "class_distribution": dict(class_counts),
        "top_symptoms_per_class": {
            disease: symptoms.most_common(10)
            for disease, symptoms in class_symptoms.items()
        },
        "cross_class_overlaps": {
            disease: [
                {"symptom": s, "also_in": classes}
                for s, classes in overlaps_list[:5]
            ]
            for disease, overlaps_list in overlaps.items()
        },
        "unique_indicator_analysis": indicator_analysis,
        "response_consistency": consistency
    }
    
    # Print summary
    print("\n" + "=" * 60)
    print("ANALYSIS SUMMARY")
    print("=" * 60)
    
    print("\n📊 CLASS DISTRIBUTION:")
    for cls, count in sorted(class_counts.items()):
        pct = count / len(training_data) * 100
        print(f"  {cls:<15} {count:>5} ({pct:>5.1f}%)")
    
    print("\n🔍 TOP SYMPTOMS PER CLASS:")
    for disease, symptoms in sorted(class_symptoms.items()):
        top_3 = symptoms.most_common(3)
        print(f"\n  {disease}:")
        for symptom, count in top_3:
            print(f"    - {symptom}: {count}")
    
    print("\n⚠️  POTENTIAL CONFUSION RISKS (shared symptoms):")
    for disease, overlap_list in sorted(overlaps.items()):
        if overlap_list[:3]:
            print(f"\n  {disease} shares symptoms with:")
            for symptom, other_classes in overlap_list[:3]:
                print(f"    '{symptom}' also in: {', '.join(other_classes)}")
    
    print("\n✅ UNIQUE INDICATOR USAGE:")
    for disease, analysis in indicator_analysis.items():
        correct = analysis["correct_mentions"]
        total = analysis["total_class_samples"]
        wrong = analysis["wrong_class_mentions"]
        print(f"\n  {disease}:")
        print(f"    Correct usage: {correct}/{total} samples")
        if wrong:
            print(f"    ⚠️  Wrong class mentions: {dict(wrong)}")
    
    print(f"\n📝 RESPONSE CONSISTENCY:")
    print(f"  Unique base images: {consistency['unique_base_images']}")
    print(f"  Images with multiple responses: {consistency['images_with_multiple_responses']}")
    print(f"  Avg responses per image: {consistency['avg_responses_per_image']:.1f}")
    
    # Save if output path provided
    if output_path:
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n📁 Results saved to: {output_path}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Analyze training data for hallucination patterns")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="Output JSON file path")
    args = parser.parse_args()
    
    output_path = args.output or str(Path(__file__).parent / "results" / "training_analysis.json")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    run_analysis(output_path)


if __name__ == "__main__":
    main()

