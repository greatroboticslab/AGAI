#!/usr/bin/env python3
"""
Semantic Hallucination Evaluation

Uses sentence embeddings for:
1. Training data alignment - does output match training style?
2. Symptom grounding - are correct symptoms semantically present?
3. Cross-disease contamination - does it describe wrong disease symptoms?

Usage:
    python evaluate_semantic.py --results results/improved_eval.json
    python evaluate_semantic.py --run-fresh --num-samples 10
"""

import argparse
import json
import os
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from semantic_evaluator import SemanticEvaluator, EMBEDDINGS_AVAILABLE


def evaluate_existing_results(results_path: str, output_path: str = None):
    """
    Re-evaluate existing results with semantic metrics.
    """
    if not EMBEDDINGS_AVAILABLE:
        print("ERROR: sentence-transformers required. Install with: pip install sentence-transformers")
        return
    
    print("=" * 60)
    print("SEMANTIC RE-EVALUATION")
    print("=" * 60)
    
    # Load existing results
    with open(results_path, 'r') as f:
        data = json.load(f)
    
    individual_results = data.get("individual_results", [])
    print(f"Re-evaluating {len(individual_results)} responses...")
    
    # Initialize semantic evaluator
    evaluator = SemanticEvaluator()
    
    semantic_results = []
    
    for i, result in enumerate(individual_results):
        print(f"  [{i+1}/{len(individual_results)}] {Path(result['image_path']).name}")
        
        sem_result = evaluator.evaluate_response(
            image_path=result["image_path"],
            predicted_label=result["predicted_label"],
            generated_text=result["generated_text"],
            ground_truth_label=result.get("ground_truth_label")
        )
        
        semantic_results.append(sem_result.to_dict())
    
    # Compute summary statistics
    training_alignments = [r["training_alignment_score"] for r in semantic_results]
    grounding_scores = [r["symptom_grounding_score"] for r in semantic_results]
    quality_scores = [r["semantic_quality_score"] for r in semantic_results]
    
    summary = {
        "total_samples": len(semantic_results),
        "avg_training_alignment": round(sum(training_alignments) / len(training_alignments), 3),
        "avg_symptom_grounding": round(sum(grounding_scores) / len(grounding_scores), 3),
        "avg_semantic_quality": round(sum(quality_scores) / len(quality_scores), 3),
        "low_alignment_count": sum(1 for s in training_alignments if s < 0.5),
        "low_grounding_count": sum(1 for s in grounding_scores if s < 0.3),
    }
    
    # Per-class breakdown
    per_class = {}
    for r in semantic_results:
        cls = r["ground_truth_label"]
        if cls not in per_class:
            per_class[cls] = {"alignments": [], "groundings": [], "qualities": []}
        per_class[cls]["alignments"].append(r["training_alignment_score"])
        per_class[cls]["groundings"].append(r["symptom_grounding_score"])
        per_class[cls]["qualities"].append(r["semantic_quality_score"])
    
    for cls, scores in per_class.items():
        per_class[cls] = {
            "count": len(scores["alignments"]),
            "avg_alignment": round(sum(scores["alignments"]) / len(scores["alignments"]), 3),
            "avg_grounding": round(sum(scores["groundings"]) / len(scores["groundings"]), 3),
            "avg_quality": round(sum(scores["qualities"]) / len(scores["qualities"]), 3),
        }
    
    summary["per_class"] = per_class
    
    # Output results
    output = {
        "metadata": {
            "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "source_file": results_path,
            "evaluator": "semantic"
        },
        "summary": summary,
        "individual_results": semantic_results
    }
    
    # Save results
    if output_path is None:
        output_path = results_path.replace(".json", "_semantic.json")
    
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    # Print summary
    print("\n" + "=" * 60)
    print("SEMANTIC EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Total samples: {summary['total_samples']}")
    print(f"\n📊 Training Alignment (style match):")
    print(f"   Average: {summary['avg_training_alignment']:.1%}")
    print(f"   Low (<50%): {summary['low_alignment_count']}")
    print(f"\n🎯 Symptom Grounding (semantic):")
    print(f"   Average: {summary['avg_symptom_grounding']:.1%}")
    print(f"   Low (<30%): {summary['low_grounding_count']}")
    print(f"\n⭐ Overall Semantic Quality:")
    print(f"   Average: {summary['avg_semantic_quality']:.1%}")
    print(f"\n📋 Per-Class Breakdown:")
    print(f"   {'Class':<15} {'Align':>8} {'Ground':>8} {'Quality':>8}")
    print("   " + "-" * 45)
    for cls, stats in sorted(per_class.items()):
        print(f"   {cls:<15} {stats['avg_alignment']:>7.1%} {stats['avg_grounding']:>7.1%} {stats['avg_quality']:>7.1%}")
    
    print(f"\nResults saved to: {output_path}")
    
    return output


def main():
    parser = argparse.ArgumentParser(description="Semantic hallucination evaluation")
    parser.add_argument("--results", type=str, default=None,
                        help="Path to existing evaluation results JSON")
    parser.add_argument("--output", type=str, default=None,
                        help="Output path for semantic results")
    
    args = parser.parse_args()
    
    if args.results:
        evaluate_existing_results(args.results, args.output)
    else:
        # Default to latest results
        results_dir = Path(__file__).parent / "results"
        latest = sorted(results_dir.glob("*.json"), key=os.path.getmtime)
        if latest:
            # Skip semantic results files
            for f in reversed(latest):
                if "_semantic" not in f.name:
                    print(f"Using latest results: {f}")
                    evaluate_existing_results(str(f), args.output)
                    break
        else:
            print("No results found. Run evaluate.py first.")


if __name__ == "__main__":
    main()

