#!/usr/bin/env python3
"""
Report Generator for Hallucination Evaluation Results

Generates human-readable reports (text, markdown, and tables) from evaluation JSON files.

Usage:
    python report_generator.py results/eval_20231217_123456.json
    python report_generator.py results/eval_*.json --compare
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional
from collections import defaultdict
from datetime import datetime


def load_results(path: str) -> Dict:
    """Load evaluation results from JSON file."""
    with open(path, 'r') as f:
        return json.load(f)


def generate_text_report(results: Dict, output_path: Optional[str] = None) -> str:
    """Generate a detailed text report from evaluation results."""
    
    meta = results.get("metadata", {})
    resnet = results.get("resnet_performance", {})
    summary = results.get("hallucination_summary", {})
    individual = results.get("individual_results", [])
    
    lines = []
    lines.append("=" * 80)
    lines.append("HALLUCINATION EVALUATION REPORT")
    lines.append("=" * 80)
    lines.append("")
    
    # Metadata
    lines.append("📋 EVALUATION CONFIGURATION")
    lines.append("-" * 40)
    lines.append(f"  Timestamp:        {meta.get('timestamp', 'N/A')}")
    lines.append(f"  Total samples:    {meta.get('num_samples', 'N/A')}")
    lines.append(f"  Runs per image:   {meta.get('runs_per_image', 'N/A')}")
    lines.append(f"  Total evals:      {meta.get('total_evaluations', 'N/A')}")
    lines.append(f"  Data source:      {'Holdout' if meta.get('use_holdout') else 'Training'}")
    lines.append(f"  Label source:     {'ResNet' if meta.get('use_resnet_label') else 'Ground Truth'}")
    lines.append(f"  Temperature:      {meta.get('temperature', 'N/A')}")
    lines.append("")
    
    # ResNet Performance
    lines.append("🔬 RESNET CLASSIFIER PERFORMANCE")
    lines.append("-" * 40)
    lines.append(f"  Accuracy:         {resnet.get('accuracy', 0):.1%}")
    lines.append(f"  Avg Confidence:   {resnet.get('avg_confidence', 0):.1%}")
    lines.append(f"  Samples tested:   {resnet.get('samples', 0)}")
    lines.append("")
    
    # Wrong-Disease Hallucination
    wd = summary.get("wrong_disease_hallucination", {})
    lines.append("⚠️  WRONG-DISEASE HALLUCINATION (Misdiagnosis Text)")
    lines.append("-" * 40)
    lines.append(f"  Instances:        {wd.get('count', 0)}")
    lines.append(f"  Rate:             {wd.get('rate', 0):.1%}")
    lines.append("")
    if wd.get("most_confused_pairs"):
        lines.append("  Most common confusions:")
        for i, (pred, confused, cnt) in enumerate(wd["most_confused_pairs"][:5], 1):
            lines.append(f"    {i}. {pred} → {confused}: {cnt} times")
    lines.append("")
    
    # Visibility Hallucination
    vh = summary.get("visibility_hallucination", {})
    lines.append("👁️  VISIBILITY HALLUCINATION (Non-Apparent Claims)")
    lines.append("-" * 40)
    lines.append(f"  Instances:        {vh.get('count', 0)}")
    lines.append(f"  Rate:             {vh.get('rate', 0):.1%}")
    lines.append("")
    if vh.get("most_common_claims"):
        lines.append("  Most common invisible region claims:")
        for region, cnt in vh["most_common_claims"][:5]:
            lines.append(f"    - {region}: {cnt} times")
    lines.append("")
    
    # Grounding
    gr = summary.get("grounding", {})
    lines.append("📈 SYMPTOM GROUNDING")
    lines.append("-" * 40)
    lines.append(f"  Avg Score:        {gr.get('avg_score', 0):.1%}")
    lines.append(f"  Low grounding:    {gr.get('low_grounding_count', 0)} samples (<30%)")
    lines.append("")
    
    # Overall
    ov = summary.get("overall", {})
    lines.append("📊 OVERALL HALLUCINATION METRICS")
    lines.append("-" * 40)
    lines.append(f"  Any hallucination:     {ov.get('any_hallucination_count', 0)} ({ov.get('any_hallucination_rate', 0):.1%})")
    lines.append(f"  Avg hallucin. score:   {ov.get('avg_hallucination_score', 0):.3f} (0=clean, 1=severe)")
    lines.append("")
    
    # Per-class breakdown
    per_class = summary.get("per_class", {})
    if per_class:
        lines.append("📋 PER-CLASS BREAKDOWN")
        lines.append("-" * 40)
        lines.append(f"  {'Class':<15} {'Count':>6} {'WD Rate':>10} {'Vis Rate':>10} {'Grounding':>10}")
        lines.append("  " + "-" * 55)
        for cls, stats in sorted(per_class.items()):
            lines.append(f"  {cls:<15} {stats.get('count', 0):>6} {stats.get('wrong_disease_rate', 0):>9.1%} {stats.get('visibility_rate', 0):>9.1%} {stats.get('avg_grounding', 0):>9.1%}")
    lines.append("")
    
    # Sample of problematic responses
    lines.append("=" * 80)
    lines.append("SAMPLE PROBLEMATIC RESPONSES")
    lines.append("=" * 80)
    
    # Find responses with hallucinations
    problematic = [r for r in individual if r.get("has_wrong_disease_hallucination") or r.get("has_visibility_hallucination")]
    
    for i, r in enumerate(problematic[:5], 1):
        lines.append(f"\n--- Sample {i} ---")
        lines.append(f"Image: {Path(r.get('image_path', '')).name}")
        lines.append(f"Ground Truth: {r.get('ground_truth_label', 'N/A')}")
        lines.append(f"Predicted: {r.get('predicted_label', 'N/A')}")
        lines.append(f"Hallucination Score: {r.get('total_hallucination_score', 0):.3f}")
        
        if r.get("wrong_disease_mentions"):
            lines.append(f"Wrong-disease mentions: {r['wrong_disease_mentions']}")
        if r.get("invisible_region_claims"):
            lines.append(f"Invisible region claims: {r['invisible_region_claims']}")
        
        text = r.get("generated_text", "")[:500]
        lines.append(f"Response preview: {text}...")
    
    lines.append("")
    lines.append("=" * 80)
    lines.append("END OF REPORT")
    lines.append("=" * 80)
    
    report = "\n".join(lines)
    
    if output_path:
        with open(output_path, 'w') as f:
            f.write(report)
        print(f"Report saved to: {output_path}")
    
    return report


def generate_markdown_report(results: Dict, output_path: Optional[str] = None) -> str:
    """Generate a markdown report suitable for documentation."""
    
    meta = results.get("metadata", {})
    resnet = results.get("resnet_performance", {})
    summary = results.get("hallucination_summary", {})
    per_class = summary.get("per_class", {})
    
    lines = []
    lines.append("# Hallucination Evaluation Report")
    lines.append("")
    lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append("")
    
    # Summary table
    lines.append("## Summary Metrics")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    lines.append(f"| Total Samples | {meta.get('num_samples', 0)} |")
    lines.append(f"| ResNet Accuracy | {resnet.get('accuracy', 0):.1%} |")
    lines.append(f"| Wrong-Disease Rate | {summary.get('wrong_disease_hallucination', {}).get('rate', 0):.1%} |")
    lines.append(f"| Visibility Hallucination Rate | {summary.get('visibility_hallucination', {}).get('rate', 0):.1%} |")
    lines.append(f"| Average Grounding Score | {summary.get('grounding', {}).get('avg_score', 0):.1%} |")
    lines.append(f"| Overall Hallucination Rate | {summary.get('overall', {}).get('any_hallucination_rate', 0):.1%} |")
    lines.append("")
    
    # Per-class table
    if per_class:
        lines.append("## Per-Class Breakdown")
        lines.append("")
        lines.append("| Disease Class | Samples | Wrong-Disease | Visibility | Grounding |")
        lines.append("|--------------|---------|---------------|------------|-----------|")
        for cls, stats in sorted(per_class.items()):
            lines.append(f"| {cls.replace('_', ' ').title()} | {stats.get('count', 0)} | {stats.get('wrong_disease_rate', 0):.1%} | {stats.get('visibility_rate', 0):.1%} | {stats.get('avg_grounding', 0):.1%} |")
    lines.append("")
    
    # Confusion analysis
    wd = summary.get("wrong_disease_hallucination", {})
    if wd.get("most_confused_pairs"):
        lines.append("## Disease Confusion Analysis")
        lines.append("")
        lines.append("Most common disease confusions in generated text:")
        lines.append("")
        for pred, confused, cnt in wd["most_confused_pairs"][:5]:
            lines.append(f"- **{pred}** confused with **{confused}**: {cnt} instances")
    lines.append("")
    
    # Methodology
    lines.append("## Evaluation Methodology")
    lines.append("")
    lines.append("### Wrong-Disease Hallucination")
    lines.append("Detects when the model mentions symptoms unique to a disease different from the diagnosis.")
    lines.append("Uses a knowledge base of disease-specific visual indicators.")
    lines.append("")
    lines.append("### Visibility Hallucination")
    lines.append("Detects when the model claims to see plant parts not visible in the training ground truth.")
    lines.append("Cross-references with training data descriptions.")
    lines.append("")
    lines.append("### Grounding Score")
    lines.append("Measures how many correct symptoms for the diagnosed disease are mentioned.")
    lines.append("Higher is better (1.0 = all expected symptoms mentioned).")
    lines.append("")
    
    report = "\n".join(lines)
    
    if output_path:
        with open(output_path, 'w') as f:
            f.write(report)
        print(f"Markdown report saved to: {output_path}")
    
    return report


def generate_csv_summary(results: Dict, output_path: str):
    """Generate CSV summary of individual results."""
    import csv
    
    individual = results.get("individual_results", [])
    
    if not individual:
        print("No individual results to export.")
        return
    
    fieldnames = [
        "image", "ground_truth", "predicted", 
        "wrong_disease", "visibility", "grounding_score",
        "hallucination_score", "wrong_mentions", "invisible_claims"
    ]
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for r in individual:
            writer.writerow({
                "image": Path(r.get("image_path", "")).name,
                "ground_truth": r.get("ground_truth_label", ""),
                "predicted": r.get("predicted_label", ""),
                "wrong_disease": r.get("has_wrong_disease_hallucination", False),
                "visibility": r.get("has_visibility_hallucination", False),
                "grounding_score": r.get("grounding_score", 0),
                "hallucination_score": r.get("total_hallucination_score", 0),
                "wrong_mentions": "; ".join(r.get("wrong_disease_mentions", [])),
                "invisible_claims": "; ".join(r.get("invisible_region_claims", []))
            })
    
    print(f"CSV summary saved to: {output_path}")


def compare_evaluations(result_files: List[str], output_path: Optional[str] = None) -> str:
    """Compare multiple evaluation runs."""
    
    lines = []
    lines.append("=" * 80)
    lines.append("EVALUATION COMPARISON")
    lines.append("=" * 80)
    lines.append("")
    
    header = f"{'Eval':<25} {'Samples':>8} {'WD Rate':>10} {'Vis Rate':>10} {'Ground':>10} {'Overall':>10}"
    lines.append(header)
    lines.append("-" * 80)
    
    for path in result_files:
        try:
            results = load_results(path)
            name = Path(path).stem[:24]
            meta = results.get("metadata", {})
            summary = results.get("hallucination_summary", {})
            
            wd_rate = summary.get("wrong_disease_hallucination", {}).get("rate", 0)
            vis_rate = summary.get("visibility_hallucination", {}).get("rate", 0)
            grounding = summary.get("grounding", {}).get("avg_score", 0)
            overall = summary.get("overall", {}).get("any_hallucination_rate", 0)
            
            lines.append(f"{name:<25} {meta.get('total_evaluations', 0):>8} {wd_rate:>9.1%} {vis_rate:>9.1%} {grounding:>9.1%} {overall:>9.1%}")
        except Exception as e:
            lines.append(f"{Path(path).stem:<25} ERROR: {e}")
    
    lines.append("")
    report = "\n".join(lines)
    
    if output_path:
        with open(output_path, 'w') as f:
            f.write(report)
    
    print(report)
    return report


def main():
    parser = argparse.ArgumentParser(description="Generate reports from hallucination evaluation results")
    parser.add_argument("results", nargs="+", help="Path(s) to evaluation result JSON file(s)")
    parser.add_argument("--format", choices=["text", "markdown", "csv", "all"], default="text",
                        help="Output format (default: text)")
    parser.add_argument("--compare", action="store_true",
                        help="Compare multiple evaluation results")
    parser.add_argument("--output", type=str, default=None,
                        help="Output file path (optional)")
    
    args = parser.parse_args()
    
    if args.compare and len(args.results) > 1:
        output_path = args.output or "comparison_report.txt"
        compare_evaluations(args.results, output_path)
    else:
        results = load_results(args.results[0])
        base_name = Path(args.results[0]).stem
        
        if args.format == "text" or args.format == "all":
            output_path = args.output or f"{base_name}_report.txt"
            report = generate_text_report(results, output_path)
            print(report)
        
        if args.format == "markdown" or args.format == "all":
            output_path = args.output or f"{base_name}_report.md"
            generate_markdown_report(results, output_path)
        
        if args.format == "csv" or args.format == "all":
            output_path = args.output or f"{base_name}_summary.csv"
            generate_csv_summary(results, output_path)


if __name__ == "__main__":
    main()

