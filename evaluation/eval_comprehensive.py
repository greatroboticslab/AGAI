#!/usr/bin/env python3
"""
Comprehensive evaluation script for Plant Diagnostic System paper.
Evaluates ResNet classification, MiniGPT explanations, and system-level metrics.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import logging

import numpy as np
import torch
from torch.utils.data import DataLoader
from PIL import Image
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report
)
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from minigpt4.common.config import Config
from minigpt4.common.registry import registry
from minigpt4.conversation.conversation import CONV_VISION_minigptv2
from resnet_classifier import load_resnet, diagnose_or_none

# Import modules for registration
from minigpt4.datasets.builders import *
from minigpt4.models import *
from minigpt4.processors import *
from minigpt4.runners import *
from minigpt4.tasks import *

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ComprehensiveEvaluator:
    """Comprehensive evaluation for Plant Diagnostic System."""
    
    def __init__(self, cfg_path: str, resnet_path: str, device: str = "cuda:0"):
        self.device = torch.device(device)
        self.cfg = Config(cfg_path)
        
        # Load ResNet
        logger.info(f"Loading ResNet from {resnet_path}")
        self.resnet_model = load_resnet(resnet_path)
        
        # Load MiniGPT
        logger.info("Loading MiniGPT model...")
        self.task = setup_task(self.cfg)
        self.datasets = self.task.build_datasets(self.cfg)
        self.model = registry.get_model_class(self.cfg.model_cfg.arch).from_config(self.cfg.model_cfg)
        self.model = self.model.to(self.device).eval()
        
        # Load checkpoint
        ckpt_path = self.cfg.model_cfg.get("ckpt")
        if ckpt_path and os.path.exists(ckpt_path):
            logger.info(f"Loading checkpoint from {ckpt_path}")
            checkpoint = torch.load(ckpt_path, map_location=self.device, weights_only=False)
            state = checkpoint.get("model", checkpoint)
            self.model.load_state_dict(state, strict=False)
        
        # Get processors
        self.vis_processor = registry.get_processor_class("blip2_image_eval").from_config(
            self.cfg.datasets_cfg.strawberry_diagnostic.vis_processor.eval
        )
        self.text_processor = registry.get_processor_class("blip_caption").from_config(
            self.cfg.datasets_cfg.strawberry_diagnostic.text_processor.eval
        )
        
        # Class names (7 classes)
        self.class_names = ["healthy", "overwatering", "root_rot", "drought", 
                           "frost_injury", "gray_mold", "white_mold"]
        
        # Results storage
        self.results = {
            "resnet_metrics": {},
            "minigpt_metrics": {},
            "system_metrics": {},
            "per_image_results": []
        }
    
    def evaluate_resnet(self, image_paths: List[str], ground_truth: List[str]) -> Dict:
        """Evaluate ResNet classification performance."""
        logger.info(f"Evaluating ResNet on {len(image_paths)} images...")
        
        predictions = []
        confidences = []
        all_probs = []
        
        for img_path in image_paths:
            result = diagnose_or_none(self.resnet_model, img_path)
            if result:
                predictions.append(result["label"])
                confidences.append(result["p1"])
                # Store full probability distribution if available
                all_probs.append(result.get("probs", [result["p1"]]))
            else:
                predictions.append("unknown")
                confidences.append(0.0)
                all_probs.append([0.0] * len(self.class_names))
        
        # Normalize labels
        pred_normalized = [self._normalize_label(p) for p in predictions]
        gt_normalized = [self._normalize_label(gt) for gt in ground_truth]
        
        # Calculate metrics
        accuracy = accuracy_score(gt_normalized, pred_normalized)
        precision, recall, f1, support = precision_recall_fscore_support(
            gt_normalized, pred_normalized, labels=self.class_names, average=None, zero_division=0
        )
        
        # Macro and weighted averages
        precision_macro = np.mean(precision)
        recall_macro = np.mean(recall)
        f1_macro = np.mean(f1)
        
        precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
            gt_normalized, pred_normalized, labels=self.class_names, average='weighted', zero_division=0
        )
        
        # Confusion matrix
        cm = confusion_matrix(gt_normalized, pred_normalized, labels=self.class_names)
        
        # Per-class metrics
        per_class_metrics = {}
        for i, class_name in enumerate(self.class_names):
            per_class_metrics[class_name] = {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "support": int(support[i])
            }
        
        results = {
            "overall_accuracy": float(accuracy),
            "macro_avg": {
                "precision": float(precision_macro),
                "recall": float(recall_macro),
                "f1": float(f1_macro)
            },
            "weighted_avg": {
                "precision": float(precision_weighted),
                "recall": float(recall_weighted),
                "f1": float(f1_weighted)
            },
            "per_class": per_class_metrics,
            "confusion_matrix": cm.tolist(),
            "avg_confidence": float(np.mean(confidences))
        }
        
        self.results["resnet_metrics"] = results
        logger.info(f"ResNet Accuracy: {accuracy:.4f}")
        return results
    
    def evaluate_minigpt(self, image_paths: List[str], resnet_predictions: List[str]) -> Dict:
        """Evaluate MiniGPT explanation quality."""
        logger.info(f"Evaluating MiniGPT on {len(image_paths)} images...")
        
        conv_temp = CONV_VISION_minigptv2.copy()
        conv_temp.system = ""
        
        diagnosis_agreements = []
        explanations = []
        inference_times = []
        
        for img_path, resnet_pred in zip(image_paths, resnet_predictions):
            try:
                # Load and process image
                image = Image.open(img_path).convert("RGB")
                image_tensor = self.vis_processor(image).unsqueeze(0).to(self.device)
                
                # Create prompt with ResNet diagnosis
                prompt = f"Analyze this strawberry plant image. The diagnosis is: {resnet_pred}. Provide a medical report with: 1) Diagnosis, 2) Visible cues, 3) Recommendations."
                text = self.text_processor(prompt)
                texts = [text]
                
                # Prepare conversation
                conv = conv_temp.copy()
                conv.append_message(conv.roles[0], prompt)
                conv.append_message(conv.roles[1], None)
                prompt_text = conv.get_prompt()
                
                # Generate explanation
                start_time = time.time()
                with torch.no_grad():
                    explanation = self.model.generate(
                        image_tensor,
                        [prompt_text],
                        max_new_tokens=200,
                        do_sample=False
                    )[0]
                inference_time = time.time() - start_time
                inference_times.append(inference_time)
                
                # Check diagnosis agreement
                explanation_lower = explanation.lower()
                resnet_pred_lower = resnet_pred.lower()
                
                # Simple agreement check (can be improved)
                agreement = (
                    resnet_pred_lower in explanation_lower or
                    any(alias in explanation_lower for alias in self._get_aliases(resnet_pred))
                )
                diagnosis_agreements.append(agreement)
                
                explanations.append({
                    "image": str(img_path),
                    "resnet_prediction": resnet_pred,
                    "explanation": explanation,
                    "agreement": agreement,
                    "inference_time": inference_time
                })
                
            except Exception as e:
                logger.error(f"Error processing {img_path}: {e}")
                explanations.append({
                    "image": str(img_path),
                    "resnet_prediction": resnet_pred,
                    "explanation": f"ERROR: {str(e)}",
                    "agreement": False,
                    "inference_time": 0.0
                })
                diagnosis_agreements.append(False)
        
        # Calculate metrics
        agreement_rate = np.mean(diagnosis_agreements) if diagnosis_agreements else 0.0
        avg_inference_time = np.mean(inference_times) if inference_times else 0.0
        
        # Check structured output compliance
        structured_compliant = 0
        for exp in explanations:
            exp_text = exp["explanation"].lower()
            has_diagnosis = any(keyword in exp_text for keyword in ["diagnosis", "diagnosed", "condition"])
            has_cues = any(keyword in exp_text for keyword in ["visible", "cue", "symptom", "sign"])
            has_recommendation = any(keyword in exp_text for keyword in ["recommend", "treatment", "action", "suggest"])
            if has_diagnosis and has_cues and has_recommendation:
                structured_compliant += 1
        
        structured_compliance = structured_compliant / len(explanations) if explanations else 0.0
        
        results = {
            "diagnosis_agreement_rate": float(agreement_rate),
            "structured_output_compliance": float(structured_compliance),
            "avg_inference_time_ms": float(avg_inference_time * 1000),
            "total_explanations": len(explanations),
            "explanations": explanations
        }
        
        self.results["minigpt_metrics"] = results
        self.results["per_image_results"] = explanations
        logger.info(f"MiniGPT Diagnosis Agreement: {agreement_rate:.4f}")
        logger.info(f"Structured Output Compliance: {structured_compliance:.4f}")
        return results
    
    def evaluate_system_end_to_end(self, image_paths: List[str], ground_truth: List[str]) -> Dict:
        """Evaluate end-to-end system performance."""
        logger.info("Evaluating end-to-end system...")
        
        # Get ResNet predictions
        resnet_preds = []
        for img_path in image_paths:
            result = diagnose_or_none(self.resnet_model, img_path)
            resnet_preds.append(result["label"] if result else "unknown")
        
        # Get MiniGPT explanations
        minigpt_results = self.evaluate_minigpt(image_paths, resnet_preds)
        
        # Calculate end-to-end accuracy (both correct)
        correct_both = 0
        correct_resnet_only = 0
        correct_minigpt_only = 0
        correct_neither = 0
        
        gt_normalized = [self._normalize_label(gt) for gt in ground_truth]
        resnet_normalized = [self._normalize_label(p) for p in resnet_preds]
        
        for i, (gt, resnet_pred, exp_data) in enumerate(zip(gt_normalized, resnet_normalized, minigpt_results["explanations"])):
            resnet_correct = (gt == resnet_pred)
            minigpt_agrees = exp_data["agreement"]
            
            if resnet_correct and minigpt_agrees:
                correct_both += 1
            elif resnet_correct:
                correct_resnet_only += 1
            elif minigpt_agrees:
                correct_minigpt_only += 1
            else:
                correct_neither += 1
        
        total = len(ground_truth)
        results = {
            "end_to_end_accuracy": float(correct_both / total) if total > 0 else 0.0,
            "correct_both": int(correct_both),
            "correct_resnet_only": int(correct_resnet_only),
            "correct_minigpt_only": int(correct_minigpt_only),
            "correct_neither": int(correct_neither),
            "total": int(total),
            "resnet_accuracy": float(np.mean([gt == pred for gt, pred in zip(gt_normalized, resnet_normalized)]))
        }
        
        self.results["system_metrics"] = results
        logger.info(f"End-to-End Accuracy: {results['end_to_end_accuracy']:.4f}")
        return results
    
    def _normalize_label(self, label: str) -> str:
        """Normalize label to canonical form."""
        label = (label or "").strip().lower().replace("-", " ").replace("_", " ")
        label = " ".join(label.split())
        
        # Map aliases to canonical names
        alias_map = {
            "healthy": "healthy",
            "overwatering": "overwatering",
            "over-watering": "overwatering",
            "root rot": "root_rot",
            "root_rot": "root_rot",
            "drought": "drought",
            "frost": "frost_injury",
            "frost injury": "frost_injury",
            "frost_injury": "frost_injury",
            "gray mold": "gray_mold",
            "grey mold": "gray_mold",
            "gray_mold": "gray_mold",
            "white mold": "white_mold",
            "white_mold": "white_mold"
        }
        
        return alias_map.get(label, label.replace(" ", "_"))
    
    def _get_aliases(self, label: str) -> List[str]:
        """Get aliases for a label."""
        alias_map = {
            "healthy": ["healthy", "normal"],
            "overwatering": ["overwatering", "over-watering", "over watering"],
            "root_rot": ["root rot", "root_rot"],
            "drought": ["drought", "dry"],
            "frost_injury": ["frost", "frost injury", "frost_injury"],
            "gray_mold": ["gray mold", "grey mold", "gray_mold", "grey_mold"],
            "white_mold": ["white mold", "white_mold"]
        }
        return alias_map.get(label.lower(), [label])
    
    def save_results(self, output_dir: str):
        """Save all results to files."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save main results JSON
        results_file = output_path / "comprehensive_evaluation_results.json"
        with open(results_file, "w") as f:
            json.dump(self.results, f, indent=2)
        logger.info(f"Saved results to {results_file}")
        
        # Save confusion matrix as CSV
        if "confusion_matrix" in self.results["resnet_metrics"]:
            cm = np.array(self.results["resnet_metrics"]["confusion_matrix"])
            cm_df = pd.DataFrame(cm, index=self.class_names, columns=self.class_names)
            cm_file = output_path / "confusion_matrix.csv"
            cm_df.to_csv(cm_file)
            logger.info(f"Saved confusion matrix to {cm_file}")
        
        # Save per-image results as JSONL
        if self.results["per_image_results"]:
            jsonl_file = output_path / "per_image_results.jsonl"
            with open(jsonl_file, "w") as f:
                for result in self.results["per_image_results"]:
                    f.write(json.dumps(result) + "\n")
            logger.info(f"Saved per-image results to {jsonl_file}")
        
        # Save summary report
        summary_file = output_path / "evaluation_summary.txt"
        with open(summary_file, "w") as f:
            f.write("=" * 80 + "\n")
            f.write("COMPREHENSIVE EVALUATION SUMMARY\n")
            f.write("=" * 80 + "\n\n")
            
            f.write("RESNET CLASSIFICATION METRICS\n")
            f.write("-" * 80 + "\n")
            if "resnet_metrics" in self.results:
                rm = self.results["resnet_metrics"]
                f.write(f"Overall Accuracy: {rm.get('overall_accuracy', 0):.4f}\n")
                f.write(f"Macro-Avg F1: {rm.get('macro_avg', {}).get('f1', 0):.4f}\n")
                f.write(f"Weighted-Avg F1: {rm.get('weighted_avg', {}).get('f1', 0):.4f}\n")
                f.write(f"Average Confidence: {rm.get('avg_confidence', 0):.4f}\n\n")
            
            f.write("MINIGPT EXPLANATION METRICS\n")
            f.write("-" * 80 + "\n")
            if "minigpt_metrics" in self.results:
                mm = self.results["minigpt_metrics"]
                f.write(f"Diagnosis Agreement Rate: {mm.get('diagnosis_agreement_rate', 0):.4f}\n")
                f.write(f"Structured Output Compliance: {mm.get('structured_output_compliance', 0):.4f}\n")
                f.write(f"Avg Inference Time: {mm.get('avg_inference_time_ms', 0):.2f} ms\n\n")
            
            f.write("SYSTEM-LEVEL METRICS\n")
            f.write("-" * 80 + "\n")
            if "system_metrics" in self.results:
                sm = self.results["system_metrics"]
                f.write(f"End-to-End Accuracy: {sm.get('end_to_end_accuracy', 0):.4f}\n")
                f.write(f"ResNet Accuracy: {sm.get('resnet_accuracy', 0):.4f}\n")
                f.write(f"Both Correct: {sm.get('correct_both', 0)} / {sm.get('total', 0)}\n")
        
        logger.info(f"Saved summary to {summary_file}")


def load_test_data(dataset_path: str, images_dir: str, max_samples: Optional[int] = None):
    """Load test dataset."""
    import json
    
    with open(dataset_path, "r") as f:
        data = json.load(f)
    
    image_paths = []
    ground_truth = []
    
    for item in data[:max_samples] if max_samples else data:
        if "image" in item:
            img_path = Path(images_dir) / item["image"]
            if img_path.exists():
                image_paths.append(str(img_path))
                # Extract ground truth from annotations
                if "conversations" in item:
                    # Try to extract from conversations
                    for conv in item["conversations"]:
                        if "value" in conv and "diagnosis" in conv["value"].lower():
                            # Extract diagnosis (simplified)
                            ground_truth.append("healthy")  # Default, should be improved
                            break
                    else:
                        ground_truth.append("unknown")
                else:
                    ground_truth.append("unknown")
    
    return image_paths, ground_truth


def main():
    parser = argparse.ArgumentParser(description="Comprehensive evaluation for Plant Diagnostic System")
    parser.add_argument("--cfg-path", required=True, help="Path to config file")
    parser.add_argument("--resnet-path", required=True, help="Path to ResNet checkpoint")
    parser.add_argument("--dataset", help="Path to test dataset JSON")
    parser.add_argument("--images-dir", help="Path to images directory")
    parser.add_argument("--image-list", nargs="+", help="List of image paths to evaluate")
    parser.add_argument("--ground-truth", nargs="+", help="Ground truth labels (must match image-list)")
    parser.add_argument("--output-dir", default="evaluation/results", help="Output directory for results")
    parser.add_argument("--device", default="cuda:0", help="Device to use")
    parser.add_argument("--max-samples", type=int, help="Maximum number of samples to evaluate")
    
    args = parser.parse_args()
    
    # Load test data
    if args.image_list:
        image_paths = args.image_list
        ground_truth = args.ground_truth or ["unknown"] * len(image_paths)
    elif args.dataset and args.images_dir:
        image_paths, ground_truth = load_test_data(args.dataset, args.images_dir, args.max_samples)
    else:
        raise ValueError("Must provide either --image-list or --dataset + --images-dir")
    
    if len(image_paths) != len(ground_truth):
        raise ValueError(f"Number of images ({len(image_paths)}) must match number of ground truth labels ({len(ground_truth)})")
    
    logger.info(f"Evaluating {len(image_paths)} images...")
    
    # Initialize evaluator
    evaluator = ComprehensiveEvaluator(args.cfg_path, args.resnet_path, args.device)
    
    # Run evaluations
    resnet_results = evaluator.evaluate_resnet(image_paths, ground_truth)
    system_results = evaluator.evaluate_system_end_to_end(image_paths, ground_truth)
    
    # Save results
    evaluator.save_results(args.output_dir)
    
    logger.info("Evaluation complete!")


if __name__ == "__main__":
    main()



