#!/usr/bin/env python3
"""
Model validation and confidence calibration utilities for the Plant Diagnostic System.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)


class ModelValidator:
    """Model validation and confidence calibration utilities."""
    
    def __init__(self, model, device: str = "cuda"):
        self.model = model
        self.device = device
        self.calibration_data = []
        self.validation_results = {}
    
    def validate_model_performance(self, dataloader, class_names: List[str]) -> Dict:
        """Validate model performance on test data."""
        logger.info("Validating model performance...")
        
        self.model.eval()
        all_predictions = []
        all_labels = []
        all_probabilities = []
        
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(dataloader):
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                # Get model predictions
                outputs = self.model(images)
                probabilities = torch.softmax(outputs, dim=1)
                predictions = torch.argmax(probabilities, dim=1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
        
        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_predictions)
        precision, recall, f1, support = precision_recall_fscore_support(
            all_labels, all_predictions, average='weighted'
        )
        
        # Per-class metrics
        precision_per_class, recall_per_class, f1_per_class, support_per_class = precision_recall_fscore_support(
            all_labels, all_predictions, average=None
        )
        
        # Confusion matrix
        cm = confusion_matrix(all_labels, all_predictions)
        
        results = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'support': support,
            'per_class_metrics': {
                'precision': precision_per_class,
                'recall': recall_per_class,
                'f1_score': f1_per_class,
                'support': support_per_class
            },
            'confusion_matrix': cm,
            'class_names': class_names
        }
        
        self.validation_results = results
        logger.info(f"Validation complete - Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
        
        return results
    
    def calibrate_confidence(self, dataloader, method: str = 'isotonic') -> nn.Module:
        """Calibrate model confidence scores."""
        logger.info(f"Calibrating confidence using {method} method...")
        
        self.model.eval()
        all_probabilities = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in dataloader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(images)
                probabilities = torch.softmax(outputs, dim=1)
                
                all_probabilities.extend(probabilities.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Convert to numpy arrays
        X_cal = np.array(all_probabilities)
        y_cal = np.array(all_labels)
        
        # Create calibrated classifier
        calibrated_model = CalibratedClassifierCV(
            base_estimator=None,  # We'll use the probabilities directly
            method=method,
            cv=3
        )
        
        # Fit calibration
        calibrated_model.fit(X_cal, y_cal)
        
        logger.info("Confidence calibration complete")
        return calibrated_model
    
    def evaluate_calibration(self, dataloader, calibrated_model=None) -> Dict:
        """Evaluate calibration quality using reliability diagrams."""
        logger.info("Evaluating calibration quality...")
        
        self.model.eval()
        all_probabilities = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in dataloader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(images)
                probabilities = torch.softmax(outputs, dim=1)
                
                all_probabilities.extend(probabilities.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        X_test = np.array(all_probabilities)
        y_test = np.array(all_labels)
        
        if calibrated_model is not None:
            calibrated_probs = calibrated_model.predict_proba(X_test)
        else:
            calibrated_probs = X_test
        
        # Calculate Expected Calibration Error (ECE)
        ece = self._calculate_ece(calibrated_probs, y_test)
        
        # Calculate Maximum Calibration Error (MCE)
        mce = self._calculate_mce(calibrated_probs, y_test)
        
        results = {
            'ece': ece,
            'mce': mce,
            'calibrated_probabilities': calibrated_probs,
            'true_labels': y_test
        }
        
        logger.info(f"Calibration evaluation complete - ECE: {ece:.4f}, MCE: {mce:.4f}")
        return results
    
    def _calculate_ece(self, probabilities: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
        """Calculate Expected Calibration Error."""
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (probabilities.max(axis=1) > bin_lower) & (probabilities.max(axis=1) <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = (labels == probabilities.argmax(axis=1))[in_bin].mean()
                avg_confidence_in_bin = probabilities[in_bin].max(axis=1).mean()
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        
        return ece
    
    def _calculate_mce(self, probabilities: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
        """Calculate Maximum Calibration Error."""
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        mce = 0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (probabilities.max(axis=1) > bin_lower) & (probabilities.max(axis=1) <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = (labels == probabilities.argmax(axis=1))[in_bin].mean()
                avg_confidence_in_bin = probabilities[in_bin].max(axis=1).mean()
                mce = max(mce, np.abs(avg_confidence_in_bin - accuracy_in_bin))
        
        return mce
    
    def plot_confusion_matrix(self, save_path: Optional[str] = None):
        """Plot confusion matrix."""
        if not self.validation_results:
            logger.warning("No validation results available. Run validate_model_performance first.")
            return
        
        cm = self.validation_results['confusion_matrix']
        class_names = self.validation_results['class_names']
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_reliability_diagram(self, calibration_results: Dict, save_path: Optional[str] = None):
        """Plot reliability diagram for calibration evaluation."""
        probabilities = calibration_results['calibrated_probabilities']
        labels = calibration_results['true_labels']
        
        # Get max probabilities and corresponding predictions
        max_probs = probabilities.max(axis=1)
        predictions = probabilities.argmax(axis=1)
        
        # Create reliability diagram
        n_bins = 10
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        bin_centers = []
        bin_accuracies = []
        bin_confidences = []
        bin_counts = []
        
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (max_probs > bin_lower) & (max_probs <= bin_upper)
            prop_in_bin = in_bin.sum()
            
            if prop_in_bin > 0:
                bin_centers.append((bin_lower + bin_upper) / 2)
                bin_accuracies.append((labels == predictions)[in_bin].mean())
                bin_confidences.append(max_probs[in_bin].mean())
                bin_counts.append(prop_in_bin)
        
        # Plot
        plt.figure(figsize=(10, 8))
        plt.bar(bin_centers, bin_accuracies, width=0.1, alpha=0.7, label='Accuracy')
        plt.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
        plt.xlabel('Confidence')
        plt.ylabel('Accuracy')
        plt.title('Reliability Diagram')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_validation_report(self, save_path: Optional[str] = None) -> str:
        """Generate a comprehensive validation report."""
        if not self.validation_results:
            return "No validation results available."
        
        report = []
        report.append("=" * 80)
        report.append("PLANT DIAGNOSTIC SYSTEM - MODEL VALIDATION REPORT")
        report.append("=" * 80)
        report.append("")
        
        # Overall metrics
        report.append("OVERALL PERFORMANCE METRICS:")
        report.append(f"  Accuracy: {self.validation_results['accuracy']:.4f}")
        report.append(f"  Precision: {self.validation_results['precision']:.4f}")
        report.append(f"  Recall: {self.validation_results['recall']:.4f}")
        report.append(f"  F1-Score: {self.validation_results['f1_score']:.4f}")
        report.append("")
        
        # Per-class metrics
        report.append("PER-CLASS METRICS:")
        class_names = self.validation_results['class_names']
        precision_per_class = self.validation_results['per_class_metrics']['precision']
        recall_per_class = self.validation_results['per_class_metrics']['recall']
        f1_per_class = self.validation_results['per_class_metrics']['f1_score']
        support_per_class = self.validation_results['per_class_metrics']['support']
        
        for i, class_name in enumerate(class_names):
            report.append(f"  {class_name}:")
            report.append(f"    Precision: {precision_per_class[i]:.4f}")
            report.append(f"    Recall: {recall_per_class[i]:.4f}")
            report.append(f"    F1-Score: {f1_per_class[i]:.4f}")
            report.append(f"    Support: {support_per_class[i]}")
            report.append("")
        
        # Confusion matrix
        report.append("CONFUSION MATRIX:")
        cm = self.validation_results['confusion_matrix']
        report.append("  " + " ".join([f"{name:>8}" for name in class_names]))
        for i, row in enumerate(cm):
            report.append(f"  {class_names[i]:>8} " + " ".join([f"{val:>8}" for val in row]))
        
        report_text = "\n".join(report)
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report_text)
            logger.info(f"Validation report saved to {save_path}")
        
        return report_text


def validate_resnet_model(model, test_dataloader, class_names: List[str]) -> Dict:
    """Validate ResNet model performance."""
    validator = ModelValidator(model)
    return validator.validate_model_performance(test_dataloader, class_names)


def validate_minigpt_model(model, test_dataloader, class_names: List[str]) -> Dict:
    """Validate MiniGPT model performance."""
    validator = ModelValidator(model)
    return validator.validate_model_performance(test_dataloader, class_names)


if __name__ == "__main__":
    # Example usage
    print("Model validation utilities loaded!")
    print("Use ModelValidator class to validate and calibrate your models.")

