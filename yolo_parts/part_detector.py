#!/usr/bin/env python3
"""
Strawberry Plant Part Detector using YOLOv8.

Detects: leaf, fruit, flower, crown, stem
Used to constrain MiniGPT's descriptions to only visible parts.
"""

import os
from typing import List, Dict, Optional, Tuple
from PIL import Image
import numpy as np

# Class names matching strawberry_parts.yaml (7 classes, no calyx)
CLASS_NAMES = ['leaf', 'fruit', 'flower', 'crown', 'stem', 'root', 'soil']

# Confidence threshold for detections
DEFAULT_CONFIDENCE = 0.25


class StrawberryPartDetector:
    """Detect strawberry plant parts using YOLOv8."""
    
    def __init__(
        self,
        model_path: str = "/data/AGAI/MiniGPT-4/yolo_parts/models/strawberry_parts/weights/best.pt",
        confidence: float = DEFAULT_CONFIDENCE,
        device: str = "cuda:0"
    ):
        self.model_path = model_path
        self.confidence = confidence
        self.device = device
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load YOLO model."""
        if not os.path.exists(self.model_path):
            print(f"[PartDetector] Model not found: {self.model_path}")
            print("[PartDetector] Running without part detection (all parts assumed visible)")
            return
        
        try:
            from ultralytics import YOLO
            self.model = YOLO(self.model_path)
            print(f"[PartDetector] Loaded model from {self.model_path}")
        except Exception as e:
            print(f"[PartDetector] Failed to load model: {e}")
            self.model = None
    
    def detect(self, image: Image.Image) -> Dict[str, float]:
        """
        Detect plant parts in image.
        
        Args:
            image: PIL Image
            
        Returns:
            Dict mapping part name to confidence score.
            e.g., {'leaf': 0.95, 'fruit': 0.87}
        """
        if self.model is None:
            # Return all parts as visible if no model
            return {name: 1.0 for name in CLASS_NAMES}
        
        # Run detection
        results = self.model(image, conf=self.confidence, verbose=False)
        
        # Aggregate detections (take max confidence per class)
        detected = {}
        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                name = CLASS_NAMES[cls] if cls < len(CLASS_NAMES) else f"class_{cls}"
                
                if name not in detected or conf > detected[name]:
                    detected[name] = conf
        
        return detected
    
    def get_visible_parts(self, image: Image.Image) -> List[str]:
        """Get list of visible part names."""
        detected = self.detect(image)
        return list(detected.keys())
    
    def get_not_visible_parts(self, image: Image.Image) -> List[str]:
        """Get list of parts NOT detected in image."""
        detected = self.detect(image)
        return [name for name in CLASS_NAMES if name not in detected]
    
    def get_prompt_constraint(self, image: Image.Image) -> str:
        """
        Generate prompt constraint based on detected parts.
        
        Returns a string to prepend to MiniGPT prompt that tells it
        what parts are/aren't visible.
        """
        detected = self.detect(image)
        visible = list(detected.keys())
        not_visible = [name for name in CLASS_NAMES if name not in detected]
        
        if not not_visible:
            # All parts visible, no constraint needed
            return ""
        
        if not visible:
            # Nothing detected (unlikely), don't constrain
            return ""
        
        # Build constraint message
        visible_str = ", ".join(visible)
        not_visible_str = ", ".join(not_visible)
        
        constraint = (
            f"VISIBLE in this image: {visible_str}. "
            f"NOT visible: {not_visible_str}. "
            f"Describe symptoms ONLY on visible parts ({visible_str}). "
            f"Do NOT mention {not_visible_str}."
        )
        
        return constraint
    
    def filter_response(self, response: str, image: Image.Image) -> str:
        """
        Filter MiniGPT response to remove mentions of non-visible parts.
        
        Args:
            response: MiniGPT's generated text
            image: The image that was analyzed
            
        Returns:
            Filtered response with sentences about non-visible parts removed
        """
        not_visible = self.get_not_visible_parts(image)
        
        if not not_visible:
            return response
        
        # Split into sentences
        sentences = response.replace('\n', ' ').split('. ')
        
        # Filter sentences that mention non-visible parts
        filtered = []
        for sentence in sentences:
            sentence_lower = sentence.lower()
            mentions_hidden = any(part in sentence_lower for part in not_visible)
            
            if not mentions_hidden:
                filtered.append(sentence)
        
        return '. '.join(filtered)


# Singleton instance for easy import
_detector = None


def get_detector() -> StrawberryPartDetector:
    """Get or create singleton detector instance."""
    global _detector
    if _detector is None:
        _detector = StrawberryPartDetector()
    return _detector


def detect_parts(image: Image.Image) -> Dict[str, float]:
    """Convenience function to detect parts."""
    return get_detector().detect(image)


def get_prompt_constraint(image: Image.Image) -> str:
    """Convenience function to get prompt constraint."""
    return get_detector().get_prompt_constraint(image)


def filter_response(response: str, image: Image.Image) -> str:
    """Convenience function to filter response."""
    return get_detector().filter_response(response, image)


# Test
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python part_detector.py <image_path>")
        sys.exit(1)
    
    img_path = sys.argv[1]
    img = Image.open(img_path).convert("RGB")
    
    detector = StrawberryPartDetector()
    
    print(f"\nAnalyzing: {img_path}")
    print(f"Detected parts: {detector.get_visible_parts(img)}")
    print(f"Not visible: {detector.get_not_visible_parts(img)}")
    print(f"\nPrompt constraint:")
    print(detector.get_prompt_constraint(img))
