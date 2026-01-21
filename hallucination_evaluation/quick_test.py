#!/usr/bin/env python3
"""
Quick Test Script for Hallucination Evaluation

Runs a minimal evaluation on 5 images to verify the pipeline works.
This is much faster than the full evaluation.

Usage:
    python quick_test.py
    python quick_test.py --images 10
"""

import argparse
import sys
import os
from pathlib import Path

# Suppress warnings before other imports
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import HOLDOUT_DIR, OUTPUT_DIR


def quick_test(num_images: int = 5, verbose: bool = True):
    """
    Run a quick test of the hallucination evaluation pipeline.
    
    Args:
        num_images: Number of images to test
        verbose: Print detailed output
    """
    print("=" * 60)
    print("QUICK HALLUCINATION EVALUATION TEST")
    print("=" * 60)
    print(f"Testing with {num_images} images...")
    
    # Collect a few test images
    test_images = []
    for class_dir in HOLDOUT_DIR.iterdir():
        if not class_dir.is_dir():
            continue
        for img in class_dir.glob("*"):
            if img.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp"]:
                test_images.append({
                    "path": str(img),
                    "class": class_dir.name
                })
                if len(test_images) >= num_images:
                    break
        if len(test_images) >= num_images:
            break
    
    if not test_images:
        print("ERROR: No test images found!")
        return False
    
    print(f"\nFound {len(test_images)} test images:")
    for img in test_images:
        print(f"  - {img['class']}: {Path(img['path']).name}")
    
    # Import checker (lightweight, no model needed)
    print("\n[1/3] Testing hallucination checker...")
    try:
        from hallucination_checker import HallucinationChecker
        checker = HallucinationChecker()
        print("  ✓ Checker loaded successfully")
        
        # Test with a sample response
        test_response = """
        Diagnosis: Drought. The plant shows curled leaves with dry margins.
        The soil appears cracked and dusty. Petioles are limp and drooping.
        Recommended: Water immediately and apply mulch.
        """
        
        result = checker.check_response(
            image_path=test_images[0]["path"],
            predicted_label=test_images[0]["class"],
            generated_text=test_response
        )
        
        print(f"  ✓ Sample check completed:")
        print(f"    - Wrong-disease hallucination: {result.has_wrong_disease_hallucination}")
        print(f"    - Grounding score: {result.grounding_score:.2f}")
        print(f"    - Total hallucination score: {result.total_hallucination_score:.2f}")
        
    except Exception as e:
        print(f"  ✗ Checker test failed: {e}")
        return False
    
    # Try loading models (this is the heavy part)
    print("\n[2/3] Testing model loading (this may take a moment)...")
    try:
        # First check if CUDA is available
        import torch
        if not torch.cuda.is_available():
            print("  ⚠ CUDA not available, skipping model test")
            skip_model = True
        else:
            print(f"  ✓ CUDA available: {torch.cuda.get_device_name(0)}")
            skip_model = False
            
        if not skip_model:
            from evaluate import ModelEvaluator
            evaluator = ModelEvaluator(gpu_id=0, verbose=False)
            print("  ✓ Models loaded successfully")
            
            # Run one inference
            print("\n[3/3] Testing inference...")
            img = test_images[0]
            
            # ResNet classification
            label, conf = evaluator.classify_image(img["path"])
            print(f"  ✓ ResNet: {label} ({conf:.1%})")
            
            # MiniGPT generation
            response = evaluator.generate_explanation(img["path"], label)
            print(f"  ✓ MiniGPT: Generated {len(response)} characters")
            
            # Check for hallucinations
            result = checker.check_response(
                image_path=img["path"],
                predicted_label=label,
                generated_text=response
            )
            
            print(f"\n  📊 Evaluation Result:")
            print(f"     Image: {Path(img['path']).name}")
            print(f"     Ground truth: {img['class']}")
            print(f"     Predicted: {label}")
            print(f"     Response preview: {response[:150]}...")
            print(f"     Wrong-disease hallucination: {result.has_wrong_disease_hallucination}")
            if result.wrong_disease_mentions:
                print(f"       Mentions: {result.wrong_disease_mentions[:3]}")
            print(f"     Visibility hallucination: {result.has_visibility_hallucination}")
            print(f"     Grounding score: {result.grounding_score:.2f}")
            print(f"     Total score: {result.total_hallucination_score:.2f}")
            
    except Exception as e:
        print(f"  ✗ Model test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "=" * 60)
    print("✓ QUICK TEST COMPLETED SUCCESSFULLY")
    print("=" * 60)
    print("\nTo run full evaluation:")
    print("  python evaluate.py --num-samples 50 --runs-per-image 3")
    
    return True


def main():
    parser = argparse.ArgumentParser(description="Quick test of hallucination evaluation")
    parser.add_argument("--images", type=int, default=5, help="Number of images to test")
    parser.add_argument("--quiet", action="store_true", help="Reduce output")
    args = parser.parse_args()
    
    success = quick_test(num_images=args.images, verbose=not args.quiet)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

