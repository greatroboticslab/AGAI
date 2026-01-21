#!/usr/bin/env python3
"""
Hallucination Evaluation Script

Runs the full pipeline (ResNet + MiniGPT-v2) on test images and evaluates
hallucination rates across multiple response generations.

Usage:
    python evaluate.py --num-samples 50 --runs-per-image 3
    python evaluate.py --use-holdout --num-samples 20
"""

import argparse
import json
import os
import sys
import time
import random
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Suppress warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore')

# Add parent directory for imports BEFORE any other imports
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import torch
from PIL import Image

from hallucination_checker import HallucinationChecker, HallucinationResult, compute_summary
from config import (
    PROJECT_ROOT, EVAL_CONFIG_PATH, OUTPUT_DIR, 
    TRAINING_DATA_PATH, HOLDOUT_DIR, TRAIN_DIR,
    DISEASE_CLASSES, DEFAULT_NUM_SAMPLES, DEFAULT_TEMPERATURE, MAX_NEW_TOKENS
)

# Import MiniGPT components at module level (required for import *)
# These will be loaded lazily when ModelEvaluator is instantiated
_MINIGPT_IMPORTED = False

def _import_minigpt():
    """Lazy import of MiniGPT components."""
    global _MINIGPT_IMPORTED
    if _MINIGPT_IMPORTED:
        return
    
    # These imports register models/processors with the registry
    import minigpt4.datasets.builders
    import minigpt4.models
    import minigpt4.processors
    import minigpt4.runners
    import minigpt4.tasks
    
    _MINIGPT_IMPORTED = True


class ModelEvaluator:
    """
    Handles model loading and inference for evaluation.
    """
    
    def __init__(self, gpu_id: int = 0, verbose: bool = True):
        self.gpu_id = gpu_id
        self.verbose = verbose
        self.device = f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"
        
        self.model = None
        self.vis_processor = None
        self.text_processor = None
        self.chat = None
        self.resnet_classifier = None
        
        self._load_models()
    
    def _log(self, msg: str):
        if self.verbose:
            print(f"[Evaluator] {msg}")
    
    def _load_models(self):
        """Load MiniGPT-v2 and ResNet models."""
        self._log("Loading models...")
        
        # Import MiniGPT components (lazy import)
        _import_minigpt()
        
        from minigpt4.common.config import Config
        from minigpt4.common.registry import registry
        from minigpt4.conversation.conversation import CONV_VISION_minigptv2, Chat
        
        # Load config
        class Args:
            cfg_path = str(EVAL_CONFIG_PATH)
            options = None
            gpu_id = self.gpu_id
        
        args = Args()
        cfg = Config(args)
        
        # Initialize model
        torch.cuda.set_device(self.gpu_id)
        
        model_config = cfg.model_cfg
        model_cls = registry.get_model_class(model_config.arch)
        self.model = model_cls.from_config(model_config).to(self.device)
        self.model.eval()
        
        # Initialize processors
        # Get the first available dataset config
        datasets_cfg = cfg.datasets_cfg
        dataset_name = list(datasets_cfg.keys())[0]  # e.g., 'strawberry_diagnostic'
        vis_processor_cfg = datasets_cfg[dataset_name].vis_processor
        
        # Try eval processor first, then train
        if hasattr(vis_processor_cfg, 'eval'):
            vp_cfg = vis_processor_cfg.eval
        elif hasattr(vis_processor_cfg, 'train'):
            vp_cfg = vis_processor_cfg.train
        else:
            # Fallback to default BLIP2 processor
            from omegaconf import OmegaConf
            vp_cfg = OmegaConf.create({"name": "blip2_image_eval", "image_size": 448})
        
        self.vis_processor = registry.get_processor_class(vp_cfg.name).from_config(vp_cfg)
        
        # Apply cache_position patch (compatibility fix for newer transformers)
        self._patch_cache_position(self.model)
        
        # Initialize chat
        self.conv_template = CONV_VISION_minigptv2.copy()
        self.chat = Chat(self.model, self.vis_processor, device=self.device)
    
    def _patch_cache_position(self, model):
        """Patch model to remove cache_position from forward calls (transformers compatibility)."""
        wrapped = 0
        for module in model.modules():
            f = getattr(module, "forward", None)
            if f is None:
                continue
            if getattr(f, "_drops_cachepos", False):
                continue

            def wrapped_forward(*args, __orig=f, **kwargs):
                kwargs.pop("cache_position", None)
                return __orig(*args, **kwargs)

            setattr(wrapped_forward, "_drops_cachepos", True)
            try:
                module.forward = wrapped_forward
                wrapped += 1
            except Exception:
                pass
        self._log(f"Patched {wrapped} modules for cache_position compatibility")
        
        # Load ResNet classifier
        from resnet_classifier import load_resnet, diagnose_or_none
        self.resnet_classifier = load_resnet(str(PROJECT_ROOT / "plant_diagnostic" / "models" / "resnet_strawberry.pth"))
        self._diagnose_fn = diagnose_or_none
        
        self._log("Models loaded successfully.")
    
    def classify_image(self, image_path: str) -> Tuple[str, float]:
        """Run ResNet classification on an image."""
        result = self._diagnose_fn(self.resnet_classifier, image_path)
        if result is None:
            return "unknown", 0.0
        return result["label"], result["p1"]
    
    def generate_explanation(
        self,
        image_path: str,
        disease_label: str,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = MAX_NEW_TOKENS
    ) -> str:
        """
        Generate a diagnostic explanation for an image.
        
        Args:
            image_path: Path to the image
            disease_label: The disease label (from ResNet or ground truth)
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            
        Returns:
            Generated explanation text
        """
        from minigpt4.conversation.conversation import CONV_VISION_minigptv2
        
        # Load image
        image = Image.open(image_path).convert("RGB")
        
        # Setup conversation
        conv = CONV_VISION_minigptv2.copy()
        conv.system = f"""<<SYS>>You are a plant diagnostician. The diagnosis has been determined: {disease_label.title()}
Your task is to examine the image and explain why this diagnosis is correct.
Provide a detailed medical report including:
1) Diagnosis
2) Visible cues: Describe the visual symptoms you observe
3) Recommendation: Provide treatment steps
<</SYS>>"""
        
        # Create prompt
        prompt = f"[vqa] Examine this image. The diagnosis is {disease_label.title()}. Describe the visible symptoms and provide treatment recommendations."
        
        # Process through chat
        chat_state = conv.copy()
        img_list = []
        
        # Upload and encode image (required steps)
        self.chat.upload_img(image, chat_state, img_list)
        self.chat.encode_img(img_list)
        
        # Ask question
        self.chat.ask(prompt, chat_state)
        
        # Generate response
        response = self.chat.answer(
            conv=chat_state,
            img_list=img_list,
            temperature=temperature,
            max_new_tokens=max_tokens,
            max_length=2000
        )[0]
        
        return response


def collect_test_images(
    use_holdout: bool = True,
    num_samples: int = DEFAULT_NUM_SAMPLES,
    balance_classes: bool = True
) -> List[Dict]:
    """
    Collect test images for evaluation.
    
    Args:
        use_holdout: Use holdout set (True) or sample from training (False)
        num_samples: Total number of samples to collect
        balance_classes: Whether to balance samples across classes
        
    Returns:
        List of dicts with 'path' and 'ground_truth' keys
    """
    samples = []
    
    if use_holdout:
        # Use holdout set
        print(f"Collecting samples from holdout set: {HOLDOUT_DIR}")
        for class_dir in HOLDOUT_DIR.iterdir():
            if not class_dir.is_dir():
                continue
            class_name = class_dir.name.lower()
            for img_file in class_dir.glob("*"):
                if img_file.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp"]:
                    samples.append({
                        "path": str(img_file),
                        "ground_truth": class_name
                    })
    else:
        # Sample from training data
        print(f"Sampling from training data: {TRAINING_DATA_PATH}")
        with open(TRAINING_DATA_PATH, 'r') as f:
            train_data = json.load(f)
        
        for entry in train_data:
            img_path = entry.get("image", "")
            # Extract class from path
            path_parts = Path(img_path).parts
            for part in path_parts:
                if any(cls in part.lower() for cls in DISEASE_CLASSES):
                    class_name = part.lower()
                    break
            else:
                class_name = "unknown"
            
            # Fix path if needed
            if not os.path.exists(img_path):
                # Try local path
                local_path = str(PROJECT_ROOT / "plant_diagnostic" / "data" / "train_aug" / 
                               class_name / Path(img_path).name)
                if os.path.exists(local_path):
                    img_path = local_path
            
            if os.path.exists(img_path):
                samples.append({
                    "path": img_path,
                    "ground_truth": class_name
                })
    
    # Balance classes if requested
    if balance_classes and samples:
        from collections import defaultdict
        by_class = defaultdict(list)
        for s in samples:
            by_class[s["ground_truth"]].append(s)
        
        balanced = []
        samples_per_class = num_samples // len(by_class)
        for class_name, class_samples in by_class.items():
            random.shuffle(class_samples)
            balanced.extend(class_samples[:samples_per_class])
        
        # Fill remaining from random samples
        remaining = num_samples - len(balanced)
        if remaining > 0:
            all_remaining = [s for s in samples if s not in balanced]
            random.shuffle(all_remaining)
            balanced.extend(all_remaining[:remaining])
        
        samples = balanced
    else:
        random.shuffle(samples)
        samples = samples[:num_samples]
    
    print(f"Collected {len(samples)} samples")
    return samples


def run_evaluation(
    num_samples: int = DEFAULT_NUM_SAMPLES,
    runs_per_image: int = 3,
    use_holdout: bool = True,
    use_resnet_label: bool = True,
    temperature: float = DEFAULT_TEMPERATURE,
    gpu_id: int = 0,
    output_name: Optional[str] = None
) -> Dict:
    """
    Run full hallucination evaluation.
    
    Args:
        num_samples: Number of images to evaluate
        runs_per_image: Number of response generations per image (for consistency check)
        use_holdout: Use holdout set vs training samples
        use_resnet_label: Use ResNet prediction (True) or ground truth label (False)
        temperature: Generation temperature
        gpu_id: GPU to use
        output_name: Optional name for output files
        
    Returns:
        Evaluation results dictionary
    """
    # Setup output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_name = output_name or f"eval_{timestamp}"
    
    print("=" * 60)
    print("HALLUCINATION EVALUATION")
    print("=" * 60)
    print(f"Samples: {num_samples}")
    print(f"Runs per image: {runs_per_image}")
    print(f"Use ResNet labels: {use_resnet_label}")
    print(f"Temperature: {temperature}")
    print(f"Output: {output_name}")
    print("=" * 60)
    
    # Initialize components
    print("\n[1/4] Loading models...")
    evaluator = ModelEvaluator(gpu_id=gpu_id)
    checker = HallucinationChecker()
    
    # Collect test images
    print("\n[2/4] Collecting test images...")
    test_samples = collect_test_images(
        use_holdout=use_holdout,
        num_samples=num_samples,
        balance_classes=True
    )
    
    if not test_samples:
        print("ERROR: No test samples found!")
        return {}
    
    # Run evaluation
    print(f"\n[3/4] Running evaluation on {len(test_samples)} images...")
    all_results = []
    resnet_stats = {"correct": 0, "total": 0, "confidences": []}
    
    for i, sample in enumerate(test_samples):
        img_path = sample["path"]
        gt_label = sample["ground_truth"]
        
        print(f"\n--- Image {i+1}/{len(test_samples)}: {Path(img_path).name} ---")
        print(f"Ground truth: {gt_label}")
        
        # Get ResNet prediction
        try:
            resnet_label, resnet_conf = evaluator.classify_image(img_path)
            print(f"ResNet prediction: {resnet_label} ({resnet_conf:.2%})")
            
            resnet_stats["total"] += 1
            resnet_stats["confidences"].append(resnet_conf)
            if resnet_label.lower().replace("_", " ") == gt_label.lower().replace("_", " "):
                resnet_stats["correct"] += 1
        except Exception as e:
            print(f"ResNet error: {e}")
            resnet_label = gt_label
            resnet_conf = 0.0
        
        # Determine which label to use for generation
        use_label = resnet_label if use_resnet_label else gt_label
        
        # Generate multiple responses
        for run in range(runs_per_image):
            try:
                start_time = time.time()
                response = evaluator.generate_explanation(
                    img_path, use_label, temperature=temperature
                )
                gen_time = time.time() - start_time
                
                print(f"  Run {run+1}: {len(response)} chars, {gen_time:.1f}s")
                
                # Check for hallucinations
                result = checker.check_response(
                    image_path=img_path,
                    predicted_label=use_label,
                    generated_text=response,
                    ground_truth_label=gt_label
                )
                
                # Add metadata
                result_dict = result.to_dict()
                result_dict["run_number"] = run + 1
                result_dict["resnet_confidence"] = resnet_conf
                result_dict["generation_time_s"] = gen_time
                
                all_results.append(result)
                
                if result.has_wrong_disease_hallucination:
                    print(f"    ⚠️  Wrong-disease: {result.wrong_disease_mentions[:2]}")
                if result.has_visibility_hallucination:
                    print(f"    ⚠️  Visibility: {result.invisible_region_claims[:2]}")
                    
            except Exception as e:
                print(f"  Run {run+1} ERROR: {e}")
    
    # Compute summary
    print("\n[4/4] Computing summary statistics...")
    summary = compute_summary(all_results)
    
    # Add ResNet stats
    resnet_accuracy = resnet_stats["correct"] / resnet_stats["total"] if resnet_stats["total"] > 0 else 0
    avg_conf = sum(resnet_stats["confidences"]) / len(resnet_stats["confidences"]) if resnet_stats["confidences"] else 0
    
    # Prepare output
    output = {
        "metadata": {
            "timestamp": timestamp,
            "num_samples": num_samples,
            "runs_per_image": runs_per_image,
            "total_evaluations": len(all_results),
            "use_holdout": use_holdout,
            "use_resnet_label": use_resnet_label,
            "temperature": temperature
        },
        "resnet_performance": {
            "accuracy": round(resnet_accuracy, 3),
            "avg_confidence": round(avg_conf, 3),
            "samples": resnet_stats["total"]
        },
        "hallucination_summary": summary.to_dict(),
        "individual_results": [r.to_dict() for r in all_results]
    }
    
    # Save results
    output_path = OUTPUT_DIR / f"{output_name}.json"
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to: {output_path}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Total samples: {summary.total_samples}")
    print(f"\n📊 ResNet Performance:")
    print(f"   Accuracy: {resnet_accuracy:.1%}")
    print(f"   Avg confidence: {avg_conf:.1%}")
    print(f"\n🔍 Wrong-Disease Hallucination:")
    print(f"   Count: {summary.wrong_disease_count}")
    print(f"   Rate: {summary.wrong_disease_rate:.1%}")
    if summary.most_confused_pairs:
        print(f"   Top confusions: {summary.most_confused_pairs[:3]}")
    print(f"\n👁️  Visibility Hallucination:")
    print(f"   Count: {summary.visibility_hallucination_count}")
    print(f"   Rate: {summary.visibility_hallucination_rate:.1%}")
    print(f"\n📈 Grounding:")
    print(f"   Average score: {summary.avg_grounding_score:.1%}")
    print(f"   Low grounding (<30%): {summary.low_grounding_count}")
    print(f"\n⚠️  Overall Hallucination:")
    print(f"   Any hallucination: {summary.any_hallucination_count} ({summary.any_hallucination_rate:.1%})")
    print(f"   Average score: {summary.avg_hallucination_score:.3f}")
    print("\n" + "=" * 60)
    
    return output


def main():
    parser = argparse.ArgumentParser(description="Hallucination Evaluation for Plant Diagnostic Model")
    parser.add_argument("--num-samples", type=int, default=DEFAULT_NUM_SAMPLES,
                        help=f"Number of images to evaluate (default: {DEFAULT_NUM_SAMPLES})")
    parser.add_argument("--runs-per-image", type=int, default=3,
                        help="Number of response generations per image (default: 3)")
    parser.add_argument("--use-holdout", action="store_true", default=True,
                        help="Use holdout test set (default: True)")
    parser.add_argument("--use-train", action="store_true",
                        help="Use training set samples instead of holdout")
    parser.add_argument("--use-gt-label", action="store_true",
                        help="Use ground truth label instead of ResNet prediction")
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE,
                        help=f"Generation temperature (default: {DEFAULT_TEMPERATURE})")
    parser.add_argument("--gpu", type=int, default=0,
                        help="GPU ID to use (default: 0)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output filename (without extension)")
    
    args = parser.parse_args()
    
    run_evaluation(
        num_samples=args.num_samples,
        runs_per_image=args.runs_per_image,
        use_holdout=not args.use_train,
        use_resnet_label=not args.use_gt_label,
        temperature=args.temperature,
        gpu_id=args.gpu,
        output_name=args.output
    )


if __name__ == "__main__":
    main()

