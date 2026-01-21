#!/usr/bin/env python3
"""
Test different prompt styles to see which produces more image-specific responses.

Compares:
1. No system prompt (matches training format)
2. Current system prompt (elaborate instructions)
3. Minimal system prompt

Then measures symptom overlap with training data.
"""

import os
import sys
import json
import re
import time
from pathlib import Path
from datetime import datetime

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from PIL import Image


def load_models():
    """Load MiniGPT-v2 model."""
    print("[1/3] Loading models...")
    
    # Import MiniGPT components
    import minigpt4.datasets.builders
    import minigpt4.models
    import minigpt4.processors
    import minigpt4.runners
    import minigpt4.tasks
    
    from minigpt4.common.config import Config
    from minigpt4.common.registry import registry
    from minigpt4.conversation.conversation import CONV_VISION_minigptv2, Chat
    
    class Args:
        cfg_path = "/data/AGAI/MiniGPT-4/eval_configs/minigptv2_eval.yaml"
        options = None
        gpu_id = 0
    
    cfg = Config(Args())
    
    device = "cuda:0"
    torch.cuda.set_device(0)
    
    model_config = cfg.model_cfg
    model_cls = registry.get_model_class(model_config.arch)
    model = model_cls.from_config(model_config).to(device)
    model.eval()
    
    # Get vis processor
    datasets_cfg = cfg.datasets_cfg
    dataset_name = list(datasets_cfg.keys())[0]
    vis_processor_cfg = datasets_cfg[dataset_name].vis_processor
    vp_cfg = vis_processor_cfg.eval if hasattr(vis_processor_cfg, 'eval') else vis_processor_cfg.train
    vis_processor = registry.get_processor_class(vp_cfg.name).from_config(vp_cfg)
    
    # Patch cache_position
    for module in model.modules():
        f = getattr(module, "forward", None)
        if f is None or getattr(f, "_drops_cachepos", False):
            continue
        def wrapped_forward(*args, __orig=f, **kwargs):
            kwargs.pop("cache_position", None)
            return __orig(*args, **kwargs)
        setattr(wrapped_forward, "_drops_cachepos", True)
        try:
            module.forward = wrapped_forward
        except:
            pass
    
    chat = Chat(model, vis_processor, device=device)
    
    # Load ResNet
    from resnet_classifier import load_resnet, diagnose_or_none
    resnet = load_resnet("/data/AGAI/MiniGPT-4/plant_diagnostic/models/resnet_strawberry.pth")
    
    print("  Models loaded.")
    return chat, vis_processor, resnet, diagnose_or_none, CONV_VISION_minigptv2


def generate_response(chat, vis_processor, conv_template, image, system_prompt, user_prompt, device="cuda:0"):
    """Generate a response with specific prompts."""
    from minigpt4.conversation.conversation import CONV_VISION_minigptv2
    
    conv = CONV_VISION_minigptv2.copy()
    conv.system = system_prompt
    
    chat_state = conv.copy()
    img_list = []
    
    chat.upload_img(image, chat_state, img_list)
    chat.encode_img(img_list)
    
    if user_prompt:
        chat.ask(user_prompt, chat_state)
    
    response = chat.answer(
        conv=chat_state,
        img_list=img_list,
        temperature=0.2,
        max_new_tokens=500,
        max_length=2000
    )[0]
    
    return response


def extract_symptoms(text):
    """Extract symptom keywords from text."""
    symptoms = set()
    text_lower = text.lower()
    
    checks = [
        ('cracked', 'cracked'),
        ('limp', 'limp'),
        ('brown', 'brown'),
        ('crisp', 'crisp'),
        ('powdery', 'powdery'),
        ('dry', 'dry'),
        ('wilted', 'wilted'),
        ('curl', 'curled'),
        ('shrivel', 'shriveled'),
        ('yellow', 'yellow'),
        ('pale', 'pale'),
        ('moist', 'moist'),
        ('soft', 'soft'),
        ('mushy', 'mushy'),
        ('fuzz', 'fuzzy'),
        ('gray', 'gray'),
        ('spot', 'spots'),
        ('rot', 'rot'),
        ('black', 'black'),
        ('dull', 'dull'),
        ('glossy', 'glossy'),
        ('droop', 'drooping'),
        ('hang', 'hanging'),
        ('upright', 'upright'),
        ('prostrate', 'prostrate'),
        ('flaccid', 'flaccid'),
    ]
    
    for keyword, label in checks:
        if keyword in text_lower:
            symptoms.add(label)
    
    return symptoms


def load_training_data():
    """Load training data for comparison."""
    train_path = "/data/AGAI/MiniGPT-4/plant_diagnostic/datasets/stage2_train_7class_fixed.json"
    with open(train_path, 'r') as f:
        data = json.load(f)
    
    train_map = {}
    for entry in data:
        filename = Path(entry['image']).stem
        base = re.sub(r'_aug\d+', '', filename)
        for conv in entry.get('conversations', []):
            if conv.get('from') == 'assistant':
                if base not in train_map:
                    train_map[base] = conv.get('value', '')
                break
    
    return train_map


def run_comparison():
    """Run comparison of different prompt styles."""
    
    print("=" * 70)
    print("PROMPT STYLE COMPARISON TEST")
    print("=" * 70)
    
    # Load models
    chat, vis_processor, resnet, diagnose_fn, conv_template = load_models()
    train_map = load_training_data()
    
    # Define prompt styles to test
    prompt_styles = {
        "no_prompt": {
            "system": "",
            "user": ""  # Just the image, like training
        },
        "minimal": {
            "system": "",
            "user": "Describe what you see in this plant image."
        },
        "current": {
            "system": """<<SYS>>You are a plant diagnostician. 
Provide a detailed medical report including:
1) Diagnosis
2) Visible cues: Describe the visual symptoms you observe
3) Recommendation: Provide treatment steps
<</SYS>>""",
            "user": "Examine this image and provide a diagnosis with visible symptoms."
        },
        "training_style": {
            "system": "",
            "user": ""  # Empty - matches training exactly
        }
    }
    
    # Get holdout images
    holdout_dir = Path("/data/AGAI/MiniGPT-4/plant_diagnostic/data/holdout")
    test_images = []
    for class_dir in holdout_dir.iterdir():
        if not class_dir.is_dir():
            continue
        for img_path in list(class_dir.glob("*"))[:2]:  # 2 per class
            if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.webp']:
                test_images.append({
                    "path": str(img_path),
                    "class": class_dir.name,
                    "filename": img_path.stem
                })
    
    print(f"\n[2/3] Testing {len(test_images)} images with {len(prompt_styles)} prompt styles...")
    
    results = {style: [] for style in prompt_styles}
    
    for i, img_info in enumerate(test_images[:6]):  # Limit to 6 for speed
        print(f"\n--- Image {i+1}: {img_info['filename']} ({img_info['class']}) ---")
        
        image = Image.open(img_info['path']).convert('RGB')
        
        # Get training reference
        train_ref = None
        for train_name, train_text in train_map.items():
            if img_info['filename'] in train_name or train_name in img_info['filename']:
                train_ref = train_text
                break
        
        if train_ref:
            train_symptoms = extract_symptoms(train_ref)
            print(f"  Training symptoms: {sorted(train_symptoms)}")
        else:
            train_symptoms = set()
            print(f"  No training reference found")
        
        for style_name, prompts in prompt_styles.items():
            start = time.time()
            
            response = generate_response(
                chat, vis_processor, conv_template, image,
                prompts["system"], prompts["user"]
            )
            
            elapsed = time.time() - start
            model_symptoms = extract_symptoms(response)
            
            # Calculate overlap
            if train_symptoms:
                overlap = len(train_symptoms & model_symptoms) / len(train_symptoms)
            else:
                overlap = 0
            
            results[style_name].append({
                "image": img_info['filename'],
                "class": img_info['class'],
                "response_length": len(response),
                "symptoms": list(model_symptoms),
                "train_overlap": overlap,
                "response_preview": response[:200]
            })
            
            print(f"  {style_name}: {len(model_symptoms)} symptoms, {overlap*100:.0f}% overlap, {elapsed:.1f}s")
    
    # Compute summary
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    
    summary = {}
    for style_name, style_results in results.items():
        overlaps = [r['train_overlap'] for r in style_results if r['train_overlap'] > 0]
        symptom_counts = [len(r['symptoms']) for r in style_results]
        
        summary[style_name] = {
            "avg_overlap": sum(overlaps) / len(overlaps) if overlaps else 0,
            "avg_symptoms": sum(symptom_counts) / len(symptom_counts) if symptom_counts else 0,
            "samples": len(style_results)
        }
        
        print(f"\n{style_name}:")
        print(f"  Average training overlap: {summary[style_name]['avg_overlap']*100:.1f}%")
        print(f"  Average symptoms detected: {summary[style_name]['avg_symptoms']:.1f}")
    
    # Save detailed results
    output = {
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "summary": summary,
        "detailed_results": results
    }
    
    output_path = Path(__file__).parent / "results" / "prompt_comparison.json"
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\nDetailed results saved to: {output_path}")
    
    # Determine winner
    best_style = max(summary.items(), key=lambda x: x[1]['avg_overlap'])
    print(f"\n🏆 BEST STYLE: {best_style[0]} ({best_style[1]['avg_overlap']*100:.1f}% overlap)")
    
    return output


if __name__ == "__main__":
    run_comparison()

