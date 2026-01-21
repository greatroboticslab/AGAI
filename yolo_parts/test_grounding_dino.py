#!/usr/bin/env python3
"""
Test Grounding DINO auto-labeling on sample images.
See if it can detect strawberry plant parts.
"""

import os
import sys
import torch
from PIL import Image
import random

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')

def test_groundingdino():
    try:
        from groundingdino.util.inference import load_model, load_image, predict, annotate
    except ImportError:
        print("Trying alternative import...")
        try:
            import groundingdino
            print(f"groundingdino version: {groundingdino.__version__}")
            print("Module found but inference not available - need full install")
            return False
        except:
            print("groundingdino not properly installed")
            return False
    
    print("Grounding DINO loaded!")
    
    # Download model weights if needed
    model_config = "groundingdino/config/GroundingDINO_SwinT_OGC.py"
    model_weights = "groundingdino_swint_ogc.pth"
    
    if not os.path.exists(model_weights):
        print(f"Downloading model weights...")
        os.system(f"wget -q https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/{model_weights}")
    
    # Load model
    model = load_model(model_config, model_weights)
    
    # Test on a few images
    test_dir = "/data/AGAI/MiniGPT-4/yolo_parts/to_annotate"
    images = [f for f in os.listdir(test_dir) if f.endswith(('.jpg', '.png', '.jpeg', '.webp'))]
    
    # Sample 5 random images
    test_images = random.sample(images, min(5, len(images)))
    
    # Prompts for plant parts
    text_prompt = "leaf . fruit . flower . stem . root . crown . berry . soil"
    
    for img_name in test_images:
        img_path = os.path.join(test_dir, img_name)
        
        image_source, image = load_image(img_path)
        
        boxes, logits, phrases = predict(
            model=model,
            image=image,
            caption=text_prompt,
            box_threshold=0.25,
            text_threshold=0.25
        )
        
        print(f"\n{img_name}:")
        print(f"  Expected parts: {img_name.split('_')[2:]}")
        print(f"  Detected: {phrases} (confidence: {[f'{l:.2f}' for l in logits.tolist()]})")
        
        # Save annotated image
        annotated = annotate(image_source, boxes, logits, phrases)
        out_path = f"/data/AGAI/MiniGPT-4/yolo_parts/test_output_{img_name}"
        Image.fromarray(annotated).save(out_path)
        print(f"  Saved: {out_path}")
    
    return True

if __name__ == "__main__":
    test_groundingdino()
