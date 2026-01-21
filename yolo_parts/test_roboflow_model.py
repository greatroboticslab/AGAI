#!/usr/bin/env python3
"""
Test your deployed Roboflow strawberry parts model

Usage:
    python test_roboflow_model.py path/to/image.jpg
    python test_roboflow_model.py  # Uses a sample image from dataset
"""

import sys
import os
from pathlib import Path
import requests
import base64
import json

# Your Roboflow credentials
API_KEY = "mefICSeTxIoUcveWqE9h"
WORKSPACE = "mtsu-2h73y"
PROJECT = "agai-g-w"
MODEL_NAME = "strawberry-parts-seg"


def run_inference(image_path: str, save_output: bool = True):
    """Run inference on an image using Roboflow hosted API."""
    
    print(f"\n🔍 Running inference on: {image_path}")
    
    # Method 1: Try using inference SDK (for versionless models)
    try:
        from inference_sdk import InferenceHTTPClient
        
        client = InferenceHTTPClient(
            api_url="https://detect.roboflow.com",
            api_key=API_KEY
        )
        
        # For versionless models, use workspace/model_name format
        result = client.infer(image_path, model_id=f"{WORKSPACE}/{MODEL_NAME}")
        prediction = result
        
    except ImportError:
        # Method 2: Use REST API directly
        print("Using REST API (inference_sdk not installed)...")
        
        # Read and encode image
        with open(image_path, "rb") as f:
            image_data = base64.b64encode(f.read()).decode("utf-8")
        
        # Call Roboflow API
        url = f"https://detect.roboflow.com/{PROJECT}/{MODEL_NAME}"
        params = {
            "api_key": API_KEY,
            "confidence": 40,
        }
        
        response = requests.post(
            url,
            params=params,
            data=image_data,
            headers={"Content-Type": "application/x-www-form-urlencoded"}
        )
        
        if response.status_code != 200:
            print(f"API Error: {response.status_code} - {response.text}")
            return None
            
        prediction = response.json()
    
    # Display results
    print(f"\n📊 Detection Results:")
    print(f"   Image: {prediction.get('image', {}).get('width', '?')}x{prediction.get('image', {}).get('height', '?')}")
    print(f"   Objects detected: {len(prediction.get('predictions', []))}")
    
    if prediction.get('predictions'):
        print(f"\n   Detected parts:")
        class_counts = {}
        for pred in prediction['predictions']:
            cls = pred['class']
            conf = pred['confidence']
            class_counts[cls] = class_counts.get(cls, 0) + 1
            print(f"   - {cls}: {conf*100:.1f}% confidence")
        
        print(f"\n   Summary:")
        for cls, count in sorted(class_counts.items()):
            print(f"   - {cls}: {count} instance(s)")
    else:
        print("   No objects detected. Try a different image or lower confidence threshold.")
    
    # Save annotated image
    if save_output:
        output_path = Path(image_path).stem + "_roboflow_result.jpg"
        
        # Try to save visualization
        try:
            # Use roboflow's built-in visualization
            model.predict(image_path, confidence=40).save(output_path)
            print(f"\n💾 Annotated image saved to: {output_path}")
        except Exception as e:
            print(f"\n⚠️  Could not save annotated image: {e}")
    
    return prediction


def main():
    # Determine image path
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        # Use a sample image from the dataset
        sample_images = list(Path("/data/AGAI/MiniGPT-4/yolo_parts/dataset/images/val").glob("*.jpg"))
        if sample_images:
            image_path = str(sample_images[0])
            print(f"No image specified, using sample: {image_path}")
        else:
            print("Usage: python test_roboflow_model.py path/to/image.jpg")
            sys.exit(1)
    
    if not os.path.exists(image_path):
        print(f"Error: Image not found: {image_path}")
        sys.exit(1)
    
    run_inference(image_path)


if __name__ == "__main__":
    main()
