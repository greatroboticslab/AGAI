#!/usr/bin/env python3
"""
Train YOLOv8 for strawberry plant part detection.

Usage:
    python train_yolo.py                    # Train from scratch
    python train_yolo.py --resume           # Resume training
    python train_yolo.py --test image.jpg   # Test on single image
"""

import argparse
import os
from pathlib import Path


def train(resume: bool = False):
    """Train YOLOv8 model."""
    try:
        from ultralytics import YOLO
    except ImportError:
        print("Installing ultralytics...")
        os.system("pip install ultralytics")
        from ultralytics import YOLO
    
    # Paths
    data_yaml = "/data/AGAI/MiniGPT-4/yolo_parts/strawberry_parts.yaml"
    output_dir = "/data/AGAI/MiniGPT-4/yolo_parts/models"
    
    if resume:
        # Resume from last checkpoint
        model = YOLO(f"{output_dir}/strawberry_parts/weights/last.pt")
    else:
        # Start from pretrained YOLOv8 nano (fast, good for this task)
        model = YOLO("yolov8n.pt")
    
    # Train
    results = model.train(
        data=data_yaml,
        epochs=100,
        imgsz=640,
        batch=16,
        patience=20,  # Early stopping
        project=output_dir,
        name="strawberry_parts",
        exist_ok=True,
        pretrained=True,
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        augment=True,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=10,
        translate=0.1,
        scale=0.5,
        fliplr=0.5,
        mosaic=0.5,
    )
    
    print(f"\nTraining complete!")
    print(f"Best model: {output_dir}/strawberry_parts/weights/best.pt")
    return results


def test(image_path: str):
    """Test model on a single image."""
    from ultralytics import YOLO
    
    model_path = "/data/AGAI/MiniGPT-4/yolo_parts/models/strawberry_parts/weights/best.pt"
    
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        print("Train the model first with: python train_yolo.py")
        return
    
    model = YOLO(model_path)
    results = model(image_path)
    
    # Print detections
    for r in results:
        print(f"\nDetected parts in {image_path}:")
        for box in r.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            name = model.names[cls]
            print(f"  - {name}: {conf:.2%}")
        
        # Save annotated image
        output_path = Path(image_path).stem + "_detected.jpg"
        r.save(filename=output_path)
        print(f"\nSaved annotated image: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Train YOLO for strawberry parts")
    parser.add_argument("--resume", action="store_true", help="Resume training")
    parser.add_argument("--test", type=str, help="Test on image")
    args = parser.parse_args()
    
    if args.test:
        test(args.test)
    else:
        train(resume=args.resume)


if __name__ == "__main__":
    main()
