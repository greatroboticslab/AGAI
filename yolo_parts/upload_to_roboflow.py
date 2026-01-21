#!/usr/bin/env python3
"""
Upload YOLO strawberry parts model to Roboflow

Usage:
    python upload_to_roboflow.py --api-key YOUR_API_KEY --workspace YOUR_WORKSPACE --project YOUR_PROJECT

Before running:
    pip install --upgrade roboflow>=1.1.53
"""

import argparse
import os
from pathlib import Path


def upload_model(api_key: str, workspace: str, project_id: str, model_name: str, 
                 model_type: str = "yolov8", use_segmentation: bool = True):
    """Upload YOLO model weights to Roboflow."""
    
    from roboflow import Roboflow
    
    # Initialize Roboflow
    rf = Roboflow(api_key=api_key)
    ws = rf.workspace(workspace)
    
    # Determine model path based on type
    base_path = Path(__file__).parent / "models"
    
    if use_segmentation:
        model_path = base_path / "strawberry_seg_50ep"
        print(f"Using segmentation model: {model_path}")
    else:
        model_path = base_path / "strawberry_parts_test"
        print(f"Using detection model: {model_path}")
    
    weights_path = model_path / "weights" / "best.pt"
    
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights not found at: {weights_path}")
    
    print(f"\n📦 Uploading model to Roboflow...")
    print(f"   Workspace: {workspace}")
    print(f"   Project: {project_id}")
    print(f"   Model name: {model_name}")
    print(f"   Model type: {model_type}")
    print(f"   Weights: {weights_path}")
    
    # Upload using versionless deploy
    try:
        ws.deploy_model(
            model_type=model_type,
            model_path=str(model_path),
            project_ids=[project_id],
            model_name=model_name,
            filename="weights/best.pt"
        )
        print("\n✅ Model uploaded successfully!")
        print(f"\nView your model at: https://app.roboflow.com/{workspace}/{project_id}/models")
        
    except AttributeError:
        # Fallback for older roboflow versions - use versioned upload
        print("\n⚠️  Versionless deploy not available. Using versioned upload...")
        project = rf.workspace(workspace).project(project_id)
        version = project.version(1)  # You may need to adjust version number
        version.deploy(model_type=model_type, model_path=str(model_path))
        print("\n✅ Model uploaded successfully (versioned)!")


def main():
    parser = argparse.ArgumentParser(description="Upload YOLO model to Roboflow")
    parser.add_argument("--api-key", required=True, help="Your Roboflow API key")
    parser.add_argument("--workspace", required=True, help="Your Roboflow workspace name/ID")
    parser.add_argument("--project", required=True, help="Your Roboflow project ID")
    parser.add_argument("--model-name", default="strawberry-parts-detector", 
                        help="Name for the model (default: strawberry-parts-detector)")
    parser.add_argument("--detection", action="store_true", 
                        help="Use detection model instead of segmentation")
    parser.add_argument("--model-type", default="yolov8", 
                        choices=["yolov8", "yolov8-seg"],
                        help="Model type for Roboflow (default: yolov8)")
    
    args = parser.parse_args()
    
    # Auto-detect model type based on which model is being used
    model_type = args.model_type
    if not args.detection and model_type == "yolov8":
        model_type = "yolov8-seg"  # Segmentation model
    
    upload_model(
        api_key=args.api_key,
        workspace=args.workspace,
        project_id=args.project,
        model_name=args.model_name,
        model_type=model_type,
        use_segmentation=not args.detection
    )


if __name__ == "__main__":
    main()
