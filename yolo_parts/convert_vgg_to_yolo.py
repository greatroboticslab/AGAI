#!/usr/bin/env python3
"""
Convert VGG JSON polygon annotations to YOLO bounding box format.

Usage:
    python convert_vgg_to_yolo.py annotations.json images_dir output_dir
    
Example:
    python convert_vgg_to_yolo.py vgg_export.json to_annotate/ dataset/
"""

import json
import os
import sys
import shutil
from PIL import Image
from pathlib import Path

# Class mapping - must match strawberry_parts.yaml (7 classes, no calyx)
CLASS_MAP = {
    'leaf': 0,
    'leaves': 0,  # plural variant
    'fruit': 1,
    'flower': 2,
    'crown': 3,
    'stem': 4,
    'root': 5,
    'soil': 6,
}


def polygon_to_bbox(all_points_x, all_points_y):
    """Convert polygon points to bounding box."""
    x_min = min(all_points_x)
    x_max = max(all_points_x)
    y_min = min(all_points_y)
    y_max = max(all_points_y)
    return x_min, y_min, x_max, y_max


def bbox_to_yolo(x_min, y_min, x_max, y_max, img_width, img_height):
    """Convert bbox to YOLO format (normalized center x, y, width, height)."""
    x_center = (x_min + x_max) / 2 / img_width
    y_center = (y_min + y_max) / 2 / img_height
    width = (x_max - x_min) / img_width
    height = (y_max - y_min) / img_height
    return x_center, y_center, width, height


def convert_vgg_to_yolo(json_path, images_dir, output_dir, train_split=0.8):
    """Convert VGG JSON annotations to YOLO format."""
    
    # Load VGG JSON
    with open(json_path) as f:
        vgg_data = json.load(f)
    
    # Create output directories
    for split in ['train', 'val']:
        os.makedirs(f"{output_dir}/images/{split}", exist_ok=True)
        os.makedirs(f"{output_dir}/labels/{split}", exist_ok=True)
    
    # Process each image
    all_images = list(vgg_data.keys())
    n_train = int(len(all_images) * train_split)
    
    stats = {'total': 0, 'train': 0, 'val': 0, 'annotations': 0, 'skipped': 0}
    class_counts = {name: 0 for name in CLASS_MAP}
    
    for i, key in enumerate(all_images):
        entry = vgg_data[key]
        filename = entry.get('filename', key)
        
        # Find image file
        img_path = None
        for ext in ['', '.jpg', '.jpeg', '.png', '.webp']:
            test_path = os.path.join(images_dir, filename + ext if ext and not filename.endswith(ext) else filename)
            if os.path.exists(test_path):
                img_path = test_path
                break
        
        if not img_path:
            print(f"  Warning: Image not found: {filename}")
            stats['skipped'] += 1
            continue
        
        # Get image dimensions
        with Image.open(img_path) as img:
            img_width, img_height = img.size
        
        # Determine split
        split = 'train' if i < n_train else 'val'
        stats[split] += 1
        stats['total'] += 1
        
        # Process annotations
        yolo_lines = []
        regions = entry.get('regions', [])
        
        # Handle both list and dict formats
        if isinstance(regions, dict):
            regions = list(regions.values())
        
        for region in regions:
            shape = region.get('shape_attributes', {})
            attrs = region.get('region_attributes', {})
            
            # Get label
            label = attrs.get('label', attrs.get('class', attrs.get('name', '')))
            label = label.lower().strip()
            
            if label not in CLASS_MAP:
                print(f"  Warning: Unknown label '{label}' in {filename}")
                continue
            
            class_id = CLASS_MAP[label]
            class_counts[label] += 1
            
            # Get bounding box from polygon or rect
            shape_type = shape.get('name', '')
            
            if shape_type == 'polygon':
                all_x = shape.get('all_points_x', [])
                all_y = shape.get('all_points_y', [])
                if not all_x or not all_y:
                    continue
                x_min, y_min, x_max, y_max = polygon_to_bbox(all_x, all_y)
            
            elif shape_type == 'rect':
                x_min = shape.get('x', 0)
                y_min = shape.get('y', 0)
                x_max = x_min + shape.get('width', 0)
                y_max = y_min + shape.get('height', 0)
            
            else:
                print(f"  Warning: Unknown shape type '{shape_type}' in {filename}")
                continue
            
            # Convert to YOLO format
            x_c, y_c, w, h = bbox_to_yolo(x_min, y_min, x_max, y_max, img_width, img_height)
            
            # Validate
            if not (0 <= x_c <= 1 and 0 <= y_c <= 1 and 0 < w <= 1 and 0 < h <= 1):
                print(f"  Warning: Invalid bbox in {filename}: {x_c:.3f} {y_c:.3f} {w:.3f} {h:.3f}")
                continue
            
            yolo_lines.append(f"{class_id} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}")
            stats['annotations'] += 1
        
        if not yolo_lines:
            print(f"  Warning: No valid annotations for {filename}")
            continue
        
        # Copy image
        img_ext = Path(img_path).suffix
        new_img_name = Path(filename).stem + img_ext
        shutil.copy(img_path, f"{output_dir}/images/{split}/{new_img_name}")
        
        # Write YOLO label file
        label_name = Path(filename).stem + '.txt'
        with open(f"{output_dir}/labels/{split}/{label_name}", 'w') as f:
            f.write('\n'.join(yolo_lines))
    
    # Print summary
    print(f"\n{'='*50}")
    print(f"Conversion Complete!")
    print(f"{'='*50}")
    print(f"Total images: {stats['total']} (train: {stats['train']}, val: {stats['val']})")
    print(f"Total annotations: {stats['annotations']}")
    print(f"Skipped: {stats['skipped']}")
    print(f"\nClass distribution:")
    for name, count in sorted(class_counts.items(), key=lambda x: -x[1]):
        if count > 0:
            print(f"  {name}: {count}")
    print(f"\nOutput saved to: {output_dir}/")
    print(f"\nNext step: python train_yolo.py")


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python convert_vgg_to_yolo.py <vgg_json> <images_dir> <output_dir>")
        print("\nExample:")
        print("  python convert_vgg_to_yolo.py annotations.json to_annotate/ dataset/")
        sys.exit(1)
    
    json_path = sys.argv[1]
    images_dir = sys.argv[2]
    output_dir = sys.argv[3]
    
    convert_vgg_to_yolo(json_path, images_dir, output_dir)
