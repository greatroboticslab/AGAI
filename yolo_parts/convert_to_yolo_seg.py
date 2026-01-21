#!/usr/bin/env python3
"""
Convert all annotations to YOLO Segmentation format.
Preserves polygon precision - no conversion to bounding boxes.

YOLO Seg format: class_id x1 y1 x2 y2 x3 y3 ... xn yn
All coordinates normalized to 0-1.
"""

import json
import os
import shutil
from PIL import Image
from pathlib import Path

# Class mapping - matches strawberry_parts.yaml (7 classes, no calyx)
CLASS_MAP = {
    'leaf': 0, 'leaves': 0,
    'fruit': 1,
    'flower': 2,
    'crown': 3,
    'stem': 4,
    'root': 5,
    'soil': 6,
}

# Rectangle YOLO class mapping (from the 2 txt files)
RECT_CLASS_MAP = {
    0: 0,  # class_0 -> leaf
    1: 1,  # class_1 -> fruit
}


def normalize_polygon(all_x, all_y, width, height):
    """Normalize polygon coordinates to 0-1 range, clamping to image bounds."""
    normalized = []
    for x, y in zip(all_x, all_y):
        # Clamp to image bounds
        x = max(0, min(width, x))
        y = max(0, min(height, y))
        # Normalize
        nx = x / width
        ny = y / height
        normalized.extend([nx, ny])
    return normalized


def rect_to_polygon(x_center, y_center, w, h):
    """Convert YOLO bbox to polygon (4 corners)."""
    x1 = x_center - w/2
    y1 = y_center - h/2
    x2 = x_center + w/2
    y2 = y_center + h/2
    # Return as polygon: top-left, top-right, bottom-right, bottom-left
    return [x1, y1, x2, y1, x2, y2, x1, y2]


def process_json_file(json_path, images_dir, output_annotations):
    """Process a VGG JSON file and extract polygon annotations."""
    
    with open(json_path) as f:
        data = json.load(f)
    
    print(f"\n  Processing: {os.path.basename(json_path)}")
    
    processed = 0
    skipped = 0
    
    for key, entry in data.items():
        filename = entry.get('filename', key)
        
        # Find image
        img_path = os.path.join(images_dir, filename)
        if not os.path.exists(img_path):
            continue
        
        # Get image dimensions
        with Image.open(img_path) as img:
            width, height = img.size
        
        regions = entry.get('regions', {})
        if isinstance(regions, dict):
            regions = list(regions.values())
        
        annotations = []
        
        for r in regions:
            label = r.get('region_attributes', {}).get('label', '').lower().strip()
            
            # Skip calyx and unknown labels
            if label not in CLASS_MAP:
                if label and label != 'calyx':
                    print(f"    Warning: Unknown label '{label}' in {filename}")
                skipped += 1
                continue
            
            class_id = CLASS_MAP[label]
            
            shape = r.get('shape_attributes', {})
            shape_type = shape.get('name', '')
            
            if shape_type == 'polygon':
                all_x = shape.get('all_points_x', [])
                all_y = shape.get('all_points_y', [])
                
                if len(all_x) < 3:
                    skipped += 1
                    continue
                
                # Normalize polygon
                norm_coords = normalize_polygon(all_x, all_y, width, height)
                
                # Check if polygon is valid (has area)
                xs = norm_coords[0::2]
                ys = norm_coords[1::2]
                if max(xs) - min(xs) < 0.001 or max(ys) - min(ys) < 0.001:
                    skipped += 1
                    continue
                
                annotations.append((class_id, norm_coords))
                processed += 1
            
            elif shape_type == 'rect':
                x = shape.get('x', 0)
                y = shape.get('y', 0)
                w = shape.get('width', 0)
                h = shape.get('height', 0)
                
                if w < 1 or h < 1:
                    skipped += 1
                    continue
                
                # Convert to polygon (4 corners)
                x1, y1 = x / width, y / height
                x2, y2 = (x + w) / width, (y + h) / height
                norm_coords = [x1, y1, x2, y1, x2, y2, x1, y2]
                
                annotations.append((class_id, norm_coords))
                processed += 1
        
        if annotations:
            base_name = Path(filename).stem
            if base_name not in output_annotations:
                output_annotations[base_name] = {
                    'filename': filename,
                    'annotations': annotations
                }
            else:
                # Update with newer annotations (later files take precedence)
                output_annotations[base_name] = {
                    'filename': filename,
                    'annotations': annotations
                }
    
    print(f"    Processed: {processed}, Skipped: {skipped}")
    return processed, skipped


def process_yolo_txt_file(txt_path, output_annotations):
    """Process YOLO txt files (rectangle format) and convert to polygon."""
    
    base_name = Path(txt_path).stem
    
    with open(txt_path) as f:
        lines = [l.strip() for l in f if l.strip()]
    
    annotations = []
    
    for line in lines:
        parts = line.split()
        if len(parts) < 5:
            continue
        
        old_class_id = int(parts[0])
        x_c, y_c, w, h = map(float, parts[1:5])
        
        # Map old class to new class
        if old_class_id not in RECT_CLASS_MAP:
            continue
        
        class_id = RECT_CLASS_MAP[old_class_id]
        
        # Convert bbox to polygon
        norm_coords = rect_to_polygon(x_c, y_c, w, h)
        annotations.append((class_id, norm_coords))
    
    if annotations:
        # These images have .jpg extension
        filename = base_name + '.jpg'
        output_annotations[base_name] = {
            'filename': filename,
            'annotations': annotations
        }
    
    print(f"    {base_name}: {len(annotations)} rectangles -> polygons")
    return len(annotations)


def write_yolo_seg_dataset(output_annotations, images_dir, output_dir, train_ratio=0.8):
    """Write YOLO Segmentation format dataset."""
    
    # Create directories
    for split in ['train', 'val']:
        os.makedirs(f"{output_dir}/images/{split}", exist_ok=True)
        os.makedirs(f"{output_dir}/labels/{split}", exist_ok=True)
    
    # Sort for reproducibility
    items = sorted(output_annotations.items())
    n_train = int(len(items) * train_ratio)
    
    stats = {'train': 0, 'val': 0, 'annotations': 0}
    
    for i, (base_name, data) in enumerate(items):
        filename = data['filename']
        annotations = data['annotations']
        
        # Find image
        img_path = os.path.join(images_dir, filename)
        if not os.path.exists(img_path):
            # Try alternate extensions
            for ext in ['.jpg', '.jpeg', '.png', '.webp']:
                alt_path = os.path.join(images_dir, base_name + ext)
                if os.path.exists(alt_path):
                    img_path = alt_path
                    filename = base_name + ext
                    break
        
        if not os.path.exists(img_path):
            print(f"  Warning: Image not found: {filename}")
            continue
        
        split = 'train' if i < n_train else 'val'
        stats[split] += 1
        
        # Copy image
        img_ext = Path(img_path).suffix
        dest_img = f"{output_dir}/images/{split}/{base_name}{img_ext}"
        shutil.copy2(img_path, dest_img)
        
        # Write label file (YOLO Seg format)
        label_lines = []
        for class_id, coords in annotations:
            # Format: class_id x1 y1 x2 y2 ... xn yn
            coords_str = ' '.join(f'{c:.6f}' for c in coords)
            label_lines.append(f"{class_id} {coords_str}")
            stats['annotations'] += 1
        
        label_path = f"{output_dir}/labels/{split}/{base_name}.txt"
        with open(label_path, 'w') as f:
            f.write('\n'.join(label_lines))
    
    return stats


def main():
    print("="*70)
    print("YOLO SEGMENTATION DATASET CREATION")
    print("Preserving polygon precision - NO bounding box conversion")
    print("="*70)
    
    # Paths
    images_dir = 'to_annotate'
    output_dir = 'dataset_seg'
    
    # JSON files with polygon annotations (in order of precedence - later overrides earlier)
    json_files = [
        'original_makesense/labels_my-project-name_2026-01-12-11-31-15.json',
        'original_makesense/labels_my-project-name_2026-01-13-12-46-12.json',
        'original_makesense/labels_my-project-name_2026-01-13-01-09-08.json',  # Latest for 0241
    ]
    
    # YOLO txt files (rectangle annotations)
    yolo_txt_files = [
        'labels_my-project-name_2026-01-12-11-30-56/0193_healthy_crown_fruit_leaf_root_soil_ste.txt',
        'labels_my-project-name_2026-01-12-11-30-56/0235_healthy_calyx_fruit_leaf_root_soil.txt',
    ]
    
    output_annotations = {}
    
    # Process rectangle annotations first (will be overwritten by polygons if both exist)
    print("\n📦 Processing rectangle annotations:")
    for txt_file in yolo_txt_files:
        if os.path.exists(txt_file):
            process_yolo_txt_file(txt_file, output_annotations)
    
    # Process polygon annotations
    print("\n📦 Processing polygon annotations:")
    total_processed = 0
    total_skipped = 0
    for json_file in json_files:
        if os.path.exists(json_file):
            p, s = process_json_file(json_file, images_dir, output_annotations)
            total_processed += p
            total_skipped += s
    
    print(f"\n  Total polygon annotations: {total_processed}")
    print(f"  Skipped (invalid/calyx): {total_skipped}")
    
    # Write dataset
    print("\n📦 Writing YOLO Segmentation dataset...")
    stats = write_yolo_seg_dataset(output_annotations, images_dir, output_dir)
    
    print("\n" + "="*70)
    print("COMPLETE!")
    print("="*70)
    print(f"\nDataset: {output_dir}/")
    print(f"  Train images: {stats['train']}")
    print(f"  Val images: {stats['val']}")
    print(f"  Total annotations: {stats['annotations']}")
    print(f"\nFormat: YOLO Segmentation (polygon masks)")
    print(f"  class_id x1 y1 x2 y2 ... xn yn")
    print(f"\nTrain with: yolo segment train data=strawberry_parts_seg.yaml model=yolov8n-seg.pt")


if __name__ == '__main__':
    main()
