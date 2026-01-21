#!/usr/bin/env python3
"""
Select diverse images from training data for YOLO annotation.
Prioritizes images with roots (for root rot) and diverse part combinations.
"""

import json
import os
import shutil
import random
from pathlib import Path
from collections import defaultdict

# Paths
TRAIN_JSON = "/data/AGAI/MiniGPT-4/plant_diagnostic/datasets/stage2_train_7class_fixed.json"
OUTPUT_DIR = "/data/AGAI/MiniGPT-4/yolo_parts/to_annotate"

# Target: ~250 images total
TARGET_TOTAL = 250


def get_parts(description: str) -> set:
    """Extract plant parts mentioned in description."""
    desc = description.lower()
    
    parts = set()
    if any(w in desc for w in ['leaf', 'leaves', 'foliage']): parts.add('leaf')
    if any(w in desc for w in ['berry', 'berries', 'fruit']): parts.add('fruit')
    if any(w in desc for w in ['flower', 'bloom', 'blossom', 'petal']): parts.add('flower')
    if 'crown' in desc: parts.add('crown')
    if any(w in desc for w in ['stem', 'petiole', 'runner']): parts.add('stem')
    if 'root' in desc: parts.add('root')
    if 'calyx' in desc: parts.add('calyx')
    if any(w in desc for w in ['soil', 'mulch']): parts.add('soil')
    
    return parts


def get_disease(img_path: str) -> str:
    """Extract disease class from image path."""
    for disease in ['drought', 'frost', 'gray_mold', 'white_mold', 'root_rot', 'overwatered', 'healthy']:
        if disease in img_path.lower():
            return disease
    return 'unknown'


def main():
    # Load training data
    with open(TRAIN_JSON) as f:
        data = json.load(f)
    
    print(f"Loaded {len(data)} training samples")
    
    # Analyze all images
    images = []
    for entry in data:
        img_path = entry['image']
        if not os.path.exists(img_path):
            continue
            
        description = entry['conversations'][1]['value']
        parts = get_parts(description)
        disease = get_disease(img_path)
        
        images.append({
            'path': img_path,
            'parts': parts,
            'disease': disease,
            'description': description[:150]
        })
    
    print(f"Found {len(images)} valid images")
    
    # Priority selection strategy:
    # 1. Images with ROOTS (critical for root rot) - get 50
    # 2. Images from each disease class - get 20 each
    # 3. Fill rest with diverse part combinations
    
    selected = []
    selected_paths = set()
    
    # Priority 1: Images with roots
    root_images = [img for img in images if 'root' in img['parts']]
    print(f"\nImages with roots: {len(root_images)}")
    for img in random.sample(root_images, min(50, len(root_images))):
        if img['path'] not in selected_paths:
            selected.append(img)
            selected_paths.add(img['path'])
    print(f"  Selected: {len([s for s in selected if 'root' in s['parts']])}")
    
    # Priority 2: Each disease class (for disease-specific part visibility)
    diseases = defaultdict(list)
    for img in images:
        diseases[img['disease']].append(img)
    
    print(f"\nDisease distribution:")
    for disease, imgs in diseases.items():
        n_select = min(25, len(imgs))
        sampled = random.sample(imgs, n_select)
        for img in sampled:
            if img['path'] not in selected_paths:
                selected.append(img)
                selected_paths.add(img['path'])
        print(f"  {disease}: {len(imgs)} total, added {n_select}")
    
    # Priority 3: Ensure we have images WITHOUT certain parts (negative examples)
    # Important for YOLO to learn what's NOT there
    print(f"\nAdding negative examples (images missing common parts):")
    
    for missing_part in ['leaf', 'fruit', 'root', 'crown']:
        no_part = [img for img in images if missing_part not in img['parts'] 
                   and img['path'] not in selected_paths]
        n_add = min(15, len(no_part))
        for img in random.sample(no_part, n_add):
            selected.append(img)
            selected_paths.add(img['path'])
        print(f"  Without {missing_part}: added {n_add}")
    
    # Trim to target if over
    if len(selected) > TARGET_TOTAL:
        selected = random.sample(selected, TARGET_TOTAL)
    
    print(f"\nFinal selection: {len(selected)} images")
    
    # Summary of parts coverage
    part_counts = defaultdict(int)
    for img in selected:
        for part in img['parts']:
            part_counts[part] += 1
    
    print("\nPart coverage in selected images:")
    for part in ['leaf', 'fruit', 'flower', 'crown', 'stem', 'root', 'calyx', 'soil']:
        count = part_counts.get(part, 0)
        pct = count / len(selected) * 100
        print(f"  {part}: {count} ({pct:.0f}%)")
    
    # Copy images to output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Clear existing
    for f in os.listdir(OUTPUT_DIR):
        os.remove(os.path.join(OUTPUT_DIR, f))
    
    manifest = []
    for i, item in enumerate(selected):
        src = item['path']
        ext = Path(src).suffix
        parts_str = '_'.join(sorted(item['parts']))[:30]
        dst_name = f"{i:04d}_{item['disease']}_{parts_str}{ext}"
        dst = os.path.join(OUTPUT_DIR, dst_name)
        
        shutil.copy2(src, dst)
        manifest.append({
            'filename': dst_name,
            'original': src,
            'disease': item['disease'],
            'parts': list(item['parts']),
            'description': item['description']
        })
    
    # Save manifest
    manifest_path = os.path.join(OUTPUT_DIR, 'manifest.json')
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    
    print(f"\n✓ Images copied to: {OUTPUT_DIR}")
    print(f"✓ Manifest saved to: {manifest_path}")
    print(f"\n=== NEXT STEPS ===")
    print(f"1. Upload {OUTPUT_DIR}/ to Roboflow")
    print(f"2. Annotate with 8 classes: leaf, fruit, flower, crown, stem, root, calyx, soil")
    print(f"3. Export → YOLOv8 format")
    print(f"4. Place in yolo_parts/dataset/images/ and labels/")


if __name__ == "__main__":
    main()
