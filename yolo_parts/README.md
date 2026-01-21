# YOLO Strawberry Part Detection

Detects plant parts (leaf, fruit, flower, crown, stem) to constrain MiniGPT's output to only describe visible parts.

## Pipeline Flow

```
Image → ResNet (disease) → YOLO (parts) → MiniGPT (constrained description)
                ↓              ↓                    ↓
           "drought"    [leaf, fruit]    "Describe drought on leaf and fruit only"
```

## Setup

### Step 1: Annotate Images

511 diverse images have been selected in `to_annotate/`.

**Option A: Roboflow (Recommended)**
1. Create account at https://roboflow.com
2. Create new project → Object Detection
3. Upload images from `to_annotate/`
4. Annotate with 8 classes: `leaf`, `fruit`, `flower`, `crown`, `stem`, `root`, `calyx`, `soil`
5. Use Roboflow's AI-assist to speed up annotation
6. Export → YOLOv8 format
7. Download and extract to `dataset/`

**Option B: Label Studio (Local)**
```bash
pip install label-studio
label-studio start
# Upload images, annotate, export YOLO format
```

### Step 2: Organize Dataset

After annotation, structure should be:
```
yolo_parts/
├── dataset/
│   ├── images/
│   │   ├── train/    # 80% of annotated images
│   │   └── val/      # 20% of annotated images
│   └── labels/
│       ├── train/    # Corresponding .txt files
│       └── val/
├── strawberry_parts.yaml
└── train_yolo.py
```

### Step 3: Train

```bash
cd /data/AGAI/MiniGPT-4/yolo_parts
python train_yolo.py
```

Training takes ~30-60 minutes on GPU. Best model saved to:
`models/strawberry_parts/weights/best.pt`

### Step 4: Test

```bash
python train_yolo.py --test /path/to/image.jpg
```

## Integration with demo_v5.py

Already integrated! After training, the part detector will automatically:
1. Detect visible parts in uploaded images
2. Add constraints to MiniGPT prompts
3. Filter out hallucinated plant parts from responses

## Classes

| ID | Name   | Training Mentions | Why Important |
|----|--------|-------------------|---------------|
| 0  | leaf   | 75% | Most diseases affect leaves |
| 1  | fruit  | 61% | Berry condition/ripeness |
| 2  | flower | 20% | Frost damage shows here first |
| 3  | crown  | 66% | Crown rot, overall health |
| 4  | stem   | 35% | Includes petioles, runners |
| 5  | root   | 45% | **CRITICAL for root rot!** |
| 6  | calyx  | 33% | Green cap on berries, drought indicator |
| 7  | soil   | 94% | Moisture context (dry/wet/mulched) |

## Files

- `select_images_for_annotation.py` - Selects diverse images for annotation
- `train_yolo.py` - Training script
- `part_detector.py` - Detection module for integration
- `strawberry_parts.yaml` - Dataset config
- `to_annotate/` - Images ready for annotation
- `dataset/` - Annotated dataset (after annotation)
- `models/` - Trained models
