# RF-DETR Training Dataset (not tracked)

This directory holds the COCO-format dataset used to fine-tune the
RF-DETR Small plant-part detector. The image files and their COCO
annotations are **excluded from version control** (see
[`.gitignore`](../../.gitignore)) because of size. This README and the
placeholder directory layout are tracked.

## Expected layout

```
rfdetr_parts/dataset/
├── train/
│   ├── _annotations.coco.json   # 753 images, 19 487 instance annotations
│   └── ... (image files)
└── valid/
    ├── _annotations.coco.json   #  322 images,  8 582 instance annotations
    └── ... (image files)
```

## Six-class plant-part vocabulary

| ID | Class  |
|----|--------|
| 1  | flower |
| 2  | fruit  |
| 3  | leaf   |
| 4  | root   |
| 5  | soil   |
| 6  | stem   |

## Reproducing

See [`../prepare_dataset.py`](../prepare_dataset.py) for the dataset
preparation script and [`../train_rfdetr.py`](../train_rfdetr.py) for
training. Hyperparameters are pinned in
[`../rfdetr_config.yaml`](../rfdetr_config.yaml). The deployed
checkpoint hash is listed in the reproducibility manifest of
[`../../paper/main.tex`](../../paper/main.tex).
