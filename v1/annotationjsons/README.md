# `v1/annotationjsons/`: a plant-part annotation export from January 2026

Part of v1, the original system (2025 to April 2026). One COCO-format annotation file of strawberry plant parts,
committed on 21 January 2026 in the same commit as the lab's YOLOv8 detection experiments, which were removed from the
repository on 12 April 2026 when the RF-DETR detector ([`../rfdetr_parts/`](../rfdetr_parts/)) was added. No code in
the repository reads it, and the images it describes are not in the repository.

## Files

| File | What it is |
|---|---|
| [`labels_my-project-name_2026-01-20-06-10-25.json`](labels_my-project-name_2026-01-20-06-10-25.json) | COCO export (`info.description`: "my-project-name"), stored with Git LFS: 53 images (44 JPG, 7 JPEG, 2 WebP; file names such as `0017_drought_crown_fruit_leaf_soil_stem.jpg` name a condition and plant parts) with 360 polygon annotations in seven categories: fruit (130), stem (67), leaf (123), soil (5), root (24), flower (10) and crown (1) |

## Notes

- The categories are not those of the RF-DETR detector: the file adds `crown` and numbers the classes in another
  order (fruit 1, stem 2, leaf 3, soil 4, root 5, flower 6, crown 7), while the detector's data has six parts numbered
  flower 1 to stem 6 ([`../rfdetr_parts/dataset/README.md`](../rfdetr_parts/dataset/README.md)).
- The detector was trained on a separate Roboflow export (753 training and 322 validation images,
  [`../rfdetr_parts/dataset/README.md`](../rfdetr_parts/dataset/README.md)); whether these 53 images are among them
  is not recorded.
- The folder has been listed in [`.gitignore`](../../.gitignore) (`annotationjsons/`) since 25 April 2026, after this
  file was committed; the file stays tracked, and new files here are ignored.
