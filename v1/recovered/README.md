# `v1/recovered/`: code recovered from the lab server

v1 code that existed only in the working copy of the lab's GPU server, where it was last changed in April 2026. It was
recovered from an export of that working copy made on 19 October 2026 and added here during the v2 revival, on
17 November 2026. Nothing in the repository imports or runs these files from this folder.

## Contents

| Path | Contents |
|---|---|
| [`labbox_2026-10-19/`](labbox_2026-10-19/) | 16 code files, byte for byte at their paths in the lab server's layout, and a README with their dates, purpose, use and SHA-256 checksums: the April 2026 versions of the ResNet-50 training, evaluation, cross-validation, inference and folder-prediction scripts; scripts that find and import external photographs and rebuild the training and holdout sets, with and without synthetic images; two scripts that evaluate fixed checkpoints; the April technical report's MiniGPT-v2 holdout run and its rubric scoring; a multi-caption instruction-data generator |

The lab server's layout is the layout of `v1/`: `labbox_2026-10-19/plant_diagnostic/...` corresponds to
`v1/plant_diagnostic/...`, and `labbox_2026-10-19/paper/scripts/` to `v1/paper/scripts/`. The recovered
`plant_diagnostic/scripts/resnetscripts/train_resnet_straw5.py` is the version that trained every ResNet-50 of the v2
revival's paper; the older same-named files under `v1/plant_diagnostic/` are the 2025 versions.
