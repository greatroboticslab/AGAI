# `v1/recovered/labbox_2026-10-19/`: code recovered from the lab server, 19 October 2026

These 16 files existed only in the working copy of the lab's GPU server (`/data/AGAI/MiniGPT-4`, at commit `468951f`)
when Dr. Zhang exported it on 19 October 2026 (archive `agai_worktree.tgz`, SHA-256 `061952891fe2cb5d3ad8066d2cf4fc0c300dbf87194c812a0371f034f9948c97`). None of them
was in this repository. Each is copied byte for byte, at its original path relative to the root of that working copy
(the layout of `v1/` in this repository), and the checksums at the end of this page match the files in the archive.

Five of them are later versions of scripts that also exist at the main paths under `v1/`, where the versions committed
in 2025 stay unchanged; this folder keeps the lab server's set complete and byte for byte. The code directory that the
2026 paper's runs used on the MTSU cluster (`/projects/<user>/agai/MiniGPT-4` on Hamilton) was extracted from the same
archive on 19 October 2026, so those runs used the versions in this folder: every ResNet-50 trained for the paper
(the split v1 runs, the synthetic-data ablations and the 512-pixel control, from 20 October on) was trained with this
folder's `train_resnet_straw5.py`, whose checksum the cluster's copy matches, and the training logs show its progress
bars.

## What is here

| File (under this folder) | Last changed on the lab server | What it does |
|---|---|---|
| `plant_diagnostic/scripts/resnetscripts/train_resnet_straw5.py` | 26 Apr 2026 | ResNet-50 training, the version that trained every ResNet-50 of the 2026 paper: square resize to the input size instead of a random 90 to 100 % crop (training) and resize plus centre crop (validation); no random erasing; progress bars |
| `plant_diagnostic/scripts/resnetscripts/eval_resnet_straw5.py` | 26 Apr 2026 | evaluation with the same square resize |
| `plant_diagnostic/scripts/resnetscripts/crossval_resnet_straw.py` | 27 Apr 2026 | cross-validation; only the example checkpoint path differs from the main copy |
| `plant_diagnostic/src/resnet_classifier.py` | 27 Apr 2026 | inference; default checkpoint `resnet_strawberry.pth` instead of `resnet_straw7.pth` |
| `plant_diagnostic/src/predict_folder.py` | 26 Apr 2026 | folder prediction with two views (square resize and its mirror) instead of centre crops at three scales |
| `plant_diagnostic/scripts/resnetscripts/eval_resnet_kfold_compare.py` | 26 Apr 2026 | stratified k-fold comparison of fixed checkpoints |
| `plant_diagnostic/scripts/resnetscripts/eval_resnet_train_aug_subsets.py` | 27 Apr 2026 | accuracy of fixed checkpoints on the real and the synthetic parts of `train_aug` |
| `plant_diagnostic/scripts/release_holdout_to_train_and_rebuild_aug.py` | 25 Apr 2026 | moves a fixed number of holdout images back into `train` and rebuilds `train_aug` |
| `plant_diagnostic/scripts/find_external_strawberry_candidates.py` | 25 Apr 2026 | downloads candidate web images and screens them for exact and near duplicates of the collection |
| `plant_diagnostic/scripts/import_curated_external_candidates.py` | 25 Apr 2026 | imports the audited external images into `train` and `train_aug` |
| `plant_diagnostic/scripts/add_synthetic_rebuild_holdout15.py` | 26 Apr 2026 | re-pools the real images, takes a 15 % holdout per class and rebuilds `train_aug` as real plus synthetic images |
| `plant_diagnostic/scripts/build_mixed_holdout_eval.py` | 27 Apr 2026 | builds an evaluation holdout of the real holdout plus a 15 % sample of the synthetic images |
| `plant_diagnostic/scripts/build_mixed_holdout_train_exclusive.py` | 27 Apr 2026 | the same, plus a training tree that leaves out the sampled synthetic images |
| `plant_diagnostic/scripts/gptscripts/json_generation/generate_multicaption_jsonl.py` | 26 Apr 2026 | instruction data with several conversation styles per image, written by OpenAI's o3 |
| `paper/scripts/run_minigpt_on_holdout.py` | 25 Apr 2026 | runs the deployed MiniGPT-v2 on the April technical report's 70-image holdout |
| `paper/scripts/score_minigpt_rubric.py` | 25 Apr 2026 | automatic rubric scoring of those outputs |

## What they were used for

On the lab server: the data export holds 17 ResNet-50 checkpoints. The late-April ones, v5 to v12 ("synth" and
"synthOOD"), date from 27 April 2026 on, after these scripts were last changed, and the default checkpoint of the
application, `resnet_strawberry.pth`, is v12 (`synthOODmixed_exclusive`). The lab data also contains the folders that
the split-building scripts create (`holdout_mixed_synth15_20260427T…`, `holdout_mixed_exclusive_synth15_20260427T141500Z`).
No output of `generate_multicaption_jsonl.py` is in the export, so it was probably never run at scale.

In the 2026 paper: `train_resnet_straw5.py` trained every ResNet-50 (see above). `resnet_classifier.py` differs from the
main copy only in its default checkpoint, and the cluster scripts pass the checkpoint explicitly; the end-to-end run of
v1 as deployed (`v2/vlm/latency_v1.py`) loads the application's checkpoint, `resnet_strawberry.pth`, that is v12.
No other file in this folder is behind a result in the paper.

## Running them

The scripts are written for the lab server's layout, and several default to `/data/AGAI/MiniGPT-4`, so pass the paths
explicitly (`python <script> --help` lists the options). The data scripts move and copy image files: run them on a
copy of `v1/plant_diagnostic/data`. `run_minigpt_on_holdout.py` finds the repository from its own location, so it only
works from its original path (`paper/scripts/` of the lab server's layout, which is `v1/paper/scripts/` here). To
recreate the lab server's code as a whole, restore this folder over a checkout of `468951f`, which is in this
repository's history (from the repository root):

```bash
git worktree add ../agai-labserver 468951f
cp -r v1/recovered/labbox_2026-10-19/. ../agai-labserver/
```

Examples, from `v1/`:

```bash
# the ResNet-50 recipe of the 2026 paper's runs (square resize, no random erasing)
python recovered/labbox_2026-10-19/plant_diagnostic/scripts/resnetscripts/train_resnet_straw5.py \
    --train_root <train folder> --val_root <validation folder> --out <checkpoint.pth> --seed 42

# instruction data in several conversation styles; needs OPENAI_API_KEY in the environment, never in a file
python recovered/labbox_2026-10-19/plant_diagnostic/scripts/gptscripts/json_generation/generate_multicaption_jsonl.py \
    --img-root plant_diagnostic/data/train_aug --out plant_diagnostic/datasets/strawberry_multicaption.jsonl \
    --variants 3 --model o3
```

`generate_multicaption_jsonl.py` writes one record per image and conversation style (clinical, quick triage,
treatment, differential, severity, layperson), keyed by image and style so that an interrupted run resumes; it skips
failed calls instead of writing a fallback. `--always-include-clinical` keeps the style of the original reports in
every image's set.

## Not included

The data and weights (in the lab's data export, not public); the JSON outputs of the external-image and April holdout
scripts (in the export); and the only other code change on the lab server, a Gradio 4 compatibility fix in
`demo_v5.py` (`sources=["upload"]` instead of `source="upload"`).

## Checksums (SHA-256)

```
21cae2c7bd697c97fa9556dc05c805151d106f0942ac1ee005890113eccac857  plant_diagnostic/scripts/resnetscripts/train_resnet_straw5.py
b6c87421880399eee7658a9e43808e76bb3e13e405ade66b34729075b261ec18  plant_diagnostic/scripts/resnetscripts/eval_resnet_straw5.py
e208b5c7dc1ab881f96964e4b6c7cc3586456fb0658c994eefa3be726e5c0106  plant_diagnostic/scripts/resnetscripts/crossval_resnet_straw.py
ca45325ea7e8a21afa7d7bb1c6f1befac13fea60a448f294db9bbd8ffc720b36  plant_diagnostic/src/resnet_classifier.py
720475f592f9c1303fa5a90b1c0b2da82da510c0e24763f86509349bf99f35e2  plant_diagnostic/src/predict_folder.py
1e458d186e82611f36abd3090d74a907b1d4f324120332bacb07aabed223bf41  plant_diagnostic/scripts/resnetscripts/eval_resnet_kfold_compare.py
14184ab7b53400421a873f5742be9598ba784f3dbfb5571a39c6bdc6c377e231  plant_diagnostic/scripts/resnetscripts/eval_resnet_train_aug_subsets.py
ebff17256010450fe49e974b751995e3cdced1d4487cea6c26fd89f58a837e9b  plant_diagnostic/scripts/release_holdout_to_train_and_rebuild_aug.py
4fbf6b6537b452059d4061c393366d62db4fdc3ba6091873654cee0c578bda37  plant_diagnostic/scripts/find_external_strawberry_candidates.py
3a1d3d200fdf9c50e7b6ea381e20b285ba2ba734cbb3406ff0292cd9d16b6f89  plant_diagnostic/scripts/import_curated_external_candidates.py
139e61d0f8bf4773854e457024270e408e9d8b6e484a4c468b9deff032fa9cdb  plant_diagnostic/scripts/add_synthetic_rebuild_holdout15.py
949f3560265466ebb5a3266f0bfbcc0ea9c001dac12d13b21d9ea2a0b804f9a0  plant_diagnostic/scripts/build_mixed_holdout_eval.py
1e294ce9ad4a5c04837bc2c2b757d6c67974590840b75dab51aed0ff84fbfa8e  plant_diagnostic/scripts/build_mixed_holdout_train_exclusive.py
03db447756c888ee470460a6ffa0fc783057d76c6d4d7a90c84586a2697c27a6  plant_diagnostic/scripts/gptscripts/json_generation/generate_multicaption_jsonl.py
46253a93771387aed055e46aaf8e177aceb676bea8f3be0f0b43d442df04202c  paper/scripts/run_minigpt_on_holdout.py
80b2bc583a7fa0425bddf6ff0746c68193ceade4dfb812fda10b95ffc622aea8  paper/scripts/score_minigpt_rubric.py
```
