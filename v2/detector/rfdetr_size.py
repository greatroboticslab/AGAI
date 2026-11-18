"""RF-DETR plant-part detector: evaluate the lab's checkpoint, or train another model size with the lab's own
hyperparameters (rfdetr_parts/rfdetr_config.yaml) and evaluate it, on the lab's dataset split
(rfdetr_parts/prepare_dataset.py: 753 train / 322 valid images, 6 classes). Writes COCO metrics to
evaluation/rfdetr_<name>_valid.json. Each size trains at its own default resolution (Small 512, Medium 576,
Large 704); the lab's yaml says 560, which current RF-DETR rejects for patch 16 x 2 windows.
  python cluster/detector/rfdetr_size.py --eval-only --size small --weights rfdetr_parts/output/checkpoint_best_total.pth --name lab_small
  python cluster/detector/rfdetr_size.py --size large --batch-size 4 --accum 4 --name large
"""
import argparse
import json
import os
import time

import yaml
from rfdetr import RFDETRLarge, RFDETRMedium, RFDETRSmall

SIZES = {"small": RFDETRSmall, "medium": RFDETRMedium, "large": RFDETRLarge}
ap = argparse.ArgumentParser()
ap.add_argument("--size", choices=SIZES, required=True)
ap.add_argument("--name", required=True)
ap.add_argument("--weights", default="", help="checkpoint to evaluate (--eval-only) or to start from")
ap.add_argument("--eval-only", action="store_true")
ap.add_argument("--epochs", type=int, default=0, help="0 = the lab's yaml (100)")
ap.add_argument("--batch-size", type=int, default=0)
ap.add_argument("--accum", type=int, default=0)
ap.add_argument("--config", default="rfdetr_parts/rfdetr_config.yaml")
ap.add_argument("--dataset-dir", default="", help="COCO dataset root (train/valid[/test]); default: the yaml's")
args = ap.parse_args()

cfg = yaml.safe_load(open(args.config))
dataset_dir = args.dataset_dir or cfg["dataset_dir"]
out_dir = f"/projects/{os.environ['USER']}/agai/rfdetr_runs/{args.name}"
os.makedirs(out_dir, exist_ok=True)
Model = SIZES[args.size]
t0 = time.time()
if args.eval_only:
    model = Model(pretrain_weights=args.weights)
else:
    train_kw = {k: v for k, v in cfg.items() if k not in ("resolution", "dataset_dir", "output_dir")}
    train_kw.update(dataset_dir=dataset_dir, output_dir=out_dir, progress_bar=False)
    if args.epochs:
        train_kw["epochs"] = args.epochs
    if args.batch_size:
        train_kw["batch_size"] = args.batch_size
    if args.accum:
        train_kw["grad_accum_steps"] = args.accum
    model = Model(pretrain_weights=args.weights) if args.weights else Model()
    allowed = set(model._train_config_class.model_fields)   # current RF-DETR rejects some of the lab's yaml keys
    dropped = {k: v for k, v in train_kw.items() if k not in allowed}
    train_kw = {k: v for k, v in train_kw.items() if k in allowed}
    print("train kwargs:", train_kw, "| dropped (not accepted by this RF-DETR version):", dropped, flush=True)
    model.train(**train_kw)
    best = os.path.join(out_dir, "checkpoint_best_total.pth")
    print(f"trained in {(time.time() - t0) / 3600:.2f} h; reloading {best}", flush=True)
    model = Model(pretrain_weights=best)
metrics = model.evaluate(split="val", dataset_dir=dataset_dir)
res = {"name": args.name, "size": args.size, "weights": args.weights or "trained here", "eval_only": args.eval_only,
       "resolution": model.model_config.resolution, "dataset": dataset_dir, "split": "valid (322 images)",
       "hours": round((time.time() - t0) / 3600, 2), "metrics": {k: (float(v) if hasattr(v, "__float__") else v)
                                                                 for k, v in (metrics or {}).items()}}
json.dump(res, open(f"evaluation/rfdetr_{args.name}_valid.json", "w"), indent=1)
print(json.dumps(res, indent=1), flush=True)
