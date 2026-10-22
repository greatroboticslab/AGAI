"""Which test images did the lab's fine-tuned MiniGPT (checkpoint 2025-12-18) train on?
Its training JSON (plant_diagnostic/datasets/stage2_train_7class_fixed.json, 2159 entries) lists train_aug paths;
their content is the LFS oid of those paths at origin/main (GitHub-era train_aug, also 2159 files).
Writes evaluation/minigpt_lab_train_overlap.json: {set: [test image paths seen in training]}.
Run from the repo root with system python3.
"""
import hashlib
import json
import subprocess

train = json.load(open("plant_diagnostic/datasets/stage2_train_7class_fixed.json"))
rel = {x["image"].split("/MiniGPT-4/", 1)[1] for x in train}

ls = subprocess.run(["git", "ls-tree", "-r", "origin/main", "--", "plant_diagnostic/data/train_aug"],
                    capture_output=True, text=True, check=True).stdout
blob_of = {line.split("\t", 1)[1]: line.split()[2] for line in ls.splitlines()}
blobs = [blob_of[p] for p in rel if p in blob_of]
cat = subprocess.run(["git", "cat-file", "--batch"], input="\n".join(blobs) + "\n",
                     capture_output=True, text=True, errors="replace", check=True).stdout
train_oids = {l[len("oid sha256:"):].strip() for l in cat.splitlines() if l.startswith("oid sha256:")}
print(f"training JSON images: {len(rel)}, found in origin/main: {len(blobs)}, unique contents: {len(train_oids)}")


def sha(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


tests = {
    "paper68": [r["found_at"] for r in json.load(open("evaluation/holdout/repro_resnet_v4_hamilton.json"))["rows"]],
    "resnet161": [p for ps in json.load(open("evaluation/clean_eval_sets.json"))["resnet_v4_clean"].values() for p in ps],
}
overlap = {name: [p for p in paths if sha(p) in train_oids] for name, paths in tests.items()}
for name, paths in tests.items():
    print(f"{name}: {len(overlap[name])}/{len(paths)} images were in the lab MiniGPT's training set")
json.dump(overlap, open("evaluation/minigpt_lab_train_overlap.json", "w"), indent=1)
