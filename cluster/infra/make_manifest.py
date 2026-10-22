"""Write MODEL_DIR/SHA256SUMS (sha256sum format) for every weight file of a model directory, so staged copies can be
verified on each node (cluster/infra/verify_staged.py). Reads the whole model once over NFS: run it as a CPU job.
  python3 cluster/infra/make_manifest.py /projects/$USER/agai/models/GLM-4.6V-FP8"""
import hashlib
import os
import sys

WEIGHTS = (".safetensors", ".bin", ".pth", ".pt", ".gguf")

src = sys.argv[1].rstrip("/")
lines = []
for name in sorted(os.listdir(src)):
    path = os.path.join(src, name)
    if not os.path.isfile(path) or not name.endswith(WEIGHTS):
        continue
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(16 << 20), b""):
            h.update(chunk)
    lines.append(f"{h.hexdigest()}  {name}\n")
    print(name, h.hexdigest()[:16], flush=True)
tmp = os.path.join(src, "SHA256SUMS.tmp")
open(tmp, "w").writelines(lines)
os.replace(tmp, os.path.join(src, "SHA256SUMS"))
print(f"{len(lines)} files -> {src}/SHA256SUMS")
