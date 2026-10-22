"""Check staged model copies against a per-model SHA256 manifest, reading them from DISK, not from the page cache.
Why: on 2026-10-22 node c5's /scratch was found to return corrupted data for correctly copied weight files (same size,
different bytes). Right after a copy the reads are served from the (correct) page cache, so a naive check passes; and
vLLM does not verify quantized weights, so a corrupted file loads silently and the model emits garbage. Each file is
fsync'ed and evicted (posix_fadvise DONTNEED) before hashing, which forces a read from the device.
  python3 cluster/infra/make_manifest.py /projects/$USER/agai/models/GLM-4.6V-FP8        # once per model, on a good node
  python3 cluster/infra/verify_staged.py /projects/$USER/agai/models/GLM-4.6V-FP8 /scratch/$USER/models/GLM-4.6V-FP8 \
      [--files a.safetensors b.safetensors] [--fix]
Exit 0 if every checked file matches (or no manifest exists), 1 otherwise; --fix deletes mismatching copies so the
caller can copy them again."""
import argparse
import hashlib
import os
import sys

MANIFEST = "SHA256SUMS"


def read_manifest(src):
    path = os.path.join(src, MANIFEST)
    if not os.path.exists(path):
        return None
    out = {}
    for line in open(path):
        h, name = line.strip().split(None, 1)
        out[name.lstrip("*")] = h
    return out


def disk_sha256(path, chunk=16 << 20):
    """SHA256 of a file as stored on the device: flush, evict its cached pages, then read."""
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        h = hashlib.sha256()
        while True:
            b = os.read(fd, chunk)
            if not b:
                break
            h.update(b)
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        return h.hexdigest()
    finally:
        os.close(fd)


def verify(src, dst, names=None, fix=False):
    """Names of staged files whose on-disk content differs from the manifest (None if src has no manifest)."""
    man = read_manifest(src)
    if man is None:
        return None
    bad = []
    for name in sorted(names or man):
        if name not in man or not os.path.exists(os.path.join(dst, name)):
            continue
        if disk_sha256(os.path.join(dst, name)) != man[name]:
            bad.append(name)
            if fix:
                os.remove(os.path.join(dst, name))
    return bad


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--files", nargs="*")
    ap.add_argument("--fix", action="store_true")
    a = ap.parse_args()
    bad = verify(a.src.rstrip("/"), a.dst.rstrip("/"), a.files, a.fix)
    host = os.uname().nodename.split(".")[0]
    if bad is None:
        print(f"{host}: no {MANIFEST} in {a.src}, nothing verified", file=sys.stderr)
        sys.exit(0)
    print(f"{host}: {len(bad)} staged file(s) differ from the manifest" + (f": {' '.join(bad)}" if bad else ""),
          file=sys.stderr)
    sys.exit(1 if bad else 0)
