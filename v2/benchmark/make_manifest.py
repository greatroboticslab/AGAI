"""SHA-256 manifest of the evaluation data: the split and set definitions, the result files behind the paper, and
every split v1 image present in the repository.

    python -P v2/benchmark/make_manifest.py --out evaluation/MANIFEST.sha256

Check a copy with `sha256sum -c evaluation/MANIFEST.sha256` from the repository root (paths are relative to it), in a
checkout with the Git LFS objects pulled (`git lfs pull`): the images are stored in LFS, and their hashes here are of
the images themselves (taken from the LFS pointers when the objects are not checked out). Text files (JSON, CSV, TXT,
Markdown) are hashed with LF line endings, as the repository stores them: on Windows, check a checkout made with
`core.autocrlf false`.
Images listed in the split but absent from the repository (evaluation/split_v1_coverage.json) are not in the manifest.
"""
import argparse
import glob
import hashlib
import json
import os


LFS_HEAD = b"version https://git-lfs"
TEXT_EXT = {".json", ".csv", ".txt", ".md"}


def sha(p):
    """SHA-256 of the file's content. For a Git LFS pointer (a checkout without the LFS objects) this is the pointer's
    oid, which Git LFS defines as the SHA-256 of the real file, so the manifest is the same with or without them."""
    with open(p, "rb") as f:
        head = f.read(len(LFS_HEAD))
        if head == LFS_HEAD:
            rest = (head + f.read(1000)).decode("ascii", "replace")
            for line in rest.splitlines():
                if line.startswith("oid sha256:"):
                    return line.split(":", 1)[1].strip()
            raise ValueError(f"LFS pointer without an oid: {p}")
    data = open(p, "rb").read()
    if os.path.splitext(p)[1].lower() in TEXT_EXT:
        # text files are hashed as the repository stores them (LF line endings), whatever the checkout converted them to
        data = data.replace(b"\r\n", b"\n")
    return hashlib.sha256(data).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    files = sorted(set(glob.glob("evaluation/*.json") + glob.glob("evaluation/*.csv") +
                       glob.glob("evaluation/paper_data/*")) - {a.out})
    s = json.load(open("evaluation/split_v1.json", encoding="utf-8"))
    imgs = sorted({os.path.join("v1", p).replace("\\", "/") for part in ("train", "val", "test")
                   for ps in s[part].values() for p in ps if os.path.exists(os.path.join("v1", p))})
    rows = [(sha(p), p.replace("\\", "/")) for p in files if os.path.isfile(p)] + [(sha(p), p) for p in imgs]
    with open(a.out, "w", encoding="utf-8", newline="\n") as f:
        f.write("# SHA-256 of the evaluation files and of the split v1 images present in this repository.\n"
                "# Check with: git lfs pull, then sha256sum -c evaluation/MANIFEST.sha256 (from the repository root).\n"
                "# Images absent from the repository are listed in evaluation/split_v1_coverage.json.\n")
        for h, p in rows:
            f.write(f"{h}  {p}\n")
    print(f"{len(rows)} entries ({len(rows) - len(imgs)} evaluation files, {len(imgs)} images) -> {a.out}")


if __name__ == "__main__":
    main()
