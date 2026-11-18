#!/usr/bin/env bash
# Rebuild the single-folder working copy that the v2 scripts were written for and run in on the cluster.
#
# On Hamilton every v2 script ran from the root of one working copy: the v1 tree, with evaluation/ inside it and the
# v2 scripts in cluster/. The scripts open evaluation/split_v1.json, the photographs it lists
# (plant_diagnostic/data/...), v1 modules (resnet_classifier, grounding, knowledge_graph), v1 configs and
# rfdetr_parts/ relative to the working directory. This script builds such a folder out of symbolic links into this
# repository, so the commands in the lab notebook (python cluster/<script>.py) and in the READMEs
# (python cluster/<study>/<script>.py) work as written.
#
#   bash v2/infra/make_workdir.sh /projects/$USER/agai-work
#   cd /projects/$USER/agai-work
#   python cluster/vlm/score_faithfulness.py --help
#
# DEST then holds (every entry is a link into the repository, except the two real folders):
#   <each entry of v1/>  the v1 tree: minigpt4/, grounding/, plant_diagnostic/, rfdetr_parts/, train.py, ...
#   evaluation/          real folder: links to the entries of evaluation/ and of v1/evaluation/
#   cluster/             real folder: cluster/<study>/ for each folder of v2/, cluster/<script> for every file in
#                        those folders (the notebook's flat names), and cluster/EXPERIMENT_LOG.md
#   docs/                docs/
# New files a script writes directly into evaluation/ stay in DEST/evaluation, as they stayed in the cluster's
# working copy (copy the ones to keep into evaluation/paper_data/). Writes into evaluation/paper_data/,
# evaluation/holdout/, plant_diagnostic/ or rfdetr_parts/ go through the links into the repository.
# infra/agai_env.sh ends with a cd into the cluster's own working copy; after sourcing it, cd into DEST.
# Needs a file system with symbolic links (Linux or macOS).
set -euo pipefail
dest=${1:?usage: make_workdir.sh DEST}
repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
if [ -e "$dest" ] || [ -L "$dest" ]; then echo "make_workdir.sh: $dest exists; give a new folder" >&2; exit 1; fi
mkdir -p "$dest/cluster" "$dest/evaluation"
dest=$(cd "$dest" && pwd)

link() {
  if [ -e "$2" ] || [ -L "$2" ]; then echo "make_workdir.sh: name clash at $2" >&2; exit 1; fi
  ln -s "$1" "$2"
}

shopt -s nullglob dotglob
for e in "$repo"/v1/*; do
  n=$(basename "$e")
  case $n in README.md|evaluation) continue ;; esac
  link "$e" "$dest/$n"
done
for e in "$repo"/evaluation/* "$repo"/v1/evaluation/*; do
  n=$(basename "$e")
  [ "$n" = README.md ] && continue
  link "$e" "$dest/evaluation/$n"
done
for s in "$repo"/v2/*/; do
  s=${s%/}
  link "$s" "$dest/cluster/$(basename "$s")"
  for f in "$s"/*; do
    [ -f "$f" ] || continue
    n=$(basename "$f")
    [ "$n" = README.md ] && continue
    link "$f" "$dest/cluster/$n"
  done
done
link "$repo/v2/EXPERIMENT_LOG.md" "$dest/cluster/EXPERIMENT_LOG.md"
link "$repo/docs" "$dest/docs"
echo "working copy ready: $dest"
