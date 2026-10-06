#!/bin/bash
# Clone the baselines' code at the commits our results were made with, into baselines/repos.
# The adapters in run_baselines.py read from here (PAPILLON's module, Loiseau et al.'s prompts).
#
#   bash baselines/fetch_repos.sh
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
# repository                                   pinned commit
REPOS=(
  "Columbia-NLP-Lab/PAPILLON                    594e7ebe5ab87090916a53425a2c2ac0f4ec6239"
  "gabrielloiseau/adaptive-text-anonymization   315ed31fc8560053dd782fa21fbd98c8470deb4e"
)

mkdir -p "$HERE/repos"
for entry in "${REPOS[@]}"; do
  read -r repo commit <<< "$entry"
  dir="$HERE/repos/$(basename "$repo")"
  [ -d "$dir" ] || git clone -q "https://github.com/$repo.git" "$dir"
  git -C "$dir" fetch -q origin "$commit"
  git -C "$dir" checkout -q "$commit"
  echo "$repo at $(git -C "$dir" rev-parse --short HEAD)"
done
