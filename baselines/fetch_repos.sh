#!/bin/bash
# Clone the baselines' code at the commits our results were made with, into baselines/repos.
# The adapters in run_baselines.py read from here (PAPILLON's module, Loiseau et al.'s prompts,
# Staab et al.'s main.py). A repo with a file in baselines/patches gets that patch applied on
# a clean checkout; every patch only touches plumbing and is listed in the deviations table.
#
#   bash baselines/fetch_repos.sh
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
# repository                                   pinned commit
REPOS=(
  "Columbia-NLP-Lab/PAPILLON                    594e7ebe5ab87090916a53425a2c2ac0f4ec6239"
  "eth-sri/llm-anonymization                    5dafa6d1a9d2694ef81f88ca9e3e54058497c27a"
  "gabrielloiseau/adaptive-text-anonymization   315ed31fc8560053dd782fa21fbd98c8470deb4e"
)

mkdir -p "$HERE/repos"
for entry in "${REPOS[@]}"; do
  read -r repo commit <<< "$entry"
  name=$(basename "$repo")
  dir="$HERE/repos/$name"
  [ -d "$dir" ] || git clone -q "https://github.com/$repo.git" "$dir"
  git -C "$dir" fetch -q origin "$commit"
  git -C "$dir" checkout -q "$commit"
  git -C "$dir" checkout -q -- .            # drop local edits, so a patch applies to the pinned code
  patch="$HERE/patches/$name.patch"
  [ -f "$patch" ] && git -C "$dir" apply "$patch" && echo "  applied $(basename "$patch")"
  echo "$repo at $(git -C "$dir" rev-parse --short HEAD)"
done

# Staab et al.'s setup step (their README): copy credentials_clean.py to credentials.py.
# openai 0.28 refuses an empty key; our local server ignores it, and OPENAI_API_BASE points there.
STAAB="$HERE/repos/llm-anonymization"
sed 's/^openai_api_key = ""/openai_api_key = "local-server-no-key"/' "$STAAB/credentials_clean.py" > "$STAAB/credentials.py"
