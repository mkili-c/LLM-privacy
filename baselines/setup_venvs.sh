#!/bin/bash
# Build a baseline's venv from its frozen requirements, so every rewrite can be reproduced.
# Venvs live on permanent storage (/work); the requirements files next to this script pin
# every package version (pip freeze of the venv the results were made with).
#
#   bash baselines/setup_venvs.sh presidio    # x86 login node, Python 3.12.3; no GPU needed
set -euo pipefail

VENVS=/work/projects/mkilic/venvs
HERE=$(cd "$(dirname "$0")" && pwd)
METHOD=${1:?usage: setup_venvs.sh <method>}

python3 -m venv "$VENVS/$METHOD"
"$VENVS/$METHOD/bin/pip" install -q --upgrade pip
"$VENVS/$METHOD/bin/pip" install -q -r "$HERE/requirements-$METHOD.txt"
echo "built $VENVS/$METHOD with $("$VENVS/$METHOD/bin/python" --version)"
