#!/bin/bash --login
# One-time setup of the x86 venv ~/.llmvenv: the Laya jobs on the RTX 5060 Ti, the dev-set utility
# measurements, and every analysis and plotting script (results/, figures/). The GB10 nodes are
# ARM and use their own venv (setup_gb10.sh); the baselines use theirs (baselines/setup_venvs.sh).
#
# Only installs packages, so it may run on the login node:
#
#     bash /work/projects/mkilic/LLM-privacy/setup_x86.sh
#
# torch comes from the CUDA 12.8 wheel index and is resolved together with the frozen
# requirements: they pin torch's own CUDA packages (cuda-toolkit, nvidia-*, triton), which
# conflict with any other torch build. Then laya with --no-deps, so pip cannot replace torch.
set -euo pipefail

VENV="${VENV:-/home/mkilic/.llmvenv}"
REPO_ROOT="${REPO_ROOT:-/work/projects/mkilic/LLM-privacy}"

if [ "$(uname -m)" != "x86_64" ]; then
    echo "This is $(uname -m), not x86_64: on a GB10 node use setup_gb10.sh." >&2
    exit 1
fi

python3 -m venv "$VENV"
source "$VENV/bin/activate"
pip install --upgrade pip

pip install --extra-index-url https://download.pytorch.org/whl/cu128 \
    "torch==2.11.0+cu128" -r "$REPO_ROOT/requirements-x86.txt"
pip install --no-deps laya==0.3.21    # --no-deps: keep pip from replacing torch

# imports only: the login node has no GPU, so CUDA is reported, not required
python - <<'EOF'
import torch, transformers, matplotlib, laya
print("torch", torch.__version__, "| transformers", transformers.__version__,
      "| matplotlib", matplotlib.__version__)
print("CUDA available:", torch.cuda.is_available())
print("laya import OK")
EOF
echo "x86 venv ready: $VENV"
