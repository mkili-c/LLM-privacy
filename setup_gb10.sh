#!/bin/bash --login
# One-time setup of the aarch64 venv for the GB10 nodes.
#
# The GB10 nodes are ARM, so the x86 venv ~/.llmvenv cannot be used there and
# the venv must be built ON a GB10 node, not on the login node. Run inside an
# interactive GB10 session (a login shell, so the login node's module paths
# are not inherited):
#
#     srun --gpus gb10:1 --pty -A projects -t 60 bash --login
#     bash /work/projects/mkilic/LLM-privacy/setup_gb10.sh
#
# The venv goes to /work/projects (permanent); /home is nearly full.
set -euo pipefail

VENV="${VENV:-/work/projects/mkilic/.llmvenv-gb10}"
REPO_ROOT="${REPO_ROOT:-/work/projects/mkilic/LLM-privacy}"

if [ "$(uname -m)" != "aarch64" ]; then
    echo "This is $(uname -m), not a GB10 node: run it inside a GB10 session." >&2
    exit 1
fi

python3 -m venv "$VENV"
source "$VENV/bin/activate"
pip install --upgrade pip

# The torch CUDA wheels bundle their own CUDA runtime, so no cuda module is
# needed (not all module CUDA versions exist on the GB10 nodes anyway).
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r "$REPO_ROOT/requirements-gb10.txt"
pip install --no-deps laya==0.3.21    # --no-deps: keep pip from replacing torch

python - <<'EOF'
import torch, transformers
print("torch", torch.__version__, "| transformers", transformers.__version__)
print("CUDA available:", torch.cuda.is_available())
name = torch.cuda.get_device_name(0)
print("GPU:", name, "| capability:", torch.cuda.get_device_capability(0))
x = torch.randn(1024, 1024, device="cuda", dtype=torch.bfloat16)
print("bf16 matmul OK:", torch.isfinite(x @ x).all().item())
import laya
print("laya import OK")
EOF
echo "GB10 venv ready: $VENV"
