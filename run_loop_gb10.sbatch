#!/bin/bash --login
#SBATCH --job-name=pu_loop_gb10
#SBATCH --account=projects
#SBATCH --gpus=gb10:1
#SBATCH --time=02:00:00
#SBATCH --export=NONE
#SBATCH --output=pu_loop_gb10_%j.out

# Privacy-utility loop on a GB10 node, unquantized bf16 model.
# Run ./setup_gb10.sh once first. GB10 jobs are capped at 24 h.
#     sbatch run_loop_gb10.sbatch
#     sbatch run_loop_gb10.sbatch /scratch/users/mkilic/Qwen2.5-14B-Instruct
# (an argument, not MODEL_PATH=... sbatch: --export=NONE drops that variable)
# --export=NONE and the login shell: the job must not inherit the login
# node's environment (its module paths do not work on the GB10 nodes).

set -euo pipefail

source /work/projects/mkilic/.llmvenv-gb10/bin/activate
REPO_ROOT="${REPO_ROOT:-/work/projects/mkilic/LLM-privacy}"
MODEL_PATH="${1:-/scratch/users/mkilic/Qwen2.5-7B-Instruct}"
cd "$REPO_ROOT"

# CPU and GPU share the node's memory; the buffer cache would take part of it.
/usr/bin/drop-caches || true
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# print() to a file is block-buffered: without this the log lags minutes behind.
export PYTHONUNBUFFERED=1

echo "Job ${SLURM_JOB_ID:-manual} on $(hostname) ($(uname -m)), model: $MODEL_PATH"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

python privacy_utility_loop.py \
    --model_path "$MODEL_PATH" \
    --dtype bfloat16 \
    --text "I hurt my eyes while I was watching the fireworks on 1st of August, what should I do?" \
    --k 4

echo "Done."
