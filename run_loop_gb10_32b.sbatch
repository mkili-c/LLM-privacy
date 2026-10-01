#!/bin/bash --login
#SBATCH --job-name=pu_loop_32b
#SBATCH --account=projects
#SBATCH --gpus=gb10:1
#SBATCH --time=02:00:00
#SBATCH --export=NONE
#SBATCH --output=pu_loop_32b_%j.out

# Privacy-utility loop with Qwen2.5-32B-Instruct (bf16, ~65 GB) on a GB10 node,
# with reduced settings: generation runs at only a few tokens/s at this size.
# Estimated 10-15 min for a message that converges in one iteration, up to
# ~1 h for hard ones (extrapolated from the 7B run, job 347).
#     sbatch run_loop_gb10_32b.sbatch
# Download first:
#     hf download Qwen/Qwen2.5-32B-Instruct --local-dir /scratch/users/mkilic/Qwen2.5-32B-Instruct

set -euo pipefail

source /work/projects/mkilic/.llmvenv-gb10/bin/activate
REPO_ROOT=/work/projects/mkilic/LLM-privacy
MODEL_PATH=/scratch/users/mkilic/Qwen2.5-32B-Instruct
cd "$REPO_ROOT"

# CPU and GPU share the node's memory; the buffer cache would take part of it.
/usr/bin/drop-caches || true
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# print() to a file is block-buffered: without this the log lags minutes behind.
export PYTHONUNBUFFERED=1

echo "Job ${SLURM_JOB_ID:-manual} on $(hostname) ($(uname -m)), model: $MODEL_PATH"
free -g | head -2

python privacy_utility_loop.py \
    --model_path "$MODEL_PATH" \
    --dtype bfloat16 \
    --text "I hurt my eyes while I was watching the fireworks on 1st of August, what should I do?" \
    --max_iters 1 \
    --privacy_cand_rounds 1 \
    --utility kl_multi \
    --utility_args "--n_steps 2 --k 2 --cand_rounds 1 --cand_max_new_tokens 80"

echo "Done."
