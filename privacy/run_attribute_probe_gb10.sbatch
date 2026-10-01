#!/bin/bash --login
#SBATCH --job-name=attr_probe_gb10
#SBATCH --account=projects
#SBATCH --gpus=gb10:1
#SBATCH --time=00:45:00
#SBATCH --export=NONE
#SBATCH --output=attr_probe_gb10_7_%j.out

# Attribute probe sweep over prompt_rewrites_100.csv on a GB10 node, for
# comparing larger Qwen models with the 3B sweep (job 324, 61 s on a 5060ti).
# 1200 single forward passes, no generation: ~10 min for the 32B.
#     sbatch run_attribute_probe_gb10.sbatch                     # 32B
#     sbatch run_attribute_probe_gb10.sbatch /scratch/users/mkilic/Qwen2.5-7B-Instruct
# (an argument, not MODEL_PATH=... sbatch: --export=NONE drops that variable)

set -euo pipefail

source /work/projects/mkilic/.llmvenv-gb10/bin/activate
REPO_ROOT=/work/projects/mkilic/LLM-privacy
MODEL_PATH="${1:-/scratch/users/mkilic/Qwen2.5-7B-Instruct}"
cd "$REPO_ROOT/privacy"

/usr/bin/drop-caches || true
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

echo "Job ${SLURM_JOB_ID:-manual} on $(hostname) ($(uname -m)), model: $MODEL_PATH"

python attribute_probe_sweep.py \
    --model_path "$MODEL_PATH" \
    --csv ../utility/data/prompt_rewrites_100.csv \
    --save_suffix "$(basename "$MODEL_PATH")_attribute_probe_${SLURM_JOB_ID:-manual}"

echo "Done."
