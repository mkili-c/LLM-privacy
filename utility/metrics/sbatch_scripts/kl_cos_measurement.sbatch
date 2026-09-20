#!/bin/bash
#SBATCH --job-name=kl_cos_measurement
#SBATCH --account=projects
#SBATCH --gpus=5060ti:1
#SBATCH --time=24:00:00
#SBATCH --output=floor_100_kl_cos_centroid_k10_256tok%j.out

# Measures cos both with empirical diff of means and unbiased (without diagonal)
# AND the total per token divergence (KL) for the same samples


# Required first line so `module` works in a batch job
. /etc/profile.d/modules.sh
module load cuda/12.4


set -euo pipefail

source /home/mkilic/.llmvenv/bin/activate
# Repo root -- override with `REPO_ROOT=/path/to/LLM-privacy sbatch <script>` if
# the repo lives elsewhere. Every path below is relative to it, so the job no
# longer depends on which directory you submit from.
REPO_ROOT="${REPO_ROOT:-/work/projects/mkilic/LLM-privacy}"
cd "$REPO_ROOT/utility"

#Expandable segments let the allocator grow/shrink a single segment instead of 
#stranding fixed-size blocks, so a failure stays local to the cell that caused it.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "Job ${SLURM_JOB_ID:-manual} on $(hostname), GPU: ${CUDA_VISIBLE_DEVICES:-none}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

python metrics/test_cos_kl.py \
    --csv data/privacy_prompt_rewrites_floor_100.csv \
    --k_grid 10 \
    --max_new_tokens 256 \
    --save_suffix "floor_100_kl_cos_centroid${SLURM_JOB_ID:-manual}"

echo "Done."
