#!/bin/bash
#SBATCH --job-name=attr_probe
#SBATCH --account=projects
#SBATCH --gpus=5060ti:1
#SBATCH --time=01:00:00
#SBATCH --output=attr_probe_%j.out

# Attribute probe over prompt_rewrites_100.csv: every row x message column,
# on that row's own protected attributes. 100 rows x 6 columns x 2 attributes
# = 1200 single forward passes, no generation -- a few minutes of GPU time.

# Required first line so `module` works in a batch job
. /etc/profile.d/modules.sh
module load cuda/12.4

set -euo pipefail

source /home/mkilic/.llmvenv/bin/activate
# Repo root -- override with `REPO_ROOT=/path/to/LLM-privacy sbatch <script>` if
# the repo lives elsewhere.
REPO_ROOT="${REPO_ROOT:-/work/projects/mkilic/LLM-privacy}"
cd "$REPO_ROOT/privacy"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "Job ${SLURM_JOB_ID:-manual} on $(hostname), GPU: ${CUDA_VISIBLE_DEVICES:-none}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

python attribute_probe_sweep.py \
    --csv ../utility/data/prompt_rewrites_100.csv \
    --save_suffix "qwen_attribute_probe_${SLURM_JOB_ID:-manual}"

echo "Done."
