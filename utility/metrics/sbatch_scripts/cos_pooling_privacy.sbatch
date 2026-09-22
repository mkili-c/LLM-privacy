#!/bin/bash
#SBATCH --job-name=cos_pooling_privacy
#SBATCH --account=projects
#SBATCH --gpus=5060ti:1
#SBATCH --time=24:00:00
#SBATCH --output=privacy_cos_pooling_k15and30_256tok_%j.out

# Pooling sweep (mean / last-token / position-weighted) of the cos measurement
# (udist2, rel_udist2, cos_centroid, dist2) over the 100 privacy rewrites,
# final hidden layer


# Required first line so `module` works in a batch job
. /etc/profile.d/modules.sh
module load cuda/12.4


set -euo pipefail

source /home/mkilic/.llmvenv/bin/activate
# Repo root -- override with `REPO_ROOT=/path/to/LLM-privacy sbatch <script>` if
# the repo lives elsewhere. Every path below is relative to it.
REPO_ROOT="${REPO_ROOT:-/work/projects/mkilic/LLM-privacy}"
cd "$REPO_ROOT/utility"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "Job ${SLURM_JOB_ID:-manual} on $(hostname), GPU: ${CUDA_VISIBLE_DEVICES:-none}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

python metrics/cos_measurement_pooling_privacy.py \
    --csv data/prompt_rewrites_100.csv \
    --k_grid 15 30\
    --max_new_tokens 256 \
    --layer -1 \
    --save_suffix "privacy_cos_pooling_k15and30_tok256_${SLURM_JOB_ID:-manual}"

echo "Done."
