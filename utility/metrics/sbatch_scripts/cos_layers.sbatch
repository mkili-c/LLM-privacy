#!/bin/bash
#SBATCH --job-name=cos_measurement_layers
#SBATCH --account=projects
#SBATCH --gpus=5060ti:1
#SBATCH --time=12:00:00
#SBATCH --output=paraphrases_cos_layers_k10_256tok_%j.out

# Measures cos both with empirical diff of means and unbiased (without diagonal)
#for the whole dataset


# Required first line so `module` works in a batch job
. /etc/profile.d/modules.sh
module load cuda/12.4


set -euo pipefail

source /home/mkilic/.llmvenv/bin/activate
cd "${SLURM_SUBMIT_DIR:-/work/projects/mkilic/privacy-utility/utility}"

#Expandable segments let the allocator grow/shrink a single segment instead of 
#stranding fixed-size blocks, so a failure stays local to the cell that caused it.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "Job ${SLURM_JOB_ID:-manual} on $(hostname), GPU: ${CUDA_VISIBLE_DEVICES:-none}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

python cos_measurement_layers.py \
    --csv data/prompt_paraphrases_100.csv \
    --k_grid 10 \
    --max_new_tokens 256 \
    --save_suffix "paraphrases_cos_layers_${SLURM_JOB_ID:-manual}"

echo "Done."
