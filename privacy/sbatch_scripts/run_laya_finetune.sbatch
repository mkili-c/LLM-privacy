#!/bin/bash
#SBATCH --job-name=laya_ft
#SBATCH --account=projects
#SBATCH --gpus=5060ti:1
#SBATCH --time=01:00:00
#SBATCH --output=laya_ft_%j.out

# Fine-tune Laya on privacy/data/laya_ft/train.csv (laya_finetune.py, the creators'
# notebook recipe). 70 pairs (402 items) are held out for the temperature fit; the
# checkpoint goes to /scratch/users/mkilic/laya-privacy-n<pairs>[-head].
#
#   sbatch run_laya_finetune.sbatch                                  # all 1,930 train pairs
#   N_PAIRS=500 sbatch run_laya_finetune.sbatch                      # learning-curve point
#   EXTRA_ARGS="--freeze_encoder" sbatch run_laya_finetune.sbatch    # head only
#
# Learning curve: for n in 250 500 1000; do N_PAIRS=$n sbatch run_laya_finetune.sbatch; done
# (one job runs at a time, the others wait in the queue).

# Required first line so `module` works in a batch job
. /etc/profile.d/modules.sh
module load cuda/12.4

set -euo pipefail

source /home/mkilic/.llmvenv/bin/activate
REPO_ROOT="${REPO_ROOT:-/work/projects/mkilic/LLM-privacy}"
cd "$REPO_ROOT/privacy"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "Job ${SLURM_JOB_ID:-manual} on $(hostname), GPU: ${CUDA_VISIBLE_DEVICES:-none}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

N_ARG=""
if [ -n "${N_PAIRS:-}" ]; then N_ARG="--n_pairs $N_PAIRS"; fi

python laya_finetune.py $N_ARG ${EXTRA_ARGS:-}

echo "Done."
