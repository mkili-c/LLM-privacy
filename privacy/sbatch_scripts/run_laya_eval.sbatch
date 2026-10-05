#!/bin/bash
#SBATCH --job-name=laya_eval
#SBATCH --account=projects
#SBATCH --gpus=5060ti:1
#SBATCH --time=01:00:00
#SBATCH --output=laya_eval_%j.out

# Base vs fine-tuned Laya (laya_eval.py) on privacy/data/laya_ft/test.csv and,
# out of distribution, utility/data/prompt_rewrites_100.csv. Base Laya is also
# scored on train.csv once, for the "base + bias" row.
# Results -> privacy/results/laya_eval/laya_eval_<job>{.json,_synthetic.csv,_ood.csv}
#
#   sbatch run_laya_eval.sbatch
#   CKPTS="/scratch/users/mkilic/laya-privacy-n1930" sbatch run_laya_eval.sbatch

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

S=/scratch/users/mkilic
CKPTS="${CKPTS:-$S/laya-privacy-n250 $S/laya-privacy-n500 $S/laya-privacy-n1000 $S/laya-privacy-n1930}"

python laya_eval.py --checkpoints $CKPTS --save_suffix "laya_eval_${SLURM_JOB_ID:-manual}"

echo "Done."
