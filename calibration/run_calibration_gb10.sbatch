#!/bin/bash --login
#SBATCH --job-name=calibration
#SBATCH --account=projects
#SBATCH --gpus=gb10:1
#SBATCH --time=08:00:00
#SBATCH --export=NONE
#SBATCH --output=/work/projects/mkilic/LLM-privacy/calibration/calibration_%j.out

# Calibration measurement (calibration/calibrate_thresholds.py, ROADMAP B1) on a GB10 node with
# Qwen2.5-7B-Instruct in bf16, the model of the evaluation. All arguments go to the script:
#     sbatch calibration/run_calibration_gb10.sbatch --limit 2      (smoke test; then the full run)
# The script skips dev rows already written, so submitting again continues it.

set -euo pipefail

source /work/projects/mkilic/.llmvenv-gb10/bin/activate
REPO=/work/projects/mkilic/LLM-privacy
MODEL=/scratch/users/mkilic/Qwen2.5-7B-Instruct   # revision a09a35458c702b33eeacc393d103063234e8bc28, as the baselines
cd "$REPO"

/usr/bin/drop-caches || true
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

echo "Job ${SLURM_JOB_ID:-manual} on $(hostname) ($(uname -m)), args: $*"
python calibration/calibrate_thresholds.py --model_path "$MODEL" --dtype bfloat16 "$@"
echo "Done $(date)"
