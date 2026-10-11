#!/bin/bash --login
#SBATCH --job-name=probe_speed
#SBATCH --account=projects
#SBATCH --gpus=gb10:1
#SBATCH --time=01:00:00
#SBATCH --export=NONE
#SBATCH --output=/work/projects/mkilic/LLM-privacy/privacy/sbatch_scripts/check_probe_speed_%j.out

# The probe speed-ups (output layer on the last positions only; the message encoded once for all
# attribute questions) against the probe as it was: same P, and how much faster. ~25 min.
#     sbatch privacy/sbatch_scripts/run_check_probe_speed_gb10.sbatch

set -euo pipefail

source /work/projects/mkilic/.llmvenv-gb10/bin/activate
cd /work/projects/mkilic/LLM-privacy

/usr/bin/drop-caches || true
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

echo "Job ${SLURM_JOB_ID:-manual} on $(hostname) ($(uname -m))"
python privacy/check_probe_speed.py "$@"
echo "Done $(date)"
