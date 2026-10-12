#!/bin/bash --login
#SBATCH --job-name=profile_probe
#SBATCH --account=projects
#SBATCH --gpus=gb10:1
#SBATCH --time=01:00:00
#SBATCH --export=NONE
#SBATCH --output=/work/projects/mkilic/LLM-privacy/privacy/sbatch_scripts/profile_probe_%j.out

# Where the attacker probe's time goes on the GB10 (privacy/profile_probe.py), and whether a change
# that keeps P identical could make it faster. Nothing here changes a result. ~25-35 min.
#     sbatch privacy/sbatch_scripts/run_profile_probe_gb10.sbatch

set -euo pipefail

source /work/projects/mkilic/.llmvenv-gb10/bin/activate
cd /work/projects/mkilic/LLM-privacy

/usr/bin/drop-caches || true
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

echo "Job ${SLURM_JOB_ID:-manual} on $(hostname) ($(uname -m))"
python privacy/profile_probe.py "$@"
echo "Done $(date)"
