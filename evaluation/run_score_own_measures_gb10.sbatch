#!/bin/bash --login
#SBATCH --job-name=own_measures
#SBATCH --account=projects
#SBATCH --gpus=gb10:1
#SBATCH --time=06:00:00
#SBATCH --export=NONE
#SBATCH --output=/work/projects/mkilic/LLM-privacy/evaluation/own_measures_%j.out

# The loop's own privacy and utility measures on every method's rewrites of PUPA-SD and on the
# originals (evaluation/score_own_measures.py). ~3.5 h. A rerun continues where it stopped.
#     sbatch evaluation/run_score_own_measures_gb10.sbatch            (all methods)
#     sbatch evaluation/run_score_own_measures_gb10.sbatch --out_dir results/evaluation_smoke --limit 2
# (Arguments, not VAR=... sbatch: --export=NONE drops such variables.)

set -euo pipefail

source /work/projects/mkilic/.llmvenv-gb10/bin/activate
cd /work/projects/mkilic/LLM-privacy

/usr/bin/drop-caches || true
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

echo "Job ${SLURM_JOB_ID:-manual} on $(hostname) ($(uname -m)), args: $*"
python evaluation/score_own_measures.py "$@"
echo "Done $(date)"
