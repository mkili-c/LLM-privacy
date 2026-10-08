#!/bin/bash --login
#SBATCH --job-name=loop_sweep
#SBATCH --account=projects
#SBATCH --gpus=gb10:1
#SBATCH --time=24:00:00
#SBATCH --export=NONE
#SBATCH --output=/work/projects/mkilic/LLM-privacy/pu_loop_sweep_%j.out

# The privacy-utility loop over an evaluation CSV (run_loop_sweep.py) on a GB10 node, with
# Qwen2.5-7B-Instruct in bf16, the backbone of every baseline. All arguments go to the script:
#     sbatch run_loop_sweep_gb10.sbatch --out results/rewrites/dev_loop.csv --limit 3
# GB10 jobs are capped at 24 h; the sweep skips the rows already written, so submitting the
# same command again continues it.
# (Arguments, not VAR=... sbatch: --export=NONE drops such variables.)

set -euo pipefail

source /work/projects/mkilic/.llmvenv-gb10/bin/activate
REPO=/work/projects/mkilic/LLM-privacy
MODEL=/scratch/users/mkilic/Qwen2.5-7B-Instruct   # revision a09a35458c702b33eeacc393d103063234e8bc28, as the baselines
cd "$REPO"

# CPU and GPU share the node's memory; the buffer cache would take part of it.
/usr/bin/drop-caches || true
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# print() to a file is block-buffered: without this the log lags minutes behind.
export PYTHONUNBUFFERED=1

echo "Job ${SLURM_JOB_ID:-manual} on $(hostname) ($(uname -m)), args: $*"
python run_loop_sweep.py --model_path "$MODEL" --dtype bfloat16 "$@"
echo "Done $(date)"
