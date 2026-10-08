#!/bin/bash --login
#SBATCH --job-name=baselines
#SBATCH --account=projects
#SBATCH --gpus=gb10:1
#SBATCH --time=24:00:00
#SBATCH --export=NONE
#SBATCH --output=/work/projects/mkilic/LLM-privacy/baselines/logs/sweep_%j.out

# Serve Qwen2.5-7B-Instruct with `transformers serve` on a GB10 node and run baselines
# through it, one request at a time (no continuous batching: Qwen decodes exactly as its
# generation_config says, and the fixed seed and order make reruns reproducible).
#     sbatch baselines/run_sweep_gb10.sbatch "loiseau_pupa loiseau_synthpai papillon" --limit 3
# First argument: the methods, space-separated; the rest goes to run_baselines.py.
# (Arguments, not VAR=... sbatch: --export=NONE drops such variables.)

set -euo pipefail

REPO=/work/projects/mkilic/LLM-privacy
VENVS=/work/projects/mkilic/venvs
MODEL=/scratch/users/mkilic/Qwen2.5-7B-Instruct   # revision a09a35458c702b33eeacc393d103063234e8bc28
METHODS=$1
shift
cd "$REPO"

# print() to a file is block-buffered: without this the log lags minutes behind.
export PYTHONUNBUFFERED=1
# DSPy caches answers on disk (default ~/.dspy_cache, i.e. /home). A fresh, per-job cache:
# a hit from an earlier job would skip a server call and shift the seeded sampling of every
# later request, so a rerun would no longer reproduce the outputs.
export DSPY_CACHEDIR=/scratch/users/mkilic/dspy_cache/${SLURM_JOB_ID:-manual_$$}
# The baselines pass api_key="" (as in PAPILLON's scripts); the OpenAI client treats that as
# missing and litellm then falls back to this variable. The local server ignores the key.
export OPENAI_API_KEY=local-server-no-key
# Staab et al.'s openai 0.28 reads its endpoint from here (the DSPy baselines pass api_base).
export OPENAI_API_BASE=http://localhost:8000/v1
SERVE_LOG=baselines/logs/serve_${SLURM_JOB_ID:-manual}.log

echo "Job ${SLURM_JOB_ID:-manual} on $(hostname) ($(uname -m)), methods: $METHODS"
/work/projects/mkilic/.llmvenv-gb10/bin/transformers serve "$MODEL" --default-seed 0 --port 8000 > "$SERVE_LOG" 2>&1 &
SERVER=$!
trap 'kill $SERVER 2>/dev/null || true' EXIT
until curl -s localhost:8000/health > /dev/null; do
    kill -0 $SERVER 2>/dev/null || { echo "server died:"; tail -20 "$SERVE_LOG"; exit 1; }
    sleep 5
done
echo "server up $(date)"

for method in $METHODS; do
    case $method in
        loiseau_*) venv=loiseau ;;
        *)         venv=$method ;;
    esac
    echo "== $method (venv $venv) $(date)"
    "$VENVS/$venv/bin/python" run_baselines.py --method "$method" "$@"
done
echo "Done $(date)"
