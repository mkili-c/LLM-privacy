#!/bin/bash
#SBATCH --job-name=priv_adv
#SBATCH --account=projects
#SBATCH --gpus=5060ti:1
#SBATCH --time=00:40:00
#SBATCH --output=priv_adv_%j.out

# Adversarial advantage 2|Pr[f_hat = f] - 1/2| for AGE (adult vs minor) and
# GENDER (woman vs not), over all five columns of privacy_prompt_rewrites_100.csv.
#
# Every Yes/No surface form is a single token under the Qwen2.5 tokenizer, so
# each readout is ONE prefill of a ~100-token prompt -- no generation. Each of
# the six runs below is 100 rows x 5 columns x 2 polarities = 1000 forward
# passes, under a minute of GPU time; the wall clock is six model loads plus slack.

# Required first line so `module` works in a batch job
. /etc/profile.d/modules.sh
module load cuda/12.4

set -euo pipefail

source /home/mkilic/.llmvenv/bin/activate
# Repo root -- override with `REPO_ROOT=/path/to/LLM-privacy sbatch <script>` if
# the repo lives elsewhere. Every path below is relative to it, so the job no
# longer depends on which directory you submit from.
REPO_ROOT="${REPO_ROOT:-/work/projects/mkilic/LLM-privacy}"
cd "$REPO_ROOT/privacy"

# confidence_assesment.py puts ../utility/metrics on sys.path itself, so running
# this directory needs no PYTHONPATH. Results land in ./results/privacy_advantage
# (the path is relative to the cwd), keeping them out of utility/results.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "Job ${SLURM_JOB_ID:-manual} on $(hostname), GPU: ${CUDA_VISIBLE_DEVICES:-none}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

CSV=../utility/data/privacy_prompt_rewrites_100.csv

# Both attributes are read out of the profile string under `id` ("Marta, 34,
# female, midwife, ..."): the age as the one comma-separated field that is a
# bare number, the gender from woman/not-a-woman markers. All 100 rows parse
# under either rule.
LABEL_COLUMN=id

# original is the attack surface; the four rewrites are what is supposed to
# shrink the advantage. Ordered so the two privacy rewrites sit last, next to
# each other in the summary table.
COLUMNS=(original
         util_preserving_rewrite
         util_paraphrase_rewrite
         priv_preserving_rewrite
         priv_targeted_rewrite)

# The attribute axis. Each entry is <tag>:<args> -- the tag names the output
# file, the args select the attribute. Deliberately a list and not a product
# with a threshold sweep, since the threshold only means anything for age.
#
#   age18   the threshold that matters, and the one this file has NO variance
#           on -- its ages run 19..71, so every row is an adult, the base rate
#           is 1.0 and the best constant adversary already scores advantage
#           1.0. The script warns and runs anyway: the per-row probabilities
#           and the WITHOUT PROMPT prior are still worth having, and the
#           accuracy column is a readout of the model's answering bias on age
#           questions. What it cannot show is leakage. Read nothing else into
#           it. Fixing this needs profiles of minors in the CSV, not a
#           different estimator.
#   age38   the median age, so the split is 53/47 and the baseline advantage is
#           0.06. This is the age run that measures something: still age
#           leakage, just at the only cut point this population supports.
#   gender  one-vs-rest, "is the user a woman?", with P(not a woman) = 1 - p by
#           construction. 49/51 and a baseline of 0.02 -- the best-conditioned
#           of the three, so it is the one to read first if the numbers
#           disagree.
ATTRIBUTES=("age18:--attribute age --age_threshold 18"
            "age38:--attribute age --age_threshold 38"
            "gender:--attribute gender")

# Both baseline conditions, because the whole measurement is a difference
# against one of them and they are not interchangeable:
#   dropped  the message block is absent      -- the model's unconditioned prior
#   empty    the block is present and empty   -- tighter control, since only the
#            information changes and not the wording of the query
# Only the no-context readout differs between them (build_query ignores
# ablate_context whenever a real message is passed), so the WITH PROMPT numbers
# must agree row for row across the two JSONs for a given attribute. They are a
# free consistency check -- if they disagree, something is nondeterministic.
for SPEC in "${ATTRIBUTES[@]}"; do
    TAG=${SPEC%%:*}
    # read -ra rather than an unquoted expansion: the split into argv happens
    # here, once and visibly, instead of at the call site under `set -u`.
    read -ra ATTR_ARGS <<< "${SPEC#*:}"

    for PRIOR in dropped empty; do
        ABLATE=()
        [[ $PRIOR == empty ]] && ABLATE=(--ablate_context)

        echo
        echo "============ $TAG   |   prior condition: $PRIOR ============"
        python confidence_assesment.py \
            --csv "$CSV" \
            --columns "${COLUMNS[@]}" \
            --label_column "$LABEL_COLUMN" \
            "${ATTR_ARGS[@]}" \
            --counterbalance \
            "${ABLATE[@]}" \
            --save_suffix "qwen_${SLURM_JOB_ID:-manual}_${TAG}_prior_${PRIOR}"
    done
done

echo "Done."
