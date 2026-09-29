#!/usr/bin/env bash
# Demo of the privacy tools in this directory, on one GPU:
#   1. attribute_probe.py     Qwen 2.5-3B: P(attribute can be inferred) from the Yes/No logits
#   2. privacy_token_grad.py  Qwen: occlusion + gradient attribution of sum_a -log(1 - P_a)
#   3. laya_probe.py          Laya: same question as a calibrated classifier, word occlusion
#
# Usage:  bash demo_privacy.sh        (requests the GPU itself via srun)
set -euo pipefail
cd "$(dirname "$0")"

if [ -z "${SLURM_JOB_ID:-}" ]; then
    exec srun -A projects --gpus 5060ti:1 -t 00:20:00 bash "$0" "$@"
fi
source /home/mkilic/.llmvenv/bin/activate
export PYTHONWARNINGS="ignore::RuntimeWarning"   # laya's clamped choice:11+ temperature

NEUTRAL="How can I manage stress?"
LOCATION="Best places to eat near Zurich HB? I just moved here from Munich."
AGE="Can you help with my 7th grade math homework? My mom says I have to finish before dinner."
ALL="I am a 25 year old woman from Zurich and my period is late, could I be pregnant?"

quiet() { "$@" 2> >(tr "\r" "\n" | grep -v -e "Loading weights" -e "clean_up_tokenization" -e "^$" >&2); }

section() { printf '\n\n########## %s ##########\n' "$1"; }

section "1. Qwen probe: can the attribute be inferred? (one attribute per run)"
for a in gender age "geographic location"; do
    quiet python attribute_probe.py --attribute "$a" --text "$NEUTRAL" "$LOCATION" "$AGE" "$ALL" \
        | grep -E "^question|^P\(yes\)|^  '"
done

section "2. Qwen attribution: which tokens carry the leak?"
quiet python privacy_token_grad.py --text "$ALL" --top_k 8 | grep -v "^Loading"

section "3. Laya probe: all attributes in one pass, word-level occlusion"
quiet python laya_probe.py --check --top_k 8 --text "$ALL"
quiet python laya_probe.py --top_k 5 --text "$LOCATION"
