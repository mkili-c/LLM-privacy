# LLM-privacy

Protecting user privacy against inference capabilites of large language models.

Prompt anonymizer using Kullback-Leibler divergence for utility measurement with a Qwen backbone and Qwen and Laya for privacy measurement. It finds the spans of a prompt that reveal personal attributes, lets an LLM rewrite them, and accepts each rewrite only if a utility check passes.

## Reproduce

Everything the thesis reports is in this repository: code, data (PUPA-SD, the development sets, Laya's training data) and every result (rewrites, our measures, the judge's items, keys and answers). The tables and figures can be regenerated from these files on a CPU, without rerunning any model (step 5). Rerunning the experiments needs the environments and models below.

### 1. Environments

| Environment | Where it runs | What runs in it | Build it with |
|---|---|---|---|
| `~/.llmvenv` | x86 (login node, RTX 5060 Ti) | Laya jobs, dev-set utility measurements, all analysis and plotting | `bash setup_x86.sh` (on the login node) |
| `/work/projects/mkilic/.llmvenv-gb10` | GB10 (aarch64) | the loop, calibration, our measures, the model server for the baselines | `bash setup_gb10.sh` (inside a GB10 session, see the script) |
| `/work/projects/mkilic/venvs/<method>` | presidio: x86; loiseau, papillon, staab: GB10 | one per baseline, since they pin conflicting packages | `bash baselines/setup_venvs.sh <method>` |

All three are built from frozen requirement files (`requirements-x86.txt`, `requirements-gb10.txt`, `baselines/requirements-<method>.txt`). The baselines' code is cloned at pinned commits, with our one patch applied, by `bash baselines/fetch_repos.sh`.

### 2. Models

    python models/download_models.py --check   # the pinned revisions still exist
    python models/download_models.py           # Qwen2.5-7B, Qwen2.5-3B, base Laya

Each model is downloaded at the exact revision of our runs, to the path the scripts read. The fine-tuned Laya (`/work/projects/mkilic/models/laya-privacy`, the loop's second attacker) is ours and not on the Hub: restore it from its backup, or retrain it with `sbatch privacy/sbatch_scripts/run_laya_finetune.sbatch` (all 1,930 training pairs, seed 0) and copy `/scratch/users/mkilic/laya-privacy-n1930` there. A retrained model is close to, but not bit-identical with, the one of our runs.

### 3. Runs (GPU)

    sbatch run_loop_sweep_gb10.sbatch --data data/pupa_sd/pupa_sd_eval.csv --out results/rewrites/pupa_sd_loop.csv
    sbatch baselines/run_sweep_gb10.sbatch "loiseau_pupa loiseau_synthpai papillon staab"
    srun -A projects -p interactive-cpu -t 00:15:00 /work/projects/mkilic/venvs/presidio/bin/python run_baselines.py --method presidio --data data/pupa_sd/pupa_sd_eval.csv
    sbatch evaluation/run_score_own_measures_gb10.sbatch
    sbatch calibration/run_calibration_gb10.sbatch

GB10 jobs stop after 24 h; every sweep skips the rows it has already written, so submitting the same command again continues it. Development-set measurements of the attackers: `privacy/sbatch_scripts/run_attribute_probe_gb10.sbatch` (Qwen) and `privacy/sbatch_scripts/run_laya_probe.sbatch` (Laya; `--laya_path ... --save_suffix laya_privacy_probe` for the fine-tuned one).

### 4. The LLM judge

    python evaluation/write_judge_items.py          # results/evaluation/judge_upload/ (guess_* and utility folders)
    # each folder -> its own private repo -> one Claude Code session (see judge_upload/README.md)
    # copy each answers.jsonl to results/evaluation/judge_answers/<folder>/
    python evaluation/check_judge_answers.py <folder>
    python evaluation/write_compare_items.py        # the compare folder, after all guesses are back
    # compare folder -> its own session -> results/evaluation/judge_answers/compare/

### 5. Tables and figures (CPU, from the stored results)

    python evaluation/read_judge_answers.py          # main, per-attribute and agreement tables; privacy_utility.png
    python evaluation/plot_attacker_recall.py        # Qwen vs Laya per attribute
    python calibration/analyze_calibration.py        # the R >= 100% rule
    python privacy/plot_probe_rewrites.py            # attackers on the development rewrites
    python privacy/plot_laya_learning_curve.py
    python utility/metrics/plotting/plot_k_sweep.py --ranking_pair

All from the repository root, in the x86 venv.

### What a rerun reproduces

The same hardware, software and seeds should give the same outputs: the loop resets its seed (23) per prompt, and the baselines' server uses seed 0 and answers one request at a time. On other GPUs, bf16 kernels make reruns close but not bit-identical. The judge (Claude in cloud sessions) is not deterministic: our numbers are reproducible from its stored answers, and a new judge run gives close but not identical ones.
