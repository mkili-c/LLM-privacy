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

## Environment details

The versions all results in `results/` come from.

### Hardware and OS

| Machine | Used for |
|---|---|
| Login node: x86_64, no GPU, Ubuntu 24.04.5 LTS (kernel 7.0.0) | analysis and plotting, Presidio |
| NVIDIA GB10 (Grace Blackwell: Blackwell GPU + 20-core Arm Grace CPU, 128 GB unified memory, compute capability 12.1), aarch64 | everything with the 7B: the loop, the baselines' model server, our measures, calibration, 7B probe sweeps |
| NVIDIA GeForce RTX 5060 Ti (16 GB, compute capability 12.0), x86_64 | Laya (probe sweeps, fine-tuning, evaluation), dev-set utility measurements with the 3B |

Slurm account `projects`; GPU types `gb10` and `5060ti`. Each `*.sbatch` script names its GPU type, time limit and venv. The GPU driver version was not recorded; it must support CUDA 12.8.

### Python and packages

All venvs use the system Python 3.12.3 (`/usr/bin/python3.12`). Venvs are not relocatable, because `activate`, the console-script shebangs and `pyvenv.cfg` hold absolute paths. Rebuild them with the setup scripts instead of copying them to another path or machine. The GB10 venvs only run on aarch64, the others only on x86_64.

| Venv | Key packages (the full lists are the frozen requirement files) |
|---|---|
| `~/.llmvenv` (x86) | torch 2.11.0+cu128, transformers 5.8.0, accelerate 1.13.0, numpy 2.4.4, matplotlib 3.11.1, laya 0.3.21 |
| `.llmvenv-gb10` (aarch64) | the same versions; torch 2.11.0+cu128 is the aarch64 build |
| `venvs/presidio` (x86) | presidio-analyzer / -anonymizer 2.2.364, spaCy 3.8.16, `en_core_web_lg` 3.8.0 |
| `venvs/loiseau` (aarch64) | DSPy 3.0.4 (as in their paper), litellm 1.104.0, openai 2.54.0 |
| `venvs/papillon` (aarch64) | DSPy 3.1.2 (their dspy-ai 2.5.35 environment no longer installs), litellm 1.104.0, openai 2.54.0 |
| `venvs/staab` (aarch64) | openai 0.28.1 (as in their `environment.yaml`), plus a CPU-only torch 2.14.1 that their code imports but never uses |

### CUDA

torch comes from the CUDA 12.8 wheel index (`cu128`) and bundles its own CUDA libraries, the same in both GPU venvs: CUDA runtime 12.8.90, cuBLAS 12.8.4.1, cuDNN 9.19.0.56, NCCL 2.28.9. No system CUDA toolkit is needed. Some older 5060 Ti scripts load the cluster module `cuda/12.4`, but torch uses its bundled libraries. Both GPUs are Blackwell (compute capability 12.0 and 12.1), so they need a torch build with Blackwell support, such as 2.11 cu128. laya is installed with `--no-deps`, so pip cannot replace this torch.

### Settings that affect the outputs

- **Precision:** Qwen runs in bfloat16 on the GB10 (the loop, the server, our measures). Laya scores in float32.
- **Model server:** `transformers serve`, without continuous batching, because it would drop Qwen's repetition penalty. A baseline that does not set a decoding parameter gets Qwen's `generation_config.json` defaults: sampling, temperature 0.7, top-p 0.8, top-k 20, repetition penalty 1.05. Staab et al. set temperature 0.1.
- **Other seeds:** Staab et al.'s config uses seed 10; the Laya fine-tuning uses seed 0.
- **Not pinned:** `sentence-transformers/all-MiniLM-L6-v2`, which Staab et al.'s code loads from the Hugging Face cache at import (`src/utils/string_utils.py`). It maps their attacker's answers to attribute types (`src/reddit/reddit.py`), so it can affect their outputs. Its revision was not recorded.
- **Paths:** the paths above are hard-coded in the job scripts; on another account or cluster, change them there.
