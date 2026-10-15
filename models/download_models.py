"""
Download the models the experiments use, each at the exact Hugging Face revision of our runs, to
the paths the scripts read them from. The weights are not in git (.gitignore).

    Qwen2.5-7B-Instruct   backbone of the loop, the baselines and the evaluation
                          (on /scratch, which may be cleaned: download it again from here)
    Qwen2.5-3B-Instruct   the dev-set utility measurements and DEFAULT_MODEL of the probe scripts
    laya                  the base Laya checkpoint, the starting point of the fine-tuning

The fine-tuned Laya (models/laya-privacy, the loop's second attacker) is ours and not on the Hub.
Restore it from its backup; retraining it (privacy/laya_finetune.py, seed 0, the n1930 run on
privacy/data/laya_ft/train.csv) gives a close but not bit-identical model.

Run it in the x86 venv (setup_x86.sh); it only downloads, so the login node is fine:
    python models/download_models.py --check          # revisions exist, nothing downloaded
    python models/download_models.py                  # all three
    python models/download_models.py qwen7b           # one of them
"""
import argparse

from huggingface_hub import HfApi, snapshot_download

# name -> (Hub repo, revision of our runs, local directory the scripts read)
MODELS = {
    "qwen7b": ("Qwen/Qwen2.5-7B-Instruct", "a09a35458c702b33eeacc393d103063234e8bc28",
               "/scratch/users/mkilic/Qwen2.5-7B-Instruct"),
    "qwen3b": ("Qwen/Qwen2.5-3B-Instruct", "aa8e72537993ba99e69dfaafa59ed015b17504d1",
               "/work/projects/mkilic/models/Qwen2.5-3B-Instruct"),
    "laya": ("convaiinnovations/laya", "55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851",
             "/work/projects/mkilic/models/laya"),
}


def get_args():
    p = argparse.ArgumentParser(description="Download the models at the revisions of our runs")
    p.add_argument("names", nargs="*", help=f"models to fetch, of {', '.join(MODELS)} (default: all)")
    p.add_argument("--check", action="store_true", help="only check that each revision exists")
    args = p.parse_args()
    # checked here: argparse's `choices` rejects a list default for nargs="*"
    unknown = [n for n in args.names if n not in MODELS]
    if unknown:
        p.error(f"unknown model(s) {', '.join(unknown)}; choose from {', '.join(MODELS)}")
    args.names = args.names or list(MODELS)
    return args


def check(repo, revision):
    """Raises if the revision is gone from the Hub."""
    info = HfApi().model_info(repo, revision=revision)
    print(f"ok    {repo} @ {info.sha[:7]} ({len(info.siblings)} files)")


def download(repo, revision, local_dir):
    path = snapshot_download(repo_id=repo, revision=revision, local_dir=local_dir)
    print(f"done  {repo} @ {revision[:7]} -> {path}")


def main():
    args = get_args()
    for name in args.names:
        repo, revision, local_dir = MODELS[name]
        if args.check:
            check(repo, revision)
        else:
            download(repo, revision, local_dir)


if __name__ == "__main__":
    main()
