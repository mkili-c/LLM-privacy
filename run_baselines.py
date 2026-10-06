"""Run one anonymization baseline over an evaluation set and record its rewrites.

    python run_baselines.py --method presidio --data data/pupa_sd/pupa_sd_eval.csv

Reads `id` and `original` from the evaluation CSV, gives each prompt to the method's
adapter, and appends one row per prompt to the output CSV:
    id, method, rewrite, error     the rewrite, or the exception if the call failed
    seconds                        wall-clock time of this anonymization
    calls, prompt_tokens,          LLM calls it made and the tokens the server read and
    completion_tokens              generated for them (as the server reports; 0 for Presidio)
A row whose call fails gets an empty rewrite and the exception in `error`. Ids with a
successful rewrite in the output are skipped, so an interrupted run resumes where it stopped
and retries the failures; for each id the last row in the file counts.

Each method runs in its own venv (/work/projects/mkilic/venvs/<venv>), because the
baselines pin conflicting packages: presidio -> presidio, loiseau_* -> loiseau
(DSPy 3.0.4, as in their paper), papillon -> papillon (DSPy 3.1.2: PAPILLON's README says
its code runs with the newest DSPy for inference, and the authors' 2026 follow-up uses 3.1.2).
The LLM baselines talk to Qwen2.5-7B-Instruct through `transformers serve` at SERVER
(baselines/run_sweep_gb10.sbatch starts it). The adapters only feed the prompt to the
released code (baselines/repos, pinned by baselines/fetch_repos.sh) and read its
output; they never change the method.
"""
import argparse
import csv
import os
import re
import sys
import time
from pathlib import Path

METHODS = ["presidio", "loiseau_pupa", "loiseau_synthpai", "papillon"]  # staab follows
OUT_TEMPLATE = "results/rewrites/{data}_{method}.csv"  # {data}: the eval CSV's folder name
HERE = Path(__file__).resolve().parent
REPOS = HERE / "baselines" / "repos"
SERVER = "http://localhost:8000/v1"
SERVED_MODEL = "/scratch/users/mkilic/Qwen2.5-7B-Instruct"  # the server answers to the path it loaded
# Loiseau et al.'s GEPA-optimized instructions for Qwen2.5-7B, one per task they optimized on
LOISEAU_PROMPTS = {
    "loiseau_pupa": REPOS / "adaptive-text-anonymization" / "prompts" / "pupa" / "qwen_2_5_7b.md",
    "loiseau_synthpai": REPOS / "adaptive-text-anonymization" / "prompts" / "synthpai" / "qwen_2_5_7b.md",
}


def get_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--method", choices=METHODS, required=True)
    p.add_argument("--data", default="data/pupa_sd/pupa_sd_eval.csv")
    p.add_argument("--out", default=None, help=f"default: {OUT_TEMPLATE}")
    p.add_argument("--limit", type=int, default=None, help="only the first N rows (smoke test)")
    return p.parse_args()


def load_presidio():
    """Presidio as released: default recognizers, default `replace` operator (entity -> <TYPE>)."""
    from presidio_analyzer import AnalyzerEngine
    from presidio_anonymizer import AnonymizerEngine
    analyzer, anonymizer = AnalyzerEngine(), AnonymizerEngine()

    def rewrite(text):
        results = analyzer.analyze(text=text, language="en")
        return anonymizer.anonymize(text=text, analyzer_results=results).text, 0, 0, 0
    return rewrite


def call_counting_usage(lm, call):
    """Run call() and return (its result, LLM calls, prompt tokens, completion tokens) from the
    usage the server reported to `lm`. The history is cleared first: it is capped, so its
    length alone cannot tell how many entries one call added."""
    lm.history.clear()
    result = call()
    usage = [entry.get("usage") or {} for entry in lm.history]
    return (result, len(usage), sum(u.get("prompt_tokens") or 0 for u in usage),
            sum(u.get("completion_tokens") or 0 for u in usage))


def load_loiseau(method):
    """Loiseau et al. as released: their program is dspy.Predict("text:str -> anonymized_text:str")
    (optimizer.py), and the .md file holds its optimized signature.instructions (export_prompts.py).
    LM settings as in their models.py: max_tokens 25,000, everything else DSPy's default."""
    import dspy
    md = LOISEAU_PROMPTS[method].read_text(encoding="utf-8")
    instructions = re.search(r"## Optimized prompt\n\n```\n(.*)\n```", md, re.S).group(1)
    program = dspy.Predict(dspy.Signature("text:str -> anonymized_text:str").with_instructions(instructions))
    lm = dspy.LM(f"openai/{SERVED_MODEL}", api_base=SERVER, api_key="", max_tokens=25_000)
    program.set_lm(lm)
    return lambda text: call_counting_usage(lm, lambda: program(text=text).anonymized_text)


def load_papillon():
    """PAPILLON (Columbia-NLP-Lab/PAPILLON, papillon/run_llama_dspy.py) zero-shot at prompt level:
    the local model's prompt_creater output (the prompt it would send to the remote model) is the
    rewrite; the remote model and the aggregator are never called. LM settings as in their
    papillon/evaluate_papillon.py."""
    import dspy
    sys.path.insert(0, str(REPOS / "PAPILLON" / "papillon"))
    from run_llama_dspy import PAPILLON
    lm = dspy.LM(f"openai/{SERVED_MODEL}", api_base=SERVER, api_key="", max_tokens=4000)
    dspy.configure(lm=lm)
    pipeline = PAPILLON(untrusted_model=None)
    return lambda text: call_counting_usage(lm, lambda: pipeline.prompt_creater(userQuery=text).createdPrompt)


def load_adapter(method):
    """Load the method's models once; return a function
    prompt -> (rewrite, LLM calls, prompt tokens, completion tokens)."""
    if method == "presidio":
        return load_presidio()
    if method in LOISEAU_PROMPTS:
        return load_loiseau(method)
    if method == "papillon":
        return load_papillon()
    raise ValueError(f"no adapter for {method}")


def read_rows(path, limit):
    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return rows[:limit] if limit else rows


def done_ids(path):
    """Ids with a successful rewrite. Failed rows are retried on the next run and appended
    again, so for each id the last row in the file counts."""
    if not os.path.exists(path):
        return set()
    with open(path, encoding="utf-8") as f:
        return {r["id"] for r in csv.DictReader(f) if not r.get("error")}


def run(rows, rewrite, method, out_path):
    """Append one rewrite per row not yet in out_path; flush after each, so a crash loses at most one.
    A failed call is recorded (empty rewrite, exception in `error`), never silently dropped:
    an empty rewrite would otherwise look perfectly private to the scorer."""
    done = done_ids(out_path)
    todo = [r for r in rows if r["id"] not in done]
    print(f"{method}: {len(done)} rows already done, {len(todo)} to go -> {out_path}")
    new_file = not os.path.exists(out_path)
    failed = 0
    with open(out_path, "a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        if new_file:
            writer.writerow(["id", "method", "rewrite", "error", "seconds",
                             "calls", "prompt_tokens", "completion_tokens"])
        for n, row in enumerate(todo, 1):
            start = time.perf_counter()
            try:
                text, calls, prompt_tokens, completion_tokens = rewrite(row["original"])
                error = ""
            except Exception as e:
                text, calls, prompt_tokens, completion_tokens = "", "", "", ""
                error = f"{type(e).__name__}: {e}"
                failed += 1
                print(f"  id {row['id']} failed: {error[:200]}")
            seconds = round(time.perf_counter() - start, 2)
            writer.writerow([row["id"], method, text, error, seconds,
                             calls, prompt_tokens, completion_tokens])
            f.flush()
            if n % 20 == 0 or n == len(todo):
                print(f"  {n}/{len(todo)} ({failed} failed)")


def main():
    args = get_args()
    data_name = os.path.basename(os.path.dirname(os.path.abspath(args.data)))
    out_path = args.out or OUT_TEMPLATE.format(data=data_name, method=args.method)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    rows = read_rows(args.data, args.limit)
    rewrite = load_adapter(args.method)
    run(rows, rewrite, args.method, out_path)


if __name__ == "__main__":
    main()
