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

Each method runs in its own venv (/work/projects/p23-mkilic/venvs/<venv>), because the
baselines pin conflicting packages: presidio -> presidio, loiseau_* -> loiseau
(DSPy 3.0.4, as in their paper), papillon* -> papillon (DSPy 3.1.2: PAPILLON's README says
its code runs with the newest DSPy for inference, and the authors' 2026 follow-up uses 3.1.2),
staab -> staab (openai 0.28.1, as in their environment.yaml).
papillon_t0 is PAPILLON with greedy decoding, as in their own environment: dspy-ai 2.5.35's
dspy.LM defaults to temperature 0.0, while DSPy 3.1.2's sends none, so `papillon` (run
2026-10-09, kept for comparison) sampled with Qwen's generation_config (found 2026-11-06).
The LLM baselines talk to Qwen2.5-7B-Instruct through `transformers serve` at SERVER
(baselines/run_sweep_gb10.sbatch starts it). The adapters only feed the prompt to the
released code (baselines/repos, pinned by baselines/fetch_repos.sh) and read its
output; they never change the method.

Staab et al. are not a per-prompt adapter: their main.py loops over all prompts in each
iteration. run_staab runs main.py unchanged once per prompt (so each prompt gets its own time)
and reads the results back into the same CSV format (tokens empty: their code does not keep
the usage).

staab_md is Staab et al. with baselines/staab_main.py in place of main.py, which puts Qwen's
Markdown-formatted attacker answers into the plain layout their parser reads. It reruns only the
prompts in STAAB_BLIND_IDS (or --ids); the other rows are copied from the `staab` run, since the
parser worked there and the fix would change nothing.
"""
import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

METHODS = ["presidio", "loiseau_pupa", "loiseau_synthpai", "papillon", "papillon_t0", "staab", "staab_md"]
COLUMNS = ["id", "method", "rewrite", "error", "seconds", "calls", "prompt_tokens", "completion_tokens"]
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
STAAB_REPO = REPOS / "llm-anonymization"
STAAB_RUNS = HERE / "results" / "staab_runs"   # their inputs and per-iteration outputs, one folder per run
STAAB_ITERATIONS = 3                           # as in their configs
STAAB_TEMPERATURE = 0.1                        # as in their configs
STAAB_TIMESTAMP = 1700000000                   # their prompts print each comment's date: one fixed date for all
STAAB_MAIN = HERE / "baselines" / "staab_main.py"   # main.py with the Markdown fix (staab_md)
# PUPA-SD prompts where the attacker's Markdown answer left the anonymizer without inferences in
# at least one of rounds 0-2 (results/staab_runs/pupa_sd/<id>/out/inference_<round>.jsonl)
STAAB_BLIND_IDS = ["4", "140", "198", "199", "266", "339", "458", "542"]
# Our 10 attributes, under Staab et al.'s key where they have one, so their own description and
# answer options apply (reddit_utils.type_to_str / type_to_options); the other three pass
# through by name. Every prompt gets all 10: their attacker targets the types in `reviews`.
STAAB_ATTRIBUTES = {
    "gender": "gender", "age": "age", "geographic location": "location",
    "profession": "occupation", "education level": "education", "socioeconomic status": "income",
    "relationship status": "married", "health condition": "health condition",
    "family situation": "family situation", "nationality": "nationality",
}


def get_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--method", choices=METHODS, required=True)
    p.add_argument("--data", default="data/pupa_sd/pupa_sd_eval.csv")
    p.add_argument("--out", default=None, help=f"default: {OUT_TEMPLATE}")
    p.add_argument("--limit", type=int, default=None, help="only the first N rows (smoke test)")
    p.add_argument("--ids", nargs="+", default=None, help="only these ids (smoke test on chosen prompts)")
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


def load_papillon(greedy):
    """PAPILLON (Columbia-NLP-Lab/PAPILLON, papillon/run_llama_dspy.py) zero-shot at prompt level:
    the local model's prompt_creater output (the prompt it would send to the remote model) is the
    rewrite; the remote model and the aggregator are never called. LM settings as in their
    papillon/evaluate_papillon.py; greedy passes the temperature 0.0 their dspy-ai 2.5.35 defaults to."""
    import dspy
    sys.path.insert(0, str(REPOS / "PAPILLON" / "papillon"))
    from run_llama_dspy import PAPILLON
    decoding = {"temperature": 0.0} if greedy else {}   # the server decodes greedily at temperature 0
    lm = dspy.LM(f"openai/{SERVED_MODEL}", api_base=SERVER, api_key="", max_tokens=4000, **decoding)
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
    if method in ("papillon", "papillon_t0"):
        return load_papillon(greedy=method == "papillon_t0")
    raise ValueError(f"no adapter for {method}")


def read_rows(path, limit, ids):
    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if ids:
        rows = [r for r in rows if r["id"] in set(ids)]
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
            writer.writerow(COLUMNS)
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


def write_staab_profiles(rows, path):
    """Each prompt as one of their profiles. Its non-empty lines are the profile's comments:
    their parser splits the anonymizer's answer into one comment per line, so a multi-line
    comment would come back misaligned. `reviews` lists our 10 attributes with empty values,
    so their attacker targets exactly those (hardness and certainty 1 pass their filter)."""
    reviews = {"human": {key: {"estimate": "", "hardness": 1, "certainty": 1}
                         for key in STAAB_ATTRIBUTES.values()}}
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            lines = [line.strip() for line in row["original"].split("\n") if line.strip()]
            comments = [{"text": line, "subreddit": "synthetic", "user": row["id"],
                         "timestamp": STAAB_TIMESTAMP, "pii": {}} for line in lines]
            f.write(json.dumps({"username": row["id"], "comments": comments, "reviews": reviews}) + "\n")


def write_staab_config(profiles_path, run_dir, config_path):
    """Their SynthPAI config (configs/anonymization/synthpai/*.yaml) with every model set to our
    server, one request at a time, and no separate evaluation model. JSON is valid YAML."""
    model = {"name": SERVED_MODEL, "provider": "openai", "args": {"temperature": STAAB_TEMPERATURE}}
    config = {
        "output_dir": str(run_dir / "logs"), "seed": 10, "task": "ANONYMIZED", "dryrun": False,
        "save_prompts": True, "timeout": 0.0, "max_workers": 1,
        "task_config": {
            "profile_path": str(profiles_path), "outpath": str(run_dir / "out"),
            "anonymizer": {"anon_type": "llm", "target_mode": "single", "max_workers": 1},
            "anon_model": model, "inference_model": model, "utility_model": model,
            "eval_inference_model": None,
            "profile_filter": {"hardness": 1, "certainty": 1},
            "max_num_iterations": STAAB_ITERATIONS, "use_ner": False,
            "offset": 0, "num_profiles": 100_000,
        },
        "gen_model": model,
    }
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")


def staab_progress(profile):
    """(comment sets, filled inferences + utility scores) of one saved profile state. A finished
    profile has 1 + STAAB_ITERATIONS comment sets; its calls = anonymizations + filled."""
    filled = sum(SERVED_MODEL in c.get("predictions", {}) for c in profile["comments"])
    filled += sum(SERVED_MODEL in c.get("utility", {}) for c in profile["comments"])
    return len(profile["comments"]), filled


def staab_final_profiles(out_dir):
    """The most advanced saved state of each profile across their per-iteration files."""
    final = {}
    for path in out_dir.glob("*.jsonl"):
        for line in path.read_text(encoding="utf-8").splitlines():
            profile = json.loads(line)
            best = final.get(profile["username"])
            if best is None or staab_progress(profile) > staab_progress(best):
                final[profile["username"]] = profile
    return final


def staab_rewrite(original, profile):
    """Their final comments put back into the prompt's layout, one per non-empty line."""
    texts = iter(c["text"] for c in profile["comments"][-1]["comments"])
    return "\n".join(next(texts) if line.strip() else line for line in original.split("\n"))


def run_staab_prompt(row, prompt_dir, script):
    """One prompt through their code (script: their main.py, or STAAB_MAIN), in its own folder.
    Returns its final saved profile (None if there is none), the seconds the run took and the exit
    code. A finished prompt is not run again; an unfinished one starts over, so its time is whole."""
    seconds_path, out_dir = prompt_dir / "seconds.txt", prompt_dir / "out"
    if seconds_path.exists():
        profile = staab_final_profiles(out_dir).get(row["id"])
        if profile is not None and staab_progress(profile)[0] > STAAB_ITERATIONS:
            return profile, float(seconds_path.read_text()), 0
    shutil.rmtree(prompt_dir, ignore_errors=True)
    prompt_dir.mkdir(parents=True)
    profiles_path, config_path = prompt_dir / "profiles.jsonl", prompt_dir / "config.yaml"
    write_staab_profiles([row], profiles_path)
    write_staab_config(profiles_path, prompt_dir, config_path)
    start = time.perf_counter()
    done = subprocess.run([sys.executable, script, "--config_path", str(config_path)], cwd=STAAB_REPO)
    seconds = round(time.perf_counter() - start, 2)
    seconds_path.write_text(str(seconds))
    return staab_final_profiles(out_dir).get(row["id"]), seconds, done.returncode


def run_staab(rows, out_path, run_name, method="staab", rerun=None, copy_from=None):
    """Each prompt through their code separately (STAAB_RUNS/run_name/<id>/), so every prompt
    gets its own time. Their main.py handles all prompts together in each step, and its loop never
    mixes profiles, so one prompt per run is the same method; only the order of calls differs.
    The seconds include ~10-20 s of process startup (their imports and embedding model).
    The CSV is rewritten from all prompt folders on every run, so a rerun resumes.
    With rerun (ids) only those prompts run; the other rows are copied from the CSV copy_from."""
    run_dir = STAAB_RUNS / run_name
    script = "main.py" if method == "staab" else str(STAAB_MAIN)
    copied = {}
    if rerun is not None:
        with open(copy_from, encoding="utf-8") as f:
            copied = {r["id"]: r for r in csv.DictReader(f) if not r.get("error")}
    print(f"{method}: {len(rows)} prompts, their files in {run_dir}/<id>/")
    incomplete = 0
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(COLUMNS)
        for n, row in enumerate(rows, 1):
            if rerun is not None and row["id"] not in rerun:
                writer.writerow([{**copied[row["id"]], "method": method}[c] for c in COLUMNS])
                continue
            profile, seconds, exit_code = run_staab_prompt(row, run_dir / row["id"], script)
            sets, filled = staab_progress(profile) if profile else (0, 0)
            if sets <= STAAB_ITERATIONS:
                incomplete += 1
                error = (f"incomplete: {max(0, sets - 1)} of {STAAB_ITERATIONS} anonymizations "
                         f"(main.py exit code {exit_code})")
                writer.writerow([row["id"], method, "", error, seconds, "", "", ""])
            else:
                writer.writerow([row["id"], method, staab_rewrite(row["original"], profile), "",
                                 seconds, sets - 1 + filled, "", ""])
            f.flush()
            print(f"  {n}/{len(rows)} id {row['id']}: {seconds} s ({incomplete} incomplete so far)")
    print(f"{method}: wrote {out_path} ({incomplete} incomplete)")


def main():
    args = get_args()
    data_name = os.path.basename(os.path.dirname(os.path.abspath(args.data)))
    out_path = args.out or OUT_TEMPLATE.format(data=data_name, method=args.method)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    rows = read_rows(args.data, args.limit, args.ids)
    if args.method in ("staab", "staab_md"):
        # a smoke test (--limit / --ids) gets its own folder, so it never mixes into the full run's files
        run_name = data_name if args.method == "staab" else f"{data_name}_md"
        if args.ids:
            run_name += "_ids_" + "_".join(args.ids)
        elif args.limit:
            run_name += f"_first{args.limit}"
        rerun, copy_from = None, None
        if args.method == "staab_md":
            rerun = set(args.ids) if args.ids else set(STAAB_BLIND_IDS)
            copy_from = OUT_TEMPLATE.format(data=data_name, method="staab")
        run_staab(rows, out_path, run_name, args.method, rerun, copy_from)
        return
    rewrite = load_adapter(args.method)
    run(rows, rewrite, args.method, out_path)


if __name__ == "__main__":
    main()
