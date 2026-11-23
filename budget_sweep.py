"""
The loop at a lower utility budget (--max_share below 1.0), rerunning only the prompts it can change.

The budget enters the loop in one place: privacy_rewrite.utility_vetted lets a rewrite in iff its
R < --max_share. The loop is deterministic (its seed is reset per prompt; job 1984 reproduced job
663 exactly). So a prompt whose checked R stayed below the new budget at every rewrite of the 1.0
run takes exactly the same path again and keeps its output. Those R values are in the 1.0 run's
logs: one "utility R ..%" line per checked rewrite (jobs 665 and 674 on PUPA-SD).

    python budget_sweep.py plan --max_share 0.5      # the prompts to rerun, and the sbatch command
    (run the printed command)
    python budget_sweep.py merge --max_share 0.5     # -> results/rewrites/pupa_sd_loop_r050.csv

plan: a prompt is rerun if its largest checked R, as logged in whole percents, is at least the
budget minus one point (the log rounds). CONTROLS prompts it would not rerun are rerun too, as a
check that their output is unchanged. Before that, every prompt of the evaluation set must have
exactly one complete block in the logs, whose final text equals its rewrite in the 1.0 run.
merge: the rerun rows for the rerun prompts, the 1.0 run's rows for all others, in the evaluation
set's order, as method loop_r<budget>; stops if a rerun is missing or failed or a control changed.
"""
import argparse
import ast
import csv
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent
DATA = REPO / "data" / "pupa_sd" / "pupa_sd_eval.csv"
BASE_RUN = REPO / "results" / "rewrites" / "pupa_sd_loop.csv"                   # the loop at --max_share 1.0
BASE_LOGS = [REPO / "pu_loop_sweeps" / f"pu_loop_sweep_{job}.out" for job in (665, 674)]
CONTROLS = 2
CHECKED_R = re.compile(r"^\s+utility R (\d+)%")                                # one per checked rewrite
FINAL = re.compile(r"^final \(.*privacy step \d+ s\): (.*)$")
COLUMNS = ["id", "method", "rewrite", "error", "seconds", "calls", "prompt_tokens", "completion_tokens"]


def get_args():
    p = argparse.ArgumentParser(description="The loop at a lower utility budget, rerunning only what it can change")
    p.add_argument("step", choices=["plan", "merge"])
    p.add_argument("--max_share", type=float, required=True)
    return p.parse_args()


def last_rows(path):
    """{id: row}, the last row per id (a failed row is retried and appended)."""
    with open(path, encoding="utf-8") as f:
        return {r["id"]: r for r in csv.DictReader(f)}


def logged_checks(originals):
    """{id: ([checked R in %], final text)} from the 1.0 run's logs, one complete block per prompt.
    A block ends with "original: <prompt>" and "final (...): <text>"; a block a cancelled job left
    unfinished has neither and is skipped."""
    by_text = {text: i for i, text in originals.items()}
    out, checks, original = {}, [], None
    for log in BASE_LOGS:
        for line in log.read_text(encoding="utf-8").splitlines():
            if line.startswith("  sampling the reference answers"):
                checks, original = [], None
            elif CHECKED_R.match(line):
                checks.append(int(CHECKED_R.match(line).group(1)))
            elif line.startswith("original: "):
                original = ast.literal_eval(line[len("original: "):])
            elif FINAL.match(line) and original in by_text:
                i = by_text[original]
                if i in out:
                    sys.exit(f"id {i} has two complete blocks in {BASE_LOGS}")
                out[i] = (checks, ast.literal_eval(FINAL.match(line).group(1)))
    return out


def tag(max_share):
    return f"r{round(max_share * 100):03d}"


def rerun_ids(max_share):
    """(prompts the budget can change, control prompts), after checking the logs against the 1.0 run."""
    with open(DATA, encoding="utf-8") as f:
        originals = {r["id"]: r["original"] for r in csv.DictReader(f)}
    base = last_rows(BASE_RUN)
    logged = logged_checks(originals)
    missing = [i for i in originals if i not in logged]
    differs = [i for i in originals if i in logged and logged[i][1] != base[i]["rewrite"]]
    if missing or differs:
        sys.exit(f"logs do not match the 1.0 run: no block for {missing}, other final text for {differs}")
    limit = round(max_share * 100) - 1
    affected = [i for i in originals if max(logged[i][0], default=0) >= limit]
    # controls: prompts with at least one checked rewrite that the budget leaves alone, the fastest first
    others = [i for i in originals if i not in affected and logged[i][0]]
    controls = sorted(others, key=lambda i: float(base[i]["seconds"]))[:CONTROLS]
    return affected, controls, base, logged


def plan(max_share):
    affected, controls, base, logged = rerun_ids(max_share)
    hours = sum(float(base[i]["seconds"]) for i in affected + controls) / 3600
    print(f"budget {max_share:.0%}: {len(affected)} of {len(logged)} prompts can change "
          f"(largest checked R >= {round(max_share * 100) - 1}%), {len(controls)} controls {controls}")
    print(f"  their time at 1.0: {hours:.1f} h (a lower budget adds retries, so expect more)")
    out = REPO / "results" / "rewrites" / f"pupa_sd_loop_{tag(max_share)}_rerun.csv"
    print("\nsbatch run_loop_sweep_gb10.sbatch --data data/pupa_sd/pupa_sd_eval.csv "
          f"--max_share {max_share} --out {out.relative_to(REPO)} --ids {' '.join(affected + controls)}")


def merge(max_share):
    affected, controls, base, _ = rerun_ids(max_share)
    method = f"loop_{tag(max_share)}"
    rerun = last_rows(REPO / "results" / "rewrites" / f"pupa_sd_loop_{tag(max_share)}_rerun.csv")
    bad = [i for i in affected + controls if i not in rerun or rerun[i]["error"]]
    if bad:
        sys.exit(f"reruns missing or failed: {bad}")
    changed = [i for i in controls if rerun[i]["rewrite"] != base[i]["rewrite"]]
    if changed:
        sys.exit(f"controls changed although the budget cannot change them: {changed}; the shortcut does not hold")
    out = REPO / "results" / "rewrites" / f"pupa_sd_{method}.csv"
    with open(DATA, encoding="utf-8") as f:
        ids = [r["id"] for r in csv.DictReader(f)]
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(COLUMNS)
        for i in ids:
            row = rerun[i] if i in affected else base[i]
            w.writerow([{**row, "method": method}[c] for c in COLUMNS])
    same = sum(rerun[i]["rewrite"] == base[i]["rewrite"] for i in affected)
    print(f"{out.relative_to(REPO)}: {len(ids)} rows, {len(affected)} rerun ({same} of them unchanged), "
          f"{len(ids) - len(affected)} copied from the 1.0 run; controls unchanged: {controls}")


def main():
    args = get_args()
    plan(args.max_share) if args.step == "plan" else merge(args.max_share)


if __name__ == "__main__":
    main()
