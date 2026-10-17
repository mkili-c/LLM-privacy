"""
D3: did Staab et al.'s comment parser move lines in our multi-line prompts?

Each prompt ran as one of their profiles whose comments are its non-empty lines
(run_baselines.write_staab_profiles). When the anonymizer returns a different number of lines,
their parser (llm_anonymizers.filter_and_align_comments) matches every old line to its most similar
returned line (Jaro-Winkler > 0.5, each returned line used once) and keeps the old text where
nothing matches. A line can then receive another line's text, as in the smoke test (id 714:
"About" overwritten by line 22). Their saved comment sets (results/staab_runs/pupa_sd/<id>/out),
the original and one per round, show every round's lines, so each round is checked against the
one before:
    copied      line i of a round equals line j != i of the round before, and not line i's own text
    misaligned  line i of a round is closer to line j != i of the round before than to its own
                (difflib ratio, above MIN_SIM), so it most likely took line j's place
Single-line prompts cannot move lines; when the answer has several lines, the parser keeps only
the most similar one, which their saved files cannot show.

A second artifact: their prompts print each comment's date (our fixed STAAB_TIMESTAMP), the
anonymizer echoes it, and their parser strips "YYYY-MM-DD:" only within a line's first 11
characters, so " 2023-11-14: ..." (a leading space) stays in the rewrite. Prompts whose rewrite
contains the date and whose original does not are flagged too.

The sensitivity check recomputes the judge's leak counts and mean utility loss without the flagged
prompts (moved lines, the date, either), for every method on the same remaining prompts.

    python evaluation/staab_line_check.py
"""
import argparse
import csv
import difflib
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "evaluation"))

from run_baselines import STAAB_TIMESTAMP, staab_final_profiles, staab_rewrite   # noqa: E402
from read_judge_answers import METHODS                                    # noqa: E402

MIN_SIM = 0.5              # their parser's similarity floor for a match
DATE = datetime.fromtimestamp(STAAB_TIMESTAMP, timezone.utc).strftime("%Y-%m-%d")   # 2023-11-14


def get_args():
    p = argparse.ArgumentParser(description="Staab et al.'s line alignment on multi-line prompts")
    p.add_argument("--data", default=str(REPO / "data" / "pupa_sd" / "pupa_sd_eval.csv"))
    p.add_argument("--runs", default=str(REPO / "results" / "staab_runs" / "pupa_sd"))
    p.add_argument("--rewrites", default=str(REPO / "results" / "rewrites" / "pupa_sd_staab.csv"))
    p.add_argument("--eval_dir", default=str(REPO / "results" / "evaluation"))
    return p.parse_args()


def read_csv(path):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def sim(a, b):
    return difflib.SequenceMatcher(None, a, b).ratio()


def moved_lines(sets):
    """[(round, line i, kind, line j, text)] for every line that took another line's place."""
    found = []
    for t in range(1, len(sets)):
        prev, cur = sets[t - 1], sets[t]
        for i, line in enumerate(cur):
            others = [j for j in range(len(prev)) if j != i and prev[j] != prev[i]]
            copied = [j for j in others if line == prev[j] and line != prev[i]]
            if copied:
                found.append((t, i, "copied", copied[0], line))
                continue
            if others:
                j = max(others, key=lambda j: sim(line, prev[j]))
                if sim(line, prev[j]) > max(sim(line, prev[i]), MIN_SIM):
                    found.append((t, i, "misaligned", j, line))
    return found


def check_prompts(rows, runs_dir, rewrites):
    """{id: moved lines} for the multi-line prompts; also checks every rewrite against the CSV."""
    flagged, n_multi, mismatched = {}, 0, []
    for row in rows:
        profile = staab_final_profiles(runs_dir / row["id"] / "out").get(row["id"])
        if staab_rewrite(row["original"], profile) != rewrites[row["id"]]:
            mismatched.append(row["id"])
        sets = [[c["text"] for c in s["comments"]] for s in profile["comments"]]   # original + one per round
        if len(sets[0]) < 2:
            continue
        n_multi += 1
        moved = moved_lines(sets)
        print(f"  id {row['id']:>5}: {len(sets[0])} lines, {len(moved)} moved")
        for t, i, kind, j, text in moved:
            print(f"      round {t}, line {i + 1} {kind} from line {j + 1}: {text[:80]!r}")
        if moved:
            flagged[row["id"]] = moved
    return flagged, n_multi, mismatched


def sensitivity(eval_dir, exclude):
    """The judge's leak count (match) and mean utility loss per method, with and without the
    flagged prompts."""
    outcomes = read_csv(eval_dir / "judge_outcomes.csv")
    utility = read_csv(eval_dir / "judge_utility.csv")
    counted = {(r["id"], r["attribute"]) for r in outcomes if r["version"] == "original" and r["outcome"] == "match"}
    leak = {(r["id"], r["version"], r["attribute"]): r["outcome"] == "match" for r in outcomes}
    for label, drop in (("all prompts", set()), (f"without {len(exclude)} prompts", exclude)):
        pairs = [(i, a) for i, a in counted if i not in drop]
        print(f"  {label}: {len(pairs)} counted pairs")
        for m in METHODS:
            k = sum(leak[(i, m, a)] for i, a in pairs)
            losses = [int(r["loss"]) for r in utility if r["method"] == m and r["id"] not in drop]
            print(f"    {m:<14} leak {k}/{len(pairs)} ({k / len(pairs):.1%})   "
                  f"utility loss {sum(losses) / len(losses):.2f} (n {len(losses)})")


def main():
    args = get_args()
    rows = read_csv(args.data)
    rewrites = {r["id"]: r["rewrite"] for r in read_csv(args.rewrites) if r.get("rewrite")}
    print(f"Staab et al.'s lines, round by round, on the multi-line prompts of {len(rows)}:")
    flagged, n_multi, mismatched = check_prompts(rows, Path(args.runs), rewrites)
    print(f"\n{len(flagged)} of {n_multi} multi-line prompts have moved lines: {sorted(flagged, key=int)}; "
          f"rewrites that differ from {Path(args.rewrites).name}: {len(mismatched)} {mismatched}")
    dated = {r["id"] for r in rows if DATE in rewrites[r["id"]] and DATE not in r["original"]}
    print(f"{len(dated)} of {len(rows)} rewrites contain the echoed comment date {DATE}: {sorted(dated, key=int)}")
    for label, drop in (("moved lines", set(flagged)), ("echoed date", dated), ("either", set(flagged) | dated)):
        print(f"\n=== sensitivity: the judge's results without the prompts with {label}")
        sensitivity(Path(args.eval_dir), drop)


if __name__ == "__main__":
    main()
