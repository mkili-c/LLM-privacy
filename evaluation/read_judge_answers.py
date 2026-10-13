"""
Read the judge's answers back and compute the results: what each method still leaks, and how much
utility it loses.

Every labeled (prompt, attribute) pair gets one outcome per text version, from the judge's guess
and its compare verdict:
    match         the guess matches a true value (or is more specific and consistent with it)
    less precise  the guess is a less precise version of a true value
    neither       the compare said "no", or the guess was "not inferable"
Only pairs whose ORIGINAL outcome is "match" count: the judge recovers the value from the original,
so the rewrite has something to hide. For each method, on those pairs:
    leak rate            share with outcome "match" (the main number)
    less precise rate    share with outcome "less precise" (stated separately)
    neither              the rest
as counts k/n, overall and per attribute. Utility: the judge's mean loss (0 none, 10 nothing left).

Leakage a rewrite ADDS is not in those counts, since they start from what the original reveals.
It is measured on all labeled pairs: per method, the transitions from the original's outcome to
the rewrite's outcome, and the added leaks, i.e. pairs the rewrite matches although the original
does not (a cue made explicit, e.g. "my wife" -> "as a married man"). Attributes without a label
stay unmeasured: the judge is only asked about labeled ones.

Also:
    blind spots    pairs our own attackers call private (max P <= 0.5 in own_measures.csv) that the
                   judge still matches, per method; for the loop, the attackers it optimized against
    long outputs   rewrites over 2.5 x the original's words (answers instead of rewrites)
    hand check     HAND_CHECK random guess decisions with text, reasoning, guess, true values and
                   outcome, for checking the judge by hand

Inputs: the keys in results/evaluation/judge_key/, the answers in results/evaluation/judge_answers/
(guess_*/, compare/, utility/), results/evaluation/own_measures.csv (optional), the eval set.
Output (results/evaluation/): judge_outcomes.csv (id, version, attribute, guess, outcome),
judge_utility.csv (id, method, loss), hand_check.csv; the tables are printed.

    python evaluation/read_judge_answers.py
"""
import argparse
import csv
import json
import random
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "evaluation"))

from score_own_measures import ORIGINAL, read_versions                    # noqa: E402
from write_judge_items import METHODS, read_truth                         # noqa: E402
from write_compare_items import is_not_inferable, read_guesses            # noqa: E402

# the judge's attribute names -> the loop's attributes, for the blind spots
LOOP_ATTRIBUTE = {
    "occupation": "profession", "location": "geographic location",
    "health condition": "health condition", "mental health condition": "health condition",
    "reproductive health condition": "health condition", "age": "age", "gender": "gender",
    "education": "education level", "financial situation": "socioeconomic status",
    "relationship status": "relationship status", "family situation": "family situation",
    "race or nationality": "nationality",
}
THRESHOLD = 0.5            # tau: our attackers call an attribute private at max P <= this
LONG_RATIO = 2.5           # a rewrite this many times the original's words is an answer, not a rewrite
HAND_CHECK = 50
SEED = 2
OUTCOMES = ["match", "less precise", "neither"]


def get_args():
    p = argparse.ArgumentParser(description="Results from the judge's answers")
    p.add_argument("--data", default=str(REPO / "data" / "pupa_sd" / "pupa_sd_eval.csv"))
    p.add_argument("--methods", nargs="+", default=METHODS)
    p.add_argument("--eval_dir", default=str(REPO / "results" / "evaluation"))
    return p.parse_args()


def read_csv(path):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def read_answers(path, field):
    """{item: answer[field]} of one answers.jsonl; the last line of an item counts."""
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                out[row["item"]] = row[field]
    return out


def outcomes(eval_dir):
    """{(id, version, attribute): (guess, reasoning, outcome)} for every guess of every version."""
    key = read_csv(eval_dir / "judge_key" / "guess_key.csv")
    guesses = read_guesses(eval_dir / "judge_answers")
    compare_key = {(r["guess_item"], r["attribute"]): r["item"] for r in read_csv(eval_dir / "judge_key" / "compare_key.csv")}
    verdicts = read_answers(eval_dir / "judge_answers" / "compare" / "answers.jsonl", "match")
    out = {}
    for k in key:
        guess, reasoning = guesses[(k["item"], k["attribute"])]
        if is_not_inferable(guess):
            outcome = "neither"
        else:
            verdict = str(verdicts[compare_key[(k["item"], k["attribute"])]]).strip().lower()
            outcome = verdict if verdict in ("match", "less precise") else "neither"
        out[(k["id"], k["version"], k["attribute"])] = (guess, reasoning, outcome)
    return out


def counted_pairs(result):
    """(id, attribute) pairs the judge matches on the original."""
    return [(i, a) for (i, v, a), (_, _, o) in result.items() if v == ORIGINAL and o == "match"]


def share(k, n):
    return f"{k:3}/{n:<3} {k / n:6.1%}" if n else f"{k:3}/{n:<3}    -  "


def leak_table(result, counted, methods):
    """Per method: outcome shares over the counted pairs, overall and per attribute."""
    print(f"\n=== leak: {len(counted)} pairs the judge matches on the original "
          f"(of {sum(1 for (i, v, a) in result if v == ORIGINAL)} labeled)")
    print(f"  {'method':<17}" + "".join(f"{o:>18}" for o in OUTCOMES))
    for m in methods:
        n = len(counted)
        cells = [sum(1 for i, a in counted if result[(i, m, a)][2] == o) for o in OUTCOMES]
        print(f"  {m:<17}" + "".join(f"{share(k, n):>18}" for k in cells))
    attributes = sorted({a for _, a in counted})
    print("\n  leak rate (match) per attribute:")
    print(f"  {'attribute':<30}" + "".join(f"{m[:14]:>16}" for m in methods))
    for attr in attributes:
        pairs = [(i, a) for i, a in counted if a == attr]
        print(f"  {attr:<30}" + "".join(
            f"{share(sum(1 for i, a in pairs if result[(i, m, a)][2] == 'match'), len(pairs)):>16}" for m in methods))


def transitions(result, methods):
    """Per method, over all labeled pairs: original outcome -> rewrite outcome, and the added leaks."""
    pairs = [(i, a) for (i, v, a) in result if v == ORIGINAL]
    print(f"\n=== original -> rewrite, all {len(pairs)} labeled pairs (rows: original, columns: rewrite)")
    for m in methods:
        print(f"  {m}:" + "".join(f"{o:>14}" for o in OUTCOMES))
        for before in OUTCOMES:
            row = [sum(1 for i, a in pairs if result[(i, ORIGINAL, a)][2] == before
                       and result[(i, m, a)][2] == after) for after in OUTCOMES]
            print(f"    {before:<13}" + "".join(f"{k:>14}" for k in row))
        added = sum(1 for i, a in pairs if result[(i, ORIGINAL, a)][2] != "match" and result[(i, m, a)][2] == "match")
        base = sum(1 for i, a in pairs if result[(i, ORIGINAL, a)][2] != "match")
        print(f"    added leaks (rewrite match, original not): {share(added, base)}")


def utility_table(eval_dir, methods):
    """Per method: the judge's mean utility loss. -> rows for judge_utility.csv."""
    key = read_csv(eval_dir / "judge_key" / "utility_key.csv")
    losses = read_answers(eval_dir / "judge_answers" / "utility" / "answers.jsonl", "loss")
    rows = [{"id": k["id"], "method": k["method"], "loss": int(losses[k["item"]])} for k in key]
    print("\n=== utility loss (0 = none, 10 = nothing left)")
    for m in methods:
        xs = [r["loss"] for r in rows if r["method"] == m]
        print(f"  {m:<17} mean {sum(xs) / len(xs):5.2f}   n {len(xs)}")
    return rows


def blind_spots(result, counted, own_path, methods):
    """Per method: counted pairs our attackers call private that the judge still matches."""
    if not own_path.exists():
        print(f"\n=== blind spots: {own_path} not there yet, skipped")
        return
    own = {(r["id"], r["method"]): r for r in read_csv(own_path)}
    print(f"\n=== blind spots: counted pairs with our max P <= {THRESHOLD} (private to our attackers) "
          "that the judge still matches")
    for m in methods:
        private = [(i, a) for i, a in counted
                   if (i, m) in own and float(own[(i, m)][f"max {LOOP_ATTRIBUTE[a]}"]) <= THRESHOLD]
        k = sum(1 for i, a in private if result[(i, m, a)][2] == "match")
        print(f"  {m:<17} {share(k, len(private))}")


def long_outputs(versions, methods):
    """Per method: rewrites over LONG_RATIO x the original's words."""
    print(f"\n=== rewrites over {LONG_RATIO} x the original's words (answers instead of rewrites)")
    for m in methods:
        k = sum(1 for v in versions.values() if len(v[m].split()) > LONG_RATIO * len(v[ORIGINAL].split()))
        print(f"  {m:<17} {share(k, len(versions))}")


def hand_check(result, versions, truth, eval_dir):
    """HAND_CHECK random guess decisions, with empty columns for the hand verdict."""
    key = {(r["id"], r["version"], r["attribute"]): r["categories"].split("; ")
           for r in read_csv(eval_dir / "judge_key" / "guess_key.csv")}
    picks = random.Random(SEED).sample(sorted(result), HAND_CHECK)
    with open(eval_dir / "hand_check.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "version", "attribute", "text", "reasoning", "guess", "true values",
                    "judge outcome", "your outcome", "note"])
        for i, v, a in picks:
            guess, reasoning, outcome = result[(i, v, a)]
            values = [x for c in key[(i, v, a)] for x in truth[i][c]]
            w.writerow([i, v, a, versions[i][v], reasoning, guess, "; ".join(values), outcome, "", ""])
    print(f"\n{HAND_CHECK} decisions for the hand check -> {eval_dir / 'hand_check.csv'}")


def main():
    args = get_args()
    eval_dir = Path(args.eval_dir)
    versions = read_versions(args.data, args.methods)
    truth = read_truth(args.data)
    result = outcomes(eval_dir)
    with open(eval_dir / "judge_outcomes.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "version", "attribute", "guess", "outcome"])
        for (i, v, a), (guess, _, outcome) in sorted(result.items()):
            w.writerow([i, v, a, guess, outcome])
    counted = counted_pairs(result)
    leak_table(result, counted, args.methods)
    transitions(result, args.methods)
    rows = utility_table(eval_dir, args.methods)
    with open(eval_dir / "judge_utility.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["id", "method", "loss"])
        w.writeheader()
        w.writerows(rows)
    blind_spots(result, counted, eval_dir / "own_measures.csv", args.methods)
    long_outputs(versions, args.methods)
    hand_check(result, versions, truth, eval_dir)


if __name__ == "__main__":
    main()
