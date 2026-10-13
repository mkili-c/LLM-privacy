"""
Read the judge's answers back and compute the results: what each method still leaks, and how much
utility it loses, by the judge and by our own measures side by side.

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

Our own measures (results/evaluation/own_measures.csv) on the same pairs: our attackers leak an
attribute iff max P(Qwen, Laya) > THRESHOLD, and utility is R = D / D(empty) per text, with the
loop's rule "broken iff R >= MAX_SHARE".

Tables, in the order printed:
    main table      per version: the judge's outcomes and utility loss, our leak rate and R
    per attribute   leak rate per attribute, the judge's and ours
    transitions     original outcome -> rewrite outcome on all labeled pairs, and the added leaks:
                    pairs the rewrite matches although the original does not (a cue made explicit,
                    e.g. "my wife" -> "as a married man"). Unlabeled attributes stay unmeasured.
    agreement       judge vs our attackers per counted pair (2 x 2); "judge only" are our blind
                    spots: private to our attackers, but the judge still matches; for the loop,
                    the attackers it optimized against
    utility         judge vs R: within each prompt, every pair of rewrites the judge rates
                    differently, ordered the same way by R? (pair accuracy); and the judge's loss
                    on the texts our rule calls broken vs the rest
    long outputs    rewrites over LONG_RATIO x the original's words (answers instead of rewrites)
Also: HAND_CHECK random guess decisions with text, reasoning, guess, true values and outcome, for
checking the judge by hand, and the privacy-utility plot (one point per version).

Inputs: the keys in results/evaluation/judge_key/, the answers in results/evaluation/judge_answers/
(guess_*/, compare/, utility/), results/evaluation/own_measures.csv, the eval set.
Output (results/evaluation/): judge_outcomes.csv (id, version, attribute, guess, outcome),
judge_utility.csv (id, method, loss), hand_check.csv, privacy_utility.png; the tables are printed.

    python evaluation/read_judge_answers.py
"""
import argparse
import csv
import itertools
import json
import random
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                           # noqa: E402

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "evaluation"))

from score_own_measures import ORIGINAL, read_versions                    # noqa: E402
from write_judge_items import METHODS, read_truth                         # noqa: E402
from write_compare_items import is_not_inferable, read_guesses            # noqa: E402

# the judge's attribute names -> the loop's attributes, for our attackers' P
LOOP_ATTRIBUTE = {
    "occupation": "profession", "location": "geographic location",
    "health condition": "health condition", "mental health condition": "health condition",
    "reproductive health condition": "health condition", "age": "age", "gender": "gender",
    "education": "education level", "financial situation": "socioeconomic status",
    "relationship status": "relationship status", "family situation": "family situation",
    "race or nationality": "nationality",
}
THRESHOLD = 0.5            # tau: our attackers call an attribute private at max P <= this
MAX_SHARE = 1.0            # the loop's utility rule: a text's utility is broken iff R >= this
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


def read_utility(eval_dir):
    """{(id, method): the judge's utility loss 0-10}."""
    key = read_csv(eval_dir / "judge_key" / "utility_key.csv")
    losses = read_answers(eval_dir / "judge_answers" / "utility" / "answers.jsonl", "loss")
    return {(k["id"], k["method"]): int(losses[k["item"]]) for k in key}


def attacker_leaks(result, own):
    """{(id, version, attribute): our attackers leak it}, i.e. max P > THRESHOLD, for every key of
    the judge's result. Judge attributes that share a loop attribute (the three health ones) read
    the same P."""
    return {(i, v, a): float(own[(i, v)][f"max {LOOP_ATTRIBUTE[a]}"]) > THRESHOLD for i, v, a in result}


def share(k, n):
    return f"{k:3}/{n:<3} {k / n:6.1%}" if n else f"{k:3}/{n:<3}    -  "


def main_table(result, counted, ours, losses, own, names):
    """Per version: the judge's outcomes on the counted pairs and its mean utility loss, next to
    our attackers' leak rate on the same pairs, the mean R and the texts with R >= MAX_SHARE.
    -> {version: (judge leak rate, our leak rate, mean loss, mean R)} for the plot."""
    n = len(counted)
    n_labeled = sum(1 for (i, v, a) in result if v == ORIGINAL)
    print(f"\n=== main table: {n} pairs the judge matches on the original (of {n_labeled} labeled)")
    print(f"  judge: outcomes on those pairs, mean utility loss (0 none, 10 nothing left)")
    print(f"  ours:  leak = max P > {THRESHOLD} on the same pairs; R = D / D(empty), "
          f"broken iff R >= {MAX_SHARE:.0%}")
    print(f"  {'version':<14}" + "".join(f"{'judge ' + o:>20}" for o in OUTCOMES)
          + f"{'loss':>7}{'ours leak':>17}{'mean R':>8}{'R >= ' + f'{MAX_SHARE:.0%}':>17}")
    points = {}
    for v in names:
        judge = [sum(1 for i, a in counted if result[(i, v, a)][2] == o) for o in OUTCOMES]
        k_ours = sum(1 for i, a in counted if ours[(i, v, a)])
        rs = [float(r["R"]) for (i, m), r in own.items() if m == v]
        xs = [x for (i, m), x in losses.items() if m == v]
        loss = sum(xs) / len(xs) if xs else 0.0        # the originals are not rated: no loss by definition
        broken = sum(1 for r in rs if r >= MAX_SHARE)
        print(f"  {v:<14}" + "".join(f"{share(k, n):>20}" for k in judge)
              + (f"{loss:7.2f}" if xs else f"{'-':>7}")
              + f"{share(k_ours, n):>17}{sum(rs) / len(rs):8.1%}{share(broken, len(rs)):>17}")
        points[v] = (judge[0] / n, k_ours / n, loss, sum(rs) / len(rs))
    return points


def attribute_table(title, leaks, counted, names):
    """Per attribute and version: the share of the counted pairs that leak."""
    print(f"\n  {title}:")
    print(f"  {'attribute':<30}" + "".join(f"{v[:14]:>16}" for v in names))
    for attr in sorted({a for _, a in counted}):
        pairs = [(i, a) for i, a in counted if a == attr]
        print(f"  {attr:<30}" + "".join(
            f"{share(sum(1 for i, a in pairs if leaks[(i, v, a)]), len(pairs)):>16}" for v in names))


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


def agreement(judge, ours, counted, names):
    """Per version, on the counted pairs: judge leak (match) x our leak (max P > THRESHOLD).
    Blind spots = judge only, as a share of the pairs private to our attackers."""
    n = len(counted)
    print(f"\n=== judge vs our attackers on the {n} counted pairs (leak = judge match / our max P > {THRESHOLD})")
    print(f"  {'version':<14}{'both leak':>15}{'judge only':>15}{'ours only':>15}{'neither':>15}"
          f"{'agree':>17}{'blind spots':>17}")
    for v in names:
        both = sum(1 for i, a in counted if judge[(i, v, a)] and ours[(i, v, a)])
        judge_only = sum(1 for i, a in counted if judge[(i, v, a)] and not ours[(i, v, a)])
        ours_only = sum(1 for i, a in counted if not judge[(i, v, a)] and ours[(i, v, a)])
        neither = n - both - judge_only - ours_only
        print(f"  {v:<14}{both:>15}{judge_only:>15}{ours_only:>15}{neither:>15}"
              f"{share(both + neither, n):>17}{share(judge_only, judge_only + neither):>17}")


def utility_agreement(losses, own, methods):
    """The judge's utility loss vs our R. Pair accuracy: within each prompt, the pairs of rewrites
    the judge rates differently, and the share that R orders the same way (pairs tied on R are left
    out), overall and per method (the pairs that include it). Then the judge's mean loss on the
    texts our rule calls broken (R >= MAX_SHARE) and on the rest."""
    ids = sorted({i for i, _ in losses})
    agree = {m: 0 for m in methods}
    n = {m: 0 for m in methods}
    agree_all = n_all = 0
    for i in ids:
        for m1, m2 in itertools.combinations(methods, 2):
            d_loss = losses[(i, m1)] - losses[(i, m2)]
            d_r = float(own[(i, m1)]["R"]) - float(own[(i, m2)]["R"])
            if d_loss == 0 or d_r == 0:
                continue
            same = (d_loss > 0) == (d_r > 0)
            agree_all += same
            n_all += 1
            for m in (m1, m2):
                agree[m] += same
                n[m] += 1
    print(f"\n=== judge utility loss vs R: within-prompt pair accuracy "
          f"(pairs of rewrites the judge rates differently)")
    print(f"  {'all':<17} {share(agree_all, n_all)}")
    for m in methods:
        print(f"  {m:<17} {share(agree[m], n[m])}")
    for label, keep in ((f"R >= {MAX_SHARE:.0%}", lambda r: r >= MAX_SHARE), (f"R <  {MAX_SHARE:.0%}", lambda r: r < MAX_SHARE)):
        xs = [x for (i, m), x in losses.items() if keep(float(own[(i, m)]["R"]))]
        mean = f"{sum(xs) / len(xs):5.2f}" if xs else "    -"
        print(f"  {label}: {len(xs):3} rewrites, judge mean loss {mean}")


def long_outputs(versions, methods):
    """Per method: rewrites over LONG_RATIO x the original's words."""
    print(f"\n=== rewrites over {LONG_RATIO} x the original's words (answers instead of rewrites)")
    for m in methods:
        k = sum(1 for v in versions.values() if len(v[m].split()) > LONG_RATIO * len(v[ORIGINAL].split()))
        print(f"  {m:<17} {share(k, len(versions))}")


def plot_privacy_utility(points, path):
    """Leak rate against utility loss, one point per version: the judge (left), ours (right)."""
    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4.5))
    # same version order in both panels -> same color per version
    for v, (judge_leak, ours_leak, loss, mean_r) in points.items():
        left.scatter(loss, 100 * judge_leak, label=v)
        right.scatter(100 * mean_r, 100 * ours_leak, label=v)
    left.legend(loc="lower right")
    right.legend(loc="lower right")
    left.set(xlabel="judge: mean utility loss (0 none, 10 nothing left)", ylabel="judge: leak rate (%)",
             title="LLM judge", xlim=(-0.3, 10), ylim=(0, 105))
    right.set(xlabel="ours: mean R = D / D(empty) (%)", ylabel=f"ours: leak rate, max P > {THRESHOLD} (%)",
              title="our attackers and D", xlim=(-3, 100), ylim=(0, 105))
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"\nprivacy-utility plot -> {path}")


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
    print(f"{HAND_CHECK} decisions for the hand check -> {eval_dir / 'hand_check.csv'}")


def main():
    args = get_args()
    eval_dir = Path(args.eval_dir)
    names = [ORIGINAL] + args.methods
    versions = read_versions(args.data, args.methods)
    truth = read_truth(args.data)
    result = outcomes(eval_dir)
    losses = read_utility(eval_dir)
    own = {(r["id"], r["method"]): r for r in read_csv(eval_dir / "own_measures.csv")}
    with open(eval_dir / "judge_outcomes.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "version", "attribute", "guess", "outcome"])
        for (i, v, a), (guess, _, outcome) in sorted(result.items()):
            w.writerow([i, v, a, guess, outcome])
    with open(eval_dir / "judge_utility.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "method", "loss"])
        for (i, m), x in sorted(losses.items()):
            w.writerow([i, m, x])

    counted = counted_pairs(result)
    judge = {k: o == "match" for k, (_, _, o) in result.items()}
    ours = attacker_leaks(result, own)
    points = main_table(result, counted, ours, losses, own, names)
    print("\n=== per attribute, on the counted pairs")
    attribute_table("judge: leak rate (match)", judge, counted, names)
    attribute_table(f"ours: leak rate (max P > {THRESHOLD})", ours, counted, names)
    transitions(result, args.methods)
    agreement(judge, ours, counted, names)
    utility_agreement(losses, own, args.methods)
    long_outputs(versions, args.methods)
    plot_privacy_utility(points, eval_dir / "privacy_utility.png")
    hand_check(result, versions, truth, eval_dir)


if __name__ == "__main__":
    main()
