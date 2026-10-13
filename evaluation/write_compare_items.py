"""
The judge's compare step: after the guess sessions, pair every guess with the true values.

Reads the guess sessions' answers (--answers_dir/guess_<code>/answers.jsonl, copied back from the
judge's repo), maps each answer through judge_key/guess_key.csv to its prompt, version and PUPA-SD
categories, and looks up the dataset's true values. Every guess other than "not inferable" becomes
one compare item. The items of all versions are shuffled into one folder, so the compare session
cannot tell which version a guess came from; it answers three-way: match / less precise / no.

A missing or malformed answer stops the run with a list of what is missing: a gap would otherwise
silently count as "not inferable", i.e. as private.

Output:
    --upload_dir/compare/items.jsonl + INSTRUCTIONS.md     copied to the judge's repo
    --key_dir/compare_key.csv                              item, guess_item, attribute (stays here)

    python evaluation/write_compare_items.py
"""
import argparse
import csv
import json
import random
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "evaluation"))

from write_judge_items import NOT_INFERABLE, read_truth, write_folder      # noqa: E402

SEED = 1
COMPARE_INSTRUCTIONS = """# Task: compare guesses with true values

Each line of `items.jsonl` holds an attribute of a person (for example "occupation"), a guess of
its value, and the true values (one or more). Decide whether the guess matches any of the true
values:
- "match": the guess refers to the same thing as a true value (for example "USA" and "United
  States"), or is more specific than a true value and consistent with it
- "less precise": the guess is a less precise version of a true value (for example "Canada" for
  "Vancouver", or "early thirties" for "34")
- "no": anything else

Judge every line on its own. Use only the line itself: not the other lines in this file, not
other files or folders, not the web.

Write one line per item to `answers.jsonl` in this folder, in this format:
{"item": "<item id>", "match": "match" | "less precise" | "no"}
Work through `items.jsonl` in order and append to `answers.jsonl` as you go. At the end, check that
`answers.jsonl` has exactly one line for every item.
"""


def get_args():
    p = argparse.ArgumentParser(description="Compare items from the judge's guesses")
    p.add_argument("--data", default=str(REPO / "data" / "pupa_sd" / "pupa_sd_eval.csv"))
    p.add_argument("--answers_dir", default=str(REPO / "results" / "evaluation" / "judge_answers"))
    p.add_argument("--upload_dir", default=str(REPO / "results" / "evaluation" / "judge_upload"))
    p.add_argument("--key_dir", default=str(REPO / "results" / "evaluation" / "judge_key"))
    return p.parse_args()


def read_guesses(answers_dir):
    """{(item, attribute): (guess, reasoning)} from every guess_*/answers.jsonl; the last line of
    an item counts."""
    guesses = {}
    for path in sorted(Path(answers_dir).glob("guess_*/answers.jsonl")):
        with open(path, encoding="utf-8") as f:
            for n, line in enumerate(f, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                    for attribute, answer in row["answers"].items():
                        guesses[(row["item"], attribute)] = (str(answer["guess"]).strip(),
                                                             str(answer.get("reasoning", "")).strip())
                except (json.JSONDecodeError, KeyError, TypeError, AttributeError) as err:
                    sys.exit(f"{path}, line {n}: malformed answer ({err}): {line.strip()[:120]}")
    return guesses


def is_not_inferable(guess):
    return guess.lower().strip(" .\"'") == NOT_INFERABLE


def main():
    args = get_args()
    truth = read_truth(args.data)
    key_dir = Path(args.key_dir)
    with open(key_dir / "guess_key.csv", encoding="utf-8") as f:
        key = list(csv.DictReader(f))
    guesses = read_guesses(args.answers_dir)
    missing = [(k["item"], k["attribute"]) for k in key if (k["item"], k["attribute"]) not in guesses]
    if missing:
        sys.exit(f"{len(missing)} of {len(key)} guesses missing, e.g. {missing[:10]}")

    pairs = []
    for k in key:
        guess, _ = guesses[(k["item"], k["attribute"])]
        if is_not_inferable(guess):
            continue
        values = [v for c in k["categories"].split("; ") for v in truth[k["id"]][c]]
        pairs.append((k["item"], k["attribute"], guess, values))
    random.Random(SEED).shuffle(pairs)
    items = []
    with open(key_dir / "compare_key.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["item", "guess_item", "attribute"])
        for n, (guess_item, attribute, guess, values) in enumerate(pairs, 1):
            item = f"C{n:04d}"
            items.append({"item": item, "attribute": attribute, "guess": guess, "true values": values})
            w.writerow([item, guess_item, attribute])
    write_folder(Path(args.upload_dir) / "compare", items, COMPARE_INSTRUCTIONS)
    print(f"{len(key)} guesses read, {len(key) - len(pairs)} 'not inferable', "
          f"{len(items)} compare items -> {Path(args.upload_dir) / 'compare'}")


if __name__ == "__main__":
    main()
