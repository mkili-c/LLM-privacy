"""
Check a judge session's answers.jsonl right after the session, while it can still be asked to fix gaps.

For one folder (guess_<code>, utility or compare), compares --answers_dir/<folder>/answers.jsonl with
the items the judge was given (--upload_dir/<folder>/items.jsonl) and reports:
    missing items, items answered more than once (the last answer counts), unknown item ids,
    malformed lines, and per kind:
        guess    every listed attribute answered, with a non-empty "guess" and "reasoning"
        utility  "loss" an integer from 0 to 10, "reasoning" non-empty
        compare  "match" one of: match, less precise, no
Exits with status 1 if anything is wrong.

    python evaluation/check_judge_answers.py guess_M
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
MATCH_LABELS = {"match", "less precise", "no"}


def get_args():
    p = argparse.ArgumentParser(description="Check one judge session's answers")
    p.add_argument("folder", help="guess_<code>, utility or compare")
    p.add_argument("--upload_dir", default=str(REPO / "results" / "evaluation" / "judge_upload"))
    p.add_argument("--answers_dir", default=str(REPO / "results" / "evaluation" / "judge_answers"))
    return p.parse_args()


def read_jsonl(path):
    """-> (rows, [(line number, problem)])."""
    rows, problems = [], []
    with open(path, encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as err:
                problems.append((n, f"not JSON ({err})"))
    return rows, problems


def answer_problems(kind, item, answer):
    """Problems of one answer, given its item."""
    if kind == "guess":
        given = answer.get("answers")
        if not isinstance(given, dict):
            return ["no 'answers' object"]
        out = [f"attribute '{a}' missing" for a in item["attributes"] if a not in given]
        for a, g in given.items():
            if not isinstance(g, dict) or not str(g.get("guess", "")).strip():
                out.append(f"attribute '{a}': no guess")
            elif not str(g.get("reasoning", "")).strip():
                out.append(f"attribute '{a}': no reasoning")
        return out
    if kind == "utility":
        loss = answer.get("loss")
        out = [] if isinstance(loss, int) and 0 <= loss <= 10 else [f"loss {loss!r} is not an integer 0-10"]
        return out + ([] if str(answer.get("reasoning", "")).strip() else ["no reasoning"])
    label = str(answer.get("match", "")).strip().lower()
    return [] if label in MATCH_LABELS else [f"match {answer.get('match')!r} is not one of {sorted(MATCH_LABELS)}"]


def main():
    args = get_args()
    kind = "guess" if args.folder.startswith("guess_") else args.folder
    items, _ = read_jsonl(Path(args.upload_dir) / args.folder / "items.jsonl")
    items = {it["item"]: it for it in items}
    answers, problems = read_jsonl(Path(args.answers_dir) / args.folder / "answers.jsonl")
    counts = Counter(a.get("item") for a in answers)
    last = {a.get("item"): a for a in answers}
    report = [f"line {n}: {p}" for n, p in problems]
    report += [f"item {i}: answered {c} times (the last counts)" for i, c in counts.items() if c > 1]
    report += [f"unknown item {i!r}" for i in last if i not in items]
    missing = [i for i in items if i not in last]
    if missing:
        report.append(f"{len(missing)} items not answered, e.g. {missing[:10]}")
    for i, answer in last.items():
        if i in items:
            report += [f"item {i}: {p}" for p in answer_problems(kind, items[i], answer)]
    print(f"{args.folder}: {len(items)} items, {len(last)} answered")
    for line in report[:50]:
        print("  " + line)
    if len(report) > 50:
        print(f"  ... and {len(report) - 50} more")
    print("OK" if not report else f"{len(report)} problems")
    sys.exit(1 if report else 0)


if __name__ == "__main__":
    main()
