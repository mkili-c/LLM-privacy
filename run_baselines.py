"""Run one anonymization baseline over an evaluation set and record its rewrites.

    python run_baselines.py --method presidio --data data/pupa_sd/pupa_sd_eval.csv

Reads `id` and `original` from the evaluation CSV, gives each prompt to the method's
adapter, and appends `id, method, rewrite` to the output CSV one row at a time. Ids
already in the output are skipped, so an interrupted run resumes where it stopped.

Each method runs in its own venv, because the baselines pin conflicting packages
(ROADMAP.md, Evaluation pipeline, Step 4). The adapters only feed the prompt to the
released code and read its output; they never change the method.
"""
import argparse
import csv
import os

METHODS = ["presidio"]  # loiseau, papillon and staab follow
OUT_TEMPLATE = "results/rewrites/{data}_{method}.csv"  # {data}: the eval CSV's folder name


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
        return anonymizer.anonymize(text=text, analyzer_results=results).text
    return rewrite


def load_adapter(method):
    """Load the method's models once; return a function prompt -> rewrite."""
    if method == "presidio":
        return load_presidio()
    raise ValueError(f"no adapter for {method}")


def read_rows(path, limit):
    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return rows[:limit] if limit else rows


def done_ids(path):
    if not os.path.exists(path):
        return set()
    with open(path, encoding="utf-8") as f:
        return {r["id"] for r in csv.DictReader(f)}


def run(rows, rewrite, method, out_path):
    """Append one rewrite per row not yet in out_path; flush after each, so a crash loses at most one."""
    done = done_ids(out_path)
    todo = [r for r in rows if r["id"] not in done]
    print(f"{method}: {len(done)} rows already done, {len(todo)} to go -> {out_path}")
    new_file = not os.path.exists(out_path)
    with open(out_path, "a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        if new_file:
            writer.writerow(["id", "method", "rewrite"])
        for n, row in enumerate(todo, 1):
            writer.writerow([row["id"], method, rewrite(row["original"])])
            f.flush()
            if n % 20 == 0 or n == len(todo):
                print(f"  {n}/{len(todo)}")


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
