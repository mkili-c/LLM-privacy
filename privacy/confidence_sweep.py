"""
Run the attribute-inference measurement from confidence_assesment.py over every
row of a CSV, one advantage per text column, and optionally save a JSON.
"""
import argparse
import csv
import json
import math
import sys
from pathlib import Path

from confidence_assesment import add_common_args, advantage, setup


def load_rows(path, attr, label_column, max_rows):
    """Rows with a clear ground-truth label; unclear rows are skipped and counted."""
    rows, skipped = [], 0
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            label = attr["label"](row[label_column])
            if label is None:
                skipped += 1
                continue
            rows.append((row, label))
            if max_rows and len(rows) >= max_rows:
                break
    return rows, skipped


def get_args():
    p = argparse.ArgumentParser(description="Advantage 2|acc - 1/2| of the LLM at guessing a user attribute, over a CSV")
    add_common_args(p)
    p.add_argument("--csv", required=True)
    p.add_argument("--columns", nargs="+", default=["original"])
    p.add_argument("--label_column", default="id")
    p.add_argument("--max_rows", type=int)
    p.add_argument("--save_suffix")
    return p.parse_args()


def main():
    args = get_args()
    attr, run, prior, forms_info = setup(args)
    pos = attr["classes"][1]

    rows, skipped = load_rows(args.csv, attr, args.label_column, args.max_rows)
    print(f"\n{len(rows)} labelled rows from {args.csv} ({skipped} skipped)")
    if not rows:
        sys.exit(f"No row could be labelled from column '{args.label_column}'")

    base_rate = sum(label for _, label in rows) / len(rows)
    if base_rate in (0.0, 1.0):
        print(f"\n!! WARNING: every row is '{attr['classes'][int(base_rate)]}'. "
              "This run measures answering bias, not leakage.")

    results, per_example = {}, {}
    for col in args.columns:
        print(f"\n=== {col} ===")
        records = []
        for i, (row, label) in enumerate(rows, 1):
            r = run(row[col])
            r["label"] = label
            r["id"] = row.get(args.label_column, str(i))
            records.append(r)
            if math.isnan(r["p_pos"]):
                verdict = "?"
            else:
                verdict = "hit" if (r["p_pos"] > 0.5) == (label == 1) else "MISS"
            print(f"  [{i}/{len(rows)}] {r['id'][:34]:<34} true={attr['classes'][label]:<12} "
                  f"P({pos})={r['p_pos']:.4f} cov={r['coverage']:.3f} {verdict}", flush=True)
        per_example[col] = records
        results[col] = advantage(records)

    print(f"\n{'column':<26} {'n':>4} {'acc':>7} {'ADV':>7} {'+-':>6} "
          f"{'ADV_soft':>9} {'floor':>7} {'base':>7} {'cover':>7}")
    print("-" * 90)
    for col, s in results.items():
        if s is None:
            print(f"{col:<26}    -  (no usable readouts)")
            continue
        print(f"{col:<26} {s['n']:>4} {s['acc_hard']:>7.4f} {s['advantage_hard']:>7.4f} "
              f"{s['advantage_hard_stderr']:>6.4f} {s['advantage_soft']:>9.4f} "
              f"{s['floor']:>7.4f} {s['advantage_baseline']:>7.4f} {s['mean_coverage']:>7.4f}")
    print("\nADV counts as leakage only if it is above both `floor` and `base`.")

    if args.save_suffix:
        out_dir = Path("results/privacy_advantage")
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"{args.save_suffix}.json"
        with open(path, "w", encoding="utf-8") as fh:
            json.dump({
                "model": args.model_path,
                "attribute": {"name": attr["name"], "classes": attr["classes"],
                              "questions": attr["questions"],
                              "age_threshold": args.age_threshold if attr["name"] == "age" else None},
                "csv": args.csv,
                "counterbalance": args.counterbalance,
                "ablate_context": args.ablate_context,
                "surface_forms": forms_info,
                "prior": prior,
                "summary": results,
                "per_example": per_example,
            }, fh, indent=2, ensure_ascii=False)
        print(f"\nWrote {path}")


if __name__ == "__main__":
    main()
