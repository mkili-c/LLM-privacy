"""
Measure the KL utility metric over data/privacy_prompt_rewrites_100.csv.

The input CSV has a fixed shape:

    id, protected_attrs, original,
    util_preserving_rewrite, util_paraphrase_rewrite,
    priv_preserving_rewrite, priv_targeted_rewrite


Outputs:
    <out_dir>/<save_suffix>.csv   one flat row per cell
    stdout                        per-cell progress, then per-K mean ± std for
                                  every rewrite type plus the gap and pairwise
                                  ranking accuracy against util_preserving
"""
import os
import csv
import time
import argparse

import torch

from topk_sample import load
from kl_utility_scoring import utility_score

# Fixed CSV layout: these are the only columns we score, in this order.
REWRITE_COLS = [
    "util_preserving_rewrite",
    "util_paraphrase_rewrite",
    "priv_preserving_rewrite",
    "priv_targeted_rewrite",
]
# util_preserving_rewrite -> util_preserving; the type every other type is
# compared against in the summary.
TYPES = [c[: -len("_rewrite")] for c in REWRITE_COLS]
REFERENCE = TYPES[0]

OUT_COLS = ["id", "rewrite_type", "k", "total_divergence_per_token",
            "kl_o2r", "kl_o2r_se", "kl_r2o", "kl_r2o_se", "seconds"]


def get_args():
    p = argparse.ArgumentParser(description="KL utility over the prompt-rewrites CSV")
    p.add_argument("--model_path", default="/work/projects/mkilic/models/Qwen2.5-3B-Instruct")
    p.add_argument("--csv", default="./data/privacy_prompt_rewrites_100.csv")
    p.add_argument("--k_grid", type=int, nargs="+", default=[5, 10, 15])
    p.add_argument("--max_new_tokens", type=int, default=128)
    p.add_argument("--temperature", type=float, default=1.0,
                   help="keep 1.0 for an unbiased KL estimate")
    p.add_argument("--top_p", type=float, default=1.0,
                   help="keep 1.0 for an unbiased KL estimate")
    p.add_argument("--seed", type=int, default=23)
    p.add_argument("--out_dir", default="./results/kl_measurement")
    p.add_argument("--save_suffix", default="kl")
    return p.parse_args()


def mean_std(xs):
    if not xs:
        return float("nan"), float("nan")
    m = sum(xs) / len(xs)
    if len(xs) < 2:
        return m, 0.0
    return m, (sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5


def summarize(records, k_grid):
    """Per K, one line per rewrite type: mean ± std, and for the non-reference
    types the gap and pairwise ranking accuracy against REFERENCE over the ids
    where both types produced a score."""
    lines = []
    for k in k_grid:
        lines.append(f"\n{'=' * 76}\nSUMMARY  K={k}    (reference: {REFERENCE})")
        lines.append(f"{'rewrite type':<24} {'mean':>16} {'gap vs ref':>12} {'ranking':>16}")
        lines.append("-" * 76)

        # rewrite type -> {id: utility} for this K
        by_type = {t: {} for t in TYPES}
        for r in records:
            if r["k"] == k:
                by_type[r["rewrite_type"]][r["id"]] = r["total_divergence_per_token"]

        ref = by_type[REFERENCE]
        ref_mean, _ = mean_std(list(ref.values()))
        for t in TYPES:
            m, s = mean_std(list(by_type[t].values()))
            if t == REFERENCE:
                gap, acc = "-", "-"
            else:
                ids = sorted(set(ref) & set(by_type[t]))
                # "correct" = the reference rewrite keeps more utility hence less divergence
                correct = sum(ref[i] < by_type[t][i] for i in ids)
                gap = f"{ref_mean - m:.3f}"
                acc = f"{correct}/{len(ids)} ({100 * correct / len(ids):.0f}%)" if ids else "n/a"
            lines.append(f"{t:<24} {m:>8.3f} ± {s:<5.3f} {gap:>12} {acc:>16}")
    return "\n".join(lines)


if __name__ == "__main__":
    args = get_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    with open(args.csv, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))

    n_cells = len(rows) * len(REWRITE_COLS) * len(args.k_grid)
    print(f"Loaded {len(rows)} rows from {args.csv}")
    print(f"Rewrite types: {', '.join(TYPES)}  (reference: {REFERENCE})")
    print(f"K grid: {args.k_grid}  ->  {n_cells} cells")

    print(f"\nLoading {args.model_path} on {device} ...", flush=True)
    model, tok = load(args.model_path, device)

    records = []
    start = time.time()
    cell_i = 0
    for row_i, row in enumerate(rows):
        original = row["original"].strip()
        for col_i, col in enumerate(REWRITE_COLS):
            rewritten = row[col].strip()
            for k in args.k_grid:
                cell_i += 1
                # distinct per (row, column, K); strides leave room for
                # 100 K values, 100 columns and any number of rows
                torch.manual_seed(args.seed + k + 100 * col_i + 10_000 * row_i)

                t0 = time.time()
                r = utility_score(original, rewritten, model, tok, k=k,
                                  max_new_tokens=args.max_new_tokens,
                                  temperature=args.temperature, top_p=args.top_p)
                seconds = round(time.time() - t0, 2)

                records.append({
                    "id": row["id"], "rewrite_type": TYPES[col_i], "k": k,
                    "total_divergence_per_token": r["total_divergence_per_token"],
                    "kl_o2r": r["kl_orig_to_rew"], "kl_o2r_se": r["kl_orig_to_rew_stderr"],
                    "kl_r2o": r["kl_rew_to_orig"], "kl_r2o_se": r["kl_rew_to_orig_stderr"],
                    "seconds": seconds,
                })
                print(f"[{cell_i}/{n_cells}] id={row['id'][:24]:<24} "
                      f"{TYPES[col_i]:<20} k={k:<3} "
                      f"Total Div. pt={r['total_divergence_per_token']:.3f}  ({seconds}s)", flush=True)

    os.makedirs(args.out_dir, exist_ok=True)
    csv_path = os.path.join(args.out_dir, f"{args.save_suffix}.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=OUT_COLS)
        w.writeheader()
        w.writerows(records)

    print(summarize(records, args.k_grid))
    print(f"\nDone in {round(time.time() - start, 1)}s over {len(records)} cells.")
    print(f"CSV -> {csv_path}")
