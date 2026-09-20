"""
Measure the cosine-similarity utility metric over data/privacy_prompt_rewrites_100.csv.

The input CSV has a fixed shape:

    id, protected_attrs, original,
    util_preserving_rewrite, util_paraphrase_rewrite,
    priv_preserving_rewrite, priv_targeted_rewrite

For every (row, rewrite column, K) cell we draw K answers from the original and
K from the rewrite, embed each answer on its own (mean-pooled hidden states,
L2-normalized), and measure how far apart the two answer clouds sit:

    cross   = mean_{i,j}    cos(e_i, f_j)      e_i: embeddings of answers to the original
    self_o  = mean_{i != i'} cos(e_i, e_i')    f_j: embeddings of answers to the rewrite
    self_r  = mean_{j != j'} cos(f_j, f_j')
    udist2  = self_o + self_r - 2 * cross      = ||mu_orig - mu_rew||^2 (unbiased)
    cos_centroid = cross / sqrt(self_o * self_r)
                                               cosine between the two cloud
                                               centroids (1 = same direction);
                                               scale-free, so comparable across
                                               prompts. The headline utility.
    rel_udist2   = udist2 / sqrt(self_o * self_r)
                                               = 2 (1 - cos_centroid) + a
                                               concentration-mismatch penalty

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

from topk_sample import load, topk_answers
from cosine_utility import embed_answers, cosine_stats

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

OUT_COLS = ["id", "rewrite_type", "k", "udist2", "rel_udist2", "cos_centroid", "dist2",
            "cross", "self_orig", "self_rew", "seconds"]

# metric -> True if a LARGER value means more utility kept. Decides the
# direction of the "ranking" column (reference should keep the most utility).
HIGHER_IS_BETTER = {"udist2": False, "rel_udist2": False, "cos_centroid": True}
SUMMARY_METRICS = ["udist2", "cos_centroid"]


def get_args():
    p = argparse.ArgumentParser(description="Cosine-similarity utility over the "
                                            "prompt-rewrites CSV")
    p.add_argument("--model_path", default="/work/projects/mkilic/models/Qwen2.5-3B-Instruct")
    p.add_argument("--csv", default="./data/privacy_prompt_rewrites_100.csv")
    p.add_argument("--k_grid", type=int, nargs="+", default=[5, 10, 15],
                   help="sample counts to test (needs >= 2 per cell)")
    p.add_argument("--max_new_tokens", type=int, default=128)
    p.add_argument("--temperature", type=float, default=1.0,
                   help="keep 1.0 to match kl_measurement.py; tempering is admissible "
                        "for this metric as long as both sides stay identical")
    p.add_argument("--top_p", type=float, default=1.0,
                   help="keep 1.0 to match kl_measurement.py; tempering is admissible "
                        "for this metric as long as both sides stay identical")
    p.add_argument("--layer", type=int, default=-1,
                   help="hidden-state layer to embed answers from (-1 = final)")
    p.add_argument("--seed", type=int, default=23)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--out_dir", default="./results/cos_measurement")
    p.add_argument("--save_suffix", default="cos")
    return p.parse_args()


def mean_std(xs):
    if not xs:
        return float("nan"), float("nan")
    m = sum(xs) / len(xs)
    if len(xs) < 2:
        return m, 0.0
    return m, (sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5


def score_cell(original, rewritten, model, tok, k, args):
    """Utility of one rewrite at one K, or None if either side gave fewer than
    two non-empty answers (the self-similarity terms need a pair)."""
    a_orig = [a for a, _, _ in topk_answers(
        original, model, tok, k=k, mode="sample", max_new_tokens=args.max_new_tokens,
        temperature=args.temperature, top_p=args.top_p) if a.strip()]
    a_rew = [a for a, _, _ in topk_answers(
        rewritten, model, tok, k=k, mode="sample", max_new_tokens=args.max_new_tokens,
        temperature=args.temperature, top_p=args.top_p) if a.strip()]
    
    if len(a_orig) < 2 or len(a_rew) < 2:
        return None

    E_orig = embed_answers(a_orig, model, tok, args.batch_size, args.layer)
    E_rew = embed_answers(a_rew, model, tok, args.batch_size, args.layer)
    # self_o + self_r - 2 * cross.
    stats, _ = cosine_stats(E_orig, E_rew)
    return {"udist2": stats["udist2"], "rel_udist2": stats["rel_udist2"],
            "cos_centroid": stats["cos_centroid"],
            "dist2": stats["squared_diff_of_means"],
            "cross": stats["cross_mean"], "self_orig": stats["self_orig"],
            "self_rew": stats["self_rew"]}


def summarize(records, k_grid, metric="udist2"):
    """Per K, one line per rewrite type: mean ± std, and for the non-reference
    types the gap and pairwise ranking accuracy against REFERENCE over the ids
    where both types produced a score."""
    higher = HIGHER_IS_BETTER[metric]
    lines = []
    for k in k_grid:
        lines.append(f"\n{'=' * 76}\nSUMMARY  K={k}  metric={metric}  (reference: {REFERENCE}, "
                     f"{'higher' if higher else 'lower'} = more utility)")
        lines.append(f"{'rewrite type':<24} {'mean':>16} {'gap vs ref':>12} {'ranking':>16}")
        lines.append("-" * 76)

        # rewrite type -> {id: metric} for this K
        by_type = {t: {} for t in TYPES}
        for r in records:
            if r["k"] == k:
                by_type[r["rewrite_type"]][r["id"]] = r[metric]

        ref = by_type[REFERENCE]
        ref_mean, _ = mean_std(list(ref.values()))
        for t in TYPES:
            m, s = mean_std(list(by_type[t].values()))
            if t == REFERENCE:
                gap, acc = "-", "-"
            else:
                ids = sorted(set(ref) & set(by_type[t]))
                # "correct" = the reference rewrite keeps more utility: lower
                # distance, or higher centroid cosine
                correct = sum((ref[i] > by_type[t][i]) if higher else (ref[i] < by_type[t][i])
                              for i in ids)
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
    print(f"Metric: cosine similarity of answer embeddings from layer {args.layer} "
          f"-> squared distance between the cloud means")

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
                scores = score_cell(original, rewritten, model, tok, k, args)
                seconds = round(time.time() - t0, 2)

                head = f"[{cell_i}/{n_cells}] id={row['id'][:24]:<24} {TYPES[col_i]:<20} k={k:<3}"
                if scores is None:
                    print(f"{head} SKIP (< 2 non-empty answers)  ({seconds}s)", flush=True)
                    continue

                records.append({"id": row["id"], "rewrite_type": TYPES[col_i], "k": k,
                                "seconds": seconds, **scores})
                print(f"{head} udist2={scores['udist2']:.3f} "
                      f"cos_centroid={scores['cos_centroid']:.3f} "
                      f"dist2={scores['dist2']:.3f}  ({seconds}s)", flush=True)

    os.makedirs(args.out_dir, exist_ok=True)
    csv_path = os.path.join(args.out_dir, f"{args.save_suffix}.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=OUT_COLS)
        w.writeheader()
        w.writerows(records)

    for metric in SUMMARY_METRICS:
        print(summarize(records, args.k_grid, metric))
    print(f"\nDone in {round(time.time() - start, 1)}s over {len(records)} cells.")
    print(f"CSV -> {csv_path}")
