"""
Measures the cosine and kl metrics over data/privacy_prompt_rewrites_floor_100.csv.

The input CSV has a fixed shape:

    id, protected_attrs, original,
    util_preserving_rewrite, util_paraphrase_rewrite,
    priv_preserving_rewrite, priv_targeted_rewrite, util_broken_rewrite

For every (row, rewrite column, K) cell draws K answers from original and rewritten
answer distributions. Then calculates the cos and kl metrics on same samples

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
from kl_utility_scoring import directional_kl

# Fixed CSV layout: these are the only columns we score, in this order.
REWRITE_COLS = [
    "util_preserving_rewrite",
    "util_paraphrase_rewrite",
    "priv_preserving_rewrite",
    "priv_targeted_rewrite",
    "util_broken_rewrite",
]
# util_preserving_rewrite -> util_preserving; the type every other type is
# compared against in the summary.
TYPES = [c[: -len("_rewrite")] for c in REWRITE_COLS]
REFERENCE = TYPES[0]

OUT_COLS = ["id", "rewrite_type", "k", "total_divergence_per_token",
            "kl_orig_to_rew", "kl_orig_to_rew_stderr", "kl_rew_to_orig", "kl_rew_to_orig_stderr",
            "udist2", "rel_udist2", "cos_centroid", "dist2",
            "cross", "self_orig", "self_rew", "sample_time", "embed_time", "kl_time"]

# cosine metric -> True if a LARGER value means more utility kept (decides the
# direction of the "ranking" column in the summary)
COS_HIGHER_IS_BETTER = {"udist2": False, "rel_udist2": False, "cos_centroid": True}
COS_SUMMARY_METRICS = ["udist2", "cos_centroid"]


def get_args():
    p = argparse.ArgumentParser(description="Cosine-similarity utility over the "
                                            "prompt-rewrites CSV")
    p.add_argument("--model_path", default="/work/projects/mkilic/models/Qwen2.5-3B-Instruct")
    p.add_argument("--csv", default="./data/privacy_prompt_rewrites_floor_100.csv")
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
    p.add_argument("--embed_batch_size", type=int, default=8)
    p.add_argument("--kl_batch_size", type=int, default=8)
    p.add_argument("--out_dir", default="./results/cos_kl_test/privacy_floor_100")
    p.add_argument("--save_suffix", default="cos_vs_kl_privacy_floor_100")
    return p.parse_args()


def mean_std(xs):
    if not xs:
        return float("nan"), float("nan")
    m = sum(xs) / len(xs)
    if len(xs) < 2:
        return m, 0.0
    return m, (sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5


def score_cell(a_orig, original, rewritten, model, tok, k, args):
    """Utility of one rewrite at one K, or None if either side gave fewer than
    two non-empty answers (the self-similarity terms need a pair)."""
    t0 = time.time()
    a_rew = [a for a, _, _ in topk_answers(
        rewritten, model, tok, k=k, mode="sample", max_new_tokens=args.max_new_tokens,
        temperature=args.temperature, top_p=args.top_p) if a.strip()]
    
    if  len(a_rew) < 2:
        return None
    sampleseconds = round(time.time() - t0, 2)

    t0 = time.time()
    E_orig = embed_answers(a_orig, model, tok, args.embed_batch_size, args.layer)
    E_rew = embed_answers(a_rew, model, tok, args.embed_batch_size, args.layer)
    # self_o + self_r - 2 * cross.
    stats, _ = cosine_stats(E_orig, E_rew)
    embedseconds = round(time.time() - t0, 2)

    t0 = time.time()
    kl_o2r, se_o2r, _ = directional_kl(a_orig, original, rewritten, model, tok, batch_size=args.kl_batch_size)
    kl_r2o, se_r2o, _ = directional_kl(a_rew, rewritten, original, model, tok, batch_size=args.kl_batch_size)    
    J = kl_o2r + kl_r2o    
    klseconds = round(time.time() - t0, 2)

    return {
            "total_divergence_per_token": J,
            "kl_orig_to_rew": kl_o2r, "kl_orig_to_rew_stderr": se_o2r,
            "kl_rew_to_orig": kl_r2o, "kl_rew_to_orig_stderr": se_r2o,
            "udist2": stats["udist2"], "rel_udist2": stats["rel_udist2"],
            "cos_centroid": stats["cos_centroid"],
            "dist2": stats["squared_diff_of_means"],
            "cross": stats["cross_mean"], "self_orig": stats["self_orig"],
            "self_rew": stats["self_rew"], "sample_time": sampleseconds, "embed_time": embedseconds, "kl_time":klseconds}


def summarize_cos(records, k_grid, metric="udist2"):
    """Per K, one line per rewrite type: mean ± std, and for the non-reference
    types the gap and pairwise ranking accuracy against REFERENCE over the ids
    where both types produced a score."""
    higher = COS_HIGHER_IS_BETTER[metric]
    lines = []
    for k in k_grid:
        lines.append(f"\n{'=' * 76}\nSUMMARY COS K={k}  metric={metric}  (reference: {REFERENCE}, "
                     f"{'higher' if higher else 'lower'} = more utility)")
        lines.append(f"{'rewrite type':<24} {'mean':>16} {'paired gap ± SE':>16} {'ranking':>16}")
        lines.append("-" * 76)

        # rewrite type -> {id: metric} for this K
        by_type = {t: {} for t in TYPES}
        for r in records:
            if r["k"] == k:
                by_type[r["rewrite_type"]][r["id"]] = r[metric]

        ref = by_type[REFERENCE]
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
                # Paired: per-id differences cancel the prompt-level spread that
                # dominates the unpaired std, so this SE is the one to read.
                dm, ds = mean_std([by_type[t][i] - ref[i] for i in ids])
                gap = f"{dm:+.4f}±{ds / len(ids) ** 0.5:.4f}" if ids else "n/a"
                acc = f"{correct}/{len(ids)} ({100 * correct / len(ids):.0f}%)" if ids else "n/a"
            lines.append(f"{t:<24} {m:>8.3f} ± {s:<5.3f} {gap:>16} {acc:>16}")
    return "\n".join(lines)

def summarize_kl(records, k_grid):
    """Per K, one line per rewrite type: mean ± std, and for the non-reference
    types the gap and pairwise ranking accuracy against REFERENCE over the ids
    where both types produced a score."""
    lines = []
    for k in k_grid:
        lines.append(f"\n{'=' * 76}\nSUMMARY KL  K={k}    (reference: {REFERENCE})")
        lines.append(f"{'rewrite type':<24} {'mean':>16} {'paired gap ± SE':>16} {'ranking':>16}")
        lines.append("-" * 76)

        # rewrite type -> {id: utility} for this K
        by_type = {t: {} for t in TYPES}
        for r in records:
            if r["k"] == k:
                by_type[r["rewrite_type"]][r["id"]] = r["total_divergence_per_token"]

        ref = by_type[REFERENCE]
        for t in TYPES:
            m, s = mean_std(list(by_type[t].values()))
            if t == REFERENCE:
                gap, acc = "-", "-"
            else:
                ids = sorted(set(ref) & set(by_type[t]))
                # "correct" = the reference rewrite keeps more utility hence less divergence
                correct = sum(ref[i] < by_type[t][i] for i in ids)
                # Paired: per-id differences cancel the prompt-level spread that
                # dominates the unpaired std, so this SE is the one to read.
                dm, ds = mean_std([by_type[t][i] - ref[i] for i in ids])
                gap = f"{dm:+.4f}±{ds / len(ids) ** 0.5:.4f}" if ids else "n/a"
                acc = f"{correct}/{len(ids)} ({100 * correct / len(ids):.0f}%)" if ids else "n/a"
            lines.append(f"{t:<24} {m:>8.3f} ± {s:<5.3f} {gap:>16} {acc:>16}")
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


    os.makedirs(args.out_dir, exist_ok=True)
    csv_path = os.path.join(args.out_dir, f"{args.save_suffix}.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=OUT_COLS)
        w.writeheader()
        

    records = []
    start = time.time()
    cell_i = 0

    for row_i, row in enumerate(rows):
        for k in args.k_grid:

            torch.manual_seed(args.seed + k + 10_000 * row_i)   # original: per (row, K) 
            original = row["original"].strip()
            
            t0 = time.time()
            a_orig = [a for a, _, _ in topk_answers(
                    original, model, tok, k=k, mode="sample", max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature, top_p=args.top_p) if a.strip()]
            orig_sample_time = round(time.time() - t0, 2)

            if len(a_orig) < 2:
                print(f"a_orig too short: id={row['id'][:24]} k={k}", flush=True)
                continue
    
            for col_i, col in enumerate(REWRITE_COLS):

                rewritten = row[col].strip()
                cell_i += 1

                # distinct per (row, column, K); strides leave room for
                # 100 K values, 100 columns and any number of rows
                torch.manual_seed(args.seed + k + 100 * (col_i+1) + 10_000 * row_i)
                
                scores = score_cell(a_orig, original, rewritten, model, tok, k, args)

                if scores is None:
                    head = f"[{cell_i}/{n_cells}] id={row['id'][:24]:<24} {TYPES[col_i]:<20} k={k:<3}"
                    print(f"{head} SKIP (< 2 non-empty answers)", flush=True)
                    continue

                records.append({
                    "id": row["id"], "rewrite_type": TYPES[col_i], "k": k,
                    **scores
                })

                with open(csv_path, "a", newline="", encoding="utf-8") as f:
                    w = csv.DictWriter(f, fieldnames=OUT_COLS)
                    w.writerow(records[-1])

                tottime = scores["sample_time"]+scores["embed_time"]+scores["kl_time"]+orig_sample_time

                print(f"[{cell_i}/{n_cells}] id={row['id'][:24]:<24} "
                        f"{TYPES[col_i]:<20} k={k:<3} "
                        f"Total_div_pt={scores['total_divergence_per_token']:.3f}  "
                        f"udist2={scores['udist2']:.3f}  "
                        f"cos_centroid={scores['cos_centroid']:.3f}  ({tottime:.3f}s)", flush=True)
            



    for metric in COS_SUMMARY_METRICS:
        print(summarize_cos(records, args.k_grid, metric))
    print(summarize_kl(records,args.k_grid))
    print(f"\nDone in {round(time.time() - start, 1)}s over {len(records)} cells.")
    print(f"CSV -> {csv_path}")



