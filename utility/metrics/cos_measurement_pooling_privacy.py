"""
Pooling sweep over the privacy-rewrites CSV: same per-cell logic as
cos_measurement_layers_privacy.py, but instead of sweeping hidden layers it
fixes one layer and compares three ways of pooling the token hidden states
into one answer embedding (all from cosine_utility.py):

    mean      embed_answers           plain mean over non-pad tokens
    last      embed_answers_last      hidden state of an appended EOS token
    weighted  embed_answers_weighted  position-weighted mean, w_i = i / sum_j j

The input CSV has a fixed shape:

    id, protected_attrs, original,
    util_preserving_rewrite, util_paraphrase_rewrite,
    priv_preserving_rewrite, priv_targeted_rewrite, util_broken_rewrite

For every (row, K) we draw K answers from the original ONCE and reuse them for
all five rewrite types. Every answer set is embedded with each pooling, so all
poolings see the exact same samples: differences between poolings are purely
representational, not sampling noise. Per pooling:

    cross   = mean_{i,j}    cos(e_i, f_j)      e_i: embeddings of answers to the original
    self_o  = mean_{i != i'} cos(e_i, e_i')    f_j: embeddings of answers to the rewrite
    self_r  = mean_{j != j'} cos(f_j, f_j')
    udist2  = self_o + self_r - 2 * cross      (unbiased ||mu_orig - mu_rew||^2)
    cos_centroid = cross / sqrt(self_o * self_r)   higher = more utility
    rel_udist2   = udist2 / sqrt(self_o * self_r)

Outputs:
    <out_dir>/<save_suffix>.csv   one flat row per (id, rewrite_type, k, POOLING)
    stdout                        per-cell progress, then for every K a table
                                  with one line per pooling: mean, paired gap ± SE
                                  and ranking accuracy against the reference type.
"""
import os
import csv
import time
import argparse

import torch

from topk_sample import load
from cosine_utility import (embed_answers, embed_answers_last,
                            embed_answers_weighted, cosine_stats)
from cos_measurement_layers_privacy import (REWRITE_COLS, TYPES, REFERENCE,
                                            SUMMARY_METRICS, HIGHER_IS_BETTER,
                                            mean_std, sample_answers)

POOLINGS = {
    "mean": embed_answers,
    "last": embed_answers_last,
    "weighted": embed_answers_weighted,
}

OUT_COLS = ["id", "rewrite_type", "k", "pooling", "layer", "udist2", "rel_udist2",
            "cos_centroid", "dist2", "rel_dist2", "cross", "self_orig", "self_rew",
            "seconds"]


def get_args(description="Pooling sweep of the cosine-similarity utility over "
                         "the privacy-rewrites CSV",
             csv_default="./data/prompt_rewrites_100.csv",
             suffix_default="privacy_cos_pooling"):
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--model_path", default="/work/projects/mkilic/models/Qwen2.5-3B-Instruct")
    p.add_argument("--csv", default=csv_default)
    p.add_argument("--k_grid", type=int, nargs="+", default=[15],
                   help="sample counts to test (needs >= 2 per cell)")
    p.add_argument("--max_new_tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top_p", type=float, default=1.0)
    p.add_argument("--poolings", nargs="+", default=list(POOLINGS), choices=list(POOLINGS))
    p.add_argument("--layer", type=int, default=-1,
                   help="hidden layer to pool (0 = embeddings, -1 = final)")
    p.add_argument("--seed", type=int, default=23)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--summary_metric", default="udist2", choices=SUMMARY_METRICS,
                   help="which column the stdout summary tables rank on")
    p.add_argument("--out_dir", default="./results/cos_measurement/pooling")
    p.add_argument("--save_suffix", default=suffix_default)
    return p.parse_args()


def embed_all(answers, model, tok, poolings, args):
    """{pooling: [N, H]} for one answer set."""
    return {p: POOLINGS[p](answers, model, tok, args.batch_size, args.layer)
            for p in poolings}


def score_cell_poolings(E_orig, rewritten, model, tok, k, poolings, args):
    """Utility of one rewrite at one K, for every pooling, against the already
    embedded original answers. Returns {pooling: stats} or None if the rewrite
    gave fewer than two non-empty answers (the self-similarity terms need a pair)."""
    a_rew = sample_answers(rewritten, model, tok, k, args)
    if len(a_rew) < 2:
        return None
    E_rew = embed_all(a_rew, model, tok, poolings, args)

    out = {}
    for p in poolings:
        stats, _ = cosine_stats(E_orig[p], E_rew[p])
        out[p] = {"udist2": stats["udist2"], "rel_udist2": stats["rel_udist2"],
                  "cos_centroid": stats["cos_centroid"],
                  "dist2": stats["squared_diff_of_means"],
                  "rel_dist2": stats["rel_squared_diff_of_means"],
                  "cross": stats["cross_mean"], "self_orig": stats["self_orig"],
                  "self_rew": stats["self_rew"]}
    return out


def summarize_poolings(records, k_grid, poolings, metric="udist2",
                       types=TYPES, reference=REFERENCE):
    """Per K, one line per pooling: the reference mean plus, for every other
    rewrite type, its mean, the paired gap ± SE and the pairwise ranking
    accuracy against the reference over the ids where both types scored."""
    others = [t for t in types if t != reference]
    higher = metric in HIGHER_IS_BETTER
    lines = []
    for k in k_grid:
        width = 26 + 42 * len(others)
        lines.append(f"\n{'=' * width}\nSUMMARY COS  K={k}  metric={metric}  "
                     f"(reference: {reference}, {'higher' if higher else 'lower'} = more "
                     f"utility, one line per pooling)")
        head = f"{'pooling':>9} {reference[:9] + ' (ref)':>16}"
        for t in others:
            head += f" | {t[:9]:>9} {'gap ± SE':>15} {'ranking':>12}"
        lines.append(head)
        lines.append("-" * width)

        for p in poolings:
            # rewrite type -> {id: metric} for this (K, pooling)
            by_type = {t: {} for t in types}
            for r in records:
                if r["k"] == k and r["pooling"] == p:
                    by_type[r["rewrite_type"]][r["id"]] = r[metric]

            ref = by_type[reference]
            ref_m, _ = mean_std(list(ref.values()))
            line = f"{p:>9} {ref_m:>16.4f}"
            for t in others:
                m, _ = mean_std(list(by_type[t].values()))
                ids = sorted(set(ref) & set(by_type[t]))
                if not ids:
                    line += f" | {m:>9.4f} {'n/a':>15} {'n/a':>12}"
                    continue
                # "correct" = the reference rewrite stays closer to the
                # original: lower distance, or higher centroid cosine
                correct = sum((ref[i] > by_type[t][i]) if higher else (ref[i] < by_type[t][i])
                              for i in ids)
                # paired: per-id differences cancel the prompt-level spread
                dm, ds = mean_std([by_type[t][i] - ref[i] for i in ids])
                gap = f"{dm:+.4f}±{ds / len(ids) ** 0.5:.4f}"
                acc = f"{correct}/{len(ids)} ({100 * correct / len(ids):.0f}%)"
                line += f" | {m:>9.4f} {gap:>15} {acc:>12}"
            lines.append(line)
    return "\n".join(lines)


def run(args, rewrite_cols=REWRITE_COLS, types=TYPES, reference=REFERENCE):
    """Full sweep over one CSV. rewrite_cols are the scored columns, types
    their short names (same order), reference the type compared against."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    poolings = args.poolings

    with open(args.csv, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))

    n_cells = len(rows) * len(rewrite_cols) * len(args.k_grid)
    print(f"Loaded {len(rows)} rows from {args.csv}")
    print(f"Rewrite types: {', '.join(types)}  (reference: {reference})")
    print(f"K grid: {args.k_grid}  ->  {n_cells} cells")
    print(f"Poolings: {poolings} at layer {args.layer}  ->  "
          f"{n_cells * len(poolings)} output rows")

    print(f"\nLoading {args.model_path} on {device} ...", flush=True)
    model, tok = load(args.model_path, device)

    os.makedirs(args.out_dir, exist_ok=True)
    csv_path = os.path.join(args.out_dir, f"{args.save_suffix}.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=OUT_COLS).writeheader()

    records = []
    start = time.time()
    cell_i = 0
    for row_i, row in enumerate(rows):
        original = row["original"].strip()
        for k in args.k_grid:
            # the original is sampled and embedded once per (row, K) and reused
            # for every rewrite type; same seeds as the layer run
            torch.manual_seed(args.seed + k + 10_000 * row_i)
            t0 = time.time()
            a_orig = sample_answers(original, model, tok, k, args)
            if len(a_orig) < 2:
                cell_i += len(rewrite_cols)
                print(f"id={row['id'][:24]:<24} k={k:<3} SKIP ROW "
                      f"(< 2 non-empty original answers)", flush=True)
                continue
            E_orig = embed_all(a_orig, model, tok, poolings, args)
            orig_seconds = round(time.time() - t0, 2)

            for col_i, col in enumerate(rewrite_cols):
                rewritten = row[col].strip()
                cell_i += 1
                torch.manual_seed(args.seed + k + 100 * (col_i + 1) + 10_000 * row_i)

                t0 = time.time()
                per_pool = score_cell_poolings(E_orig, rewritten, model, tok, k,
                                               poolings, args)
                seconds = round(time.time() - t0, 2)

                head = f"[{cell_i}/{n_cells}] id={row['id'][:24]:<24} {types[col_i]:<20} k={k:<3}"
                if per_pool is None:
                    print(f"{head} SKIP (< 2 non-empty answers)  ({seconds}s)", flush=True)
                    continue

                new = [{"id": row["id"], "rewrite_type": types[col_i], "k": k,
                        "pooling": p, "layer": args.layer, "seconds": seconds,
                        **per_pool[p]}
                       for p in poolings]
                records.extend(new)
                with open(csv_path, "a", newline="", encoding="utf-8") as f:
                    csv.DictWriter(f, fieldnames=OUT_COLS).writerows(new)

                summary = " ".join(f"{p}:udist2={per_pool[p]['udist2']:.3f}/"
                                   f"cos={per_pool[p]['cos_centroid']:.3f}"
                                   for p in poolings)
                print(f"{head} {summary} ({seconds}s + {orig_seconds}s orig)", flush=True)

    for metric in ("udist2", "cos_centroid"):
        print(summarize_poolings(records, args.k_grid, poolings, metric,
                                 types, reference))
    print(f"\nDone in {round(time.time() - start, 1)}s over {len(records)} rows "
          f"({len(records) // max(len(poolings), 1)} cells).")
    print(f"CSV -> {csv_path}")


if __name__ == "__main__":
    run(get_args())
