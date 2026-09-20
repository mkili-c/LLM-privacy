"""
Privacy-rewrites version of cos_measurement_layers.py, run over
data/privacy_prompt_rewrites_floor_100.csv. Same per-cell logic and output
columns; only the CSV layout, rewrite types and default paths differ.

The input CSV has a fixed shape:

    id, protected_attrs, original,
    util_preserving_rewrite, util_paraphrase_rewrite,
    priv_preserving_rewrite, priv_targeted_rewrite, util_broken_rewrite

For every (row, K) we draw K answers from the original ONCE and reuse them for
all five rewrite types. Each answer is embedded at EVERY requested hidden
layer (one forward pass, pooled per layer) and the cosine stats are computed for
each layer separately:

    cross   = mean_{i,j}    cos(e_i, f_j)      e_i: embeddings of answers to the original
    self_o  = mean_{i != i'} cos(e_i, e_i')    f_j: embeddings of answers to the rewrite
    self_r  = mean_{j != j'} cos(f_j, f_j')
    udist2  = self_o + self_r - 2 * cross      (unbiased ||mu_orig - mu_rew||^2)
    cos_centroid = cross / sqrt(self_o * self_r)
                                               cosine between the two cloud
                                               centroids (1 = same direction),
                                               scale-free; higher = more utility
    rel_udist2   = udist2 / sqrt(self_o * self_r)

Outputs:
    <out_dir>/<save_suffix>.csv   one flat row per (id, rewrite_type, k, LAYER)
    stdout                        per-cell progress, then for every K a table
                                  with one line per layer: mean, paired gap ± SE
                                  and ranking accuracy against the reference
                                  type, so you can see which layer separated
                                  what and how.
"""
import os
import csv
import time
import argparse

import torch

from topk_sample import load, topk_answers
from cosine_utility import embed_answers_layers, cosine_stats

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

OUT_COLS = ["id", "rewrite_type", "k", "layer", "udist2", "rel_udist2", "cos_centroid",
            "dist2", "rel_dist2", "cross", "self_orig", "self_rew", "seconds"]

SUMMARY_METRICS = ["udist2", "rel_udist2", "cos_centroid", "dist2", "rel_dist2"]
# metrics where a LARGER value means more utility kept (flips the ranking test)
HIGHER_IS_BETTER = {"cos_centroid"}


def get_args():
    p = argparse.ArgumentParser(description="Layer-wise cosine-similarity utility "
                                            "over the privacy-rewrites CSV")
    p.add_argument("--model_path", default="/work/projects/mkilic/models/Qwen2.5-3B-Instruct")
    p.add_argument("--csv", default="./data/privacy_prompt_rewrites_floor_100.csv")
    p.add_argument("--k_grid", type=int, nargs="+", default=[10],
                   help="sample counts to test (needs >= 2 per cell)")
    p.add_argument("--max_new_tokens", type=int, default=128)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top_p", type=float, default=1.0)
    p.add_argument("--layers", type=int, nargs="+", default=None,
                   help="hidden layers to embed from; default = all "
                        "(0 = embeddings, ..., num_hidden_layers = final)")
    p.add_argument("--seed", type=int, default=23)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--summary_metric", default="udist2", choices=SUMMARY_METRICS,
                   help="which column the stdout summary tables rank on")
    p.add_argument("--out_dir", default="./results/cos_measurement/layers")
    p.add_argument("--save_suffix", default="cos_layers_privacy_floor_100")
    return p.parse_args()


def mean_std(xs):
    if not xs:
        return float("nan"), float("nan")
    m = sum(xs) / len(xs)
    if len(xs) < 2:
        return m, 0.0
    return m, (sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5


def sample_answers(prompt, model, tok, k, args):
    return [a for a, _, _ in topk_answers(
        prompt, model, tok, k=k, mode="sample", max_new_tokens=args.max_new_tokens,
        temperature=args.temperature, top_p=args.top_p) if a.strip()]


def score_cell_layers(E_orig, rewritten, model, tok, k, layers, args):
    """Utility of one rewrite at one K, for every layer, against the already
    embedded original answers. Returns {layer: stats} or None if the rewrite gave
    fewer than two non-empty answers (the self-similarity terms need a pair)."""
    a_rew = sample_answers(rewritten, model, tok, k, args)
    if len(a_rew) < 2:
        return None

    # Same samples for every layer: differences across layers are then purely
    # representational, not sampling noise.
    E_rew = embed_answers_layers(a_rew, model, tok, args.batch_size, layers)

    out = {}
    for l in layers:
        stats, _ = cosine_stats(E_orig[l], E_rew[l])
        out[l] = {"udist2": stats["udist2"], "rel_udist2": stats["rel_udist2"],
                  "cos_centroid": stats["cos_centroid"],
                  "dist2": stats["squared_diff_of_means"],
                  "rel_dist2": stats["rel_squared_diff_of_means"],
                  "cross": stats["cross_mean"], "self_orig": stats["self_orig"],
                  "self_rew": stats["self_rew"]}
    return out


def summarize_layers(records, k_grid, layers, metric="udist2"):
    """Per K, one line per layer: the reference mean plus, for every other
    rewrite type, its mean, the paired gap ± SE and the pairwise ranking
    accuracy against REFERENCE over the ids where both types scored."""
    others = [t for t in TYPES if t != REFERENCE]
    higher = metric in HIGHER_IS_BETTER
    lines = []
    for k in k_grid:
        width = 22 + 42 * len(others)
        lines.append(f"\n{'=' * width}\nSUMMARY COS  K={k}  metric={metric}  "
                     f"(reference: {REFERENCE}, {'higher' if higher else 'lower'} = more "
                     f"utility, one line per layer)")
        head = f"{'layer':>5} {REFERENCE[:9] + ' (ref)':>16}"
        for t in others:
            head += f" | {t[:9]:>9} {'gap ± SE':>15} {'ranking':>12}"
        lines.append(head)
        lines.append("-" * width)

        for l in layers:
            # rewrite type -> {id: metric} for this (K, layer)
            by_type = {t: {} for t in TYPES}
            for r in records:
                if r["k"] == k and r["layer"] == l:
                    by_type[r["rewrite_type"]][r["id"]] = r[metric]

            ref = by_type[REFERENCE]
            ref_m, _ = mean_std(list(ref.values()))
            line = f"{l:>5} {ref_m:>16.4f}"
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
                # Paired: per-id differences cancel the prompt-level spread that
                # dominates the unpaired std, so this SE is the one to read.
                dm, ds = mean_std([by_type[t][i] - ref[i] for i in ids])
                gap = f"{dm:+.4f}±{ds / len(ids) ** 0.5:.4f}"
                acc = f"{correct}/{len(ids)} ({100 * correct / len(ids):.0f}%)"
                line += f" | {m:>9.4f} {gap:>15} {acc:>12}"
            lines.append(line)
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

    # hidden_states has num_hidden_layers + 1 entries (0 = embedding output)
    layers = args.layers if args.layers else list(range(model.config.num_hidden_layers + 1))
    print(f"Layers: {layers}  ->  {n_cells * len(layers)} output rows")

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
            # for every rewrite type
            torch.manual_seed(args.seed + k + 10_000 * row_i)
            t0 = time.time()
            a_orig = sample_answers(original, model, tok, k, args)
            if len(a_orig) < 2:
                cell_i += len(REWRITE_COLS)
                print(f"id={row['id'][:24]:<24} k={k:<3} SKIP ROW "
                      f"(< 2 non-empty original answers)", flush=True)
                continue
            E_orig = embed_answers_layers(a_orig, model, tok, args.batch_size, layers)
            orig_seconds = round(time.time() - t0, 2)

            for col_i, col in enumerate(REWRITE_COLS):
                rewritten = row[col].strip()
                cell_i += 1
                # distinct per (row, column, K); strides leave room for
                # 100 K values, 100 columns and any number of rows
                torch.manual_seed(args.seed + k + 100 * (col_i + 1) + 10_000 * row_i)

                t0 = time.time()
                per_layer = score_cell_layers(E_orig, rewritten, model, tok, k,
                                              layers, args)
                seconds = round(time.time() - t0, 2)

                head = f"[{cell_i}/{n_cells}] id={row['id'][:24]:<24} {TYPES[col_i]:<20} k={k:<3}"
                if per_layer is None:
                    print(f"{head} SKIP (< 2 non-empty answers)  ({seconds}s)", flush=True)
                    continue

                new = [{"id": row["id"], "rewrite_type": TYPES[col_i], "k": k,
                        "layer": l, "seconds": seconds, **per_layer[l]}
                       for l in layers]
                records.extend(new)
                with open(csv_path, "a", newline="", encoding="utf-8") as f:
                    csv.DictWriter(f, fieldnames=OUT_COLS).writerows(new)

                last = layers[-1]
                print(f"{head} layer{last}: udist2={per_layer[last]['udist2']:.3f} "
                      f"cos_centroid={per_layer[last]['cos_centroid']:.3f} "
                      f"dist2={per_layer[last]['dist2']:.3f} "
                      f"({seconds}s + {orig_seconds}s orig)", flush=True)

    print(summarize_layers(records, args.k_grid, layers, args.summary_metric))
    print(f"\nDone in {round(time.time() - start, 1)}s over {len(records)} rows "
          f"({len(records) // max(len(layers), 1)} cells).")
    print(f"CSV -> {csv_path}")
