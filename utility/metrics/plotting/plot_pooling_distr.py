"""
Violin plots of the cos metrics over the personas for ONE pooling of the pooling
sweep (cos_measurement_pooling_privacy.py / _paraphrases.py), one violin per
rewrite type.

The pooling figures next to this one (plot_pooling_privacy.py,
plot_pooling_paraphrases.py) summarise every pooling as bars, heatmaps and
Cohen's d; they never show the per-persona spread. This script fills that gap:
it is plot_floor_100_centroid.py / plot_paraphrases_centroid.py applied to a
single (pooling, K) slice of the pooling CSV, so the weighted-mean pooling can be
read the same way as the original run's cross / udist2 / cos_centroid violins.

Same plotting logic, colours and reference line as those two scripts, so the
figures are directly comparable to the ones in figures/floor_100_cos_centroid_k15
and figures/paraphrases_cos_centroid_k15. Only the three metrics those figures
are read from are drawn:

    cross         mean pairwise cosine of activations to the original
    udist2        unbiased squared distance between answer centroids
    cos_centroid  cosine between answer centroids

Usage:
    python plotting/plot_pooling_distr.py                       # weighted, K=15, privacy
    python plotting/plot_pooling_distr.py --pooling mean
    python plotting/plot_pooling_distr.py --dataset paraphrases
    python plotting/plot_pooling_distr.py --dataset both --k 30
"""

import argparse
import csv
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
METRICS_DIR = os.path.dirname(HERE)              # utility/metrics
UTILITY_DIR = os.path.dirname(METRICS_DIR)       # utility
RESULTS = os.path.join(UTILITY_DIR, "results", "cos_measurement", "pooling")
FIGURES = os.path.join(UTILITY_DIR, "figures")

POOLINGS = ["mean", "last", "weighted"]

DATASETS = {
    "privacy": {
        "csv": os.path.join(RESULTS, "privacy_cos_pooling_k15and30_tok256_195.csv"),
        "out": os.path.join(FIGURES, "pooling_privacy_k15and30_tok256"),
        "unit": "rewrite type",
        "types": ["util_preserving", "util_paraphrase", "priv_preserving",
                  "priv_targeted", "util_broken"],
        "colors": ["blue", "orange", "purple", "green", "red"],
        "reference": "util_preserving",
        # util_broken spans a wide band, so the all-types figure needs fixed
        # limits to stay comparable across poolings (see YLIM).
        "fixed_ylim": True,
        # util_broken sits so far from the rest that it flattens the other four;
        # every metric is also drawn without it, autoscaled to the rest.
        "drop_for_zoom": ["util_broken"],
    },
    "paraphrases": {
        "csv": os.path.join(RESULTS, "paraphrases_cos_pooling_k15and30_tok256_196.csv"),
        "out": os.path.join(FIGURES, "pooling_paraphrases_k15and30_tok256"),
        "unit": "paraphrase type",
        "types": ["synonyms", "reordered", "both"],
        "colors": ["blue", "orange", "purple"],
        # as in plot_paraphrases_centroid.py: the reference line is the word-order
        # noise floor, not the closest paraphrase
        "reference": "reordered",
        # All three paraphrases preserve meaning, so they sit in a narrow band
        # just under 1 (cos_centroid >= 0.88, udist2 <= 0.24 at weighted/K=15).
        # The privacy limits would flatten them into three lines, and a zero
        # baseline would clip the slightly negative udist2 values that the
        # unbiased estimator legitimately produces, so everything autoscales -
        # same choice plot_paraphrases_centroid.py makes for these metrics.
        "fixed_ylim": False,
        "drop_for_zoom": [],
    },
}

# (column, y label, higher = more utility)
METRICS = [("cross", "Mean Pairwise Cosine Similarity of Activations to Original", True),
           ("udist2", "Unbiased Euclidian Distance of Answer Embedding Means to Original Mean", False),
           ("cos_centroid", "Cosine Between Answer Embedding Centroids (orig vs rew)", True)]

# Similarity metrics live in a narrow band under 1; the lower bound matches
# plot_floor_100_centroid.py so the two figure sets share a y-axis by eye, and is
# low enough for util_broken (min ~0.66 - 0.68 across the three poolings).
YLIM = {"cos_centroid": (0.65, 1.01), "cross": (0.65, 1.01)}


def load(path, k, pooling, columns):
    """-> values[metric][type] = one number per persona, for this (K, pooling)."""
    values = defaultdict(lambda: defaultdict(list))
    seen_k, seen_pool = set(), set()
    with open(path, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            seen_k.add(int(row["k"]))
            seen_pool.add(row["pooling"].strip())
            if int(row["k"]) != k or row["pooling"].strip() != pooling:
                continue
            rtype = row["rewrite_type"].strip()
            for column in columns:
                values[column][rtype].append(float(row[column]))
    if not values:
        raise SystemExit(f"{path}: no rows with k={k} and pooling={pooling} "
                         f"(K grid {sorted(seen_k)}, poolings {sorted(seen_pool)})")
    return values


def plot(values, ds, column, label, higher_is_better, types, colors, fixed_ylim,
         suffix, k, pooling, out_prefix):
    fig, ax = plt.subplots(figsize=(8, 7))
    data = [values[column][t] for t in types]
    parts = ax.violinplot(data, showextrema=False)
    # A narrow box inside each violin: quartiles, median, and the mean as a dot.
    ax.boxplot(data, widths=0.12, showfliers=False, showmeans=True,
               medianprops=dict(color="black"),
               meanprops=dict(marker="o", markerfacecolor="white",
                              markeredgecolor="black", markersize=5))
    # Reference line at the mean of the reference type, in that type's colour.
    reference = ds["reference"]
    ref = values[column][reference]
    ax.axhline(sum(ref) / len(ref), linestyle="--",
               color=ds["colors"][ds["types"].index(reference)], linewidth=1.0,
               label=f"{reference} mean")
    ax.set_xticks(range(1, len(types) + 1))
    if fixed_ylim:
        if column in YLIM:
            ax.set_ylim(*YLIM[column])
        else:
            ax.set_ylim(bottom=0.0)
    # else: autoscale to the remaining types so their spread is visible
    # Legend away from the mass of the data: the good rewrites sit at the top for
    # similarity metrics and at the bottom for distances.
    ax.legend(loc="lower left" if higher_is_better else "upper left", fontsize=9)
    ax.set_xticklabels(types, rotation=45, ha="right", fontsize=10)
    ax.set_ylabel(label)
    for body, color in zip(parts["bodies"], colors):
        body.set_color(color)

    title = f"{label}\nby {ds['unit']} - {pooling} pooling, K={k}"
    if suffix:
        title += (f"\n({', '.join(ds['drop_for_zoom'])} excluded, "
                  f"y-axis scaled to the remaining {len(types)})")
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    out = f"{out_prefix}_{column}{suffix}.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def run(name, ds, csv_path, k, pooling):
    values = load(csv_path, k, pooling, [c for c, _, _ in METRICS])
    types, colors = ds["types"], ds["colors"]
    out_dir = ds["out"]
    os.makedirs(out_dir, exist_ok=True)
    out_prefix = os.path.join(out_dir, f"distr_vio_{pooling}_k{k}")
    print(f"[{name}] {csv_path}\n[{name}] pooling={pooling} K={k}, types {types}")

    # Two variants: every type on fixed limits, then the types that are left once
    # the outlier is dropped, autoscaled to them. Colours stay bound to the type,
    # so dropping one never repaints the others.
    variants = [("", types, colors, ds["fixed_ylim"])]
    if ds["drop_for_zoom"]:
        kept = [t for t in types if t not in ds["drop_for_zoom"]]
        variants.append(("_no_broken", kept,
                         [colors[types.index(t)] for t in kept], False))

    for column, label, higher in METRICS:
        for suffix, vtypes, vcolors, fixed_ylim in variants:
            print("wrote", plot(values, ds, column, label, higher, vtypes,
                                vcolors, fixed_ylim, suffix, k, pooling,
                                out_prefix))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--dataset", choices=sorted(DATASETS) + ["both"],
                   default="privacy")
    p.add_argument("--pooling", choices=POOLINGS, default="weighted")
    p.add_argument("--k", type=int, default=15)
    p.add_argument("--csv", default=None,
                   help="override the CSV for the selected dataset")
    args = p.parse_args()

    names = sorted(DATASETS) if args.dataset == "both" else [args.dataset]
    if args.csv and len(names) > 1:
        p.error("--csv needs a single --dataset")
    for n in names:
        run(n, DATASETS[n], args.csv or DATASETS[n]["csv"], args.k, args.pooling)
