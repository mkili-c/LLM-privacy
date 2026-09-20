"""
Violin plots of the KL and cos metrics over the personas, one violin per rewrite
type, for the privacy rewrites floor_100 run that includes the centroid cosine
(cos_centroid) and the symmetric relative distance (rel_udist2).

Same plotting logic as plot_distr.py; the only differences are the two extra
metrics and per-metric y-limits so the similarity plots keep util_broken visible.

Usage:
    python plotting/plot_floor_100_centroid.py [csv] [out_prefix]
"""

import csv
import os
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CSV = sys.argv[1] if len(sys.argv) > 1 else (
    "../results/cos_kl_test/privacy_floor_100/floor_100_kl_cos_centroid131.csv")
OUT = sys.argv[2] if len(sys.argv) > 2 else "../figures/floor_100_cos_centroid/distr_vio"

TYPES = ["util_preserving", "util_paraphrase", "priv_preserving",
         "priv_targeted", "util_broken"]
COLORS = ["blue", "orange", "purple", "green", "red"]

# (column, y label, higher = more utility)
METRICS = [("cos_centroid", "Cosine Between Answer Embedding Centroids (orig vs rew)", True),
           ("rel_udist2", "Relative Unbiased Distance of Centroids (udist2 / sqrt(self_o self_r))", False),
           ("udist2", "Unbiased Euclidian Distance of Answer Embedding Means to Original Mean", False),
           ("dist2", "Biased Euclidian Distance of Answer Embedding Means to Original Mean", False),
           ("cross", "Mean Pairwise Cosine Similarity of Activations to Original", True),
           ("total_divergence_per_token", " KL(orig || rew) + KL(rew || orig) [nats per token]", False),
           ("kl_orig_to_rew", "KL(orig || rew) [nats per token]", False),
           ("kl_rew_to_orig", "KL(rew || orig) [nats per token]", False)]

# Similarity metrics live in a narrow band under 1; the lower bound is chosen so
# util_broken (min ~0.68 - 0.69 here) stays inside the axes.
YLIM = {"cos_centroid": (0.65, 1.01), "cross": (0.65, 1.01)}

# values[metric][rewrite_type] -> one number per persona
values = defaultdict(lambda: defaultdict(list))
with open(CSV) as f:
    for row in csv.DictReader(f):
        rtype = row["rewrite_type"].strip()
        for column, _, _ in METRICS:
            values[column][rtype].append(float(row[column]))

os.makedirs(os.path.dirname(OUT), exist_ok=True)

# Two variants per metric: all five types with fixed limits, and the four
# non-broken types with the y-axis autoscaled to them. util_broken is so far
# from the rest that it flattens the differences between the other four.
VARIANTS = [
    ("", TYPES, COLORS, True),                                   # suffix, types, colors, fixed ylim
    ("_no_broken", TYPES[:-1], COLORS[:-1], False),
]

# One figure per metric and variant.
for suffix, types, colors, fixed_ylim in VARIANTS:
    for column, label, higher_is_better in METRICS:
        fig, ax = plt.subplots(figsize=(8, 7))
        data = [values[column][t] for t in types]
        parts = ax.violinplot(data, showextrema=False)
        # A narrow box inside each violin: quartiles, median, and the mean as a dot.
        ax.boxplot(data, widths=0.12, showfliers=False, showmeans=True,
                   medianprops=dict(color="black"),
                   meanprops=dict(marker="o", markerfacecolor="white",
                                  markeredgecolor="black", markersize=5))
        # Reference line at the mean of the utility preserving rewrites.
        ref = values[column]["util_preserving"]
        ax.axhline(sum(ref) / len(ref), linestyle="--", color="blue", linewidth=1.0,
                   label="util_preserving mean")
        ax.set_xticks(range(1, len(types) + 1))
        if fixed_ylim:
            if column in YLIM:
                ax.set_ylim(*YLIM[column])
            else:
                ax.set_ylim(bottom=0.0)
        # else: autoscale to the four remaining types so their spread is visible
        # Legend away from the mass of the data: the good rewrites sit at the top
        # for similarity metrics and at the bottom for distances.
        ax.legend(loc="lower left" if higher_is_better else "upper left", fontsize=9)
        ax.set_xticklabels(types, rotation=45, ha="right", fontsize=10)
        ax.set_ylabel(label)
        for body, color in zip(parts["bodies"], colors):
            body.set_color(color)

        title = f"{label} by rewrite type"
        if suffix:
            title += "\n(util_broken excluded, y-axis scaled to the remaining four)"
        ax.set_title(title, fontsize=10)
        fig.tight_layout()
        out = f"{OUT}_{column}{suffix}.png"
        fig.savefig(out, dpi=200)
        plt.close(fig)
        print(f"wrote {out}")
