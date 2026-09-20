"""
Violin plots of the KL and cos metrics over the personas, one violin per
paraphrase type, for the paraphrase run that includes the centroid cosine
(cos_centroid) and the symmetric relative distance (rel_udist2).

Same plotting logic as plot_paraphrases.py / plot_floor_100_centroid.py. The
reference line is the mean of the reordered paraphrases (same words, new order),
as in plot_paraphrases.py. Cosine-based metrics autoscale because the three
types sit in a narrow band; the KL metrics keep a zero baseline.

Usage:
    python plotting/plot_paraphrases_centroid.py [csv] [out_prefix]
"""

import csv
import os
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CSV = sys.argv[1] if len(sys.argv) > 1 else (
    "../results/cos_kl_test/paraphrases/paraphrases_kl_cos_centroid_132.csv")
OUT = sys.argv[2] if len(sys.argv) > 2 else "../figures/paraphrases_cos_centroid/distr_vio"

TYPES = ["synonyms", "reordered", "both"]
COLORS = ["blue", "orange", "purple"]
REFERENCE = "reordered"

# (column, y label, higher = more utility, zero baseline)
METRICS = [("cos_centroid", "Cosine Between Answer Embedding Centroids (orig vs rew)", True, False),
           ("rel_udist2", "Relative Unbiased Distance of Centroids (udist2 / sqrt(self_o self_r))", False, False),
           ("udist2", "Unbiased Euclidian Distance of Answer Embedding Means to Original Mean", False, False),
           ("dist2", "Biased Euclidian Distance of Answer Embedding Means to Original Mean", False, False),
           ("cross", "Mean Pairwise Cosine Similarity of Activations to Original", True, False),
           ("total_divergence_per_token", " KL(orig || rew) + KL(rew || orig) [nats per token]", False, True),
           ("kl_orig_to_rew", "KL(orig || rew) [nats per token]", False, True),
           ("kl_rew_to_orig", "KL(rew || orig) [nats per token]", False, True)]

# values[metric][rewrite_type] -> one number per persona
values = defaultdict(lambda: defaultdict(list))
with open(CSV) as f:
    for row in csv.DictReader(f):
        rtype = row["rewrite_type"].strip()
        for column, _, _, _ in METRICS:
            values[column][rtype].append(float(row[column]))

os.makedirs(os.path.dirname(OUT), exist_ok=True)

# One figure per metric.
for column, label, higher_is_better, zero_baseline in METRICS:
    fig, ax = plt.subplots(figsize=(8, 7))
    data = [values[column][t] for t in TYPES]
    parts = ax.violinplot(data, showextrema=False)
    # A narrow box inside each violin: quartiles, median, and the mean as a dot.
    ax.boxplot(data, widths=0.12, showfliers=False, showmeans=True,
               medianprops=dict(color="black"),
               meanprops=dict(marker="o", markerfacecolor="white",
                              markeredgecolor="black", markersize=5))
    # Reference line at the mean of the reordered paraphrases.
    ref = values[column][REFERENCE]
    ax.axhline(sum(ref) / len(ref), linestyle="--", color="orange", linewidth=1.0,
               label=f"{REFERENCE} mean")
    ax.set_xticks(range(1, len(TYPES) + 1))
    if zero_baseline:
        ax.set_ylim(bottom=0.0)
    # Legend away from the mass of the data: the close paraphrases sit at the
    # top for similarity metrics and at the bottom for distances.
    ax.legend(loc="lower left" if higher_is_better else "upper left", fontsize=9)
    ax.set_xticklabels(TYPES, rotation=45, ha="right", fontsize=10)
    ax.set_ylabel(label)
    for body, color in zip(parts["bodies"], COLORS):
        body.set_color(color)

    ax.set_title(f"{label} by paraphrase type", fontsize=10)
    fig.tight_layout()
    out = f"{OUT}_{column}.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"wrote {out}")
