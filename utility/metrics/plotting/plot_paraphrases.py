"""
Violin plots of the KL and cos metrics over the personas, one violin per rewrite type.
For paraphrases

Usage:
    python plotting/plot_para.py [csv] [out.png]
"""

import csv
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CSV = sys.argv[1] if len(sys.argv) > 1 else (
    "../results/cos_kl_test/paraphrases/qwen_prompt_paraphrases_100_kl_cos_measurement107851.csv")
OUT = sys.argv[2] if len(sys.argv) > 2 else "../figures/paraphrases/distr_vio"

TYPES = ["synonyms", "reordered", "both"]
COLORS = ["blue", "orange", "purple", "green", "red"]

METRICS = [("total_divergence_per_token", " KL(orig || rew) + KL(rew || orig) [nats per token]"),
           ("kl_orig_to_rew", "KL(orig || rew) [nats per token]"),
           ("kl_rew_to_orig", "KL(rew || orig) [nats per token]"),
           ("udist2","Unbiased Euclidian Distance of Answer Embedding Means to Original Mean"),
           ("dist2", "Biased Euclidian Distance of Answer Embedding Means to Original Mean"),
           ("cross", "Mean Pairwise Cosine Similarity of Activations to Original")]

# values[metric][rewrite_type] -> one number per persona
values = defaultdict(lambda: defaultdict(list))
with open(CSV) as f:
    for row in csv.DictReader(f):
        rtype = row["rewrite_type"].strip()
        for column, _ in METRICS:
            values[column][rtype].append(float(row[column]))

# One figure per metric.
for column, label in METRICS:
    fig, ax = plt.subplots(figsize=(8, 7))
    data = [values[column][t] for t in TYPES]
    parts = ax.violinplot(data, showextrema=False)
    # A narrow box inside each violin: quartiles, median, and the mean as a dot.
    ax.boxplot(data, widths=0.12, showfliers=False, showmeans=True,
               medianprops=dict(color="black"),
               meanprops=dict(marker="o", markerfacecolor="white",
                              markeredgecolor="black", markersize=5))
    # Reference line at the mean of the utility preserving rewrites.
    ref = values[column]["reordered"]
    ax.axhline(sum(ref) / len(ref), linestyle="--", color="blue", linewidth=1.0,
               label="reordered rew. mean")
    ax.set_xticks(range(1, len(TYPES) + 1))
    if column == "cross": 
        ax.legend(loc="upper right", fontsize=9)
        ax.set_ylim(bottom=0.94)
    else:
        ax.legend(loc="upper left", fontsize=9)
        ax.set_ylim(bottom=0.0)
    ax.set_xticklabels(TYPES, rotation=45, ha="right", fontsize=10)
    ax.set_ylabel(label)
    for body, color in zip(parts["bodies"], COLORS):
        body.set_color(color)

    ax.set_title(f"{label} by rewrite type")
    fig.tight_layout()
    out = f"{OUT}_{column}.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"wrote {out}")
