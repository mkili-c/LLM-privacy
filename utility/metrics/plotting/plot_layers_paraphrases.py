"""
Per-layer figures for the paraphrase layer run (cos_measurement_layers.py).
Paraphrase-dataset copy of plot_layers_privacy.py: three paraphrase types,
reordered as the reference (same words, new order, so the closest control),
and no util_broken variant since there is no outlier type.

Input: the per-cell CSV, one row per (id, rewrite_type, k, layer), with at least
udist2, rel_udist2 and cos_centroid.

Three kinds of figure:

  level      one line per rewrite type: mean over personas ± SE, by layer.
             Shows the shape across depth. The absolute scale drifts with the
             layer's anisotropy, so this is NOT the figure to pick a layer from.
  separation two panels: paired Cohen's d of every non-reference type against
             the reference (scale-free, so layers are comparable), and the
             pairwise ranking accuracy (share of personas where the reference
             keeps more utility). This is the figure that picks a layer.
  heatmap    layer x rewrite type of the same Cohen's d and ranking accuracy,
             as a compact summary (no error bars).

"Loss" is oriented so that a positive paired difference always means the
rewrite kept less utility than the reference: other - ref for distances,
ref - other for cos_centroid.

Usage:
    python plotting/plot_layers_paraphrases.py [csv] [out_prefix]
"""

import csv
import os
import sys
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CSV = sys.argv[1] if len(sys.argv) > 1 else (
    "../results/cos_measurement/layers/paraphrases_cos_layers_160.csv")
OUT = sys.argv[2] if len(sys.argv) > 2 else "../figures/layers_paraphrases_k15_tok256/layers"

TYPES = ["synonyms", "reordered", "both"]
COLORS = dict(zip(TYPES, ["blue", "orange", "purple"]))
REFERENCE = "reordered"

# (column, label, higher = more utility)
METRICS = [("cos_centroid", "Cosine Between Answer Embedding Centroids (orig vs rew)", True),
           ("udist2", "Unbiased Euclidian Distance of Answer Embedding Means to Original Mean", False),
           ("rel_udist2", "Relative Unbiased Distance of Centroids (udist2 / sqrt(self_o self_r))", False)]

# suffix, types shown (single variant: no outlier type to exclude)
VARIANTS = [("", TYPES)]


# ---------------------------------------------------------------- load
# values[metric][layer][type][id] -> float
values = {m: defaultdict(lambda: defaultdict(dict)) for m, _, _ in METRICS}
with open(CSV) as f:
    for row in csv.DictReader(f):
        layer = int(row["layer"])
        rtype = row["rewrite_type"].strip()
        for column, _, _ in METRICS:
            values[column][layer][rtype][row["id"]] = float(row[column])

layers = sorted(values[METRICS[0][0]])
os.makedirs(os.path.dirname(OUT), exist_ok=True)


def level_stats(column, rtype):
    """mean and SE over personas, per layer."""
    means, ses = [], []
    for l in layers:
        xs = np.array(list(values[column][l][rtype].values()))
        means.append(xs.mean())
        ses.append(xs.std(ddof=1) / np.sqrt(len(xs)) if len(xs) > 1 else 0.0)
    return np.array(means), np.array(ses)


def separation_stats(column, rtype, higher_is_better):
    """Paired against the reference, per layer: Cohen's d of the per-persona
    loss, its SE-based error (d / sqrt(n) as a rough band), and the ranking
    accuracy (share of personas where the reference kept more utility)."""
    ds, d_errs, ranks = [], [], []
    for l in layers:
        ref, oth = values[column][l][REFERENCE], values[column][l][rtype]
        ids = sorted(set(ref) & set(oth))
        if higher_is_better:
            loss = np.array([ref[i] - oth[i] for i in ids])
        else:
            loss = np.array([oth[i] - ref[i] for i in ids])
        sd = loss.std(ddof=1) if len(loss) > 1 else 0.0
        d = loss.mean() / sd if sd > 0 else 0.0
        ds.append(d)
        d_errs.append(1.0 / np.sqrt(len(loss)) if len(loss) else 0.0)
        ranks.append(100.0 * (loss > 0).mean() if len(loss) else float("nan"))
    return np.array(ds), np.array(d_errs), np.array(ranks)


def style_layer_axis(ax):
    ax.set_xlabel("layer (0 = embeddings, %d = final block)" % layers[-1])
    ax.set_xticks(layers)
    ax.tick_params(axis="x", labelsize=7)
    ax.grid(axis="y", linewidth=0.4, alpha=0.5)


# ---------------------------------------------------------------- 1. level
for suffix, types in VARIANTS:
    for column, label, higher in METRICS:
        fig, ax = plt.subplots(figsize=(12, 7))
        for t in types:
            m, se = level_stats(column, t)
            ax.plot(layers, m, linestyle="-", color=COLORS[t], linewidth=2.0,
                    marker="o", markersize=4, label=t)
            ax.fill_between(layers, m - se, m + se, color=COLORS[t], alpha=0.15, linewidth=0)
        ax.legend(loc="lower left" if higher else "upper left", fontsize=9)
        style_layer_axis(ax)
        ax.set_ylabel(label, fontsize=9)
        # distances autoscale: udist2 sits near zero for reordered and dips
        # slightly negative, which a zero floor would hide
        title = f"{column} by layer and paraphrase type (mean ± SE over personas)"
        ax.set_title(title, fontsize=10)
        fig.tight_layout()
        out = f"{OUT}_level_{column}{suffix}.png"
        fig.savefig(out, dpi=200)
        plt.close(fig)
        print(f"wrote {out}")

# ---------------------------------------------------------------- 2. separation
for suffix, types in VARIANTS:
    others = [t for t in types if t != REFERENCE]
    for column, label, higher in METRICS:
        fig, (ax_d, ax_r) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
        for t in others:
            d, d_err, rank = separation_stats(column, t, higher)
            ax_d.plot(layers, d, linestyle="-", color=COLORS[t], linewidth=2.0,
                      marker="o", markersize=4, label=f"{t} vs {REFERENCE}")
            ax_d.fill_between(layers, d - d_err, d + d_err, color=COLORS[t], alpha=0.15, linewidth=0)
            ax_r.plot(layers, rank, linestyle="-", color=COLORS[t], linewidth=2.0,
                      marker="o", markersize=4, label=f"{t} vs {REFERENCE}")
        ax_d.axhline(0.0, color="black", linewidth=0.8)
        ax_d.set_ylabel("paired Cohen's d of utility loss vs reference\n(> 0: rewrite keeps less utility)", fontsize=9)
        ax_d.legend(loc="upper left", fontsize=9)
        ax_d.grid(axis="y", linewidth=0.4, alpha=0.5)
        ax_r.axhline(50.0, color="black", linewidth=0.8, linestyle="--", label="chance")
        ax_r.set_ylabel("ranking accuracy [% personas where\nreference keeps more utility]", fontsize=9)
        ax_r.set_ylim(0, 105)
        ax_r.legend(loc="lower left", fontsize=9)
        style_layer_axis(ax_r)
        title = f"separation from {REFERENCE} by layer, metric = {column}"
        ax_d.set_title(title, fontsize=10)
        fig.tight_layout()
        out = f"{OUT}_separation_{column}{suffix}.png"
        fig.savefig(out, dpi=200)
        plt.close(fig)
        print(f"wrote {out}")

# ---------------------------------------------------------------- 3. heatmap
for suffix, types in VARIANTS:
    others = [t for t in types if t != REFERENCE]
    for column, label, higher in METRICS:
        D = np.array([separation_stats(column, t, higher)[0] for t in others])   # [types, layers]
        R = np.array([separation_stats(column, t, higher)[2] for t in others])
        fig, (ax_d, ax_r) = plt.subplots(2, 1, figsize=(13, 2.2 + 1.1 * len(others)))
        for ax, M, name, cmap, vmin, vmax in (
                (ax_d, D, "paired Cohen's d of utility loss", "viridis", 0.0, None),
                (ax_r, R, "ranking accuracy [%]", "viridis", 50.0, 100.0)):
            im = ax.imshow(M, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
            ax.set_yticks(range(len(others)))
            ax.set_yticklabels(others, fontsize=8)
            ax.set_xticks(range(len(layers)))
            ax.set_xticklabels(layers, fontsize=7)
            ax.set_title(f"{name} vs {REFERENCE}  (metric = {column})", fontsize=9)
            fig.colorbar(im, ax=ax, fraction=0.02, pad=0.01)
            # annotate the best layer per type
            for i in range(len(others)):
                j = int(np.nanargmax(M[i]))
                ax.text(j, i, f"{M[i, j]:.2f}" if M is D else f"{M[i, j]:.0f}",
                        ha="center", va="center", fontsize=6, color="white",
                        bbox=dict(boxstyle="round,pad=0.15", facecolor="black", alpha=0.5, linewidth=0))
        ax_r.set_xlabel("layer", fontsize=9)
        fig.tight_layout()
        out = f"{OUT}_heatmap_{column}{suffix}.png"
        fig.savefig(out, dpi=200)
        plt.close(fig)
        print(f"wrote {out}")
