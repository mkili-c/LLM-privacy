"""
Per-pooling figures for the paraphrase pooling run
(cos_measurement_pooling_paraphrases.py). Pooling-sweep counterpart of
plot_layers_paraphrases.py: same three figure kinds and the same statistics, with
the pooling (mean / last / weighted) on the x-axis instead of the layer, and one
column per K.

Input: the per-cell CSV, one row per (id, rewrite_type, k, pooling), with at
least udist2, rel_udist2 and cos_centroid.

Three kinds of figure (single variant: no outlier type to exclude):

  level      grouped bars, one per rewrite type: mean over prompts ± SE, by
             pooling. The absolute scale differs between poolings (last-token
             states are more anisotropic), so this is NOT the figure to pick a
             pooling from.
  separation two rows: paired Cohen's d of every non-reference type against the
             reference (scale-free, so poolings are comparable), and the
             pairwise ranking accuracy (share of prompts where the reference
             keeps more utility). This is the figure that picks a pooling.
  heatmap    pooling x rewrite type of the same Cohen's d and ranking accuracy,
             as a compact summary (no error bars).

"Loss" is oriented so that a positive paired difference always means the
rewrite kept less utility than the reference: other - ref for distances,
ref - other for cos_centroid.

Usage:
    python plotting/plot_pooling_paraphrases.py [csv] [out_prefix]
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
    "../results/cos_measurement/pooling/paraphrases_cos_pooling_k15and30_tok256_196.csv")
OUT = sys.argv[2] if len(sys.argv) > 2 else "../figures/pooling_paraphrases_k15and30_tok256/pooling"

TYPES = ["synonyms", "reordered", "both"]
COLORS = dict(zip(TYPES, ["blue", "orange", "purple"]))
REFERENCE = "synonyms"

POOLINGS = ["mean", "last", "weighted"]

# (column, label, higher = more utility)
METRICS = [("cos_centroid", "Cosine Between Answer Embedding Centroids (orig vs rew)", True),
           ("udist2", "Unbiased Euclidian Distance of Answer Embedding Means to Original Mean", False),
           ("rel_udist2", "Relative Unbiased Distance of Centroids (udist2 / sqrt(self_o self_r))", False)]

# suffix, types shown (single variant: no outlier type to exclude)
VARIANTS = [("", TYPES)]


# ---------------------------------------------------------------- load
# values[metric][k][pooling][type][id] -> float
values = {m: defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
          for m, _, _ in METRICS}
with open(CSV) as f:
    for row in csv.DictReader(f):
        k = int(row["k"])
        pool = row["pooling"].strip()
        rtype = row["rewrite_type"].strip()
        for column, _, _ in METRICS:
            values[column][k][pool][rtype][row["id"]] = float(row[column])

KS = sorted(values[METRICS[0][0]])
poolings = [p for p in POOLINGS if p in values[METRICS[0][0]][KS[0]]]
os.makedirs(os.path.dirname(OUT), exist_ok=True)


def level_stats(column, k, rtype):
    """mean and SE over prompts, per pooling."""
    means, ses = [], []
    for p in poolings:
        xs = np.array(list(values[column][k][p][rtype].values()))
        means.append(xs.mean())
        ses.append(xs.std(ddof=1) / np.sqrt(len(xs)) if len(xs) > 1 else 0.0)
    return np.array(means), np.array(ses)


def separation_stats(column, k, rtype, higher_is_better):
    """Paired against the reference, per pooling: Cohen's d of the per-persona
    loss, its SE-based error (d / sqrt(n) as a rough band), and the ranking
    accuracy (share of prompts where the reference kept more utility)."""
    ds, d_errs, ranks = [], [], []
    for p in poolings:
        ref, oth = values[column][k][p][REFERENCE], values[column][k][p][rtype]
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


def bars(ax, series, width_total=0.8):
    """Grouped bars over the poolings: series = [(label, color, values, errs)]."""
    x = np.arange(len(poolings))
    w = width_total / len(series)
    for i, (label, color, vals, errs) in enumerate(series):
        ax.bar(x - width_total / 2 + w * (i + 0.5), vals, w, yerr=errs, capsize=2,
               color=color, alpha=0.85, label=label,
               error_kw=dict(linewidth=0.8, ecolor="black"))
    ax.set_xticks(x)
    ax.set_xticklabels(poolings)
    ax.grid(axis="y", linewidth=0.4, alpha=0.5)
    ax.set_axisbelow(True)


# ---------------------------------------------------------------- 1. level
for suffix, types in VARIANTS:
    for column, label, higher in METRICS:
        fig, axes = plt.subplots(1, len(KS), figsize=(6 * len(KS), 6), sharey=True,
                                 squeeze=False)
        stats = {(k, t): level_stats(column, k, t) for k in KS for t in types}
        # The interesting differences are much smaller than the values themselves
        # (cos_centroid sits near 1), so zoom on the data range instead of
        # starting every bar at 0. The bottom is clipped at 0 when the range
        # reaches it, so distance plots keep their true zero.
        lo = min((m - se).min() for m, se in stats.values())
        hi = max((m + se).max() for m, se in stats.values())
        span = (hi - lo) or abs(hi) or 1.0
        bottom, top = max(0.0, lo - 0.25 * span), hi + 0.35 * span
        for ax, k in zip(axes[0], KS):
            bars(ax, [(t, COLORS[t]) + tuple(stats[(k, t)]) for t in types])
            ax.set_xlabel(f"pooling  (K={k})")
            ax.set_ylim(bottom, top)
        axes[0][0].set_ylabel(label, fontsize=9)
        axes[0][0].legend(loc="upper left", fontsize=9)
        title = f"{column} by pooling and rewrite type (mean ± SE over prompts)"
        if bottom > 0:
            title += f"\n[y-axis starts at {bottom:.4g}, bars are cut off at the bottom]"
        fig.suptitle(title, fontsize=10)
        fig.tight_layout()
        out = f"{OUT}_level_{column}{suffix}.png"
        fig.savefig(out, dpi=200)
        plt.close(fig)
        print(f"wrote {out}")

# ---------------------------------------------------------------- 2. separation
for suffix, types in VARIANTS:
    others = [t for t in types if t != REFERENCE]
    for column, label, higher in METRICS:
        fig, axes = plt.subplots(2, len(KS), figsize=(6 * len(KS), 9), sharey="row",
                                 squeeze=False)
        for col, k in enumerate(KS):
            ax_d, ax_r = axes[0][col], axes[1][col]
            stats = {t: separation_stats(column, k, t, higher) for t in others}
            bars(ax_d, [(f"{t} vs {REFERENCE}", COLORS[t], stats[t][0], stats[t][1])
                        for t in others])
            bars(ax_r, [(f"{t} vs {REFERENCE}", COLORS[t], stats[t][2], None)
                        for t in others])
            ax_d.axhline(0.0, color="black", linewidth=0.8)
            ax_d.set_title(f"K={k}", fontsize=9)
            # headroom so the legend does not sit on top of the tallest bar
            ax_d.set_ylim(top=max(s[0].max() + s[1].max() for s in stats.values()) * 1.35)
            chance = ax_r.axhline(50.0, color="black", linewidth=0.8, linestyle="--",
                                  label="chance")
            ax_r.set_ylim(0, 105)
            ax_r.set_xlabel("pooling")
        axes[0][0].set_ylabel("paired Cohen's d of utility loss vs reference\n"
                              "(> 0: rewrite keeps less utility)", fontsize=9)
        axes[0][0].legend(loc="upper left", fontsize=9)
        axes[1][0].set_ylabel("ranking accuracy [% prompts where\nreference keeps more utility]",
                              fontsize=9)
        # the type colours are already in the top legend; only "chance" is new here
        axes[1][0].legend(handles=[chance], loc="lower left", fontsize=9)
        title = f"separation from {REFERENCE} by pooling, metric = {column}"
        fig.suptitle(title, fontsize=10)
        fig.tight_layout()
        out = f"{OUT}_separation_{column}{suffix}.png"
        fig.savefig(out, dpi=200)
        plt.close(fig)
        print(f"wrote {out}")

# ---------------------------------------------------------------- 3. heatmap
for suffix, types in VARIANTS:
    others = [t for t in types if t != REFERENCE]
    for column, label, higher in METRICS:
        fig, axes = plt.subplots(2, len(KS),
                                 figsize=(1.0 + 3.2 * len(others) * len(KS) / 2,
                                          2.0 + 1.1 * len(poolings) * 2),
                                 squeeze=False)
        for col, k in enumerate(KS):
            stats = {t: separation_stats(column, k, t, higher) for t in others}
            D = np.array([stats[t][0] for t in others]).T   # [poolings, types]
            R = np.array([stats[t][2] for t in others]).T
            for ax, M, name, vmin, vmax, fmt in (
                    (axes[0][col], D, "paired Cohen's d of utility loss", 0.0, None, "%.2f"),
                    (axes[1][col], R, "ranking accuracy [%]", 50.0, 100.0, "%.0f")):
                im = ax.imshow(M, aspect="auto", cmap="viridis", vmin=vmin, vmax=vmax)
                ax.set_yticks(range(len(poolings)))
                ax.set_yticklabels(poolings, fontsize=8)
                ax.set_xticks(range(len(others)))
                ax.set_xticklabels(others, fontsize=7, rotation=20, ha="right")
                ax.set_title(f"{name} vs {REFERENCE}  (K={k})", fontsize=9)
                fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
                for i in range(M.shape[0]):
                    for j in range(M.shape[1]):
                        ax.text(j, i, fmt % M[i, j], ha="center", va="center",
                                fontsize=6, color="white",
                                bbox=dict(boxstyle="round,pad=0.15", facecolor="black",
                                          alpha=0.5, linewidth=0))
        title = f"separation from {REFERENCE} by pooling, metric = {column}"
        fig.suptitle(title, fontsize=10)
        fig.tight_layout()
        out = f"{OUT}_heatmap_{column}{suffix}.png"
        fig.savefig(out, dpi=200)
        plt.close(fig)
        print(f"wrote {out}")
