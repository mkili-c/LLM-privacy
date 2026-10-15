"""
Plots for the K-sweep runs of the cos/KL utility metrics, over both the privacy
prompt rewrites and the paraphrase controls.

Adapted from plot_floor_100_centroid.py / plot_paraphrases_centroid.py. Those
scripts predate the `k` column and collapse every K into one violin; here K is a
first-class axis, and the metric set is cut down to the three that matter:

    udist2                      unbiased squared distance between answer centroids
    cos_centroid                cosine between answer centroids
    total_divergence_per_token  KL(orig||rew) + KL(rew||orig), plus both directions

The datasets rewrites_loop_kl and paraphrases_loop_kl plot the loop's own measure instead
(utility/metrics/kl_k_sweep.py writes their k_sweep.csv):

    D   KL(x_0 || y) over the whole vocabulary, nats per answer token
    R   D / D(empty message)

Three figures per dataset and metric:

    violin_<metric>.png      the old violin+box, faceted by K on a shared y-axis
    ktrend_<metric>.png      mean +- 95% CI against K, one line per rewrite type
    separation_<metric>.png  paired gap +- SE and pairwise ranking accuracy
                             against the reference type, against K

The separation figure is the one that answers "how large does K need to be":
udist2 is an unbiased estimator, so its mean is flat in K and only the paired SE
and the ranking accuracy improve as K grows. Accuracy is the same statistic
summarize_cos()/summarize_kl() in test_cos_kl.py print, so the numbers here and
in the .out files agree.

Usage:
    python plotting/plot_k_sweep.py                      # every dataset whose CSV exists
    python plotting/plot_k_sweep.py --dataset rewrites_loop_kl
    python plotting/plot_k_sweep.py --csv path.csv --dataset paraphrases
"""

import argparse
import csv
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
METRICS_DIR = os.path.dirname(HERE)              # utility/metrics
UTILITY_DIR = os.path.dirname(METRICS_DIR)       # utility
RESULTS = os.path.join(UTILITY_DIR, "results", "cos_kl_test")
LOOP_RESULTS = os.path.join(os.path.dirname(UTILITY_DIR), "results", "kl_k_sweep")
FIGURES = os.path.join(UTILITY_DIR, "figures")

# Categorical slots 1-5 of the validated default palette, used in slot order.
# The order is the colourblind-safety mechanism, so types are assigned to slots
# in their declared order and the hues are never cycled or reshuffled.
PALETTE = ["#2a78d6",   # blue
           "#eb6834",   # orange
           "#1baf7a",   # aqua
           "#eda100",   # yellow
           "#e87ba4"]   # magenta

INK = "#0b0b0b"
INK_MUTED = "#52514e"
GRID = "#d8d7d2"

DATASETS = {
    "rewrites": {
        "csv": os.path.join(RESULTS, "kl_vs_cos_measurement_256_tok_152.csv"),
        "out": os.path.join(FIGURES, "k_sweep", "rewrites"),
        "title": "privacy prompt rewrites",
        "unit": "rewrite type",
        "types": ["util_preserving", "util_paraphrase", "priv_preserving",
                  "priv_targeted", "util_broken"],
        # The type every other type is scored against, as in test_cos_kl.py.
        "reference": "util_preserving",
        # util_broken sits so far from the rest that it flattens the other four;
        # every violin figure is also drawn without it, autoscaled to the rest.
        "drop_for_zoom": ["util_broken"],
    },
    "paraphrases": {
        "csv": os.path.join(RESULTS, "paraphrases",
                            "paraphrases_kl_vs_cos_measurement_256_tok_153.csv"),
        "out": os.path.join(FIGURES, "k_sweep", "paraphrases"),
        "title": "paraphrase controls",
        "unit": "paraphrase type",
        "types": ["synonyms", "reordered", "both"],
        # test_cos_kl_paraphrases.py scores against synonyms; here the reference
        # is reordered (same words, new order), as in plot_paraphrases.py, so the
        # reference line is the word-order noise floor. All three paraphrases are
        # meaning preserving, so ranking accuracy near 50% is the *good* outcome:
        # it says the metric does not invent a difference where there is none.
        "reference": "reordered",
        "drop_for_zoom": [],
    },
    # The loop's own D and R (kl_k_sweep.py), on the same two datasets.
    "rewrites_loop_kl": {
        "csv": os.path.join(LOOP_RESULTS, "rewrites", "k_sweep.csv"),
        "out": os.path.join(FIGURES, "k_sweep", "rewrites_loop_kl"),
        "title": "privacy prompt rewrites, loop's KL",
        "unit": "rewrite type",
        "types": ["util_preserving", "util_paraphrase", "priv_preserving",
                  "priv_targeted", "util_broken"],
        "reference": "util_preserving",
        "drop_for_zoom": ["util_broken"],
        "loop_kl": True,
    },
    "paraphrases_loop_kl": {
        "csv": os.path.join(LOOP_RESULTS, "paraphrases", "k_sweep.csv"),
        "out": os.path.join(FIGURES, "k_sweep", "paraphrases_loop_kl"),
        "title": "paraphrase controls, loop's KL",
        "unit": "paraphrase type",
        "types": ["synonyms", "reordered", "both"],
        "reference": "reordered",
        "drop_for_zoom": [],
        "loop_kl": True,
    },
}

# (column, y label, higher = more utility kept, keep zero in view)
LOOP_METRICS = [
    ("D",
     "Loop's D = KL(x0 || y)  [nats per answer token]", False, True),
    ("R",
     "R = D / D(empty message)", False, True),
]
METRICS = [
    ("udist2",
     "Unbiased squared distance between answer centroids", False, True),
    ("cos_centroid",
     "Cosine between answer embedding centroids (orig vs rew)", True, False),
    ("total_divergence_per_token",
     "KL(orig || rew) + KL(rew || orig)  [nats per token]", False, True),
    ("kl_orig_to_rew",
     "KL(orig || rew)  [nats per token]", False, True),
    ("kl_rew_to_orig",
     "KL(rew || orig)  [nats per token]", False, True),
]


def load(path, metrics):
    """-> values[metric][k][type][id] = float, and the sorted K grid."""
    values = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    ks = set()
    with open(path, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            k = int(row["k"])
            ks.add(k)
            rtype = row["rewrite_type"].strip()
            for column, _, _, _ in metrics:
                values[column][k][rtype][row["id"]] = float(row[column])
    return values, sorted(ks)


def style(ax):
    """Recessive grid and axes; the data carries the ink."""
    ax.set_axisbelow(True)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_MUTED, labelsize=9)


def zero_in_view(ax, data, higher_is_better):
    """Keep 0 on the axis without clipping: udist2 is unbiased and goes slightly
    negative, so bottom=0 would hide real values."""
    low = min(v for series in data for v in series)
    if higher_is_better:
        ax.set_ylim(top=max(0.0, ax.get_ylim()[1]))
    else:
        ax.set_ylim(bottom=min(0.0, low * 1.05 if low < 0 else 0.0))


def paired(ref, other):
    """Per-id differences over the ids both types scored."""
    ids = sorted(set(ref) & set(other))
    return np.array([other[i] - ref[i] for i in ids]), ids


def zoom_note(ds, suffix):
    if not suffix:
        return ""
    return (f"  ({', '.join(ds['drop_for_zoom'])} excluded, "
            f"y-axis scaled to the rest)")


def plot_violin(values, ds, column, label, higher_is_better, zero_baseline,
                ks, types, colors, suffix, out_dir):
    """The old violin+box, one panel per K on a shared y-axis."""
    fig, axes = plt.subplots(1, len(ks), figsize=(4.0 * len(ks), 6.4),
                             sharey=True)
    axes = np.atleast_1d(axes)
    all_data = []

    for ax, k in zip(axes, ks):
        data = [list(values[column][k][t].values()) for t in types]
        all_data.extend(data)
        parts = ax.violinplot(data, showextrema=False)
        for body, color in zip(parts["bodies"], colors):
            body.set_facecolor(color)
            body.set_edgecolor(color)
            body.set_alpha(0.55)
        # A narrow box inside each violin: quartiles, median, mean as a dot.
        ax.boxplot(data, widths=0.12, showfliers=False, showmeans=True,
                   medianprops=dict(color=INK),
                   meanprops=dict(marker="o", markerfacecolor="white",
                                  markeredgecolor=INK, markersize=5))
        # Reference line at this K's mean for the reference type.
        ref = list(values[column][k][ds["reference"]].values())
        ax.axhline(float(np.mean(ref)), linestyle="--",
                   color=colors[types.index(ds["reference"])], linewidth=1.2,
                   label=f"{ds['reference']} mean")
        ax.set_xticks(range(1, len(types) + 1))
        ax.set_xticklabels(types, rotation=45, ha="right", fontsize=9)
        ax.set_title(f"K = {k}", fontsize=11, color=INK)
        style(ax)

    if zero_baseline:
        zero_in_view(axes[0], all_data, higher_is_better)
    # The x tick labels already name every type, so the only legend entry that
    # carries information is the reference line.
    axes[0].legend(loc="upper left" if not higher_is_better else "lower left",
                   fontsize=8, frameon=False)
    axes[0].set_ylabel(label, fontsize=10, color=INK)

    fig.suptitle(f"{label}\nby {ds['unit']} and K - {ds['title']}"
                 f"{zoom_note(ds, suffix)}", fontsize=11, color=INK)
    fig.tight_layout()
    out = os.path.join(out_dir, f"violin_{column}{suffix}.png")
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def plot_ktrend(values, ds, column, label, higher_is_better, zero_baseline,
                ks, types, colors, suffix, out_dir):
    """Mean +- 95% CI against K, one line per type."""
    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    all_data = []

    for t, color in zip(types, colors):
        means, halfwidths = [], []
        for k in ks:
            v = np.array(list(values[column][k][t].values()))
            all_data.append(v)
            means.append(v.mean())
            halfwidths.append(1.96 * v.std(ddof=1) / np.sqrt(len(v)))
        means, halfwidths = np.array(means), np.array(halfwidths)
        ax.fill_between(ks, means - halfwidths, means + halfwidths,
                        color=color, alpha=0.15, linewidth=0)
        ax.plot(ks, means, color=color, linewidth=2, marker="o", markersize=8,
                markeredgecolor="white", markeredgewidth=1.5, label=t)
        # Direct labels stay legible up to four series; past that the legend
        # carries identity on its own.
        if len(types) <= 4:
            ax.annotate(t, (ks[-1], means[-1]), textcoords="offset points",
                        xytext=(8, 0), va="center", fontsize=9, color=INK_MUTED)

    ax.set_xticks(ks)
    ax.set_xticklabels([str(k) for k in ks])
    ax.set_xlabel("K  (answers sampled per prompt)", fontsize=10, color=INK)
    ax.set_ylabel(label, fontsize=10, color=INK)
    ax.set_xlim(min(ks) - 1, max(ks) + (6 if len(types) <= 4 else 1))
    style(ax)
    if zero_baseline:
        zero_in_view(ax, all_data, higher_is_better)
    # Below the axes: the CI bands fill the panel, so any in-axes legend lands
    # on top of data.
    ax.legend(fontsize=9, frameon=False, ncol=min(len(types), 3),
              loc="upper center", bbox_to_anchor=(0.5, -0.11))
    ax.set_title(f"{label}\nmean +- 95% CI over personas - {ds['title']}"
                 f"{zoom_note(ds, suffix)}", fontsize=11, color=INK)
    fig.tight_layout()
    out = os.path.join(out_dir, f"ktrend_{column}{suffix}.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_separation(values, ds, column, label, higher_is_better,
                    ks, types, colors, suffix, out_dir):
    """Paired gap +- SE and pairwise ranking accuracy against the reference."""
    reference = ds["reference"]
    others = [t for t in types if t != reference]
    other_colors = [colors[types.index(t)] for t in others]

    fig, (ax_gap, ax_acc) = plt.subplots(1, 2, figsize=(11.5, 5.4))

    for t, color in zip(others, other_colors):
        gaps, ses, accs = [], [], []
        for k in ks:
            ref = values[column][k][reference]
            diff, ids = paired(ref, values[column][k][t])
            gaps.append(diff.mean())
            ses.append(diff.std(ddof=1) / np.sqrt(len(diff)))
            # "correct" = the reference keeps more utility: lower distance or
            # divergence, higher centroid cosine. Same rule as summarize_cos().
            correct = sum((ref[i] > values[column][k][t][i]) if higher_is_better
                          else (ref[i] < values[column][k][t][i]) for i in ids)
            accs.append(100.0 * correct / len(ids))

        ax_gap.errorbar(ks, gaps, yerr=ses, color=color, linewidth=2,
                        marker="o", markersize=8, markeredgecolor="white",
                        markeredgewidth=1.5, capsize=4, label=t)
        ax_acc.plot(ks, accs, color=color, linewidth=2, marker="o",
                    markersize=8, markeredgecolor="white", markeredgewidth=1.5,
                    label=t)

    ax_gap.axhline(0.0, color=INK_MUTED, linewidth=1.0, linestyle="--")
    ax_gap.set_ylabel(f"paired gap vs {reference}", fontsize=10, color=INK)
    ax_gap.set_title("Paired gap +- SE\n(per-persona difference)",
                     fontsize=10, color=INK)

    ax_acc.axhline(50.0, color=INK_MUTED, linewidth=1.0, linestyle="--")
    ax_acc.annotate("chance", (ks[0], 50.0), textcoords="offset points",
                    xytext=(0, 4), fontsize=8, color=INK_MUTED)
    ax_acc.set_ylim(0, 105)
    ax_acc.set_ylabel(f"% of personas ranked below {reference}",
                      fontsize=10, color=INK)
    ax_acc.set_title("Pairwise ranking accuracy\n(fraction of personas ordered "
                     "correctly)", fontsize=10, color=INK)

    for ax in (ax_gap, ax_acc):
        ax.set_xticks(ks)
        ax.set_xticklabels([str(k) for k in ks])
        ax.set_xlabel("K  (answers sampled per prompt)", fontsize=10, color=INK)
        style(ax)

    fig.suptitle(f"{label}\nseparation from {reference} against K - "
                 f"{ds['title']}{zoom_note(ds, suffix)}",
                 fontsize=11, color=INK)
    fig.tight_layout()
    # One legend for both panels: they carry the same series.
    handles, labels = ax_acc.get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=9, frameon=False, ncol=len(others),
               loc="upper center", bbox_to_anchor=(0.5, 0.03))
    out = os.path.join(out_dir, f"separation_{column}{suffix}.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out


def run(name, ds, csv_path):
    metrics = LOOP_METRICS if ds.get("loop_kl") else METRICS
    values, ks = load(csv_path, metrics)
    types = ds["types"]
    colors = PALETTE[:len(types)]
    out_dir = ds["out"]
    os.makedirs(out_dir, exist_ok=True)
    print(f"[{name}] {csv_path}\n[{name}] K grid {ks}, types {types}")

    # Two variants: everything, then without the types that flatten the rest.
    # Colours stay bound to the type, so dropping one never repaints the others.
    variants = [("", types, colors)]
    if ds["drop_for_zoom"]:
        kept = [t for t in types if t not in ds["drop_for_zoom"]]
        variants.append(("_zoom", kept, [colors[types.index(t)] for t in kept]))

    for column, label, higher, zero in metrics:
        for suffix, vtypes, vcolors in variants:
            # The zoom variant exists to autoscale to the surviving types, so it
            # drops the zero baseline that would otherwise flatten them again.
            zero_v = zero and not suffix
            print("wrote", plot_violin(values, ds, column, label, higher, zero_v,
                                       ks, vtypes, vcolors, suffix, out_dir))
            print("wrote", plot_ktrend(values, ds, column, label, higher, zero_v,
                                       ks, vtypes, vcolors, suffix, out_dir))
            print("wrote", plot_separation(values, ds, column, label, higher,
                                           ks, vtypes, vcolors, suffix, out_dir))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--dataset", choices=sorted(DATASETS) + ["all"],
                   default="all")
    p.add_argument("--csv", default=None,
                   help="override the CSV for the selected dataset")
    args = p.parse_args()

    names = sorted(DATASETS) if args.dataset == "all" else [args.dataset]
    if args.csv and len(names) > 1:
        p.error("--csv needs a single --dataset")
    for n in names:
        csv_path = args.csv or DATASETS[n]["csv"]
        # the loop-KL CSVs exist only after their GPU run and kl_k_sweep.py
        if not os.path.exists(csv_path):
            print(f"[{n}] skipped, no {csv_path}")
            continue
        run(n, DATASETS[n], csv_path)
