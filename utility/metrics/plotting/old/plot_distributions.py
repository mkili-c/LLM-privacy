"""
Violin + box plots of every utility metric, over the 100 personas.

The line figures show one persona at a time; these show the whole distribution
per rewrite type. Each group is a violin (kernel density) with a narrow box
plot inside it -- quartiles and median -- plus a hollow marker for the mean, so
skew is visible as the gap between mean and median.

The four RBF bandwidths share one figure, one column per sigma, as do the two
k values, one row each:

    <run>_dist_kl.pdf          KL utility,        k = 5 | 15
    <run>_dist_cos.pdf         cosine utility and MMD^2
    <run>_dist_rbf_utility.pdf RBF utility,       sigma x k grid
    <run>_dist_rbf_mmd2.pdf    RBF MMD^2,         sigma x k grid

Usage:
    python plot_distributions.py --csv results/utility_benchmark/<file>.csv
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import pandas as pd

from plot_utility_metrics import SERIES, given_name, set_style

RBF_SIGMAS = [
    ("median", r"$\sigma$ = median"),
    ("0.1", r"$\sigma$ = 0.1"),
    ("0.2", r"$\sigma$ = 0.2"),
    ("0.4", r"$\sigma$ = 0.4"),
]

INK = "#0b0b0b"
MUTED = "#52514e"


def get_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv", default="./results/utility_benchmark/"
                                    "qwen_privacy_prompt_rewrites_utility_metrics106719.csv")
    p.add_argument("--out_dir", default="./figures")
    p.add_argument("--formats", nargs="+", default=["pdf", "png"])
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument("--autoscale", action="store_true",
                   help="fit the utility y axis to the data instead of pinning it to [0, 1]")
    return p.parse_args()


def groups(df, column, k, series=SERIES):
    """The per-persona values of one metric at one k, one array per rewrite type."""
    sub = df[df["k"] == k]
    out = []
    for name, colour, _ in series:
        vals = sub.loc[sub["rewrite_type"] == name, column].dropna().to_numpy()
        if len(vals):
            out.append((name, colour, vals))
    return out


def violin_box_panel(ax, grouped, ylabel, title, ylim, zero_line=False,
                     show_xticklabels=True, tick_size=7, title_size=None,
                     label_size=None):
    """One panel: a violin per rewrite type with a box plot and mean inside it."""
    positions = list(range(1, len(grouped) + 1))
    data = [vals for _, _, vals in grouped]
    colours = [colour for _, colour, _ in grouped]

    if zero_line:
        ax.axhline(0.0, color=MUTED, linewidth=0.6, zorder=1)

    parts = ax.violinplot(data, positions=positions, widths=0.78,
                          showextrema=False, showmedians=False)
    for body, colour in zip(parts["bodies"], colours):
        body.set_facecolor(colour)
        body.set_edgecolor(colour)
        body.set_alpha(0.32)
        body.set_linewidth(0.9)

    # A narrow box inside each violin. Thin marks, dark ink -- the violin
    # already carries the colour, so the box only has to stay legible on it.
    ax.boxplot(data, positions=positions, widths=0.14, showfliers=False,
               patch_artist=True, zorder=3,
               boxprops=dict(facecolor="white", edgecolor=INK, linewidth=0.7),
               whiskerprops=dict(color=INK, linewidth=0.7),
               capprops=dict(color=INK, linewidth=0.7),
               medianprops=dict(color=INK, linewidth=1.3))

    # Mean as a hollow dot: the gap from the median reads as skew.
    ax.scatter(positions, [v.mean() for v in data], s=11, marker="o",
               facecolor="white", edgecolor=INK, linewidth=0.7, zorder=4)

    ax.set_xlim(0.4, len(grouped) + 0.6)
    ax.set_xticks(positions)
    if show_xticklabels:
        ax.set_xticklabels([name for name, _, _ in grouped], rotation=90,
                           fontsize=tick_size)
    else:
        ax.set_xticklabels([])
    if ylim is not None:
        ax.set_ylim(*ylim)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=label_size or plt.rcParams["axes.labelsize"])
    ax.grid(axis="x", visible=False)
    ax.set_title(title, loc="left", pad=6,
                 fontsize=title_size or plt.rcParams["axes.titlesize"])
    return ax


def figure_legend(fig, series=SERIES, y=0.965, size=8.5):
    """Identity by colour, plus what the box and the hollow dot mean."""
    handles = [Patch(facecolor=c, edgecolor=c, alpha=0.45, label=n)
               for n, c, _ in series]
    handles += [
        Line2D([], [], color=INK, linewidth=1.3, label="median"),
        Line2D([], [], marker="o", linestyle="none", markerfacecolor="white",
               markeredgecolor=INK, markeredgewidth=0.7, markersize=5, label="mean"),
    ]
    fig.legend(handles=handles, ncol=len(handles), loc="upper center",
               bbox_to_anchor=(0.5, y), frameon=False, fontsize=size,
               handlelength=1.6, columnspacing=1.5)


def limits(frames, pinned, autoscale):
    """Shared y limits for panels that must stay comparable."""
    lo = min(v.min() for f in frames for v in f)
    hi = max(v.max() for f in frames for v in f)
    pad = 0.06 * max(hi - lo, 1e-6)
    if pinned and not autoscale:
        return (0.0, 1.0), False
    if pinned:
        return (lo - pad, min(1.0, hi + pad)), False
    return (min(0.0, lo - pad), hi + pad), lo < 0


def save(fig, out_stem, args):
    # Keep the fitted-axis variants beside the pinned ones instead of over them.
    if args.autoscale:
        out_stem += "_autoscaled"
    for fmt in args.formats:
        fig.savefig(f"{out_stem}.{fmt}", dpi=args.dpi)
    plt.close(fig)


def simple_figure(df, ks, columns, args, run, stem, suptitle):
    """One row of panels: (column, label, pinned) x the k values."""
    ncol = len(columns) * len(ks)
    fig, axes = plt.subplots(1, ncol, figsize=(2.55 * ncol + 0.8, 4.5))
    axes = axes.ravel() if ncol > 1 else [axes]

    i = 0
    for column, label, pinned in columns:
        frames = [[v for _, _, v in groups(df, column, k)] for k in ks]
        ylim, zero = limits(frames, pinned, args.autoscale)
        for k in ks:
            violin_box_panel(axes[i], groups(df, column, k),
                             label if i % len(ks) == 0 else "",
                             f"{label} - k = {k}" if len(columns) > 1 else f"k = {k}",
                             ylim, zero_line=zero, title_size=9, label_size=9)
            i += 1

    fig.suptitle(suptitle, x=0.02, ha="left", y=0.995, fontsize=11)
    figure_legend(fig, y=0.945)
    fig.subplots_adjust(left=0.085, right=0.985, top=0.83, bottom=0.24, wspace=0.35)
    save(fig, os.path.join(args.out_dir, f"{run}_dist_{stem}"), args)


def rbf_grid(df, ks, args, run, kind):
    """Every RBF bandwidth in one figure: a column per sigma, a row per k."""
    field = "utility" if kind == "utility" else "mmd2"
    label = "RBF-MMD utility" if kind == "utility" else r"RBF-kernel MMD$^2$"
    pinned = kind == "utility"

    fig, axes = plt.subplots(len(ks), len(RBF_SIGMAS),
                             figsize=(3.15 * len(RBF_SIGMAS) + 0.6, 3.5 * len(ks) + 1.4))

    for col, (bw, siglabel) in enumerate(RBF_SIGMAS):
        column = f"rbf_{bw}_{field}"
        if column not in df.columns:
            continue
        # One scale per sigma, shared down the k rows, so k = 5 and k = 15 are
        # comparable; bandwidths keep their own range.
        frames = [[v for _, _, v in groups(df, column, k)] for k in ks]
        ylim, zero = limits(frames, pinned, args.autoscale)
        for row, k in enumerate(ks):
            ax = axes[row, col]
            violin_box_panel(ax, groups(df, column, k),
                             f"{label}   (k = {k})" if col == 0 else "",
                             siglabel if row == 0 else "", ylim, zero_line=zero,
                             show_xticklabels=(row == len(ks) - 1),
                             tick_size=6.5, title_size=9.5, label_size=8.5)

    fig.suptitle(f"{label} by rewrite type - all bandwidths",
                 x=0.02, ha="left", y=0.995, fontsize=11)
    figure_legend(fig, y=0.955)
    fig.subplots_adjust(left=0.085, right=0.985, top=0.885, bottom=0.135,
                        wspace=0.28, hspace=0.16)
    save(fig, os.path.join(args.out_dir, f"{run}_dist_rbf_{kind}"), args)


def main():
    args = get_args()
    os.makedirs(args.out_dir, exist_ok=True)

    df = pd.read_csv(args.csv)
    df["persona"] = df["id"].map(given_name)
    ks = sorted(df["k"].unique())
    run = os.path.splitext(os.path.basename(args.csv))[0]
    print(f"{args.csv}: {df['persona'].nunique()} personas, k={ks}")

    simple_figure(df, ks, [("kl_utility", "KL utility", True)], args, run,
                  "kl", "KL utility by rewrite type")
    simple_figure(df, ks,
                  [("cos_utility", "Cosine-MMD utility", True),
                   ("mmd2", r"Cosine-kernel MMD$^2$", False)],
                  args, run, "cos", "Cosine kernel by rewrite type")
    rbf_grid(df, ks, args, run, "utility")
    rbf_grid(df, ks, args, run, "mmd2")

    print(f"wrote 4 figures x {len(args.formats)} formats to {args.out_dir}/")


if __name__ == "__main__":
    set_style()
    main()
