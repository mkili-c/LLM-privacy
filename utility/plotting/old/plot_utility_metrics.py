"""
Publication-style figures for the utility benchmark CSV.

Reads a `benchmark_utility.py` results CSV (one row per persona x rewrite_type x k)
and writes one figure per (metric, k):

    utility   -- kl, cos and every rbf sigma, y in [0, 1], one line per rewrite type
    mmd2      -- the raw MMD^2 behind the cosine metric and behind every rbf sigma

The x axis is the persona (its given name, taken from the `id` column); each
figure carries four lines, one per rewrite type.

Usage:
    python plot_utility_metrics.py --csv results/utility_benchmark/<file>.csv
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

# Series order is fixed so a rewrite type keeps its colour and marker across
# every figure. Colours are slots 1, 2, 3 and 7 of the reference categorical
# palette: they clear the colour-vision-deficiency and normal-vision separation
# floors on all pairs, which the default first four do not (orange/yellow).
# Marker shape is the secondary encoding, so identity never rests on colour alone.
SERIES = [
    ("util_preserving", "#2a78d6", "o"),
    ("util_paraphrase", "#eb6834", "s"),
    ("priv_preserving", "#1baf7a", "^"),
    ("priv_targeted",   "#4a3aa7", "D"),
]

# (column, filename stem, axis label) for the utility panels, in report order.
UTILITY_METRICS = [
    ("kl_utility",         "kl",         "KL utility"),
    ("cos_utility",        "cos",        "Cosine-MMD utility"),
    ("rbf_median_utility", "rbf_median", r"RBF-MMD utility ($\sigma$ = median)"),
    ("rbf_0.1_utility",    "rbf_0.1",    r"RBF-MMD utility ($\sigma$ = 0.1)"),
    ("rbf_0.2_utility",    "rbf_0.2",    r"RBF-MMD utility ($\sigma$ = 0.2)"),
    ("rbf_0.4_utility",    "rbf_0.4",    r"RBF-MMD utility ($\sigma$ = 0.4)"),
]

# The bare `mmd2` column is the linear (cosine) kernel; the rest are Gaussian.
MMD_METRICS = [
    ("mmd2",            "cos",        r"Cosine-kernel MMD$^2$"),
    ("rbf_median_mmd2", "rbf_median", r"RBF-kernel MMD$^2$ ($\sigma$ = median)"),
    ("rbf_0.1_mmd2",    "rbf_0.1",    r"RBF-kernel MMD$^2$ ($\sigma$ = 0.1)"),
    ("rbf_0.2_mmd2",    "rbf_0.2",    r"RBF-kernel MMD$^2$ ($\sigma$ = 0.2)"),
    ("rbf_0.4_mmd2",    "rbf_0.4",    r"RBF-kernel MMD$^2$ ($\sigma$ = 0.4)"),
]


def get_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv", default="./results/utility_benchmark/"
                                    "qwen_privacy_prompt_rewrites_utility_metrics106719.csv")
    p.add_argument("--out_dir", default="./figures")
    p.add_argument("--formats", nargs="+", default=["png"])
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument("--width", type=float, default=16.0, help="figure width in inches")
    p.add_argument("--height", type=float, default=4.6, help="figure height in inches")
    p.add_argument("--autoscale", action="store_true",
                   help="fit the utility y axis to the data instead of pinning it "
                        "to [0, 1]; useful for cos, which lives above 0.84")
    p.add_argument("--sort", choices=["file", "mean"], default="file",
                   help="persona order along x: CSV order, or by mean over the "
                        "four rewrite types (per figure, makes the trend readable)")
    return p.parse_args()


def set_style():
    """Recessive grid and axes, thin marks, no chart junk."""
    plt.rcParams.update({
        "figure.dpi": 110,
        "savefig.bbox": "tight",
        "font.size": 9,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "axes.linewidth": 0.6,
        "axes.edgecolor": "#52514e",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": "#d8d7d2",
        "grid.linewidth": 0.5,
        "legend.frameon": False,
        "legend.fontsize": 9,
        "xtick.color": "#52514e",
        "ytick.color": "#52514e",
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "text.color": "#0b0b0b",
        "axes.labelcolor": "#0b0b0b",
    })


def given_name(persona_id):
    """'Marta, 34, female, midwife, Krakow, Poland' -> 'Marta'."""
    return persona_id.split(",")[0].strip()


def pivot(df, column, k):
    """personas (rows) x rewrite type (cols) for one metric at one k."""
    sub = df[df["k"] == k]
    wide = sub.pivot_table(index="persona", columns="rewrite_type",
                           values=column, sort=False)
    return wide.reindex(columns=[name for name, _, _ in SERIES])


def mean_label(name, y):
    """Default direct label: the series mean, so no point needs annotating."""
    return f"{name}  (mean {y.mean():.3f})"


def plot_panel(ax, wide, ylabel, title, ylim, zero_line=False, series=SERIES,
               xlabel="Persona", tick_size=6, label_fmt=mean_label,
               show_xticklabels=True, markersize=3.0, title_size=None,
               legend_size=None):
    """Render one panel into `ax`: the index along x, one line per `series` entry.

    `series` is a list of (column, colour, marker); `wide` is indexed by the x
    category. Used both for the standalone figures and for the report pages,
    where stacked panels share one persona axis and the upper one passes
    show_xticklabels=False.
    """
    x = range(len(wide.index))

    if zero_line:
        ax.axhline(0.0, color="#52514e", linewidth=0.6, zorder=1)

    drawn = 0
    for name, colour, marker in series:
        if name not in wide.columns:
            continue
        y = wide[name]
        ax.plot(x, y, color=colour, marker=marker, markersize=markersize,
                markeredgewidth=0, linewidth=1.2, alpha=0.9,
                label=label_fmt(name, y))
        drawn += 1

    ax.set_xlim(-0.8, len(wide.index) - 0.2)
    ax.set_xticks(list(x))
    if show_xticklabels:
        ax.set_xticklabels(wide.index, rotation=90, fontsize=tick_size)
        ax.set_xlabel(xlabel)
    else:
        ax.set_xticklabels([])
    ax.set_ylim(*ylim)
    ax.set_ylabel(ylabel)
    ax.grid(axis="x", visible=False)
    ax.set_title(title, pad=22, loc="left",
                 fontsize=title_size or plt.rcParams["axes.titlesize"])
    ax.legend(ncol=drawn, loc="lower left", bbox_to_anchor=(0, 1.005),
              handlelength=1.8, columnspacing=1.6, borderaxespad=0,
              fontsize=legend_size or plt.rcParams["legend.fontsize"])
    return ax


def draw(wide, ylabel, title, ylim, out_stem, args, zero_line=False,
         series=SERIES, xlabel="Persona", tick_size=6, label_fmt=mean_label):
    """One standalone figure per metric, written in every requested format."""
    fig, ax = plt.subplots(figsize=(args.width, args.height))
    plot_panel(ax, wide, ylabel, title, ylim, zero_line=zero_line, series=series,
               xlabel=xlabel, tick_size=tick_size, label_fmt=label_fmt)
    for fmt in args.formats:
        fig.savefig(f"{out_stem}.{fmt}", dpi=args.dpi)
    plt.close(fig)


def main():
    args = get_args()
    os.makedirs(args.out_dir, exist_ok=True)

    df = pd.read_csv(args.csv)
    df["persona"] = df["id"].map(given_name)
    ks = sorted(df["k"].unique())
    run = os.path.splitext(os.path.basename(args.csv))[0]
    print(f"{args.csv}: {len(df)} rows, {df['persona'].nunique()} personas, k={ks}")

    written = 0
    for kind, metrics in (("utility", UTILITY_METRICS), ("mmd2", MMD_METRICS)):
        for column, stem, label in metrics:
            if column not in df.columns:
                print(f"  skipping {column}: not in CSV")
                continue
            for k in ks:
                wide = pivot(df, column, k)
                if args.sort == "mean":
                    wide = wide.loc[wide.mean(axis=1).sort_values().index]

                if kind == "utility" and not args.autoscale:
                    ylim, zero_line = (0.0, 1.0), False
                elif kind == "utility":
                    lo, hi = wide.min().min(), wide.max().max()
                    pad = 0.05 * max(hi - lo, 1e-6)
                    ylim, zero_line = (lo - pad, min(1.0, hi + pad)), False
                else:
                    # MMD^2 is an unbiased estimator, so it can dip below zero;
                    # keep the baseline visible instead of clipping it away.
                    lo, hi = wide.min().min(), wide.max().max()
                    pad = 0.05 * max(hi - lo, 1e-6)
                    ylim, zero_line = (min(0.0, lo - pad), hi + pad), lo < 0

                suffix = "_autoscaled" if args.autoscale and kind == "utility" else ""
                out = os.path.join(args.out_dir, f"{run}_{kind}_{stem}_k{k}{suffix}")
                draw(wide, label, f"{label} by persona  (k = {k})", ylim, out, args,
                     zero_line=zero_line)
                written += 1

    print(f"wrote {written} figures x {len(args.formats)} formats to {args.out_dir}/")


if __name__ == "__main__":
    set_style()
    main()
