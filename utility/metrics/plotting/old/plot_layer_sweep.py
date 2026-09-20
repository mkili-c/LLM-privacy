"""
Publication-style figures for a layer sweep CSV (one row per layer).

Shares its style, palette and line specs with `plot_utility_metrics.py`, so the
sweep figures sit next to the per-persona ones without a visual seam. Writes:

    utility  -- util_para, util_priv and util_cohen_d against layer, y in [0, 1]
    mmd2     -- mmd2_para and mmd2_priv against layer

Usage:
    python plot_layer_sweep.py --csv results/layer_sweep/<file>.csv
"""

import argparse
import os

import pandas as pd

from plot_utility_metrics import draw, set_style

# Colours are slots 1, 2 and 3 of the reference categorical palette -- the three
# that clear the CVD and normal-vision separation floors on *all* pairs, not just
# adjacent ones. para keeps blue and priv keeps orange across both figures.
UTILITY_SERIES = [
    ("util_para",    "#2a78d6", "o"),
    ("util_priv",    "#eb6834", "s"),
    ("util_cohen_d", "#1baf7a", "^"),
]

MMD_SERIES = [
    ("mmd2_para", "#2a78d6", "o"),
    ("mmd2_priv", "#eb6834", "s"),
]


def get_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv", default="./results/layer_sweep/qwen_layer_sweep106793_mean.csv")
    p.add_argument("--out_dir", default="./figures")
    p.add_argument("--formats", nargs="+", default=["pdf", "png"])
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument("--width", type=float, default=9.0, help="figure width in inches")
    p.add_argument("--height", type=float, default=4.2, help="figure height in inches")
    p.add_argument("--autoscale", action="store_true",
                   help="fit the utility y axis to the data instead of pinning it to "
                        "[0, 1]; util_para and util_priv both live above 0.99, so on "
                        "the fixed axis they overplot as one flat line")
    return p.parse_args()


def extremum_label(kind):
    """Direct-label each line with its best layer -- the point of a sweep.

    Higher utility is better; lower MMD^2 is better, so the two figures report
    opposite ends.
    """
    def fmt(name, y):
        layer = (y.idxmax() if kind == "max" else y.idxmin())
        return f"{name}  ({kind} {y.loc[layer]:.4f} @ L{layer})"
    return fmt


def main():
    args = get_args()
    os.makedirs(args.out_dir, exist_ok=True)

    df = pd.read_csv(args.csv).set_index("layer").sort_index()
    run = os.path.splitext(os.path.basename(args.csv))[0]
    print(f"{args.csv}: {len(df)} layers ({df.index.min()}-{df.index.max()})")

    panels = [
        ("utility", UTILITY_SERIES, "Utility / effect size",
         "Utility and effect size by layer", "max"),
        ("mmd2", MMD_SERIES, r"MMD$^2$", r"MMD$^2$ by layer", "min"),
    ]

    for kind, series, ylabel, title, best in panels:
        cols = [name for name, _, _ in series if name in df.columns]
        missing = [name for name, _, _ in series if name not in df.columns]
        if missing:
            print(f"  skipping {missing}: not in CSV")
        if not cols:
            continue
        wide = df[cols]

        lo, hi = wide.min().min(), wide.max().max()
        pad = 0.05 * max(hi - lo, 1e-6)
        if kind == "utility" and not args.autoscale:
            ylim, zero_line = (0.0, 1.0), False
        elif kind == "utility":
            ylim, zero_line = (lo - pad, min(1.0, hi + pad)), False
        else:
            # MMD^2 is an unbiased estimator and can dip below zero; keep the
            # baseline visible rather than clipping it away.
            ylim, zero_line = (min(0.0, lo - pad), hi + pad), lo < 0

        suffix = "_autoscaled" if args.autoscale and kind == "utility" else ""
        out = os.path.join(args.out_dir, f"{run}_{kind}_by_layer{suffix}")
        draw(wide, ylabel, title, ylim, out, args, zero_line=zero_line,
             series=series, xlabel="Layer", tick_size=7,
             label_fmt=extremum_label(best))
        print(f"  wrote {out}.{{{','.join(args.formats)}}}")


if __name__ == "__main__":
    set_style()
    main()
