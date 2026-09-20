"""
Collect every utility figure into one print-ready multi-page PDF.

Pages are grouped by metric family -- KL, cosine, RBF (one sigma at a time) and
the layer sweep -- and hold at most two panels each. Stacked panels share one
persona axis, so the 100 tick labels are drawn once per page instead of twice
and each panel keeps its full height.

Usage:
    python build_report.py                       # A4 landscape
    python build_report.py --page a3             # roomier, if you can print A3
"""

import argparse
import os
from datetime import date

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd

from plot_utility_metrics import (SERIES, given_name, mean_label, pivot,
                                  plot_panel, set_style)
from plot_layer_sweep import MMD_SERIES, UTILITY_SERIES, extremum_label
from plot_distributions import (RBF_SIGMAS, figure_legend, groups, limits,
                                violin_box_panel)

PAGES = {                    # width x height in inches, landscape
    "a4": (11.69, 8.27),
    "a3": (16.54, 11.69),
    "letter": (11.0, 8.5),
}

# (section, utility column, utility label, mmd2 column or None, mmd2 label)
FAMILIES = [
    ("KL divergence", "kl_utility", "KL utility", None, None),
    ("Cosine kernel", "cos_utility", "Cosine-MMD utility",
     "mmd2", r"Cosine-kernel MMD$^2$"),
    ("RBF kernel", "rbf_median_utility", r"RBF-MMD utility ($\sigma$ = median)",
     "rbf_median_mmd2", r"RBF-kernel MMD$^2$ ($\sigma$ = median)"),
    ("RBF kernel", "rbf_0.1_utility", r"RBF-MMD utility ($\sigma$ = 0.1)",
     "rbf_0.1_mmd2", r"RBF-kernel MMD$^2$ ($\sigma$ = 0.1)"),
    ("RBF kernel", "rbf_0.2_utility", r"RBF-MMD utility ($\sigma$ = 0.2)",
     "rbf_0.2_mmd2", r"RBF-kernel MMD$^2$ ($\sigma$ = 0.2)"),
    ("RBF kernel", "rbf_0.4_utility", r"RBF-MMD utility ($\sigma$ = 0.4)",
     "rbf_0.4_mmd2", r"RBF-kernel MMD$^2$ ($\sigma$ = 0.4)"),
]


def get_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv", default="./results/utility_benchmark/"
                                    "qwen_privacy_prompt_rewrites_utility_metrics106719.csv")
    p.add_argument("--sweep_csv",
                   default="./results/layer_sweep/qwen_layer_sweep106793_mean.csv")
    p.add_argument("--out", default=None,
                   help="output PDF; defaults to ./figures/utility_report<tags>.pdf, "
                        "tagged with the options in play so variants do not collide")
    p.add_argument("--page", choices=sorted(PAGES), default="a4")
    p.add_argument("--tick_size", type=float, default=5.0,
                   help="persona tick label size; 100 labels have to fit the page width")
    p.add_argument("--autoscale", action="store_true",
                   help="fit the utility y axis to the data instead of pinning it to "
                        "[0, 1]; cos in particular is flat against the fixed axis")
    p.add_argument("--sort", choices=["file", "mean"], default="file",
                   help="persona order along x: CSV order, or ascending by mean over "
                        "the rewrite types. Does not apply to the layer sweep, where "
                        "the x axis is the layer index and reordering it is meaningless")
    args = p.parse_args()
    if args.out is None:
        tags = ("_autoscaled" if args.autoscale else "") + \
               ("_mean_ordered" if args.sort == "mean" else "")
        args.out = f"./figures/utility_report{tags}.pdf"
    return args


def save_page(pdf, fig):
    """Write one page at its full size.

    Callers must be inside `fixed_page_size()`: the shared style sets
    savefig.bbox='tight', which is right for standalone figures but would crop
    each page here to its own content, leaving every page a different size.
    """
    pdf.savefig(fig)


def fixed_page_size():
    """Disable the tight bounding box so every page comes out the paper size.

    Passing bbox_inches=None to savefig does not work -- matplotlib reads None
    as 'unspecified' and falls back to the rcParam -- so the rcParam itself has
    to be cleared for the duration of the build.
    """
    return plt.rc_context({"savefig.bbox": None})


def page_figure(pagesize, section, subtitle):
    """A page with a running header, and the plotting area below it."""
    fig = plt.figure(figsize=pagesize)
    fig.text(0.045, 0.972, section, fontsize=8.5, color="#52514e",
             va="top", ha="left")
    fig.text(0.955, 0.972, subtitle, fontsize=8.5, color="#52514e",
             va="top", ha="right")
    fig.lines.append(plt.Line2D([0.045, 0.955], [0.958, 0.958],
                                transform=fig.transFigure,
                                color="#d8d7d2", linewidth=0.6))
    return fig


def two_panel_axes(fig):
    """Two stacked panels sharing the x axis, sized for readable tick labels."""
    return fig.subplots(2, 1, sharex=True,
                        gridspec_kw=dict(left=0.075, right=0.985, top=0.895,
                                         bottom=0.175, hspace=0.42))


def cover(pdf, pagesize, args, n_personas, ks, n_layers):
    fig = plt.figure(figsize=pagesize)
    fig.text(0.08, 0.72, "Privacy-Utility Benchmark", fontsize=26, va="top")
    fig.text(0.08, 0.63, "Per-persona utility metrics and layer sweep",
             fontsize=13, color="#52514e", va="top")
    lines = [
        f"Personas: {n_personas}      Rewrite types: "
        f"{', '.join(name for name, _, _ in SERIES)}",
        f"Samples per prompt: k = {', '.join(str(k) for k in ks)}",
        f"Layer sweep: {n_layers} layers",
        "",
        "Utility y axis: " + ("fitted to the data"
                              if args.autoscale else "pinned to [0, 1]"),
        "Persona order: " + ("ascending by mean over the rewrite types, "
                             "recomputed per metric"
                             if args.sort == "mean" else "CSV order, shared by every page"),
        "",
        f"Benchmark CSV: {os.path.basename(args.csv)}",
        f"Layer sweep CSV: {os.path.basename(args.sweep_csv)}",
        f"Generated {date.today().isoformat()}",
    ]
    fig.text(0.08, 0.47, "\n".join(lines), fontsize=9.5, color="#52514e",
             va="top", linespacing=1.9, family="monospace")
    save_page(pdf, fig)
    plt.close(fig)


def distribution_pages(pdf, df, pagesize, ks, args):
    """Violin + box distributions over the personas, before the per-persona detail.

    --sort does not reach here: a distribution has no persona axis to order.
    Every RBF bandwidth shares one page, a column per sigma and a row per k.
    """
    tag = "  -  fitted axis" if args.autoscale else ""

    def grid(section, subtitle, nrow, ncol, bottom=0.20, hspace=0.18):
        fig = page_figure(pagesize, section, subtitle + tag)
        axes = fig.subplots(nrow, ncol, squeeze=False,
                            gridspec_kw=dict(left=0.075, right=0.985, top=0.845,
                                             bottom=bottom, wspace=0.34,
                                             hspace=hspace))
        figure_legend(fig, y=0.925, size=7.5)
        return fig, axes

    # KL and cosine: one row, a panel per (metric, k).
    for section, cols, subtitle in (
            ("KL divergence", [("kl_utility", "KL utility", True)],
             "KL utility by rewrite type"),
            ("Cosine kernel", [("cos_utility", "Cosine-MMD utility", True),
                               ("mmd2", r"Cosine-kernel MMD$^2$", False)],
             "Cosine kernel by rewrite type")):
        panels = [(c, lab, pin, k) for c, lab, pin in cols for k in ks
                  if c in df.columns]
        if not panels:
            continue
        fig, axes = grid(section, subtitle, 1, len(panels))
        for i, (column, label, pinned, k) in enumerate(panels):
            frames = [[v for _, _, v in groups(df, column, kk)] for kk in ks]
            ylim, zero = limits(frames, pinned, args.autoscale)
            violin_box_panel(axes[0, i], groups(df, column, k),
                             label if i % len(ks) == 0 else "",
                             f"{label} - k = {k}" if len(cols) > 1 else f"k = {k}",
                             ylim, zero_line=zero, tick_size=6.5,
                             title_size=8.5, label_size=8.5)
        save_page(pdf, fig)
        plt.close(fig)

    # RBF: every bandwidth on one page, sigma across, k down.
    for kind, field, label, pinned in (
            ("utility", "utility", "RBF-MMD utility", True),
            ("mmd2", "mmd2", r"RBF-kernel MMD$^2$", False)):
        fig, axes = grid("RBF kernel", f"{label} by rewrite type - all bandwidths",
                         len(ks), len(RBF_SIGMAS), bottom=0.165, hspace=0.14)
        for col, (bw, siglabel) in enumerate(RBF_SIGMAS):
            column = f"rbf_{bw}_{field}"
            if column not in df.columns:
                continue
            # One scale per bandwidth, shared down the k rows.
            frames = [[v for _, _, v in groups(df, column, k)] for k in ks]
            ylim, zero = limits(frames, pinned, args.autoscale)
            for row, k in enumerate(ks):
                violin_box_panel(axes[row, col], groups(df, column, k),
                                 f"{label}  (k = {k})" if col == 0 else "",
                                 siglabel if row == 0 else "", ylim,
                                 zero_line=zero,
                                 show_xticklabels=(row == len(ks) - 1),
                                 tick_size=6, title_size=8.5, label_size=7.5)
        save_page(pdf, fig)
        plt.close(fig)


def header_tag(args):
    """Say on the page which axis and ordering it is using -- both change how it reads."""
    bits = (["fitted axis"] if args.autoscale else []) + \
           (["mean-ordered"] if args.sort == "mean" else [])
    return "  -  " + ", ".join(bits) if bits else ""


def persona_xlabel(args):
    return "Persona (ascending by mean)" if args.sort == "mean" else "Persona"


def persona_pages(pdf, df, pagesize, ks, args):
    """One page per (metric, kind), holding the k = 5 and k = 15 panels."""
    for section, ucol, ulabel, mcol, mlabel in FAMILIES:
        for kind, column, ylabel in (("utility", ucol, ulabel),
                                     ("mmd2", mcol, mlabel)):
            if column is None or column not in df.columns:
                continue

            wides = [pivot(df, column, k) for k in ks]

            if args.sort == "mean":
                # ONE order for both panels. They share the x axis, so the single
                # row of tick labels at the bottom has to describe the upper panel
                # too -- sorting each panel by its own mean would mislabel it.
                # The key is the mean over every rewrite type at every k.
                order = pd.concat(wides, axis=1).mean(axis=1).sort_values().index
                wides = [w.loc[order] for w in wides]

            # Both panels always share one y scale, so k = 5 and k = 15 can be
            # compared by reading down the page.
            lo = min(w.min().min() for w in wides)
            hi = max(w.max().max() for w in wides)
            pad = 0.05 * max(hi - lo, 1e-6)
            if kind == "utility" and not args.autoscale:
                ylims, zeros = [(0.0, 1.0)] * len(ks), [False] * len(ks)
            elif kind == "utility":
                ylims = [(lo - pad, min(1.0, hi + pad))] * len(ks)
                zeros = [False] * len(ks)
            else:
                # MMD^2 is unbiased and can dip below zero; keep the baseline.
                ylims = [(min(0.0, lo - pad), hi + pad)] * len(ks)
                zeros = [lo < 0] * len(ks)

            fig = page_figure(pagesize, section,
                              f"{ylabel} by persona{header_tag(args)}")
            axes = two_panel_axes(fig)
            for ax, k, wide, ylim, zero in zip(axes, ks, wides, ylims, zeros):
                plot_panel(ax, wide, ylabel, f"k = {k}", ylim, zero_line=zero,
                           xlabel=persona_xlabel(args), tick_size=args.tick_size,
                           label_fmt=mean_label,
                           show_xticklabels=(ax is axes[-1]),
                           markersize=2.4, title_size=9.5, legend_size=7.5)
            save_page(pdf, fig)
            plt.close(fig)


def sweep_pages(pdf, sweep, pagesize, args):
    """Utility and MMD^2 against layer; both panels share the layer axis.

    --sort deliberately does not reach here: x is the layer index, an ordinal
    axis whose order carries the meaning, so reordering it would destroy the
    figure. Only the y scaling follows --autoscale.
    """
    section = "Layer sweep"

    # With --autoscale, just the fitted page; otherwise both, since the pinned
    # one is what shows that utility is essentially untouched at every layer.
    for autoscaled in ((True,) if args.autoscale else (False, True)):
        ucols = [c for c, _, _ in UTILITY_SERIES if c in sweep.columns]
        mcols = [c for c, _, _ in MMD_SERIES if c in sweep.columns]
        if not ucols or not mcols:
            continue
        uwide, mwide = sweep[ucols], sweep[mcols]

        if autoscaled:
            lo, hi = uwide.min().min(), uwide.max().max()
            pad = 0.05 * max(hi - lo, 1e-6)
            ulim = (lo - pad, min(1.0, hi + pad))
            note = "utility axis fitted to the data"
        else:
            ulim = (0.0, 1.0)
            note = "utility axis pinned to [0, 1]"

        lo, hi = mwide.min().min(), mwide.max().max()
        pad = 0.05 * max(hi - lo, 1e-6)
        mlim = (min(0.0, lo - pad), hi + pad)

        fig = page_figure(pagesize, section, f"By layer - {note}")
        axes = two_panel_axes(fig)
        plot_panel(axes[0], uwide, "Utility / effect size",
                   "Utility and effect size", ulim, series=UTILITY_SERIES,
                   xlabel="Layer", tick_size=6.5, label_fmt=extremum_label("max"),
                   show_xticklabels=False, markersize=2.8, title_size=9.5,
                   legend_size=7.5)
        plot_panel(axes[1], mwide, r"MMD$^2$", r"MMD$^2$", mlim,
                   zero_line=lo < 0, series=MMD_SERIES, xlabel="Layer",
                   tick_size=6.5, label_fmt=extremum_label("min"),
                   markersize=2.8, title_size=9.5, legend_size=7.5)
        save_page(pdf, fig)
        plt.close(fig)


def main():
    args = get_args()
    pagesize = PAGES[args.page]
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    df = pd.read_csv(args.csv)
    df["persona"] = df["id"].map(given_name)
    ks = sorted(df["k"].unique())
    sweep = pd.read_csv(args.sweep_csv).set_index("layer").sort_index()

    with fixed_page_size(), PdfPages(args.out) as pdf:
        cover(pdf, pagesize, args, df["persona"].nunique(), ks, len(sweep))
        distribution_pages(pdf, df, pagesize, ks, args)
        persona_pages(pdf, df, pagesize, ks, args)
        sweep_pages(pdf, sweep, pagesize, args)

        meta = pdf.infodict()
        meta["Title"] = "Privacy-Utility Benchmark - utility metrics and layer sweep"
        meta["Subject"] = f"{args.csv}; {args.sweep_csv}"

    print(f"wrote {args.out} ({args.page} landscape, "
          f"{pagesize[0]:.2f} x {pagesize[1]:.2f} in)")


if __name__ == "__main__":
    set_style()
    main()
