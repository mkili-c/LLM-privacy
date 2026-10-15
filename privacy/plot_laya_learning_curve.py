"""
Learning curve of the Laya fine-tuning (ROADMAP "Laya fine-tuning"): accuracy, AUROC and pair
accuracy against the number of training pairs, on the synthetic test split and the
out-of-distribution set (prompt_rewrites_100), with laya_eval.py's 95% bootstrap CIs.

Each point is a separate laya_finetune.py run from the base model on the first n training pairs
(nested subsets, jobs 404-407); n = 0 is the base model. The numbers are read from
laya_eval.py's JSON (job 408), nothing is scored again.

    python plot_laya_learning_curve.py   -> results/laya_eval/laya_learning_curve.{pdf,png}
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results" / "laya_eval" / "laya_eval_408.json"

# model name in the JSON -> training pairs it was fine-tuned on
MODELS = {"base": 0, "laya-privacy-n250": 250, "laya-privacy-n500": 500,
          "laya-privacy-n1000": 1000, "laya-privacy-n1930": 1930}
METRICS = [("acc", "Accuracy (P > 0.5)"), ("auroc", "AUROC"), ("pair_acc", "Pair accuracy")]
SETS = [("synthetic", "Synthetic test (400 pairs)"), ("ood", "Out-of-distribution (100 prompts)")]
DIRECT_LABELS = {"synthetic": "synthetic", "ood": "OOD"}
DODGE = 18                 # x-offset in pairs, so the two sets' error bars do not overlap
Y_RANGE = (0.5, 1.0)       # 0.5 = chance for AUROC and pair accuracy
FIGSIZE = (7.2, 2.7)       # inches, about the text width of a report page

# Categorical slots 1-2 of the validated default palette, as in utility/metrics/plotting
PALETTE = ["#2a78d6", "#eb6834"]   # blue, orange
INK = "#0b0b0b"
INK_MUTED = "#52514e"
GRID = "#d8d7d2"
SURFACE = "#ffffff"


def load_curves(path):
    """-> {set: {metric: (x, y, lo, hi)}}, one point per model in MODELS order."""
    result = json.loads(Path(path).read_text())
    curves = {}
    for s, _ in SETS:
        curves[s] = {}
        for m, _ in METRICS:
            x = list(MODELS.values())
            y = [result[s]["metrics"][name][m] for name in MODELS]
            lo = [result[s]["ci"][name][m][0] for name in MODELS]
            hi = [result[s]["ci"][name][m][1] for name in MODELS]
            curves[s][m] = (x, y, lo, hi)
    return curves


def style(ax):
    """Recessive grid and axes (as in plot_k_sweep.py); the data carries the ink."""
    ax.set_axisbelow(True)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_MUTED, labelsize=8)


def plot_curves(curves, out_stem):
    """One panel per metric, one line per set with its CIs as error bars."""
    fig, axes = plt.subplots(1, len(METRICS), figsize=FIGSIZE, sharey=True)
    for ax, (m, title) in zip(axes, METRICS):
        style(ax)
        for i, (s, label) in enumerate(SETS):
            x, y, lo, hi = curves[s][m]
            xd = [v + (2 * i - 1) * DODGE for v in x]
            err = [[v - l for v, l in zip(y, lo)], [h - v for v, h in zip(y, hi)]]
            ax.errorbar(xd, y, yerr=err, color=PALETTE[i], linewidth=1.5, elinewidth=1.0,
                        capsize=0, marker="o", markersize=5.5, markeredgecolor=SURFACE,
                        markeredgewidth=1.0, label=label, zorder=3)
        ax.set_title(title, color=INK, fontsize=9)
        ax.set_xticks(list(MODELS.values()))
        ax.set_xlim(-120, 2050)
        ax.set_ylim(*Y_RANGE)
        ax.set_xlabel("training pairs (0 = base Laya)", color=INK_MUTED, fontsize=8)
    axes[0].set_ylabel("score", color=INK_MUTED, fontsize=8)

    # direct labels in the first panel, in ink, just below each line midway between its
    # last two points, where there are no markers or error bars
    for s, _ in SETS:
        x, y, _, _ = curves[s][METRICS[0][0]]
        axes[0].annotate(DIRECT_LABELS[s], ((x[-2] + x[-1]) / 2, (y[-2] + y[-1]) / 2),
                         xytext=(0, -4), textcoords="offset points", ha="center", va="top",
                         fontsize=8, color=INK_MUTED)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(SETS), frameon=False, fontsize=8,
               labelcolor=INK, bbox_to_anchor=(0.5, 1.04))
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    for ext in ("pdf", "png"):
        fig.savefig(f"{out_stem}.{ext}", dpi=200, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description="Plot the Laya fine-tuning learning curve")
    p.add_argument("--results", default=str(RESULTS), help="laya_eval.py JSON")
    p.add_argument("--out", default=str(RESULTS.parent / "laya_learning_curve"),
                   help="output path without extension; writes .pdf and .png")
    args = p.parse_args()

    curves = load_curves(args.results)
    for s, _ in SETS:
        for m, _ in METRICS:
            x, y, lo, hi = curves[s][m]
            print(f"{s:<9} {m:<8} " + "  ".join(f"{n}: {v:.3f} [{l:.3f}, {h:.3f}]"
                                                  for n, v, l, h in zip(x, y, lo, hi)))
    plot_curves(curves, args.out)
    print(f"saved -> {args.out}.pdf, .png")


if __name__ == "__main__":
    main()
