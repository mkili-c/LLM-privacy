"""
Which leaks each attacker catches, per attribute, on the PUPA-SD originals: of the (prompt,
attribute) pairs whose value the judge recovers from the original (real leaks), the share each
attacker flags as inferable (P > tau), as one bar per attacker and attribute.

Inputs: results/evaluation/judge_outcomes.csv (read_judge_answers.py) for the pairs, and
results/evaluation/own_measures.csv for each attacker's P on the originals.
Output: results/evaluation/attacker_recall.{pdf,png}; the counts are printed.

    python evaluation/plot_attacker_recall.py
"""
import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                           # noqa: E402

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "evaluation"))
sys.path.insert(0, str(REPO / "privacy"))

from read_judge_answers import LOOP_ATTRIBUTE, THRESHOLD                  # noqa: E402
from plot_laya_learning_curve import INK, INK_MUTED, SURFACE, style      # noqa: E402

EVAL_DIR = REPO / "results" / "evaluation"
# judge attribute -> label; gender, then the two most common attributes
ATTRIBUTES = {"gender": "Gender", "location": "Location", "occupation": "Occupation"}
# attacker -> (own_measures.csv prefix, color as in privacy/plot_probe_rewrites.py)
ATTACKERS = {"Qwen 7B": ("qwen", "#2a78d6"), "Laya fine-tuned": ("laya", "#1baf7a")}
FIGSIZE = (4.2, 2.8)       # inches: a small figure, about 3/4 of the text width
BAR = 0.36                 # bar width, in units of the attribute spacing


def read_csv(path):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def recall(outcomes, own):
    """{attribute: {attacker: (flagged, n)}} over the pairs the judge recovers from the original."""
    p = {r["id"]: r for r in own if r["method"] == "original"}
    counts = {}
    for a in ATTRIBUTES:
        ids = [r["id"] for r in outcomes if r["version"] == "original" and r["attribute"] == a
               and r["outcome"] == "match"]
        counts[a] = {name: (sum(float(p[i][f"{prefix} {LOOP_ATTRIBUTE[a]}"]) > THRESHOLD for i in ids), len(ids))
                     for name, (prefix, _) in ATTACKERS.items()}
    return counts


def plot(counts, out_stem):
    """Grouped bars: one group per attribute, one bar per attacker, k/n above each bar."""
    fig, ax = plt.subplots(figsize=FIGSIZE)
    style(ax)
    for j, (name, (_, color)) in enumerate(ATTACKERS.items()):
        x = [i + (j - 0.5) * BAR for i in range(len(ATTRIBUTES))]
        ks = [counts[a][name] for a in ATTRIBUTES]
        ys = [100.0 * k / n for k, n in ks]
        # a surface-colored edge leaves a thin gap between the two bars of a group
        ax.bar(x, ys, width=BAR, color=color, edgecolor=SURFACE, linewidth=1.0, label=name, zorder=3)
        for xi, yi, (k, n) in zip(x, ys, ks):
            ax.annotate(f"{k}/{n}", (xi, yi), xytext=(0, 2), textcoords="offset points",
                        ha="center", va="bottom", fontsize=7, color=INK_MUTED)
    ax.set_xticks(range(len(ATTRIBUTES)))
    ax.set_xticklabels(ATTRIBUTES.values(), fontsize=8, color=INK)
    ax.set_ylim(0, 110)
    ax.set_ylabel(f"leaks flagged (P > {THRESHOLD}), %", color=INK_MUTED, fontsize=8)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=len(ATTACKERS), frameon=False,
              fontsize=8, labelcolor=INK)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(f"{out_stem}.{ext}", dpi=200, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)


def main():
    counts = recall(read_csv(EVAL_DIR / "judge_outcomes.csv"), read_csv(EVAL_DIR / "own_measures.csv"))
    for a, per in counts.items():
        print(f"{a:<11} " + "   ".join(f"{name} {k}/{n}" for name, (k, n) in per.items()))
    plot(counts, EVAL_DIR / "attacker_recall")
    print(f"saved -> {EVAL_DIR / 'attacker_recall'}.pdf, .png")


if __name__ == "__main__":
    main()
