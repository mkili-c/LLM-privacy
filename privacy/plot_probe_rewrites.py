"""
How well the loop's attackers detect the protected attributes on the development privacy
rewrites (utility/data/prompt_rewrites_100.csv), in the style of utility/metrics/plotting.

Each of the 100 personas has two protected attributes and six versions of its prompt, so every
cell (persona, attribute, version) gets one P per attacker, read from the sweeps:
    Qwen 7B          attribute_probe_sweep.py (job 350): P(yes) to the loop's probe question
    Laya             laya_probe_sweep.py (job 325): the base checkpoint, calibrated P(true)
    Laya fine-tuned  laya_probe_sweep.py --laya_path .../laya-privacy: the loop's checkpoint
    max              max(Qwen 7B, Laya fine-tuned) per cell: what the loop compares with tau
What each version does to the protected attributes:
    original, util_preserving, util_paraphrase   keep them: should leak
    priv_targeted                                removes them: should not leak
    priv_preserving                              removes most, keeps what the request needs
                                                 ("type 2 diabetes"), so not a clean "no"
    util_broken                                  off topic, may keep fragments
so the clean test is original vs priv_targeted.

Figures (in --out_dir, each as .pdf and .png):
    violin_<attacker>  one figure per attacker: P by rewrite type, with the loop's tau; the five
                types, their order and colors are those of the utility violins (plot_k_sweep.py),
                so the original, P's reference, appears in the separation figure instead
    separation  every version against the original, per cell: the mean paired drop
                P(original) - P(version) +- SE, and the ranking accuracy, the share of cells with
                P(version) < P(original) (ties count 1/2)
    detection   original (leaks) vs priv_targeted (private): AUROC, accuracy at tau and pair
                accuracy, with 95% CIs from a bootstrap over personas (a persona's cells are drawn
                together, as they share a prompt)
An attacker whose CSV is missing is skipped, together with the max that needs it. The numbers
behind every figure are printed.

    python plot_probe_rewrites.py
"""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                           # noqa: E402
import numpy as np                                                        # noqa: E402

from plot_laya_learning_curve import INK, INK_MUTED, SURFACE, style      # noqa: E402

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"

VERSIONS = ["original", "util_preserving", "util_paraphrase", "priv_preserving",
            "priv_targeted", "util_broken"]
# second line of each version's tick label: what it does to the protected attributes
NOTES = {"original": "keeps", "util_preserving": "keeps", "util_paraphrase": "keeps",
         "priv_preserving": "partly", "priv_targeted": "removes", "util_broken": "off topic"}
LEAKING, PRIVATE = "original", "priv_targeted"       # the clean test
THRESHOLD = 0.5                                       # tau: the loop calls P > tau leaking
COMBINED = ("max (loop)", "Qwen 7B", "Laya fine-tuned")      # the loop's attacker: the max of these two
N_BOOT = 1000                                         # as laya_eval.py
BOOT_SEED = 0

# Categorical slots 1-4 of the validated default palette, in attacker order
# (utility/metrics/plotting/plot_k_sweep.py, plot_laya_learning_curve.py)
COLORS = {"Qwen 7B": "#2a78d6", "Laya": "#eb6834", "Laya fine-tuned": "#1baf7a", "max (loop)": "#eda100"}
# The utility violins' rewrite types with their colors, exactly as plot_k_sweep.py (slots 1-5)
TYPE_COLORS = {"util_preserving": "#2a78d6", "util_paraphrase": "#eb6834", "priv_preserving": "#1baf7a",
               "priv_targeted": "#eda100", "util_broken": "#e87ba4"}
WIDTH = 5.6                # inches: the text width of a report page, as the learning curve
VIOLIN_HEIGHT = 3.4        # inches: one attacker per figure


def get_args():
    p = argparse.ArgumentParser(description="Plot Qwen's and Laya's detection on the dev privacy rewrites")
    p.add_argument("--qwen", default=str(RESULTS / "attribute_probe" / "Qwen2.5-7B-Instruct_attribute_probe_350.csv"))
    p.add_argument("--laya", default=str(RESULTS / "laya_probe" / "laya_probe_325.csv"))
    p.add_argument("--laya_ft", default=str(RESULTS / "laya_probe" / "laya_privacy_probe.csv"))
    p.add_argument("--out_dir", default=str(HERE / "figures" / "probe_rewrites"))
    return p.parse_args()


def read_sweep(path):
    """{(persona id, attribute): {version: P}} from one sweep CSV."""
    cells = {}
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            cells.setdefault((r["id"], r["attribute"]), {})[r["rewrite_type"]] = float(r["p"])
    return cells


def load_attackers(args):
    """{attacker: cells} for the sweeps that exist, plus the loop's max if both of its parts do."""
    attackers = {}
    for name, path in (("Qwen 7B", args.qwen), ("Laya", args.laya), ("Laya fine-tuned", args.laya_ft)):
        if Path(path).exists():
            attackers[name] = read_sweep(path)
        else:
            print(f"{name}: {path} not there, skipped")
    name, a, b = COMBINED
    if a in attackers and b in attackers:
        attackers[name] = {k: {v: max(attackers[a][k][v], attackers[b][k][v]) for v in VERSIONS}
                           for k in attackers[a]}
    return attackers


def auroc(pos, neg):
    """P(a random positive scores above a random negative), ties 1/2."""
    pos, neg = np.asarray(pos)[:, None], np.asarray(neg)[None, :]
    return float(((pos > neg) + 0.5 * (pos == neg)).mean())


def detection(cells, keys):
    """AUROC, accuracy at tau and pair accuracy of original (y = 1) vs priv_targeted (y = 0)."""
    pos = np.array([cells[k][LEAKING] for k in keys])
    neg = np.array([cells[k][PRIVATE] for k in keys])
    return {"auroc": auroc(pos, neg),
            "acc": float(np.concatenate([pos > THRESHOLD, neg <= THRESHOLD]).mean()),
            "pair_acc": float(((pos > neg) + 0.5 * (pos == neg)).mean())}


def detection_ci(cells, rng):
    """Bootstrap 95% CI of each detection metric: N_BOOT resamples of whole personas."""
    by_persona = {}
    for k in cells:
        by_persona.setdefault(k[0], []).append(k)
    personas = sorted(by_persona)
    draws = []
    for _ in range(N_BOOT):
        keys = [k for q in rng.choice(len(personas), len(personas)) for k in by_persona[personas[q]]]
        draws.append(detection(cells, keys))
    return {m: tuple(np.percentile([d[m] for d in draws], [2.5, 97.5])) for m in draws[0]}


def separation(cells, version):
    """Against the original, per cell: mean drop P(original) - P(version), its SE, and the
    ranking accuracy in % (P(version) < P(original), ties 1/2)."""
    drop = np.array([c[LEAKING] - c[version] for c in cells.values()])
    ranked = np.array([(c[version] < c[LEAKING]) + 0.5 * (c[version] == c[LEAKING]) for c in cells.values()])
    return drop.mean(), drop.std(ddof=1) / np.sqrt(len(drop)), 100.0 * ranked.mean()


def print_numbers(attackers, results):
    print(f"\nper version: mean P, share with P > {THRESHOLD}, mean drop vs original, ranking accuracy "
          f"(n = {len(next(iter(attackers.values())))} cells per version)")
    for name, cells in attackers.items():
        print(f"  {name}")
        for v in VERSIONS:
            ps = np.array([c[v] for c in cells.values()])
            line = f"    {v:<16} P {ps.mean():.3f}   flagged {(ps > THRESHOLD).mean():6.1%}"
            if v != LEAKING:
                drop, se, rank = separation(cells, v)
                line += f"   drop {drop:+.3f} +- {se:.3f}   ranked below original {rank:5.1f}%"
            print(line)
    print(f"\n{LEAKING} vs {PRIVATE}: metric [95% CI over personas]")
    for name, (point, ci) in results.items():
        print(f"  {name:<16} " + "   ".join(f"{m} {point[m]:.3f} [{ci[m][0]:.3f}, {ci[m][1]:.3f}]"
                                            for m in point))


def tick_labels(versions):
    return [f"{v}\n({NOTES[v]})" for v in versions]


def save(fig, out_stem):
    for ext in ("pdf", "png"):
        fig.savefig(f"{out_stem}.{ext}", dpi=200, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print(f"saved -> {out_stem}.pdf, .png")


def plot_violin(name, cells, out_stem):
    """One attacker: violin + box of P per rewrite type (median line, mean dot), tau dashed; the
    types, order, colors and tick labels of the utility violins (plot_k_sweep.plot_violin)."""
    types = list(TYPE_COLORS)
    fig, ax = plt.subplots(figsize=(WIDTH, VIOLIN_HEIGHT))
    style(ax)
    data = [[c[t] for c in cells.values()] for t in types]
    parts = ax.violinplot(data, showextrema=False)
    for body, t in zip(parts["bodies"], types):
        body.set_facecolor(TYPE_COLORS[t])
        body.set_edgecolor(TYPE_COLORS[t])
        body.set_alpha(0.55)
    ax.boxplot(data, widths=0.12, showfliers=False, showmeans=True,
               medianprops=dict(color=INK),
               meanprops=dict(marker="o", markerfacecolor=SURFACE, markeredgecolor=INK, markersize=5))
    ax.axhline(THRESHOLD, linestyle="--", color=INK_MUTED, linewidth=1.2, label=f"τ = {THRESHOLD}")
    ax.set_ylim(-0.03, 1.03)
    ax.set_xticks(range(1, len(types) + 1))
    ax.set_xticklabels(types, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("P (attribute inferable)", color=INK, fontsize=10)
    ax.set_title(name, color=INK, fontsize=11)
    # above the axes on the right: inside, it would sit on a violin that reaches P = 1
    ax.legend(loc="lower right", bbox_to_anchor=(1.0, 1.0), fontsize=8, frameon=False, labelcolor=INK)
    fig.tight_layout()
    save(fig, out_stem)


def plot_separation(attackers, out_stem):
    """Paired drop +- SE and ranking accuracy against the original, per version; attackers dodged."""
    versions = VERSIONS[1:]
    names = list(attackers)
    dodge = 0.18
    fig, (ax_drop, ax_rank) = plt.subplots(1, 2, figsize=(WIDTH, 3.0))
    for i, name in enumerate(names):
        stats = [separation(attackers[name], v) for v in versions]
        x = np.arange(len(versions)) + (i - (len(names) - 1) / 2) * dodge
        marker = dict(color=COLORS[name], linestyle="none", marker="o", markersize=4.5,
                      markeredgecolor=SURFACE, markeredgewidth=0.8, label=name, zorder=3)
        ax_drop.errorbar(x, [s[0] for s in stats], yerr=[s[1] for s in stats], elinewidth=1.0, capsize=0, **marker)
        ax_rank.plot(x, [s[2] for s in stats], **marker)
    ax_drop.axhline(0.0, color=INK_MUTED, linewidth=1.0, linestyle="--")
    ax_drop.set_ylabel("P(original) − P(version)", color=INK_MUTED, fontsize=8)
    ax_drop.set_title("Paired drop ± SE", color=INK, fontsize=9)
    ax_rank.axhline(50.0, color=INK_MUTED, linewidth=1.0, linestyle="--")
    ax_rank.annotate("chance", (-0.4, 50.0), xytext=(0, 3), textcoords="offset points",
                     fontsize=7, color=INK_MUTED)
    ax_rank.set_ylim(0, 105)
    ax_rank.set_ylabel("% of cells below the original", color=INK_MUTED, fontsize=8)
    ax_rank.set_title("Ranking accuracy", color=INK, fontsize=9)
    for ax in (ax_drop, ax_rank):
        style(ax)
        ax.set_xticks(range(len(versions)))
        ax.set_xticklabels(tick_labels(versions), rotation=45, ha="right", fontsize=7)
    handles, labels = ax_rank.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(names), frameon=False, fontsize=8,
               labelcolor=INK, bbox_to_anchor=(0.5, 1.06))
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save(fig, out_stem)


def plot_detection(results, out_stem):
    """Original vs priv_targeted: one panel per metric, one dot + 95% CI per attacker."""
    metrics = [("auroc", "AUROC"), ("acc", f"Accuracy (P > {THRESHOLD})"), ("pair_acc", "Pair accuracy")]
    names = list(results)
    fig, axes = plt.subplots(1, len(metrics), figsize=(WIDTH, 2.6), sharey=True)
    for ax, (m, title) in zip(axes, metrics):
        style(ax)
        for i, name in enumerate(names):
            point, ci = results[name]
            ax.errorbar([i], [point[m]], yerr=[[point[m] - ci[m][0]], [ci[m][1] - point[m]]],
                        color=COLORS[name], elinewidth=1.0, capsize=0, marker="o", markersize=5.5,
                        markeredgecolor=SURFACE, markeredgewidth=1.0, zorder=3)
        ax.axhline(0.5, color=INK_MUTED, linewidth=1.0, linestyle="--")
        ax.set_title(title, color=INK, fontsize=9)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=45, ha="right", fontsize=7)
        ax.set_xlim(-0.6, len(names) - 0.4)
        ax.set_ylim(0.4, 1.0)
    axes[0].set_ylabel("score", color=INK_MUTED, fontsize=8)
    axes[0].annotate("chance", (-0.55, 0.5), xytext=(0, 3), textcoords="offset points",
                     fontsize=7, color=INK_MUTED)
    fig.tight_layout()
    save(fig, out_stem)


def main():
    args = get_args()
    attackers = load_attackers(args)
    rng = np.random.default_rng(BOOT_SEED)
    results = {name: (detection(cells, list(cells)), detection_ci(cells, rng)) for name, cells in attackers.items()}
    print_numbers(attackers, results)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, cells in attackers.items():
        # file name from the attacker: "Laya fine-tuned" -> violin_laya_fine_tuned
        slug = "_".join(name.lower().replace("-", " ").replace("(", "").replace(")", "").split())
        plot_violin(name, cells, out_dir / f"violin_{slug}")
    plot_separation(attackers, out_dir / "separation")
    plot_detection(results, out_dir / "detection")


if __name__ == "__main__":
    main()
