"""
B2: turn B1's measurements (results/calibration/dev_D.csv) into the utility-broken rule. No model.

Per dev row, every rewrite's D is put on the scale of its own prompt:

    R(y) = D(y) / D(∅)          D(∅) = D of the empty message: how much the original shapes its answers

Prints, for raw D, R and D / D(util_paraphrase):
    1. the distribution of each column (median, middle 50%, min - max);
    2. how well each separates util_broken from the preserving rewrites (all 4 preserving columns,
       and the 2 privacy columns alone): AUROC with a bootstrap 95% CI over rows, and the accuracy
       at the accuracy-optimal threshold with the share of preserving rewrites left alone and of
       broken ones caught;
    3. the paired bootstrap of the AUROC difference R - raw D;
    4. the within-row pair accuracy: util_broken above each preserving column of its own row
       (the same for D and R, since R divides a row by one constant);
    5. the rule "broken iff R >= p" at round p, and a bootstrap 95% CI of the optimal p;
    6. the rewrites the 100% rule gets wrong, with their text;
    7. K: the 100% decisions with only the first K answers against all 10.

Bootstrap: --boot resamples of the dev rows with replacement; a row's columns are drawn together.

    python calibration/analyze_calibration.py
"""
import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
PRESERVING = ["util_preserving_rewrite", "util_paraphrase_rewrite",
              "priv_preserving_rewrite", "priv_targeted_rewrite"]
PRIVACY = ["priv_preserving_rewrite", "priv_targeted_rewrite"]
BROKEN = "util_broken_rewrite"
ROUND_P = [0.6, 0.7, 0.8, 0.9, 1.0, 1.1]   # thresholds of R shown in section 5
RULE_P = 1.0                               # R >= 100%: as far as an empty message
K_SUBSETS = [3, 5, 8]


def get_args():
    p = argparse.ArgumentParser(description="B2: the utility-broken rule from B1's dev_D.csv")
    p.add_argument("--measurements", default=str(REPO / "results" / "calibration" / "dev_D.csv"))
    p.add_argument("--data", default=str(REPO / "utility" / "data" / "prompt_rewrites_100.csv"))
    p.add_argument("--boot", type=int, default=10000, help="bootstrap resamples")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def load(path):
    """-> {id: {column: [D on each answer]}} (numpy arrays; D is their mean)."""
    kls = defaultdict(dict)
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            kls[r["id"]][r["column"]] = np.array([float(x) for x in r["kl_per_answer"].split()])
    return kls


def criterion(kls, name, k=None):
    """{id: {column: value}} of raw D ("D"), R = D / D(∅) ("R") or D / D(paraphrase) ("Dpara"),
    with D the mean over the first k answers (all if k is None)."""
    out = {}
    for i, cols in kls.items():
        d = {c: v[:k].mean() for c, v in cols.items()}
        norm = {"D": 1.0, "R": d["empty"], "Dpara": d["util_paraphrase_rewrite"]}[name]
        out[i] = {c: v / norm for c, v in d.items()}
    return out


def values(crit, ids, cols):
    return np.array([crit[i][c] for i in ids for c in cols])


def auroc(neg, pos):
    """P(pos > neg) + P(tie) / 2 over all (neg, pos) pairs."""
    return (pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean()


def best_threshold(neg, pos):
    """The cut t that maximizes the accuracy of 'broken iff value > t' -> (t, accuracy).
    Sorted once; a cut after the j-th smallest value gets right the negatives up to j and the
    positives after it."""
    v = np.concatenate([neg, pos])
    is_pos = np.concatenate([np.zeros(len(neg)), np.ones(len(pos))])
    order = np.argsort(v)
    v, is_pos = v[order], is_pos[order]
    correct = np.concatenate([[is_pos.sum()], np.cumsum(1 - is_pos) + is_pos.sum() - np.cumsum(is_pos)])
    j = int(np.argmax(correct))
    t = v[0] - 1e-9 if j == 0 else (v[j - 1] + v[j]) / 2 if j < len(v) else v[-1] + 1e-9
    return t, correct[j] / len(v)


def rates(neg, pos, t):
    """-> (accuracy, share of neg left alone, share of pos caught) of 'broken iff value >= t'."""
    acc = ((neg < t).sum() + (pos >= t).sum()) / (len(neg) + len(pos))
    return acc, (neg < t).mean(), (pos >= t).mean()


def short(c):
    return c.removesuffix("_rewrite")


# ---------------------------------------------------------------- report sections

def distributions(crits, ids):
    for name, crit in crits.items():
        scale, unit = (100, "%") if name == "R" else (1, "")
        print(f"\n--- {name}")
        for c in PRESERVING + [BROKEN] + (["empty"] if name == "D" else []):
            v = values(crit, ids, [c]) * scale
            print(f"  {short(c):<16} median {np.median(v):7.3f}{unit}   middle 50% {np.percentile(v, 25):7.3f}"
                  f" - {np.percentile(v, 75):7.3f}   min - max {v.min():7.3f} - {v.max():7.3f}")


def separation(crits, ids, rng, boot):
    print("\n=== 2. util_broken vs the preserving rewrites (every rewrite against every broken one)")
    for cols, label in [(PRESERVING, "all 4 preserving"), (PRIVACY, "2 privacy rewrites")]:
        print(f"\n  vs {label}:")
        for name, crit in crits.items():
            used = [c for c in cols if not (name == "Dpara" and c == "util_paraphrase_rewrite")]
            neg, pos = values(crit, ids, used), values(crit, ids, [BROKEN])
            draws = [auroc(values(crit, s, used), values(crit, s, [BROKEN]))
                     for s in (rng.choice(ids, len(ids)) for _ in range(boot))]
            t, _ = best_threshold(neg, pos)
            acc, alone, caught = rates(neg, pos, t)
            shown = f"{t:.1%}" if name == "R" else f"{t:.3f}"
            print(f"    {name:<6} AUROC {auroc(neg, pos):.3f} [{np.percentile(draws, 2.5):.3f}, "
                  f"{np.percentile(draws, 97.5):.3f}]   best threshold {shown}: accuracy {acc:.3f}, "
                  f"leaves {alone:.0%} alone, catches {caught:.0%}")


def auroc_difference(crits, ids, rng, boot):
    print("\n=== 3. AUROC difference R - raw D (paired bootstrap over rows)")
    for cols, label in [(PRESERVING, "all 4 preserving"), (PRIVACY, "2 privacy rewrites")]:
        def diff(s):
            return (auroc(values(crits["R"], s, cols), values(crits["R"], s, [BROKEN]))
                    - auroc(values(crits["D"], s, cols), values(crits["D"], s, [BROKEN])))
        draws = [diff(rng.choice(ids, len(ids))) for _ in range(boot)]
        print(f"  {label:<20} {diff(ids):+.3f} [{np.percentile(draws, 2.5):+.3f}, {np.percentile(draws, 97.5):+.3f}]")


def pair_accuracy(crit, ids):
    pairs = np.array([[crit[i][BROKEN] > crit[i][c] for c in PRESERVING] for i in ids])   # [rows, 4]
    print(f"\n=== 4. within-row pair accuracy: {pairs.mean():.3f} ({pairs.sum()}/{pairs.size})")
    for j, c in enumerate(PRESERVING):
        print(f"  vs {short(c):<16} {pairs[:, j].mean():.3f}")


def rule(crit, ids, rng, boot):
    print("\n=== 5. the rule 'broken iff R >= p'")
    for p in ROUND_P:
        for cols, label in [(PRESERVING, "all 4"), (PRIVACY, "privacy")]:
            acc, alone, caught = rates(values(crit, ids, cols), values(crit, ids, [BROKEN]), p)
            print(f"  p {p:4.0%}  vs {label:<8} accuracy {acc:.3f}   leaves {alone:.0%} alone   catches {caught:.0%}")
    for cols, label in [(PRESERVING, "all 4"), (PRIVACY, "privacy")]:
        draws = [best_threshold(values(crit, s, cols), values(crit, s, [BROKEN]))[0]
                 for s in (rng.choice(ids, len(ids)) for _ in range(boot))]
        print(f"  optimal p vs {label:<8} {best_threshold(values(crit, ids, cols), values(crit, ids, [BROKEN]))[0]:.1%}"
              f"   bootstrap 95% CI {np.percentile(draws, 2.5):.1%} - {np.percentile(draws, 97.5):.1%}")


def errors(crit, ids, texts):
    print(f"\n=== 6. what the {RULE_P:.0%} rule gets wrong")
    print("  preserving rewrites at or above it:")
    for i in ids:
        for c in PRESERVING:
            if crit[i][c] >= RULE_P:
                print(f"    {i[:40]:<40} {short(c):<16} R {crit[i][c]:4.0%}  {texts[i][c]!r}")
    print("  util_broken below it:")
    for i in ids:
        if crit[i][BROKEN] < RULE_P:
            print(f"    {i[:40]:<40} R {crit[i][BROKEN]:4.0%}  {texts[i][BROKEN]!r}")


def k_subsets(kls, ids):
    print(f"\n=== 7. K: {RULE_P:.0%} decisions with the first K answers against all of them")
    full = criterion(kls, "R")
    cols = PRESERVING + [BROKEN]
    for k in K_SUBSETS:
        part = criterion(kls, "R", k)
        flips = sum((part[i][c] >= RULE_P) != (full[i][c] >= RULE_P) for i in ids for c in cols)
        acc, _, _ = rates(values(part, ids, PRESERVING), values(part, ids, [BROKEN]), RULE_P)
        rel = np.array([abs(part[i][c] - full[i][c]) / full[i][c] for i in ids for c in cols])
        print(f"  K = {k}: {flips} of {len(ids) * len(cols)} decisions flip, accuracy {acc:.3f}, "
              f"|R_K - R| / R median {np.median(rel):.1%}, 90th percentile {np.percentile(rel, 90):.1%}")


def main():
    args = get_args()
    rng = np.random.default_rng(args.seed)
    kls = load(args.measurements)
    ids = sorted(kls)
    with open(args.data, encoding="utf-8") as f:
        texts = {r["id"]: r for r in csv.DictReader(f)}
    print(f"{len(ids)} dev rows, {len(next(iter(kls.values()))['empty'])} answers each, "
          f"{args.boot} bootstrap resamples")
    crits = {name: criterion(kls, name) for name in ["D", "R", "Dpara"]}
    distributions(crits, ids)
    separation(crits, ids, rng, args.boot)
    auroc_difference(crits, ids, rng, args.boot)
    pair_accuracy(crits["D"], ids)
    rule(crits["R"], ids, rng, args.boot)
    errors(crits["R"], ids, texts)
    k_subsets(kls, ids)


if __name__ == "__main__":
    main()
