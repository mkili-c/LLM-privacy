"""
K sweep of the loop's utility measure: the tests of test_cos_kl.py (K = 5, 15, 30), with D and R
measured exactly as the loop measures them (utility/metrics/utility_check.py). No model.

    D(y) = (1/K) Σ_k (1/|a_k|) Σ_t Σ_v p_t(v) · [ log p_t(v) − log q_t(v) ]     [nats per answer token]
           p_t = P(· | x_0, a_k,<t),   q_t = P(· | y, a_k,<t),   a_k ~ P(· | x_0),   v: every vocabulary token
    R(y) = D(y) / D(∅)                                                          D(∅): D of the empty message

Input: dev_D.csv of calibration/calibrate_thresholds.py run with --k 30, one row per (prompt, column)
with the D of every answer in kl_per_answer. The answers are independent samples, so D over the
first K of them is D as the loop measures it with K answers. All K therefore share answers, unlike
test_cos_kl.py, which sampled every K afresh: one sampling run instead of three, and the
comparison between K is paired. A prompt with fewer than K non-empty answers is left out at that K.

Prints per K and per measure (D, R) one line per rewrite type, as summarize_kl() in test_cos_kl.py:
    mean ± std        over prompts
    paired gap ± SE   mean over prompts of (type − reference) and its standard error
    ranking           prompts where the reference has the lower value, i.e. keeps more utility
R divides each prompt's values by one number, so its ranking equals D's; only its gap changes.

Writes k_sweep.csv next to the input (id, rewrite_type, k, D, d_empty, R), the format
plotting/plot_k_sweep.py reads (datasets rewrites_loop_kl and paraphrases_loop_kl).

    python utility/metrics/kl_k_sweep.py results/kl_k_sweep/rewrites/dev_D.csv
    python utility/metrics/kl_k_sweep.py results/kl_k_sweep/paraphrases/dev_D.csv --reference paraphrase_reordered
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO / "calibration") not in sys.path:
    sys.path.insert(0, str(REPO / "calibration"))

from analyze_calibration import load        # noqa: E402  {id: {column: D on each answer}}

K_GRID = [5, 15, 30]
ORIGINAL = "original"                        # its D must be 0: the check of the setup
EMPTY = "empty"                              # D(∅), the denominator of R
MEASURES = ["D", "R"]


def get_args():
    p = argparse.ArgumentParser(description="K sweep of the loop's D and R from a dev_D.csv")
    p.add_argument("measurements", help="dev_D.csv of calibrate_thresholds.py (run with --k 30)")
    p.add_argument("--ks", type=int, nargs="+", default=K_GRID)
    p.add_argument("--reference", default="util_preserving_rewrite",
                   help="column every other column is compared against")
    return p.parse_args()


def type_name(column):
    """util_preserving_rewrite -> util_preserving, paraphrase_both -> both (the names of test_cos_kl*.py)."""
    return column.removesuffix("_rewrite").removeprefix("paraphrase_")


def check_original(kls):
    """Largest |D(original)| over prompts, on all answers: the original against itself must give 0."""
    return max(abs(cols[ORIGINAL].mean()) for cols in kls.values() if len(cols[ORIGINAL]))


def measures_at_k(kls, columns, k):
    """D and R of every column with the first k answers -> (records, ids left out for having fewer)."""
    records, left_out = [], []
    for i, cols in kls.items():
        if len(cols[EMPTY]) < k:             # empty answers were dropped when sampling
            left_out.append(i)
            continue
        d_empty = cols[EMPTY][:k].mean()     # D(∅) = (1/k) Σ_k D_k(∅)
        for c in columns:
            D = cols[c][:k].mean()           # D(y) = (1/k) Σ_k D_k(y)
            records.append({"id": i, "rewrite_type": type_name(c), "k": k,
                            "D": D, "d_empty": d_empty, "R": D / d_empty})
    return records, left_out


def summarize(records, types, reference, k, measure):
    """The summary table of one K and one measure, as lines."""
    by_type = {t: {r["id"]: r[measure] for r in records if r["rewrite_type"] == t} for t in types}
    ref = by_type[reference]
    lines = [f"\n{'=' * 78}\nSUMMARY {measure}  K={k}    (reference: {reference}, lower = more utility)",
             f"{'rewrite type':<24} {'mean ± std':>16} {'paired gap ± SE':>18} {'ranking':>16}",
             "-" * 78]
    for t in types:
        v = np.array(list(by_type[t].values()))                     # (prompts,)
        gap = acc = "-"
        if t != reference:
            ids = sorted(set(ref) & set(by_type[t]))
            diff = np.array([by_type[t][i] - ref[i] for i in ids])   # (prompts,)  type − reference
            correct = int(sum(ref[i] < by_type[t][i] for i in ids))
            gap = f"{diff.mean():+.4f}±{diff.std(ddof=1) / np.sqrt(len(ids)):.4f}"
            acc = f"{correct}/{len(ids)} ({100 * correct / len(ids):.0f}%)"
        lines.append(f"{t:<24} {v.mean():>7.3f} ± {v.std(ddof=1):<6.3f} {gap:>18} {acc:>16}")
    return lines


def write_csv(records, path):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["id", "rewrite_type", "k", "D", "d_empty", "R"])
        w.writeheader()
        for r in records:
            w.writerow({**r, **{m: f"{r[m]:.6f}" for m in ["D", "d_empty", "R"]}})


def main():
    args = get_args()
    kls = load(args.measurements)
    columns = [c for c in next(iter(kls.values())) if c not in (ORIGINAL, EMPTY)]
    types, reference = [type_name(c) for c in columns], type_name(args.reference)
    n_answers = min(len(cols[EMPTY]) for cols in kls.values())
    print(f"{len(kls)} prompts, at least {n_answers} answers each; columns {types}")
    print(f"check: largest |D(original)| = {check_original(kls):.2e} (must be 0)")

    records = []
    for k in args.ks:
        at_k, left_out = measures_at_k(kls, columns, k)
        if left_out:
            print(f"\nK={k}: {len(left_out)} prompts left out, fewer than {k} answers: {left_out}")
        for measure in MEASURES:
            print("\n".join(summarize(at_k, types, reference, k, measure)))
        records += at_k

    out_path = Path(args.measurements).parent / "k_sweep.csv"
    write_csv(records, out_path)
    print(f"\nCSV -> {out_path}")


if __name__ == "__main__":
    main()
