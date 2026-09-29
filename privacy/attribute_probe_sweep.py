"""
Run the attribute probe over utility/data/prompt_rewrites_100.csv, for every
row and message column, on that row's own protected attributes only.

The input CSV has a fixed shape:

    id, protected_attrs, original,
    util_preserving_rewrite, util_paraphrase_rewrite,
    priv_preserving_rewrite, priv_targeted_rewrite, util_broken_rewrite

protected_attrs is a "; "-separated list ("profession; age"). For every
(row, column, attribute) cell we ask attribute_probe.py's question

    "Could someone reading this message make an informed guess about the
     user's {attribute}?"

and record the yes-vs-no logit, p = sigmoid(logit) and the coverage of the
Yes/No forms. One forward pass per cell, no generation.

The logit is the score to compare: for attributes the model is trained not to
guess (gender) it stays biased towards No, so p can sit near 0 for leaking and
private messages alike while the logit still ranks them (see attribute_probe.py).
The summary therefore compares every column to the original per (row,
attribute) pair on the logit, and reports p > --threshold only as the
absolute count privacy_rewrite.py stops on.

Outputs:
    <out_dir>/<save_suffix>.csv   one flat row per cell
    stdout                        per-row progress, then per column: mean logit
                                  and p, share above the threshold, mean logit
                                  drop vs the original and how often the
                                  column leaks less than the original; then the
                                  mean logit per attribute x column
"""
import argparse
import csv
import os
import time
from pathlib import Path

import torch

from attribute_probe import DEFAULT_MODEL, QUESTION, AttributeProbe, load

_REPO = Path(__file__).resolve().parents[1]

# Fixed CSV layout: the original is the attack surface and the reference every
# rewrite is compared against; the rewrites follow in file order.
MESSAGE_COLS = [
    "original",
    "util_preserving_rewrite",
    "util_paraphrase_rewrite",
    "priv_preserving_rewrite",
    "priv_targeted_rewrite",
    "util_broken_rewrite",
]
TYPES = [c.removesuffix("_rewrite") for c in MESSAGE_COLS]
REFERENCE = TYPES[0]

OUT_COLS = ["id", "rewrite_type", "attribute", "logit", "p", "coverage"]


def get_args():
    p = argparse.ArgumentParser(description="Attribute probe over the prompt-rewrites CSV, "
                                            "per row on its protected attributes")
    p.add_argument("--model_path", default=DEFAULT_MODEL)
    p.add_argument("--csv", default=str(_REPO / "utility" / "data" / "prompt_rewrites_100.csv"))
    p.add_argument("--columns", nargs="+", default=MESSAGE_COLS, choices=MESSAGE_COLS,
                   help="message columns to score; include 'original' for the comparison")
    p.add_argument("--question", default=QUESTION, help="template with {attribute}")
    p.add_argument("--threshold", type=float, default=0.5,
                   help="p above which an attribute counts as leaking (as in privacy_rewrite.py)")
    p.add_argument("--out_dir", default=str(Path(__file__).resolve().parent / "results" / "attribute_probe"))
    p.add_argument("--save_suffix", default="attribute_probe")
    return p.parse_args()


def protected_attrs(row):
    return [a.strip() for a in row["protected_attrs"].split(";") if a.strip()]


def mean_std(xs):
    if not xs:
        return float("nan"), float("nan")
    m = sum(xs) / len(xs)
    if len(xs) < 2:
        return m, 0.0
    return m, (sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5


def summarize(records, types, threshold):
    """Per column over all (row, attribute) cells, then mean logit per attribute."""
    cell = {(r["id"], r["attribute"], r["rewrite_type"]): r for r in records}
    ref = {(i, a): r for (i, a, t), r in cell.items() if t == REFERENCE}

    lines = [f"\n{'=' * 96}\nSUMMARY  (reference: {REFERENCE}; lower logit = less leakage)",
             f"{'column':<20} {'n':>4} {'logit':>16} {'p':>7} {'p>' + str(threshold):>7} "
             f"{'drop vs ref':>16} {'leaks less':>16}",
             "-" * 96]
    for t in types:
        rs = [r for r in records if r["rewrite_type"] == t]
        m, s = mean_std([r["logit"] for r in rs])
        mp, _ = mean_std([r["p"] for r in rs])
        above = sum(r["p"] > threshold for r in rs) / len(rs) if rs else float("nan")
        if t == REFERENCE or not ref:
            drop, less = "-", "-"
        else:
            pairs = [(ref[(r["id"], r["attribute"])]["logit"], r["logit"])
                     for r in rs if (r["id"], r["attribute"]) in ref]
            dm, ds = mean_std([o - x for o, x in pairs])
            n_less = sum(x < o for o, x in pairs)
            drop = f"{dm:+.3f} ± {ds:.3f}"
            less = f"{n_less}/{len(pairs)} ({100 * n_less / len(pairs):.0f}%)" if pairs else "n/a"
        lines.append(f"{t:<20} {len(rs):>4} {m:>+8.3f} ± {s:<5.3f} {mp:>7.3f} {above:>7.2f} "
                     f"{drop:>16} {less:>16}")

    # attribute x column table of mean logits
    attrs = sorted({r["attribute"] for r in records},
                   key=lambda a: -sum(r["attribute"] == a for r in records))
    short = [t[:12] for t in types]
    lines.append(f"\nmean logit per attribute (n = rows with that protected attribute)")
    lines.append(f"{'attribute':<22} {'n':>4}" + "".join(f" {s:>12}" for s in short))
    lines.append("-" * (27 + 13 * len(types)))
    for a in attrs:
        n = len({r["id"] for r in records if r["attribute"] == a})
        cols = []
        for t in types:
            m, _ = mean_std([r["logit"] for r in records if r["attribute"] == a and r["rewrite_type"] == t])
            cols.append(f" {m:>+12.3f}")
        lines.append(f"{a:<22} {n:>4}" + "".join(cols))
    return "\n".join(lines)


if __name__ == "__main__":
    args = get_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cols = [c for c in MESSAGE_COLS if c in args.columns]     # keep the file order
    types = [TYPES[MESSAGE_COLS.index(c)] for c in cols]

    with open(args.csv, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))

    all_attrs = sorted({a for row in rows for a in protected_attrs(row)})
    n_cells = sum(len(protected_attrs(row)) for row in rows) * len(cols)
    print(f"Loaded {len(rows)} rows from {args.csv}")
    print(f"Columns: {', '.join(types)}")
    print(f"{len(all_attrs)} protected attributes: {', '.join(all_attrs)}  ->  {n_cells} cells")
    print(f"Question: {args.question}")

    print(f"\nLoading {args.model_path} on {device} ...", flush=True)
    model, tok = load(args.model_path, device)
    model.requires_grad_(False)
    # One probe per attribute; the prefix / suffix ids are built once and reused.
    probes = {a: AttributeProbe(model, tok, args.question.format(attribute=a)) for a in all_attrs}

    records = []
    start = time.time()
    for row_i, row in enumerate(rows):
        t0 = time.time()
        attrs = protected_attrs(row)
        parts = []
        for col, t in zip(cols, types):
            text = row[col].strip()
            for a in attrs:
                r = probes[a].score(text)
                records.append({"id": row["id"], "rewrite_type": t, "attribute": a,
                                "logit": r["logit"], "p": r["p_pos"], "coverage": r["coverage"]})
            parts.append(f"{t[:12]} " + "/".join(f"{rec['logit']:+.1f}" for rec in records[-len(attrs):]))
        print(f"[{row_i + 1}/{len(rows)}] id={row['id'][:24]:<24} ({'; '.join(attrs)})  "
              f"{'  '.join(parts)}  ({time.time() - t0:.1f}s)", flush=True)

    os.makedirs(args.out_dir, exist_ok=True)
    csv_path = os.path.join(args.out_dir, f"{args.save_suffix}.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=OUT_COLS)
        w.writeheader()
        w.writerows(records)

    low_cov = [r for r in records if r["coverage"] < 0.5]
    if low_cov:
        print(f"\nwarning: {len(low_cov)} cells with Yes/No coverage < 0.5 "
              f"(the model did not answer in the expected format)")
    print(summarize(records, types, args.threshold))
    print(f"\nDone in {round(time.time() - start, 1)}s over {len(records)} cells.")
    print(f"CSV -> {csv_path}")
