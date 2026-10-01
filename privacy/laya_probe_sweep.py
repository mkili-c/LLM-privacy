"""
laya_probe.py over utility/data/prompt_rewrites_100.csv: the Laya counterpart
of attribute_probe_sweep.py, same cells, same output layout.

For every (row, message column, attribute) cell, with the attributes taken
from that row's protected_attrs only, Laya answers the noul question

    "Can the user's {attribute} be inferred from this message?"

and we record logit = (z_true - z_false) / T and p = sigmoid(logit), with T
the checkpoint's fitted noul temperature, so p is Laya's calibrated
probability rather than a relative score as for Qwen. There is no coverage
column: Laya reads a logit off each [MASK] and cannot answer off format.

Cells are grouped by attribute: one single-attribute LayaProbe per attribute,
and every message that has it as a protected attribute goes through that
probe in one batched call.

Outputs:
    <out_dir>/<save_suffix>.csv   one flat row per cell
                                  (id, rewrite_type, attribute, logit, p)
    stdout                        per-attribute progress, then the summary of
                                  attribute_probe_sweep.py: per column mean
                                  logit and p, share above --threshold, mean
                                  logit drop vs the original and how often the
                                  column leaks less; then the mean logit per
                                  attribute x column
"""
import argparse
import csv
import os
import time
from pathlib import Path

import laya
import torch

from attribute_probe_sweep import MESSAGE_COLS, TYPES, protected_attrs, summarize
from laya_probe import DEFAULT_LAYA, QUESTION, LayaProbe

_REPO = Path(__file__).resolve().parents[1]

OUT_COLS = ["id", "rewrite_type", "attribute", "logit", "p"]


def get_args():
    p = argparse.ArgumentParser(description="Laya probe over the prompt-rewrites CSV, "
                                            "per row on its protected attributes")
    p.add_argument("--laya_path", default=DEFAULT_LAYA)
    p.add_argument("--csv", default=str(_REPO / "utility" / "data" / "prompt_rewrites_100.csv"))
    p.add_argument("--columns", nargs="+", default=MESSAGE_COLS, choices=MESSAGE_COLS,
                   help="message columns to score; include 'original' for the comparison")
    p.add_argument("--question", default=QUESTION, help="template with {attribute}")
    p.add_argument("--threshold", type=float, default=0.5,
                   help="p above which an attribute counts as leaking (as in privacy_rewrite.py)")
    p.add_argument("--batch_size", type=int, default=64, help="messages per Laya forward pass")
    p.add_argument("--out_dir", default=str(Path(__file__).resolve().parent / "results" / "laya_probe"))
    p.add_argument("--save_suffix", default="laya_probe")
    return p.parse_args()


if __name__ == "__main__":
    args = get_args()
    cols = [c for c in MESSAGE_COLS if c in args.columns]     # keep the file order
    types = [TYPES[MESSAGE_COLS.index(c)] for c in cols]

    with open(args.csv, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))

    # attribute -> the rows that protect it
    by_attr = {}
    for row in rows:
        for a in protected_attrs(row):
            by_attr.setdefault(a, []).append(row)
    by_attr = dict(sorted(by_attr.items(), key=lambda kv: -len(kv[1])))
    n_cells = sum(len(rs) for rs in by_attr.values()) * len(cols)
    print(f"Loaded {len(rows)} rows from {args.csv}")
    print(f"Columns: {', '.join(types)}")
    print(f"{len(by_attr)} protected attributes: {', '.join(by_attr)}  ->  {n_cells} cells")
    print(f"Question: {args.question}")

    print(f"\nLoading Laya from {args.laya_path} ...", flush=True)
    agent = laya.load(args.laya_path)
    agent.model.requires_grad_(False)

    records = []
    start = time.time()
    for a_i, (a, a_rows) in enumerate(by_attr.items()):
        t0 = time.time()
        probe = LayaProbe(agent, [a], args.question)
        cells = [(row, t, row[c].strip()) for row in a_rows for c, t in zip(cols, types)]
        z = probe.logits([probe.encode(text) for _, _, text in cells], args.batch_size)[:, 0]
        for (row, t, _), v, p in zip(cells, z.tolist(), torch.sigmoid(z).tolist()):
            records.append({"id": row["id"], "rewrite_type": t, "attribute": a, "logit": v, "p": p})
        means = "  ".join(
            f"{t[:12]} {sum(r['logit'] for r in records[-len(cells):] if r['rewrite_type'] == t) / len(a_rows):+.2f}"
            for t in types)
        print(f"[{a_i + 1}/{len(by_attr)}] {a:<22} {len(a_rows):>3} rows  mean logit: {means}  "
              f"({time.time() - t0:.1f}s)", flush=True)

    os.makedirs(args.out_dir, exist_ok=True)
    csv_path = os.path.join(args.out_dir, f"{args.save_suffix}.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=OUT_COLS)
        w.writeheader()
        w.writerows(records)

    print(summarize(records, types, args.threshold))
    print(f"\nDone in {round(time.time() - start, 1)}s over {len(records)} cells.")
    print(f"CSV -> {csv_path}")
