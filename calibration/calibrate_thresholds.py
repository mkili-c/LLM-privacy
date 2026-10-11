"""
Calibration measurement (ROADMAP "Plan to the end", B1): the numbers that δ (--kl_threshold)
is derived from, measured with the model the evaluation uses (7B).

δ: per row of utility/data/prompt_rewrites_100.csv, K answers are sampled once from the
original, exactly as the loop samples its reference answers (utility/metrics/utility_check.py,
the loop's utility check), and every column is scored with the loop's D:

    D = (1/K) Σ_k (1/|a_k|) Σ_t KL( P(·|original, a_k,<t) ‖ P(·|column, a_k,<t) )    [nats/token]

B2 places δ between the privacy rewrites (priv_preserving, priv_targeted) and util_broken. The
original's own D must be 0, which checks the setup.

The extra column `empty` is D(∅), the D of an empty message on the same answers: how much the
original shapes its answers at all. D's scale differs between prompts (smoke job 567: a plain
paraphrase costs 0.298 for one row and 0.056 for another), so B2 also tests the per-prompt share
R = D / D(∅) against raw D, as it tests D / D(util_paraphrase).

K: each answer's own D is written too. The answers are independent samples, so the mean over the
first 5 or 8 is D as the loop would measure it with K = 5 or 8. B2 compares those with K = 10
(how often D falls on the other side of δ) to choose K, since the utility step's cost grows with K.

ε and the spikes are gone with the fluency guard (2026-10-10): both steps now rewrite with one
greedy LLM call.

Outputs (in --out_dir, default results/calibration/):
    dev_D.csv           id, column, D, kl_per_answer    one row per (dev row, column), `empty` included; D is the
                                                        mean of the space-separated kl_per_answer
    dev_answers.json    id -> the K answers, for the dev-set scoring (C3)

Dev rows are written as they finish and rows already in dev_D.csv are skipped, so a rerun
continues. A row's answers are saved before its row, so a rerun scores it on the same answers.
The seed is reset per row, as in run_loop_sweep.py.

    python calibration/calibrate_thresholds.py --model_path /scratch/users/mkilic/Qwen2.5-7B-Instruct --dtype bfloat16

analyze_calibration.py (same folder) turns the CSV into the B2 numbers.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
for _dir in (REPO / "privacy", REPO / "utility" / "metrics", REPO / "utility" / "grad_attribution"):
    if str(_dir) not in sys.path:
        sys.path.insert(0, str(_dir))

import utility_check                           # noqa: E402
from attribute_probe import DEFAULT_MODEL      # noqa: E402
from grad_based_utility_score import load      # noqa: E402

COLUMNS = ["original", "util_preserving_rewrite", "util_paraphrase_rewrite",
           "priv_preserving_rewrite", "priv_targeted_rewrite", "util_broken_rewrite"]
EMPTY = "empty"     # the column of the empty message, D(∅): not in the CSV, scored on ""


def get_args():
    p = argparse.ArgumentParser(description="D for calibrating δ (B1)")
    p.add_argument("--data", default=str(REPO / "utility" / "data" / "prompt_rewrites_100.csv"))
    p.add_argument("--out_dir", default=str(REPO / "results" / "calibration"))
    p.add_argument("--limit", type=int, default=None, help="only the first N dev rows (smoke test)")
    # 10 answers, so that B2 can compare the loop's K = 5 (or 8) with all 10 on the same answers
    p.add_argument("--k", type=int, default=10, help="answers sampled per original")
    p.add_argument("--answer_max_new_tokens", type=int, default=256)
    p.add_argument("--model_path", default=DEFAULT_MODEL)
    p.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="bfloat16")
    p.add_argument("--seed", type=int, default=23)
    return p.parse_args()


def read_csv(path, limit=None):
    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return rows[:limit] if limit else rows


def measure_row(model, tok, row, answers, device):
    """-> [(column, D, [D on each answer])] for one dev row and the empty message, against
    its original's answers."""
    columns, texts = COLUMNS + [EMPTY], [row[c] for c in COLUMNS] + [""]
    if not answers:                                   # every sampled answer was empty: no D
        return [(c, float("nan"), []) for c in columns]
    # the original's log-probs on each answer, computed once and shared by all columns
    ref_logp = utility_check.reference_logprobs(model, tok, row["original"], answers, device)
    kls = [utility_check.answer_kls(model, tok, t, answers, ref_logp, device) for t in texts]
    return [(c, sum(k) / len(k), k) for c, k in zip(columns, kls)]


def measure_dev(model, tok, rows, args, device, out_dir):
    """D of every column of every dev row, written row by row."""
    answers_path, out_path = out_dir / "dev_answers.json", out_dir / "dev_D.csv"
    answers = json.loads(answers_path.read_text(encoding="utf-8")) if answers_path.exists() else {}
    done = {r["id"] for r in read_csv(out_path)} if out_path.exists() else set()
    todo = [r for r in rows if r["id"] not in done]
    print(f"dev: {len(done)} rows already done, {len(todo)} to go -> {out_path}")
    new_file = not out_path.exists()
    with open(out_path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(["id", "column", "D", "kl_per_answer"])
        for n, row in enumerate(todo, 1):
            torch.manual_seed(args.seed)
            if row["id"] not in answers:
                answers[row["id"]] = utility_check.sample_answers(model, tok, row["original"], args.k,
                                                                  args.answer_max_new_tokens)
                answers_path.write_text(json.dumps(answers, ensure_ascii=False, indent=1), encoding="utf-8")
            results = measure_row(model, tok, row, answers[row["id"]], device)
            w.writerows([row["id"], c, f"{D:.6f}", " ".join(f"{k:.6f}" for k in kls)] for c, D, kls in results)
            f.flush()
            print(f"  {n}/{len(todo)} {row['id'][:30]!r}: "
                  + "  ".join(f"{c.removesuffix('_rewrite')} D={D:.3f}" for c, D, _ in results))


def main():
    args = get_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"K = {args.k} answers of {args.answer_max_new_tokens} tokens per original")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {args.model_path} on {device} ({args.dtype}) ...")
    model, tok = load(args.model_path, device, getattr(torch, args.dtype))
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    measure_dev(model, tok, read_csv(args.data, args.limit), args, device, out_dir)


if __name__ == "__main__":
    main()
