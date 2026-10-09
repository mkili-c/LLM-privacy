"""
Calibration measurement (ROADMAP "Plan to the end", B1): the numbers that δ (--kl_threshold)
and ε (--fluency_eps) are derived from, measured with the model the evaluation uses (7B).

δ: per row of utility/data/prompt_rewrites_100.csv, K answers are sampled once from the
original, exactly as the loop samples its reference answers (privacy_utility_loop.kl_multi_answers,
K and the answer length from kl_multi's defaults), and every column is scored with the loop's D:

    D = (1/K) Σ_k (1/|a_k|) Σ_t KL( P(·|original, a_k,<t) ‖ P(·|column, a_k,<t) )    [nats/token]

B2 places δ between the privacy rewrites (priv_preserving, priv_targeted) and util_broken. The
original's own D must be 0, which checks the setup.

ε: the spikes (disfluency) before and after each of the 21 edits from the runs in
utility/data/fluency_pairs.csv (8 clean, 3 borderline, 10 broken). B2 places ε between the clean
and the broken edits, since the guard bounds the spikes an edit adds. The spikes of every dev
column are written too.

Outputs (in --out_dir, default results/calibration/):
    fluency_pairs.csv   label, before, after, spikes_before, spikes_after, delta
    dev_D_spikes.csv    id, column, D, spikes        one row per (dev row, column)
    dev_answers.json    id -> the K answers, for the dev-set scoring (C3)

Dev rows are written as they finish and rows already in dev_D_spikes.csv are skipped, so a rerun
continues. A row's answers are saved before its row, so a rerun scores it on the same answers.
The seed is reset per row, as in run_loop_sweep.py.

    python calibrate_thresholds.py --model_path /scratch/users/mkilic/Qwen2.5-7B-Instruct --dtype bfloat16
"""
import argparse
import csv
import json
from pathlib import Path

import torch

import privacy_utility_loop as loop       # also puts privacy/ and utility/ on sys.path
import kl_mask_llm_rewrite_multi as kl_multi
from attribute_probe import DEFAULT_MODEL
from grad_based_utility_score import load
from kl_utility_scoring import disfluency

REPO = Path(__file__).resolve().parent
COLUMNS = ["original", "util_preserving_rewrite", "util_paraphrase_rewrite",
           "priv_preserving_rewrite", "priv_targeted_rewrite", "util_broken_rewrite"]


def get_args():
    p = argparse.ArgumentParser(description="D and spikes for calibrating δ and ε (B1)")
    p.add_argument("--data", default=str(REPO / "utility" / "data" / "prompt_rewrites_100.csv"))
    p.add_argument("--pairs", default=str(REPO / "utility" / "data" / "fluency_pairs.csv"))
    p.add_argument("--out_dir", default=str(REPO / "results" / "calibration"))
    p.add_argument("--limit", type=int, default=None, help="only the first N dev rows (smoke test)")
    p.add_argument("--utility_args", default="",
                   help="kl_multi's own arguments, as in the loop; K and the answer length come from them")
    p.add_argument("--model_path", default=DEFAULT_MODEL)
    p.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="bfloat16")
    p.add_argument("--seed", type=int, default=23)
    return p.parse_args()


def read_csv(path, limit=None):
    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return rows[:limit] if limit else rows


def measure_pairs(model, tok, pairs_path, out_path):
    """Spikes before and after each calibration edit, in one batched call."""
    pairs = read_csv(pairs_path)
    spikes = disfluency([p["before"] for p in pairs] + [p["after"] for p in pairs], model, tok)
    before, after = spikes[:len(pairs)], spikes[len(pairs):]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["label", "before", "after", "spikes_before", "spikes_after", "delta"])
        for p, b, a in zip(pairs, before, after):
            w.writerow([p["label"], p["before"], p["after"], f"{b:.4f}", f"{a:.4f}", f"{a - b:+.4f}"])
    print(f"{len(pairs)} calibration edits -> {out_path}")


def measure_row(model, tok, row, answers, device):
    """-> [(column, D, spikes)] for one dev row, D against its original's answers."""
    texts = [row[c] for c in COLUMNS]
    spikes = disfluency(texts, model, tok)
    if not answers:                                   # every sampled answer was empty: no D
        return [(c, float("nan"), s) for c, s in zip(COLUMNS, spikes)]
    # the original's log-probs on each answer, computed once and shared by all columns
    _, refs = kl_multi.prepare_answers(model, tok, row["original"], row["original"], answers, device)
    D = [kl_multi.text_kl(model, tok, t, answers, refs, device) for t in texts]
    return list(zip(COLUMNS, D, spikes))


def measure_dev(model, tok, rows, uargs, device, out_dir, seed):
    """D and spikes of every column of every dev row, written row by row."""
    answers_path, out_path = out_dir / "dev_answers.json", out_dir / "dev_D_spikes.csv"
    answers = json.loads(answers_path.read_text(encoding="utf-8")) if answers_path.exists() else {}
    done = {r["id"] for r in read_csv(out_path)} if out_path.exists() else set()
    todo = [r for r in rows if r["id"] not in done]
    print(f"dev: {len(done)} rows already done, {len(todo)} to go -> {out_path}")
    new_file = not out_path.exists()
    with open(out_path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(["id", "column", "D", "spikes"])
        for n, row in enumerate(todo, 1):
            torch.manual_seed(seed)
            if row["id"] not in answers:
                answers[row["id"]] = loop.kl_multi_answers(model, tok, row["original"], device, uargs)
                answers_path.write_text(json.dumps(answers, ensure_ascii=False, indent=1), encoding="utf-8")
            results = measure_row(model, tok, row, answers[row["id"]], device)
            w.writerows([row["id"], c, f"{D:.6f}", f"{s:.4f}"] for c, D, s in results)
            f.flush()
            print(f"  {n}/{len(todo)} {row['id'][:30]!r}: "
                  + "  ".join(f"{c.removesuffix('_rewrite')} D={D:.3f}" for c, D, _ in results))


def main():
    args = get_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    uargs = loop.stage_args(kl_multi.get_args, args.utility_args)
    print(f"K = {uargs.k} answers of {uargs.answer_max_new_tokens} tokens per original")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {args.model_path} on {device} ({args.dtype}) ...")
    model, tok = load(args.model_path, device, getattr(torch, args.dtype))
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    measure_pairs(model, tok, args.pairs, out_dir / "fluency_pairs.csv")      # seconds, so first
    measure_dev(model, tok, read_csv(args.data, args.limit), uargs, device, out_dir, args.seed)


if __name__ == "__main__":
    main()
