"""
Our own privacy and utility measures on every method's rewrites of PUPA-SD, and on the originals.

For every prompt of the evaluation set and every version of it (the original and each method's
rewrite):
    privacy  P_a of both of the loop's attackers for each of the 10 attributes (privacy_rewrite.py:
             Qwen's yes/no probe and Laya) and their max, the loop's leak measure
    utility  D against the original's answers, D(∅), and R = D / D(∅)  (utility/metrics/utility_check.py)
             D(y) = (1/K) Σ_k (1/|a_k|) Σ_j KL( P(·|x_0, a_k,<j) ‖ P(·|y, a_k,<j) )   [nats per answer token]

The K = 5 reference answers per prompt are sampled exactly as the loop sampled them: the seed is
reset to --seed right before utility_check.prepare, which is the loop's first random step for a
row (run_loop_sweep.py). So they reproduce the loop's answers, and the R of the loop's final text
here should equal the R in its log; the original's own D must be 0.

These are the measures the loop optimized against, so for the loop they are not independent
evidence (the external judge is). They show every method on the scale the loop itself works on.

Output (in --out_dir, default results/evaluation/):
    own_measures.csv        one row per (prompt, version): id, method, D, d_empty, R,
                            kl_per_answer (space-separated), and per attribute a the columns
                            "qwen <a>", "laya <a>", "max <a>"
    reference_answers.json  id -> the K answers D was measured on
A prompt's rows are written together once it is done, and prompts already in own_measures.csv
are skipped, so a rerun continues.

Cost on the GB10: ~3.5 h for 112 prompts x 7 versions; sampling the answers is ~2.8 h of it.

    sbatch evaluation/run_score_own_measures_gb10.sbatch
"""
import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import run_baselines                                   # noqa: E402
from run_loop_sweep import load_models                 # noqa: E402  (also puts privacy/ and utility/ on sys.path)
from privacy_rewrite import score                      # noqa: E402
import utility_check                                   # noqa: E402
from attribute_probe import ATTRIBUTES                 # noqa: E402
from laya_probe import PRIVACY_LAYA                    # noqa: E402

ORIGINAL = "original"                                  # the method name of the unedited prompt
METHODS = ["loop"] + run_baselines.METHODS             # loop, presidio, loiseau_pupa, loiseau_synthpai, papillon, staab


def get_args():
    p = argparse.ArgumentParser(description="The loop's privacy and utility measures on every method's rewrites")
    p.add_argument("--data", default=str(REPO / "data" / "pupa_sd" / "pupa_sd_eval.csv"))
    p.add_argument("--methods", nargs="+", default=METHODS)
    p.add_argument("--out_dir", default=str(REPO / "results" / "evaluation"))
    p.add_argument("--limit", type=int, default=None, help="only the first N prompts (smoke test)")
    # as in the loop's run (run_loop_sweep.py defaults), so the reference answers reproduce
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--answer_max_new_tokens", type=int, default=256)
    p.add_argument("--seed", type=int, default=23)
    p.add_argument("--attributes", nargs="+", default=ATTRIBUTES)
    p.add_argument("--model_path", default="/scratch/users/mkilic/Qwen2.5-7B-Instruct")
    p.add_argument("--laya_path", default=PRIVACY_LAYA)
    p.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="bfloat16")
    return p.parse_args()


def read_versions(data_path, methods):
    """{id: {"original": text, method: rewrite, ...}} in the evaluation set's order. A method's
    rewrite is the last successful row for the id in its CSV (failed rows are retried and
    appended); a missing rewrite stops the run, since it would look perfectly private."""
    data_name = Path(data_path).resolve().parent.name
    with open(data_path, encoding="utf-8") as f:
        versions = {r["id"]: {ORIGINAL: r["original"]} for r in csv.DictReader(f)}
    for method in methods:
        path = REPO / run_baselines.OUT_TEMPLATE.format(data=data_name, method=method)
        latest = {}
        with open(path, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if not r.get("error"):
                    latest[r["id"]] = r["rewrite"]
        missing = [i for i in versions if i not in latest]
        if missing:
            sys.exit(f"{method}: no rewrite for {len(missing)} ids, e.g. {missing[:5]} ({path})")
        for i in versions:
            versions[i][method] = latest[i]
    return versions


def utility(model, tok, texts, ref, device):
    """-> [(D, R, [D on each answer])] per text, against the original's answers in ref."""
    out = []
    for text in texts:
        kls = utility_check.answer_kls(model, tok, text, ref["answers"], ref["ref_logp"], device)
        D = sum(kls) / len(kls)
        out.append((D, D / ref["d_empty"], kls))
    return out


def score_prompt(model, tok, attackers, versions, args, device):
    """Both measures for every version of one prompt. -> (answers, rows)."""
    torch.manual_seed(args.seed)                       # the loop's state before its first random step
    ref = utility_check.prepare(model, tok, versions[ORIGINAL], args.k, args.answer_max_new_tokens, device)
    names, texts = list(versions), list(versions.values())
    per, P = score(attackers, texts)                   # {attacker: [versions, K]}, max over attackers [versions, K]
    if ref is None:                                    # every sampled answer was empty: no D
        util = [(math.nan, math.nan, [])] * len(texts)
    else:
        util = utility(model, tok, texts, ref, device)
    rows = []
    for v, name in enumerate(names):
        D, R, kls = util[v]
        row = {"method": name, "D": f"{D:.6f}", "d_empty": f"{ref['d_empty']:.6f}" if ref else "nan",
               "R": f"{R:.6f}", "kl_per_answer": " ".join(f"{k:.6f}" for k in kls)}
        for a, attribute in enumerate(args.attributes):
            for attacker, values in per.items():
                row[f"{attacker} {attribute}"] = f"{values[v, a].item():.6f}"
            row[f"max {attribute}"] = f"{P[v, a].item():.6f}"
        rows.append(row)
    return (ref["answers"] if ref else []), rows


def main():
    args = get_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path, answers_path = out_dir / "own_measures.csv", out_dir / "reference_answers.json"
    versions = read_versions(args.data, args.methods)
    columns = (["id", "method", "D", "d_empty", "R", "kl_per_answer"]
               + [f"{who} {a}" for a in args.attributes for who in ["qwen", "laya", "max"]])
    done = set()
    if out_path.exists():
        with open(out_path, encoding="utf-8") as f:
            done = {r["id"] for r in csv.DictReader(f)}
    answers = json.loads(answers_path.read_text(encoding="utf-8")) if answers_path.exists() else {}
    todo = [i for i in versions if i not in done][:args.limit]
    print(f"{len(versions)} prompts x {len(args.methods) + 1} versions; {len(done)} prompts done, "
          f"{len(todo)} to go -> {out_path}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tok, attackers = load_models(args, device)
    new_file = not out_path.exists()
    with open(out_path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=columns)
        if new_file:
            w.writeheader()
        for n, i in enumerate(todo, 1):
            start = time.perf_counter()
            answers[i], rows = score_prompt(model, tok, attackers, versions[i], args, device)
            w.writerows({"id": i, **r} for r in rows)
            f.flush()
            answers_path.write_text(json.dumps(answers, ensure_ascii=False, indent=1), encoding="utf-8")
            print(f"  {n}/{len(todo)} id {i} ({time.perf_counter() - start:.0f} s): "
                  + "  ".join(f"{r['method']} R={float(r['R']):.0%}" for r in rows))


if __name__ == "__main__":
    main()
