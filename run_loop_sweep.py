"""Run the privacy-utility loop over an evaluation set and record its rewrites, as run_baselines.py does.

    python run_loop_sweep.py --data utility/data/prompt_rewrites_100.csv --out results/rewrites/dev_loop.csv \
        --model_path /scratch/users/mkilic/Qwen2.5-7B-Instruct --dtype bfloat16

Loads Qwen and Laya once and runs privacy_utility_loop() on the `original` of every row. The
rows are written by run_baselines.run, so the output has the baselines' format:
    id, method, rewrite, error     the loop's final text (method "loop"), or the exception
    seconds                        wall-clock time of this anonymization
    calls, prompt_tokens,          model.generate calls and the tokens they read and generated,
    completion_tokens              counted the way the server counts a baseline's requests
Each row is flushed as it finishes and ids already done are skipped, so a run resumes after the
24 h limit. The seed is reset for every row, so a row's rewrite does not depend on which rows
ran before it.

calls and tokens cover the LLM's generations: the privacy step's replacements, the gap and token
fillers, the K sampled reference answers and the fluency fixes. The scoring forward passes and
gradients (attackers, D, disfluency), most of the loop's compute, are not in them, so `seconds`
is the cost that compares with the baselines.

privacy_utility_loop.py has no get_args, so its flags are copied below with the same defaults:
change them in both places.
"""
import argparse
from argparse import Namespace
from pathlib import Path

import torch

import privacy_utility_loop as loop     # also puts privacy/ and utility/ on sys.path
import run_baselines
import laya
from attribute_probe import ATTRIBUTES, DEFAULT_MODEL
from laya_probe import PRIVACY_LAYA
from privacy_rewrite import LayaAttacker, QwenAttacker
from grad_based_utility_score import load

METHOD = "loop"     # the `method` column


def get_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default="utility/data/prompt_rewrites_100.csv",
                   help="evaluation CSV with `id` and `original`")
    p.add_argument("--out", default=None,
                   help=f"default: {run_baselines.OUT_TEMPLATE}, data = the CSV's folder name")
    p.add_argument("--limit", type=int, default=None, help="only the first N rows (smoke test)")
    # the loop's flags, as in privacy_utility_loop.main(); see its --help
    p.add_argument("--attributes", nargs="+", default=ATTRIBUTES)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--max_iters", type=int, default=4)
    p.add_argument("--utility", choices=sorted(loop.UTILITY), default="kl_multi")
    p.add_argument("--utility_ref", choices=["user", "iteration"], default="user")
    p.add_argument("--utility_args", default="", help="for the utility script's own parser")
    p.add_argument("--privacy_top_m", type=int, default=2)
    p.add_argument("--privacy_max_n", type=int, default=4)
    p.add_argument("--privacy_span_penalty", type=float, default=0.05)
    p.add_argument("--privacy_max_iters", type=int, default=8)
    p.add_argument("--privacy_n_candidates", type=int, default=10)
    p.add_argument("--privacy_cand_rounds", type=int, default=2)
    p.add_argument("--privacy_cand_temperature", type=float, default=0.9)
    p.add_argument("--privacy_fluency_eps", type=float, default=4.0)
    p.add_argument("--fix_candidates", type=int, default=4)
    p.add_argument("--model_path", default=DEFAULT_MODEL)
    p.add_argument("--laya_path", default=PRIVACY_LAYA)
    p.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    p.add_argument("--seed", type=int, default=23)
    return p.parse_args()


def stage_args(args):
    """(privacy step args, utility step args), built as in privacy_utility_loop.main()."""
    pargs = Namespace(threshold=args.threshold, top_m=args.privacy_top_m,
                      max_n=args.privacy_max_n, span_penalty=args.privacy_span_penalty,
                      max_iters=args.privacy_max_iters, n_candidates=args.privacy_n_candidates,
                      cand_rounds=args.privacy_cand_rounds,
                      cand_temperature=args.privacy_cand_temperature,
                      fluency_eps=args.privacy_fluency_eps)
    uargs = loop.stage_args(loop.UTILITY[args.utility][2], args.utility_args)
    return pargs, uargs


def load_models(args, device):
    """Qwen (attacker, rewriter and scorer at once) and Laya, as in privacy_utility_loop.main().
    -> (model, tok, attackers)"""
    print(f"Loading {args.model_path} on {device} ({args.dtype}) ...")
    model, tok = load(args.model_path, device, getattr(torch, args.dtype))
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    agent = laya.load(args.laya_path)
    return model, tok, [QwenAttacker(model, tok, args.attributes), LayaAttacker(agent, args.attributes)]


def count_generations(model, tok):
    """Wrap model.generate so each call is counted the way the server counts a baseline's
    request: one call, its prompt tokens once (also with num_return_sequences > 1), and the
    generated tokens of every returned sequence, without the padding after a sequence ends.
    -> the running totals, which the caller resets per prompt."""
    usage = dict(calls=0, prompt_tokens=0, completion_tokens=0)
    generate = model.generate

    def counted(**kwargs):
        out = generate(**kwargs)
        seqs = out if torch.is_tensor(out) else out.sequences    # (returned sequences, prompt + new)
        prompt_len = kwargs["input_ids"].shape[1]                # every caller sends one prompt
        usage["calls"] += 1
        usage["prompt_tokens"] += prompt_len
        usage["completion_tokens"] += (seqs[:, prompt_len:] != tok.pad_token_id).sum().item()
        return out

    model.generate = counted
    return usage


def loop_rewriter(model, tok, attackers, pargs, uargs, device, args):
    """-> prompt -> (rewrite, LLM calls, prompt tokens, completion tokens), the adapter
    run_baselines.run expects."""
    usage = count_generations(model, tok)

    def rewrite(text):
        usage.update(calls=0, prompt_tokens=0, completion_tokens=0)
        torch.manual_seed(args.seed)
        final, history, converged = loop.privacy_utility_loop(
            model, tok, attackers, text, args.attributes, args.utility, pargs, uargs, device, args)
        loop.report(text, final, history, converged, args.attributes, args.threshold)
        return final, usage["calls"], usage["prompt_tokens"], usage["completion_tokens"]
    return rewrite


def main():
    args = get_args()
    data_name = Path(args.data).resolve().parent.name           # run_baselines.py's rule
    out_path = args.out or run_baselines.OUT_TEMPLATE.format(data=data_name, method=METHOD)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    rows = run_baselines.read_rows(args.data, args.limit)
    pargs, uargs = stage_args(args)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tok, attackers = load_models(args, device)
    rewrite = loop_rewriter(model, tok, attackers, pargs, uargs, device, args)
    run_baselines.run(rows, rewrite, METHOD, out_path)


if __name__ == "__main__":
    main()
