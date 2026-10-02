"""
Privacy-utility loop: alternate the privacy rewriter and a utility rewriter
until the utility-repaired prompt no longer leaks.

    x_0 = the user's message
    for i = 1 .. --max_iters:
        1. p_i  = privacy_rewrite(x_{i-1})
                  privacy/privacy_rewrite.py: span occlusion + LLM
                  generalization against Qwen and Laya, with its own inner
                  loop (--privacy_max_iters) and the same --threshold
        2. u_i  = utility_rewrite(reference = x_{i-1}, rewritten = p_i)
                  utility/grad_attribution/: pull P(A | p_i) back toward
                  P(A | x_{i-1}) (--utility kl_multi | cos_mask). Spans the
                  privacy step deleted reach it as [MASK] gaps; kl_multi
                  refills them with phrases if that lowers D, cos_mask drops them
        3. leak_i = max_a max_attacker P_a(u_i)
                  stop if leak_i <= --threshold, else x_i = u_i

The utility rewriters are free to re-introduce private details (they do not
see the attackers), which is why step 3 measures u_i and not p_i.

Reference of the utility step (--utility_ref):
    iteration  x_{i-1}, the input of the iteration (default). From i = 2 on
               this is already a rewrite, so utility is measured against the
               previous iterate, not the user's message, and can drift.
    user       x_0 every iteration. No drift, but x_0 carries every private
               detail, so the utility step pulls towards all of them again
               each time.

The whole pipeline runs on one Qwen model: it is the Qwen attacker, the LLM
that proposes replacements on both sides, and the model the utility
objective is measured on. Laya is the second attacker.

Arguments of the utility rewriter are parsed by that script's own get_args,
from the string in --utility_args, so its defaults stay in one place; its
--model / --original / --rewritten are ignored here. The privacy rewriter has
no get_args, so its knobs are prefixed --privacy_* flags below.


"""
import argparse
import shlex
import sys
from argparse import Namespace
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parent
for _dir in (_ROOT / "privacy", _ROOT / "utility" / "metrics", _ROOT / "utility" / "grad_attribution"):
    if str(_dir) not in sys.path:
        sys.path.insert(0, str(_dir))

import laya                                                            # noqa: E402
import privacy_rewrite                                                 # noqa: E402
from privacy_rewrite import SLOT, LayaAttacker, QwenAttacker, score    # noqa: E402
from attribute_probe import ATTRIBUTES, DEFAULT_MODEL                  # noqa: E402
from laya_probe import DEFAULT_LAYA                                    # noqa: E402
import kl_mask_llm_rewrite_multi as kl_multi                           # noqa: E402
import cos_mask_llm_rewrite as cos_mask                                # noqa: E402
from kl_mask_llm_rewrite import MASK                                   # noqa: E402
from cos_gcg_rewrite import original_embeddings                        # noqa: E402
from topk_sample import topk_answers                                   # noqa: E402
from grad_based_utility_score import load                              # noqa: E402


# ---------------------------------------------------------------- utility step

def utility_kl_multi(model, tok, reference, rewritten, device, uargs):
    """KL gradient picks positions, LLM proposes fillers, mean KL over K
    answers sampled from the reference decides."""
    answers = [a for a, _, _ in topk_answers(
        reference, model, tok, k=uargs.k, mode="sample",
        max_new_tokens=uargs.answer_max_new_tokens, temperature=1.0, top_p=1.0) if a.strip()]
    if not answers:
        print("  utility: no non-empty reference answers; rewrite left unchanged")
        return rewritten.replace(MASK, "")
    return kl_multi.run(model, tok, reference, rewritten, answers, device, uargs)


def utility_cos_mask(model, tok, reference, rewritten, device, uargs):
    """KL gradient picks positions, LLM proposes fillers, answer-centroid
    cosine against the reference decides. Gaps are not filled, just dropped."""
    rewritten = rewritten.replace(MASK, "")
    answers, E_ref = original_embeddings(model, tok, reference, device, uargs)
    return cos_mask.run(model, tok, reference, rewritten, answers[0], E_ref, device, uargs)


UTILITY = {
    "kl_multi": (utility_kl_multi, kl_multi.get_args),
    "cos_mask": (utility_cos_mask, cos_mask.get_args),
}


def stage_args(get_args, argv):
    """Run a stage script's own argparse on `argv` instead of sys.argv."""
    saved = sys.argv
    sys.argv = [saved[0]] + shlex.split(argv)
    try:
        return get_args()
    finally:
        sys.argv = saved


# ---------------------------------------------------------------- privacy measurement

def leakage(attackers, text):
    """-> (max over attributes and attackers of P_a, {attacker: [P_a per attribute]})."""
    per, P = score(attackers, [text])
    return P[0].max().item(), {n: v[0].tolist() for n, v in per.items()}


# ---------------------------------------------------------------- main loop

def privacy_utility_loop(model, tok, attackers, text, attributes, utility, pargs, uargs, device, args):
    """-> (final text, history, converged). history has one dict per iteration."""
    rewrite_utility = UTILITY[utility][0]
    x, history = text, []
    for i in range(1, args.max_iters + 1):
        print(f"\n{'#' * 30} iteration {i}: privacy step {'#' * 30}")
        p = privacy_rewrite.rewrite(model, tok, attackers, x, attributes, pargs)
        leak_p, _ = leakage(attackers, p)

        reference = x if args.utility_ref == "iteration" else text
        print(f"\n{'#' * 30} iteration {i}: utility step ({utility}) {'#' * 30}")
        u = rewrite_utility(model, tok, reference, p.replace(SLOT, MASK), device, uargs)
        leak_u, per = leakage(attackers, u)

        history.append(dict(iteration=i, input=x, privacy=p, utility=u,
                            leak_privacy=leak_p, leak_utility=leak_u, per_attacker=per))
        print(f"\n  iteration {i}: leak after privacy {leak_p:.3f}, after utility {leak_u:.3f} "
              f"(threshold {args.threshold})")
        if leak_u <= args.threshold:
            return u, history, True
        if u == x:
            print("  iteration changed nothing: stop")
            return u, history, False
        x = u
    return x, history, False


def report(text, final, history, converged, attributes, threshold):
    print(f"\n{'=' * 80}\noriginal: {text!r}")
    for h in history:
        print(f"\n  iteration {h['iteration']}")
        print(f"    privacy  leak {h['leak_privacy']:.3f}  {h['privacy']!r}")
        print(f"    utility  leak {h['leak_utility']:.3f}  {h['utility']!r}")
        for n, ps in h["per_attacker"].items():
            print(f"      {n:<5}" + "".join(f"  {a[:8]:>8} {p:.3f}" for a, p in zip(attributes, ps)))
    status = f"converged (leak <= {threshold})" if converged else "NOT converged"
    print(f"\nfinal ({status}, {len(history)} iterations): {final!r}")


def main():
    p = argparse.ArgumentParser(description="Alternate the privacy and utility rewriters until "
                                            "the utility-repaired prompt leaks no attribute")
    p.add_argument("--text", required=True)
    p.add_argument("--attributes", nargs="+", default=ATTRIBUTES)
    p.add_argument("--threshold", type=float, default=0.5,
                   help="max P_a over attributes and attackers counted as private "
                        "(also the privacy rewriter's own stopping threshold)")
    p.add_argument("--max_iters", type=int, default=4, help="privacy + utility iterations")
    p.add_argument("--utility", choices=sorted(UTILITY), default="kl_multi")
    p.add_argument("--utility_ref", choices=["iteration", "user"], default="iteration",
                   help="reference prompt of the utility step, see the module docstring")
    p.add_argument("--utility_args", default="",
                   help="extra arguments for the utility script's own parser, e.g. \"--n_steps 5\"")
    # privacy_rewrite.rewrite reads these fields of its args
    p.add_argument("--privacy_top_m", type=int, default=2)
    p.add_argument("--privacy_max_n", type=int, default=4)
    p.add_argument("--privacy_span_penalty", type=float, default=0.05)
    p.add_argument("--privacy_max_iters", type=int, default=8)
    p.add_argument("--privacy_n_candidates", type=int, default=10)
    p.add_argument("--privacy_cand_rounds", type=int, default=2)
    p.add_argument("--privacy_cand_temperature", type=float, default=0.9)
    p.add_argument("--model_path", default=DEFAULT_MODEL)
    p.add_argument("--laya_path", default=DEFAULT_LAYA)
    p.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    p.add_argument("--seed", type=int, default=23)
    args = p.parse_args()

    pargs = Namespace(threshold=args.threshold, top_m=args.privacy_top_m,
                      max_n=args.privacy_max_n, span_penalty=args.privacy_span_penalty,
                      max_iters=args.privacy_max_iters, n_candidates=args.privacy_n_candidates,
                      cand_rounds=args.privacy_cand_rounds,
                      cand_temperature=args.privacy_cand_temperature)
    uargs = stage_args(UTILITY[args.utility][1], args.utility_args)

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {args.model_path} on {device} ({args.dtype}) ...")
    model, tok = load(args.model_path, device, getattr(torch, args.dtype))
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    agent = laya.load(args.laya_path)
    attackers = [QwenAttacker(model, tok, args.attributes), LayaAttacker(agent, args.attributes)]

    final, history, converged = privacy_utility_loop(
        model, tok, attackers, args.text, args.attributes, args.utility, pargs, uargs, device, args)
    report(args.text, final, history, converged, args.attributes, args.threshold)


if __name__ == "__main__":
    main()
