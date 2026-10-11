"""
Privacy-utility loop: alternate the privacy rewriter and a utility rewriter
until the utility-repaired prompt no longer leaks.

    x_0 = the user's message
    for i = 1 .. --max_iters:
        1. p_i  = privacy_rewrite(x_{i-1})
                  privacy/privacy_rewrite.py: span occlusion against Qwen and
                  Laya blames the leaking spans, and the LLM rewrites the
                  message around them, right to left; its own inner loop
                  (--privacy_max_iters) and the same --threshold
        2. u_i  = utility_rewrite(reference = x_0, rewritten = p_i)
                  utility/grad_attribution/: pull P(A | p_i) back toward
                  P(A | x_0) (--utility kl_multi | cos_mask). kl_multi mirrors
                  the privacy step: the gradient of D marks the words of x_0
                  that p_i lost and the answers need most, the LLM puts them
                  back, and D decides whether the rewrite is kept
        3. leak_i = max_a max_attacker P_a(u_i)
                  stop if leak_i <= --threshold, else x_i = u_i

The utility rewriters are free to re-introduce private details (they do not
see the attackers), which is why step 3 measures u_i and not p_i. Neither step
has a fluency guard or a final fluency fix any more (2026-10-10): both rewrite
with one greedy LLM call that keeps the message natural.

Reference of the utility step (--utility_ref):
    user       x_0 every iteration (default): the user's message is the only
               real definition of utility. It carries every private detail,
               so the utility step may pull some back in. That is intended:
               kl_multi only edits while D > its --kl_threshold, i.e. when the
               rewrite cannot afford to lose them. The reference answers are
               sampled once and reused in every iteration, so every threshold
               decision is made against the same answers.
    iteration  x_{i-1}, the input of the iteration. From i = 2 on this is
               already a rewrite, so utility is measured against the previous
               iterate and drifts: a garbled u_1 becomes the reference of
               iteration 2. Its answers are resampled every iteration.

The whole pipeline runs on one Qwen model: it is the Qwen attacker, the LLM
that rewrites on both sides, and the model the utility objective is measured
on. Laya is the second attacker.

Arguments of the utility rewriter are parsed by that script's own get_args,
from the string in --utility_args, so its defaults stay in one place; its
--model / --original / --rewritten are ignored here. The privacy rewriter has
no get_args, so its knobs are prefixed --privacy_* flags below.


"""
import argparse
import shlex
import sys
import time
from argparse import Namespace
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parent
for _dir in (_ROOT / "privacy", _ROOT / "utility" / "metrics", _ROOT / "utility" / "grad_attribution"):
    if str(_dir) not in sys.path:
        sys.path.insert(0, str(_dir))

import laya                                                            # noqa: E402
import privacy_rewrite                                                 # noqa: E402
from privacy_rewrite import LayaAttacker, QwenAttacker, score          # noqa: E402
from attribute_probe import ATTRIBUTES, DEFAULT_MODEL                  # noqa: E402
from laya_probe import PRIVACY_LAYA                                    # noqa: E402
import kl_mask_llm_rewrite_multi as kl_multi                           # noqa: E402
import cos_mask_llm_rewrite as cos_mask                                # noqa: E402
from cos_gcg_rewrite import original_embeddings                        # noqa: E402
from topk_sample import topk_answers                                   # noqa: E402
from grad_based_utility_score import load                              # noqa: E402


# ---------------------------------------------------------------- utility step

def kl_multi_answers(model, tok, reference, device, uargs):
    """--k non-empty answers sampled from the reference: the ones D is teacher-forced on."""
    return [a for a, _, _ in topk_answers(
        reference, model, tok, k=uargs.k, mode="sample",
        max_new_tokens=uargs.answer_max_new_tokens, temperature=1.0, top_p=1.0) if a.strip()]


def utility_kl_multi(model, tok, reference, answers, rewritten, device, uargs):
    """KL gradient marks the words, the LLM rewrites with the reference in view,
    mean KL over the reference answers decides."""
    if not answers:
        print("  utility: no non-empty reference answers; rewrite left unchanged")
        return rewritten
    return kl_multi.run(model, tok, reference, rewritten, answers, device, uargs)


def utility_cos_mask(model, tok, reference, ref_samples, rewritten, device, uargs):
    """KL gradient picks positions, LLM proposes fillers, answer-centroid
    cosine against the reference decides. ref_samples = (answers, E_ref) of
    original_embeddings."""
    answers, E_ref = ref_samples
    return cos_mask.run(model, tok, reference, rewritten, answers[0], E_ref, device, uargs)


# name -> (sample the reference, rewrite against those samples, the script's own argparse)
UTILITY = {
    "kl_multi": (kl_multi_answers, utility_kl_multi, kl_multi.get_args),
    "cos_mask": (original_embeddings, utility_cos_mask, cos_mask.get_args),
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
    sample_reference, rewrite_utility, _ = UTILITY[utility]
    x, u, history, ref_samples, converged = text, text, [], None, False
    for i in range(1, args.max_iters + 1):
        print(f"\n{'#' * 30} iteration {i}: privacy step {'#' * 30}")
        start = time.perf_counter()
        p = privacy_rewrite.rewrite(model, tok, attackers, x, attributes, pargs)
        privacy_seconds = time.perf_counter() - start
        leak_p, _ = leakage(attackers, p)

        reference = x if args.utility_ref == "iteration" else text
        print(f"\n{'#' * 30} iteration {i}: utility step ({utility}) {'#' * 30}")
        start = time.perf_counter()
        # x_0 never changes, so its samples are drawn once and every threshold
        # decision is made against the same answers; an iteration reference is
        # a new prompt each time. An empty sample is not kept, it is redrawn.
        if args.utility_ref == "iteration" or not ref_samples:
            print(f"  sampling the reference answers ({args.utility_ref} reference)")
            ref_samples = sample_reference(model, tok, reference, device, uargs)
        u = rewrite_utility(model, tok, reference, ref_samples, p, device, uargs)
        utility_seconds = time.perf_counter() - start            # with the reference answers, if sampled
        leak_u, per = leakage(attackers, u)

        history.append(dict(iteration=i, input=x, privacy=p, utility=u,
                            leak_privacy=leak_p, leak_utility=leak_u, per_attacker=per))
        print(f"\n  iteration {i}: leak after privacy {leak_p:.3f}, after utility {leak_u:.3f} "
              f"(threshold {args.threshold}); privacy step {privacy_seconds:.0f} s, "
              f"utility step {utility_seconds:.0f} s")
        if leak_u <= args.threshold:
            converged = True
            break
        if u == x:
            print("  iteration changed nothing: stop")
            break
        x = u
    return u, history, converged


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
    # 2: when the two steps disagree they cycle (job 556: iterations 3-4 repeated 1-2 and
    # ended at the text of iteration 2), and the utility step, running last, wins anyway
    p.add_argument("--max_iters", type=int, default=2, help="privacy + utility iterations")
    p.add_argument("--utility", choices=sorted(UTILITY), default="kl_multi")
    p.add_argument("--utility_ref", choices=["user", "iteration"], default="user",
                   help="reference prompt of the utility step (default: %(default)s), "
                        "see the module docstring")
    p.add_argument("--utility_args", default="",
                   help="extra arguments for the utility script's own parser, e.g. \"--top_m 5\"")
    # privacy_rewrite.rewrite reads these fields of its args
    p.add_argument("--privacy_top_m", type=int, default=5,
                   help="cap on spans the privacy step rewrites per iteration (one per leaking attribute)")
    p.add_argument("--privacy_max_n", type=int, default=4)
    p.add_argument("--privacy_span_penalty", type=float, default=0.05)
    p.add_argument("--privacy_max_iters", type=int, default=8)
    p.add_argument("--model_path", default=DEFAULT_MODEL)
    p.add_argument("--laya_path", default=PRIVACY_LAYA, help="Laya attacker (default: the fine-tuned one)")
    p.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    p.add_argument("--seed", type=int, default=23)
    args = p.parse_args()

    pargs = Namespace(threshold=args.threshold, top_m=args.privacy_top_m,
                      max_n=args.privacy_max_n, span_penalty=args.privacy_span_penalty,
                      max_iters=args.privacy_max_iters)
    uargs = stage_args(UTILITY[args.utility][2], args.utility_args)

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
