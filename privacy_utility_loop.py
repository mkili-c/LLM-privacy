"""
Privacy-utility loop, interleaved: the privacy rewriter removes what leaks, and a utility check
vets each of its rewrites before it goes in.

    x_0 = the user's message
    1. reference (utility/metrics/utility_check.py): K answers sampled once from x_0, x_0's
       log-probs on them, and D(∅), the D of the empty message
    2. privacy step (privacy/privacy_rewrite.py): span occlusion against Qwen and Laya blames the
       leaking spans, and the LLM rewrites the message around them, the least utility-bearing
       span first (by R of the message with that span deleted, exact), so the most valuable
       span comes last. After each rewrite the utility check computes
           R = D(rewrite) / D(∅)
       R < --max_share: the rewrite goes in. R >= --max_share: the request is lost, since the
       answers move as far as for an empty message. The LLM is asked once more, shown its refused
       rewrite, to keep the part but make it less specific; that goes in if its R < --max_share, otherwise the
       span stays as it was and is protected, so no later iteration picks it.
    3. leak = max_a max_attacker P_a(final); converged if leak <= --threshold

--max_share 1.0 is the calibrated rule: on the dev set, R >= 100% tells the off-topic rewrites
from the privacy and utility rewrites (calibration/analyze_calibration.py: AUROC 0.993,
accuracy 0.964).

Why interleaved: the sequential version (privacy step, then a utility rewriter, alternating) had
to guess afterwards what the message had lost, and the two steps undid each other (jobs 556,
561). Checked per rewrite, a change in D belongs to exactly one span. A span the check protects
stays even if it leaks: utility wins. That earlier utility rewriter is
utility/grad_attribution/kl_mask_llm_rewrite_multi.py, kept as it was but no longer called.

The whole pipeline runs on one Qwen model: it is the Qwen attacker, the LLM that rewrites, and
the model D is measured on. Laya is the second attacker.
"""
import argparse
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
import utility_check                                                   # noqa: E402
from privacy_rewrite import LayaAttacker, QwenAttacker, score          # noqa: E402
from attribute_probe import ATTRIBUTES, DEFAULT_MODEL                  # noqa: E402
from laya_probe import PRIVACY_LAYA                                    # noqa: E402
from grad_based_utility_score import load                              # noqa: E402


def leakage(attackers, text):
    """-> (max over attributes and attackers of P_a, {attacker: [P_a per attribute]})."""
    per, P = score(attackers, [text])
    return P[0].max().item(), {n: v[0].tolist() for n, v in per.items()}


def privacy_utility_loop(model, tok, attackers, text, attributes, pargs, uargs, device):
    """-> (final text, summary dict, converged)."""
    start = time.perf_counter()
    print("\n  sampling the reference answers")
    ref = utility_check.prepare(model, tok, text, uargs.k, uargs.answer_max_new_tokens, device)
    reference_seconds = time.perf_counter() - start
    if ref is None:
        print("  every sampled answer was empty: the privacy step runs without the utility check")
    else:
        print(f"  {len(ref['answers'])} answers, D(∅) = {ref['d_empty']:.4f} ({reference_seconds:.0f} s)")

    def utility_share(t):
        """R = D(t) / D(∅) of a rewrite of this message."""
        return utility_check.share(model, tok, t, ref, device)

    start = time.perf_counter()
    final = privacy_rewrite.rewrite(model, tok, attackers, text, attributes, pargs,
                                    utility_share if ref is not None else None)
    privacy_seconds = time.perf_counter() - start
    leak, per = leakage(attackers, final)
    summary = dict(leak=leak, per_attacker=per,
                   share=utility_share(final) if ref is not None else float("nan"),
                   reference_seconds=reference_seconds, privacy_seconds=privacy_seconds)
    return final, summary, leak <= pargs.threshold


def report(text, final, summary, converged, attributes, threshold):
    print(f"\n{'=' * 80}\noriginal: {text!r}")
    for n, ps in summary["per_attacker"].items():
        print(f"  {n:<5}" + "".join(f"  {a[:8]:>8} {p:.3f}" for a, p in zip(attributes, ps)))
    status = f"converged (leak <= {threshold})" if converged else "NOT converged"
    print(f"final ({status}, leak {summary['leak']:.3f}, utility R {summary['share']:.0%}; "
          f"reference {summary['reference_seconds']:.0f} s, privacy step {summary['privacy_seconds']:.0f} s): "
          f"{final!r}")


def stage_args(args):
    """(privacy step args, utility check args) from the parsed flags."""
    pargs = Namespace(threshold=args.threshold, top_m=args.privacy_top_m, max_n=args.privacy_max_n,
                      span_penalty=args.privacy_span_penalty, max_iters=args.privacy_max_iters,
                      max_share=args.max_share)
    uargs = Namespace(k=args.k, answer_max_new_tokens=args.answer_max_new_tokens)
    return pargs, uargs


def main():
    p = argparse.ArgumentParser(description="Rewrite a message until it leaks no attribute, each "
                                            "rewrite vetted by the utility check")
    p.add_argument("--text", required=True)
    p.add_argument("--attributes", nargs="+", default=ATTRIBUTES)
    p.add_argument("--threshold", type=float, default=0.5,
                   help="max P_a over attributes and attackers counted as private")
    # privacy_rewrite.rewrite reads these fields of its args
    p.add_argument("--privacy_top_m", type=int, default=5,
                   help="cap on spans the privacy step rewrites per iteration (one per leaking attribute)")
    p.add_argument("--privacy_max_n", type=int, default=4)
    p.add_argument("--privacy_span_penalty", type=float, default=0.05)
    p.add_argument("--privacy_max_iters", type=int, default=8)
    # 1.0: as far as an empty message, the rule calibrated on the dev set
    p.add_argument("--max_share", type=float, default=1.0,
                   help="utility check: a rewrite with D >= this share of D(∅) loses the request")
    # 5 answers: on the dev set, 8 of 500 decisions at R = 100% differ from those with 10
    p.add_argument("--k", type=int, default=5, help="answers sampled from x_0 for D")
    p.add_argument("--answer_max_new_tokens", type=int, default=256)
    p.add_argument("--model_path", default=DEFAULT_MODEL)
    p.add_argument("--laya_path", default=PRIVACY_LAYA, help="Laya attacker (default: the fine-tuned one)")
    p.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    p.add_argument("--seed", type=int, default=23)
    args = p.parse_args()
    pargs, uargs = stage_args(args)

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {args.model_path} on {device} ({args.dtype}) ...")
    model, tok = load(args.model_path, device, getattr(torch, args.dtype))
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    agent = laya.load(args.laya_path)
    attackers = [QwenAttacker(model, tok, args.attributes), LayaAttacker(agent, args.attributes)]

    final, summary, converged = privacy_utility_loop(
        model, tok, attackers, args.text, args.attributes, pargs, uargs, device)
    report(args.text, final, summary, converged, args.attributes, args.threshold)


if __name__ == "__main__":
    main()
