"""
Gradient attribution of the answer-distribution KL divergence.

Idea
----
Teacher-force the same answer `a` through two forward passes -- one conditioned
on the original prompt, one on the rewritten prompt -- and compare the two sets
of logits at every answer-predicting position with a full-vocabulary KL:

    D(a) = sum_t KL( P_orig(. | a_<t)  ||  P_rew(. | a_<t) )

With a ~ P(. | T_orig), D is an unbiased estimator of the *same* forward KL
that utility_scoring.py estimates from sampled log-prob differences -- but it
integrates over the whole vocabulary at each position instead of scoring the
one sampled token, so it is the lower-variance (Rao-Blackwellized) version.

D is differentiable in the live branch's input embeddings, so backprop puts a
gradient on every prompt token: how much a small change of that token's
embedding would change the divergence between the two answer distributions.

Which branch gets the gradient (--attribute)
--------------------------------------------
    rewritten  d D / d embeds(T_rew)  -- which surviving rewrite tokens are
               load-bearing for the divergence (what not to touch when editing
               further). The reference branch (original) runs under no_grad.
    original   d D / d embeds(T_orig) -- which original details the divergence
               depends on. This side sees what *deletion* destroyed: removed
               content has no token in the rewrite to receive a gradient, but
               it does have one here.
    both       run both attributions on the same answers.

Caveats (from the design discussion)
------------------------------------
* D is conditioned on the teacher-forced answer; average over several sampled
  answers (--n_answers) rather than trusting one.
* Gradients are local: token substitution is a discrete jump, so treat the
  ranking as a candidate list to verify with real forward passes, not truth.
* Prompts of different length place the same answer token at different
  positions (RoPE), so a small slice of D is positional, not semantic.
* When the rewrite barely changes the distribution, D ~ 0 and the gradients
  are numerical noise -- check the magnitude before interpreting.
* Identity check: T_rew == T_orig gives D = 0 exactly, and gradients at the
  numerical-noise floor (~1e-7): the analytic gradient q*(sum_v p_v) - p has
  its q - p part cancel bitwise, leaving only the rounding of sum_v p_v ~ 1.
  Any real signal sits orders of magnitude above that floor.

Per-position KLs are also reported: the share of divergence inside the first
answer tokens shows how much of D is the "persona echo" at the answer opening
versus differences that permeate the whole answer.
"""
import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

# utility/metrics/ holds the shared sampling and scoring helpers, which import
# each other by bare name, so that directory goes on sys.path as a whole.
# Resolved from __file__ so the script runs from any working directory.
_METRICS = Path(__file__).resolve().parents[1] / "metrics"
if str(_METRICS) not in sys.path:
    sys.path.insert(0, str(_METRICS))

from topk_sample import topk_answers
from kl_utility_scoring import build_prompt_ids
from grad_based_utility_score import load

DEFAULT_MODEL = "/work/projects/mkilic/models/Qwen2.5-3B-Instruct"


def get_args():
    p = argparse.ArgumentParser(
        description="Backprop the full-vocab KL between original-prompt and "
                    "rewritten-prompt answer logits into prompt embeddings")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--original", default="I am a 45-year-old male doctor living in New York. "
                                         "What are the best ways to manage stress?")
    p.add_argument("--rewritten", default="What are some good general strategies for "
                                          "managing everyday stress?")
    p.add_argument("--target", default=None,
                   help="fixed answer to teacher-force; if omitted, --n_answers are "
                        "sampled from the ORIGINAL prompt (a ~ P_orig keeps D an "
                        "unbiased forward-KL estimate)")
    p.add_argument("--n_answers", type=int, default=15,
                   help="answers sampled from P_orig to average the attribution over")
    p.add_argument("--max_new_tokens", type=int, default=64)
    p.add_argument("--attribute", choices=["rewritten", "original", "both"],
                   default="both", help="which prompt's embeddings receive the gradient")
    p.add_argument("--dtype", choices=["float16", "float32"], default="float16",
                   help="float32 gives cleaner gradients but needs more VRAM")
    p.add_argument("--top_k", type=int, default=12)
    p.add_argument("--show_template", action="store_true",
                   help="rank chat-template scaffolding tokens too (default: content only)")
    p.add_argument("--front_tokens", type=int, default=15,
                   help="answer-opening window for the echo-share diagnostic")
    p.add_argument("--seed", type=int, default=23)
    return p.parse_args()


def content_span(tok, prompt):
    """
    Token span [s, e) of the raw user text inside the chat-templated prompt ids,
    found by diffing against an empty-content render (common prefix + suffix are
    the template scaffolding). BPE merging at the seams can shift it by one
    token -- fine for annotating a report. No chat template -> whole prompt.
    """
    full = build_prompt_ids(tok, prompt)
    if not getattr(tok, "chat_template", None):
        return 0, len(full)
    empty = build_prompt_ids(tok, "")
    s = 0
    while s < len(empty) and s < len(full) and full[s] == empty[s]:
        s += 1
    e_full, e_empty = len(full), len(empty)
    while e_full > s and e_empty > s and full[e_full - 1] == empty[e_empty - 1]:
        e_full -= 1
        e_empty -= 1
    return s, e_full


def answer_logits(model, prompt_ids, answer_ids, device, need_grad):
    """
    Teacher-forced forward of [prompt + answer]. Returns (logits at the
    answer-predicting positions [|a|, V], inputs_embeds or None).
    Answer tokens sit at [P, P+A), predicted by logit rows [P-1, P+A-1).
    """
    ids = torch.tensor([prompt_ids + answer_ids], device=device)
    P = len(prompt_ids)
    if need_grad:
        embeds = model.get_input_embeddings()(ids).detach().clone().requires_grad_(True)
        logits = model(inputs_embeds=embeds).logits
    else:
        with torch.no_grad():
            logits = model(input_ids=ids).logits
        embeds = None
    return logits[0, P - 1:P - 1 + len(answer_ids), :], embeds


def positionwise_kl(logits_p, logits_q):
    """KL(p || q) per answer position, softmax in float32. Gradients flow into
    whichever branch is attached to the autograd graph."""
    logp = F.log_softmax(logits_p.float(), dim=-1)
    logq = F.log_softmax(logits_q.float(), dim=-1)
    return (logp.exp() * (logp - logq)).sum(dim=-1)


@torch.enable_grad()
def kl_attribution(model, tok, original, rewritten, answer, device, attribute="rewritten"):
    """
    One answer, one side. Computes D = sum_t KL(P_orig || P_rew) with the
    reference branch under no_grad, backprops D, and returns per-token gradient
    stats over the live prompt's tokens plus the per-position KLs.
    """
    if attribute not in ("rewritten", "original"):
        raise ValueError(f"attribute must be 'rewritten' or 'original', got {attribute!r}")

    ids_orig = build_prompt_ids(tok, original)
    ids_rew = build_prompt_ids(tok, rewritten)
    a_ids = tok(answer, add_special_tokens=False).input_ids
    if not a_ids:
        raise ValueError("empty answer after tokenization")

    live_is_rew = attribute == "rewritten"
    live_ids, ref_ids = (ids_rew, ids_orig) if live_is_rew else (ids_orig, ids_rew)
    logits_live, embeds = answer_logits(model, live_ids, a_ids, device, need_grad=True)
    logits_ref, _ = answer_logits(model, ref_ids, a_ids, device, need_grad=False)
    logits_p, logits_q = (logits_ref, logits_live) if live_is_rew else (logits_live, logits_ref)

    kl_pos = positionwise_kl(logits_p, logits_q)
    D = kl_pos.sum()
    model.zero_grad(set_to_none=True)
    D.backward()

    P = len(live_ids)
    grad = embeds.grad[0, :P].float()                 # prompt slice of [L, H]
    emb = embeds[0, :P].detach().float()
    return {
        "tokens": tok.convert_ids_to_tokens(live_ids),
        "prompt_len": P,
        "kl_total": D.item(),
        "kl_per_token": D.item() / len(a_ids),
        "kl_per_pos": kl_pos.detach().float().cpu(),
        "n_answer_tokens": len(a_ids),
        "grad_norm": grad.norm(dim=-1).cpu(),
        "grad_x_input": (grad * emb).sum(dim=-1).cpu(),
    }


def aggregate(results):
    """Average per-token gradient stats and KL diagnostics over answers (the
    live prompt token grid is identical across answers)."""
    return {
        "tokens": results[0]["tokens"],
        "prompt_len": results[0]["prompt_len"],
        "kl_per_token": sum(r["kl_per_token"] for r in results) / len(results),
        "kl_totals": [r["kl_total"] for r in results],
        "grad_norm": torch.stack([r["grad_norm"] for r in results]).mean(0),
        "grad_x_input": torch.stack([r["grad_x_input"] for r in results]).mean(0),
    }


def front_share(results, n_front):
    """Mean fraction of each answer's divergence carried by its first n_front
    positions -- the 'persona echo' diagnostic."""
    shares = [r["kl_per_pos"][:n_front].sum().item() / total
              for r in results if (total := r["kl_per_pos"].sum().item()) > 1e-9]
    return sum(shares) / len(shares) if shares else float("nan")


def report(agg, span, top_k, show_template):
    tokens, P = agg["tokens"], agg["prompt_len"]
    gn, gxi = agg["grad_norm"], agg["grad_x_input"]
    s, e = span

    idxs = range(P) if show_template else range(s, e)
    share_base = sum(gn[i] for i in idxs) + 1e-9
    order = sorted(idxs, key=lambda i: gn[i], reverse=True)

    print(f"  prompt tokens: {P} total, content span [{s},{e}) "
          f"({'ranking all' if show_template else 'ranking content only'})")
    print(f"  {'rank':>4}  {'idx':>4}  {'token':<18} {'grad_norm':>10}  {'share':>7}  {'grad*input':>11}")
    print("  " + "-" * 62)
    for rank, i in enumerate(order[:top_k]):
        tk = tokens[i].replace("Ġ", "·").replace("Ċ", "\\n")
        tag = "" if s <= i < e else "  [template]"
        print(f"  {rank:>4}  {i:>4}  {tk:<18} {gn[i]:>10.4f}  {gn[i]/share_base:>6.1%}  "
              f"{gxi[i]:>11.4f}{tag}")


if __name__ == "__main__":
    args = get_args()
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = getattr(torch, args.dtype)

    print(f"Loading {args.model} on {device} ({args.dtype}) ...")
    model, tok = load(args.model, device, dtype)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    if args.target is not None:
        answers = [args.target]
    else:
        # a ~ P_orig: plain sampling so D stays an unbiased forward-KL estimate
        answers = [a for a, _, _ in topk_answers(
            args.original, model, tok, k=args.n_answers, mode="sample",
            max_new_tokens=args.max_new_tokens, temperature=1.0, top_p=1.0) if a.strip()]
    if not answers:
        raise SystemExit("no non-empty answers to teacher-force")

    print(f"\nORIGINAL : {args.original}")
    print(f"REWRITTEN: {args.rewritten}")
    print(f"Teacher-forced answers: {len(answers)} "
          f"({'fixed --target' if args.target else 'sampled from P_orig'})")

    sides = ["rewritten", "original"] if args.attribute == "both" else [args.attribute]
    for side in sides:
        results = [kl_attribution(model, tok, args.original, args.rewritten,
                                  a, device, attribute=side) for a in answers]
        agg = aggregate(results)
        span = content_span(tok, args.rewritten if side == "rewritten" else args.original)

        print(f"\n=== gradients w.r.t. the {side.upper()} prompt embeddings ===")
        kls = ", ".join(f"{x:.2f}" for x in agg["kl_totals"])
        print(f"  D = KL(P_orig || P_rew): {agg['kl_per_token']:.4f} nats/token "
              f"(per-answer totals: {kls} nats)")
        print(f"  share of D in first {args.front_tokens} answer tokens: "
              f"{front_share(results, args.front_tokens):.1%}  (persona-echo diagnostic)")
        report(agg, span, args.top_k, args.show_template)



