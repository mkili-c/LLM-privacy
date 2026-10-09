"""
Privacy-utility loop: alternate the privacy rewriter and a utility rewriter
until the utility-repaired prompt no longer leaks.

    x_0 = the user's message
    for i = 1 .. --max_iters:
        1. p_i  = privacy_rewrite(x_{i-1})
                  privacy/privacy_rewrite.py: span occlusion + LLM
                  generalization against Qwen and Laya, with its own inner
                  loop (--privacy_max_iters) and the same --threshold
        2. u_i  = utility_rewrite(reference = x_0, rewritten = p_i)
                  utility/grad_attribution/: pull P(A | p_i) back toward
                  P(A | x_0) (--utility kl_multi | cos_mask). Spans the
                  privacy step deleted reach it as [MASK] gaps; kl_multi
                  refills them with phrases if that lowers D, cos_mask drops them
        3. leak_i = max_a max_attacker P_a(u_i)
                  stop if leak_i <= --threshold, else x_i = u_i
    after the last iteration:
        4. fluency fix: the LLM proposes minimal grammar fixes of the final
           text (--fix_candidates). A fix is kept only if it has fewer
           surprisal spikes (disfluency), leaks no more than
           max(leak, --threshold) and loses no more utility than
           max(D, --kl_threshold); the most fluent such fix wins. kl_multi only,
           since the check needs its D.

The utility rewriters are free to re-introduce private details (they do not
see the attackers), which is why step 3 measures u_i and not p_i. Both
rewriters reject any edit whose surprisal spikes exceed min(spikes of the
current text, spikes of x_0) + ε nats (--privacy_fluency_eps, and --fluency_eps
in --utility_args). The loop computes spikes(x_0) once and passes it to both.
The x_0 bound stops small edits from adding up; the current-text bound stops
spikes freed by removing a rare word from paying for broken grammar later.

The loop also keeps one tried list per prompt and passes it to every privacy
step: a span that failed for an attribute (every candidate broke fluency, or
none lowered the hinge) is not tried again in a later iteration.

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
from laya_probe import PRIVACY_LAYA                                    # noqa: E402
import kl_mask_llm_rewrite_multi as kl_multi                           # noqa: E402
import cos_mask_llm_rewrite as cos_mask                                # noqa: E402
from kl_mask_llm_rewrite import MASK                                   # noqa: E402
from cos_gcg_rewrite import original_embeddings                        # noqa: E402
from topk_sample import topk_answers                                   # noqa: E402
from grad_based_utility_score import load                              # noqa: E402
from kl_utility_scoring import disfluency                              # noqa: E402

FIX_PROMPT = (
    "The message below was written by a user to an AI assistant. Words in it were "
    "removed or replaced, so it may contain grammar mistakes or broken phrases.\n"
    "Rewrite it with the fewest changes that make it fluent and grammatical. Keep its "
    "meaning and its request, and do not add any new details.\n"
    "Output only the corrected message, nothing else.\n\n"
    "Message: {message}"
)
FIX_TEMPERATURE = 0.7       # for the sampled fixes; the first one is greedy
FIX_MAX_NEW_TOKENS = 200


# ---------------------------------------------------------------- utility step

def kl_multi_answers(model, tok, reference, device, uargs):
    """--k non-empty answers sampled from the reference: the ones D is teacher-forced on."""
    return [a for a, _, _ in topk_answers(
        reference, model, tok, k=uargs.k, mode="sample",
        max_new_tokens=uargs.answer_max_new_tokens, temperature=1.0, top_p=1.0) if a.strip()]


def utility_kl_multi(model, tok, reference, answers, rewritten, device, uargs, x0_spikes):
    """KL gradient picks positions, LLM proposes fillers, mean KL over the
    reference answers decides; no edit may exceed the fluency budget of x_0."""
    if not answers:
        print("  utility: no non-empty reference answers; rewrite left unchanged")
        return rewritten.replace(MASK, "")
    return kl_multi.run(model, tok, reference, rewritten, answers, device, uargs, x0_spikes)


def utility_cos_mask(model, tok, reference, ref_samples, rewritten, device, uargs, x0_spikes):
    """KL gradient picks positions, LLM proposes fillers, answer-centroid
    cosine against the reference decides. Gaps are not filled, just dropped.
    ref_samples = (answers, E_ref) of original_embeddings. x0_spikes is unused:
    cos_mask has no fluency guard."""
    answers, E_ref = ref_samples
    rewritten = rewritten.replace(MASK, "")
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


# ---------------------------------------------------------------- fluency fix

@torch.no_grad()
def llm_fluency_fixes(model, tok, text, n):
    """n minimal grammar fixes of text proposed by the LLM: one greedy, n - 1 sampled.
    Each fix is the first non-empty line of a generation, deduped."""
    chat = tok.apply_chat_template([{"role": "user", "content": FIX_PROMPT.format(message=text)}],
                                   tokenize=False, add_generation_prompt=True)
    inputs = tok(chat, return_tensors="pt", add_special_tokens=False).to(model.device)
    outs = [model.generate(**inputs, max_new_tokens=FIX_MAX_NEW_TOKENS, do_sample=False,
                           pad_token_id=tok.pad_token_id)]
    if n > 1:
        outs.append(model.generate(**inputs, max_new_tokens=FIX_MAX_NEW_TOKENS, do_sample=True,
                                   temperature=FIX_TEMPERATURE, num_return_sequences=n - 1,
                                   pad_token_id=tok.pad_token_id))
    fixes = []
    for out in outs:
        for row in tok.batch_decode(out[:, inputs.input_ids.shape[1]:], skip_special_tokens=True):
            lines = [line.strip() for line in row.splitlines() if line.strip()]
            fix = lines[0].removeprefix("Message:").strip().strip('"').strip() if lines else ""
            if fix and fix != text and fix not in fixes:
                fixes.append(fix)
    return fixes


def fluency_fix(model, tok, attackers, reference, answers, text, device, uargs, args):
    """Keep the most fluent LLM fix of text that has fewer surprisal spikes,
    leaks no more than max(text's leak, --threshold) and has a D no higher than
    max(text's D, --kl_threshold). -> the fix, or text if none passes."""
    dis = disfluency([text], model, tok)[0]
    if dis == 0 or args.fix_candidates == 0:
        print(f"  no fix: spikes {dis:.1f}, --fix_candidates {args.fix_candidates}")
        return text
    texts = [text] + llm_fluency_fixes(model, tok, text, args.fix_candidates)
    dis_t = [dis] + disfluency(texts[1:], model, tok)
    _, P = score(attackers, texts)
    leak = P.amax(1).tolist()                                # max over attributes and attackers
    _, refs = kl_multi.prepare_answers(model, tok, reference, text, answers, device)
    D = [kl_multi.text_kl(model, tok, t, answers, refs, device) for t in texts]

    print(f"  {'spikes':>6} {'leak':>6} {'D':>6}")
    print(f"  {dis_t[0]:6.1f} {leak[0]:6.3f} {D[0]:6.3f}  now  {text!r}")
    passing = []
    for k in range(1, len(texts)):
        ok = (dis_t[k] < dis_t[0] and leak[k] <= max(leak[0], args.threshold)
              and D[k] <= max(D[0], uargs.kl_threshold))
        print(f"  {dis_t[k]:6.1f} {leak[k]:6.3f} {D[k]:6.3f}  {'ok  ' if ok else 'drop'} {texts[k]!r}")
        if ok:
            passing.append(k)
    if not passing:
        print("  no fix passes: text kept")
        return text
    return texts[min(passing, key=lambda k: dis_t[k])]


# ---------------------------------------------------------------- main loop

def privacy_utility_loop(model, tok, attackers, text, attributes, utility, pargs, uargs, device, args):
    """-> (final text, history, converged). history has one dict per iteration."""
    sample_reference, rewrite_utility, _ = UTILITY[utility]
    x, u, history, ref_samples, converged = text, text, [], None, False
    x0_spikes = disfluency([text], model, tok)[0]      # both steps' fluency bound: min(current, this) + ε
    tried = {}                                         # privacy step's failed (span, attribute)s, kept across iterations
    for i in range(1, args.max_iters + 1):
        print(f"\n{'#' * 30} iteration {i}: privacy step {'#' * 30}")
        p = privacy_rewrite.rewrite(model, tok, attackers, x, attributes, pargs, x0_spikes, tried)
        leak_p, _ = leakage(attackers, p)

        reference = x if args.utility_ref == "iteration" else text
        print(f"\n{'#' * 30} iteration {i}: utility step ({utility}) {'#' * 30}")
        # x_0 never changes, so its samples are drawn once and every threshold
        # decision is made against the same answers; an iteration reference is
        # a new prompt each time. An empty sample is not kept, it is redrawn.
        if args.utility_ref == "iteration" or not ref_samples:
            print(f"  sampling the reference answers ({args.utility_ref} reference)")
            ref_samples = sample_reference(model, tok, reference, device, uargs)
        u = rewrite_utility(model, tok, reference, ref_samples, p.replace(SLOT, MASK), device, uargs,
                            x0_spikes)
        leak_u, per = leakage(attackers, u)

        history.append(dict(iteration=i, input=x, privacy=p, utility=u,
                            leak_privacy=leak_p, leak_utility=leak_u, per_attacker=per))
        print(f"\n  iteration {i}: leak after privacy {leak_p:.3f}, after utility {leak_u:.3f} "
              f"(threshold {args.threshold})")
        if leak_u <= args.threshold:
            converged = True
            break
        if u == x:
            print("  iteration changed nothing: stop")
            break
        x = u

    # the fix checks utility with kl_multi's D on the same reference answers
    if utility == "kl_multi" and ref_samples:
        print(f"\n{'#' * 30} fluency fix {'#' * 30}")
        u = fluency_fix(model, tok, attackers, reference, ref_samples, u, device, uargs, args)
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
    p.add_argument("--max_iters", type=int, default=6, help="privacy + utility iterations")
    p.add_argument("--utility", choices=sorted(UTILITY), default="kl_multi")
    p.add_argument("--utility_ref", choices=["user", "iteration"], default="user",
                   help="reference prompt of the utility step (default: %(default)s), "
                        "see the module docstring")
    p.add_argument("--utility_args", default="",
                   help="extra arguments for the utility script's own parser, e.g. \"--n_steps 5\"")
    # privacy_rewrite.rewrite reads these fields of its args
    p.add_argument("--privacy_top_m", type=int, default=5,
                   help="cap on spans the privacy step rewrites per iteration (one per leaking attribute)")
    p.add_argument("--privacy_max_n", type=int, default=4)
    p.add_argument("--privacy_span_penalty", type=float, default=0.05)
    p.add_argument("--privacy_max_iters", type=int, default=8)
    p.add_argument("--privacy_n_candidates", type=int, default=10)
    p.add_argument("--privacy_cand_rounds", type=int, default=2)
    p.add_argument("--privacy_cand_temperature", type=float, default=0.9)
    p.add_argument("--privacy_min_hinge_drop", type=float, default=0.05,
                   help="a privacy edit must lower the hinge by at least min(this, the hinge left)")
    p.add_argument("--privacy_fluency_eps", type=float, default=4.0,
                   help="fluency guard of the privacy step: an edit may have at most min(spikes of "
                        "the current text, spikes of x_0) + this many nats (disfluency)")
    p.add_argument("--fix_candidates", type=int, default=4,
                   help="LLM fluency fixes after the last iteration (1 greedy + n - 1 sampled); 0 = none")
    p.add_argument("--model_path", default=DEFAULT_MODEL)
    p.add_argument("--laya_path", default=PRIVACY_LAYA, help="Laya attacker (default: the fine-tuned one)")
    p.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    p.add_argument("--seed", type=int, default=23)
    args = p.parse_args()

    pargs = Namespace(threshold=args.threshold, top_m=args.privacy_top_m,
                      max_n=args.privacy_max_n, span_penalty=args.privacy_span_penalty,
                      max_iters=args.privacy_max_iters, n_candidates=args.privacy_n_candidates,
                      cand_rounds=args.privacy_cand_rounds,
                      cand_temperature=args.privacy_cand_temperature,
                      fluency_eps=args.privacy_fluency_eps, min_hinge_drop=args.privacy_min_hinge_drop)
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
