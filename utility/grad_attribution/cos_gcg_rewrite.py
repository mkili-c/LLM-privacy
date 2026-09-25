"""
Hybrid of kl_gcg_rewrite.py and the centroid-cosine metric: the KL gradient
proposes token swaps, the cosine between answer centroids decides them.

The cosine metric embeds every answer standalone (cosine_utility.embed_answers),
so a teacher-forced answer's embedding does not depend on the prompt and
backprop-ing cosine into the prompt one-hot gives exactly zero. The prompt
moves cosine only through which answers get sampled. So the gradient and the
cheap shortlist stay on KL, and cosine is only paid for on the survivors:

    1. Backprop D = KL(P_original || P_rewritten), teacher-forced with one
       answer sampled from the original, into the one-hot of the rewritten
       prompt's content tokens (token_gradients, exactly as kl_gcg_rewrite.py).
    2. Pick the unblocked position with the largest |grad * input| and take
       the --topk most-negative-gradient replacement tokens.
    3. Score every candidate with one exact forward pass of D and keep the
       --shortlist lowest.
    4. For the current prompt and each shortlisted candidate, sample K answers
       with the SAME seed (common random numbers, so the candidates see
       coupled sampling noise and the comparison is paired), embed them, and
       compute cos_centroid against a fixed set of original-answer embeddings.
    5. Accept the best candidate only if it beats the current prompt's cosine
       under that seed by more than --min_gain. Otherwise block the position
       until the next accepted edit, so the deterministic gradient does not
       propose the same position forever.

The seed changes every step, so no single draw of sampling noise is optimized
against. Winner's curse still applies (the max over a shortlist of noisy
scores is biased upwards), so the final report re-scores the initial and final
prompts on --n_eval_seeds fresh seeds.
"""
import argparse
import statistics

import torch

# Importing kl_gcg_rewrite puts utility/metrics/ on sys.path as a side effect,
# so the metric helpers below resolve by bare name after it.
from kl_gcg_rewrite import (DEFAULT_MODEL, build_ids_with_answer, kl_of,
                            evaluate_candidates)
from kl_token_grad import token_gradients
from kl_grad_attribution import content_span
from grad_based_utility_score import load
from kl_utility_scoring import build_prompt_ids
from cosine_utility import embed_answers, cosine_stats


def get_args():
    p = argparse.ArgumentParser(
        description="Greedily edit the REWRITTEN prompt: KL gradient proposes "
                    "swaps, centroid cosine over sampled answers accepts them")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--original", default="I am a 45-year-old male doctor living in New York. "
                                         "What are the best ways to manage stress?")
    p.add_argument("--rewritten", default="What are some good general strategies for "
                                          "managing everyday stress?")
    p.add_argument("--n_steps", type=int, default=10, help="edit attempts, one position each")
    p.add_argument("--topk", type=int, default=64, help="gradient candidates scored by KL")
    p.add_argument("--shortlist", type=int, default=8,
                   help="lowest-KL candidates re-ranked by cosine")
    p.add_argument("--k", type=int, default=15, help="answers sampled per rewritten prompt")
    p.add_argument("--k_orig", type=int, default=30,
                   help="answers sampled once from the original for the fixed centroid")
    p.add_argument("--max_new_tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top_p", type=float, default=1.0)
    p.add_argument("--layer", type=int, default=-1, help="hidden layer used for the embedding")
    p.add_argument("--min_gain", type=float, default=0.0,
                   help="cosine margin a candidate must beat the current prompt by")
    p.add_argument("--n_eval_seeds", type=int, default=3,
                   help="fresh seeds for the final initial-vs-final comparison")
    p.add_argument("--eval_chunk_size", type=int, default=None,
                   help="candidates per KL forward pass; default scales with the "
                        "answer length to kl_gcg_rewrite.py's budget of 32 x 64 "
                        "answer tokens, since full-vocab log-probs are what run "
                        "the GPU out of memory")
    p.add_argument("--batch_size", type=int, default=8, help="embedding batch size")
    p.add_argument("--dtype", choices=["float16", "float32"], default="float16")
    p.add_argument("--seed", type=int, default=23)
    return p.parse_args()


def sample_answers(model, tok, prompt_ids, k, max_new_tokens, temperature, top_p, seed):
    """K sampled answers straight from prompt token ids. Works in id space
    because a swapped candidate need not survive a decode/re-tokenize round
    trip. Re-seeding right before generate is what makes the draws common
    random numbers across candidates."""
    torch.manual_seed(seed)
    ids = prompt_ids.unsqueeze(0)
    with torch.no_grad():
        out = model.generate(
            input_ids=ids, attention_mask=torch.ones_like(ids),
            do_sample=True, temperature=temperature, top_p=top_p,
            max_new_tokens=max_new_tokens, num_return_sequences=k,
            pad_token_id=tok.pad_token_id,
        )
    texts = tok.batch_decode(out[:, ids.shape[1]:], skip_special_tokens=True)
    return [t for t in texts if t.strip()]


def cos_of(model, tok, prompt_ids, E_orig, seed, args):
    """cos_centroid of the prompt's sampled answers against the fixed original
    embeddings; -inf if too few non-empty answers for the self terms."""
    answers = sample_answers(model, tok, prompt_ids, args.k, args.max_new_tokens,
                             args.temperature, args.top_p, seed)
    if len(answers) < 2:
        return float("-inf")
    E_rew = embed_answers(answers, model, tok, args.batch_size, args.layer)
    return cosine_stats(E_orig, E_rew)[0]["cos_centroid"]


def run(model, tok, original, rewritten, answer, E_orig, device, args):
    ref_ids, ref_target_slice, ref_loss_slice = build_ids_with_answer(tok, original, answer)
    ref_ids = torch.tensor(ref_ids, device=device)
    with torch.no_grad():
        ref_logits_p = torch.log_softmax(
            model(input_ids=ref_ids.unsqueeze(0)).logits[0, ref_loss_slice, :].float(),
            dim=-1,
        )

    live_ids, target_slice, loss_slice = build_ids_with_answer(tok, rewritten, answer)
    live_ids = torch.tensor(live_ids, device=device)
    P = target_slice.start          # prompt length; answers are sampled after it
    s, e = content_span(tok, rewritten)
    content_slice = slice(s, e)
    initial_prompt = live_ids[:P].clone()
    answer_len = target_slice.stop - target_slice.start
    eval_chunk_size = args.eval_chunk_size or max(1, (32 * 64) // answer_len)

    current_D = kl_of(model, live_ids, ref_logits_p, loss_slice)
    print(f"step 000  D={current_D:.4f} (initial)  {tok.decode(live_ids[content_slice])!r}")

    blocked = set()                 # content offsets rejected since the last accepted edit
    for step in range(1, args.n_steps + 1):
        grad = token_gradients(
            model, live_ids, ref_ids,
            content_slice, target_slice, ref_target_slice,
            loss_slice, ref_loss_slice,
        )  # [content_len, vocab]

        content_ids = live_ids[content_slice]
        grad_x_input = grad[torch.arange(grad.shape[0], device=grad.device), content_ids]
        impact = grad_x_input.abs()
        if blocked:
            impact[list(blocked)] = -1.0
        if impact.max() < 0:
            print(f"step {step:03d}  every content position rejected since the last edit; stopping")
            break
        i = impact.argmax().item()
        position = s + i

        candidate_toks = (-grad[i]).topk(args.topk).indices
        candidate_toks = candidate_toks[candidate_toks != live_ids[position]]
        cand_D = evaluate_candidates(model, live_ids, ref_logits_p, position, loss_slice,
                                     candidate_toks, eval_chunk_size)
        order = cand_D.argsort()[:args.shortlist]

        # One seed per step, shared by the current prompt and every candidate.
        seed = args.seed + step
        current_cos = cos_of(model, tok, live_ids[:P], E_orig, seed, args)
        cand_cos = []
        for j in order.tolist():
            cand_prompt = live_ids[:P].clone()
            cand_prompt[position] = candidate_toks[j]
            cand_cos.append(cos_of(model, tok, cand_prompt, E_orig, seed, args))
        best = max(range(len(cand_cos)), key=cand_cos.__getitem__)
        j = order[best].item()

        old_tok = tok.convert_ids_to_tokens([live_ids[position].item()])[0]
        new_tok_id = candidate_toks[j].item()
        new_tok = tok.convert_ids_to_tokens([new_tok_id])[0]
        gain = cand_cos[best] - current_cos
        if gain > args.min_gain:
            live_ids[position] = new_tok_id
            current_D = cand_D[j].item()
            blocked.clear()
            print(f"step {step:03d}  ACCEPT  cos {current_cos:.4f} -> {cand_cos[best]:.4f} "
                  f"(+{gain:.4f})  D={current_D:.4f}  pos={position}  {old_tok!r} -> {new_tok!r}  "
                  f"content={tok.decode(live_ids[content_slice])!r}")
        else:
            blocked.add(i)
            print(f"step {step:03d}  reject  cos {current_cos:.4f}, best candidate "
                  f"{cand_cos[best]:.4f} ({gain:+.4f})  pos={position}  "
                  f"{old_tok!r} -> {new_tok!r} not taken")

    final_report(model, tok, initial_prompt, live_ids[:P], E_orig, args)
    return tok.decode(live_ids[content_slice])


def final_report(model, tok, initial_prompt, final_prompt, E_orig, args):
    """Re-score initial vs final prompt on fresh seeds, disjoint from the
    per-step ones, so the reported gain is free of the selection bias."""
    eval_seeds = [args.seed + 10_000 + r for r in range(args.n_eval_seeds)]
    init_cos = [cos_of(model, tok, initial_prompt, E_orig, sd, args) for sd in eval_seeds]
    final_cos = [cos_of(model, tok, final_prompt, E_orig, sd, args) for sd in eval_seeds]
    diffs = [f - i for f, i in zip(final_cos, init_cos)]
    print(f"\nre-scored on {len(eval_seeds)} fresh seeds (K={args.k}):")
    print(f"  initial cos_centroid  {statistics.mean(init_cos):.4f}  {[round(c, 4) for c in init_cos]}")
    print(f"  final   cos_centroid  {statistics.mean(final_cos):.4f}  {[round(c, 4) for c in final_cos]}")
    print(f"  paired gain           {statistics.mean(diffs):+.4f}")


def original_embeddings(model, tok, original, device, args):
    """Sample --k_orig answers from the original once and embed them: the
    fixed centroid every rewrite is scored against. Returns (answers, E_orig)."""
    orig_ids = torch.tensor(build_prompt_ids(tok, original), device=device)
    answers = sample_answers(model, tok, orig_ids, args.k_orig, args.max_new_tokens,
                             args.temperature, args.top_p, args.seed)
    if len(answers) < 2:
        raise ValueError(f"only {len(answers)} non-empty original answers; raise --k_orig")
    return answers, embed_answers(answers, model, tok, args.batch_size, args.layer)


if __name__ == "__main__":
    args = get_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = getattr(torch, args.dtype)

    print(f"Loading {args.model} on {device} ({args.dtype}) ...")
    model, tok = load(args.model, device, dtype)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    orig_answers, E_orig = original_embeddings(model, tok, args.original, device, args)
    # The KL gradient is teacher-forced on one of the same original answers.
    answer = orig_answers[0]

    print(f"\nORIGINAL : {args.original}")
    print(f"REWRITTEN: {args.rewritten}")
    print(f"TARGET ANSWER (for the KL gradient): {answer}\n")

    final_rewrite = run(model, tok, args.original, args.rewritten, answer, E_orig, device, args)
    print(f"\nfinal rewrite: {final_rewrite!r}")
