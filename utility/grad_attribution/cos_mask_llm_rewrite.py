"""
Greedy rewrite of the REWRITTEN prompt's own tokens: the KL gradient picks
the position, the LLM proposes the replacements, and the cosine between answer
centroids picks the winner.

A cross of kl_mask_llm_rewrite.py (position choice and LLM candidates) and
cos_gcg_rewrite.py (cosine scoring with common random numbers). Each step:

    1. Backprop D = KL(P_original || P_rewritten), teacher-forced with one
       answer sampled from the original, into the one-hot of the rewritten
       prompt's content tokens, and rank the positions by |grad * input|
       (token_gradients, exactly as kl_mask_llm_rewrite.py).
    2. MASK the highest-impact token and ask the LLM for single-word fillers,
       showing it only the masked rewritten sentence. Keep the fillers that
       are exactly one token (llm_candidates / words_to_single_tokens).
    3. Sample K answers for the current prompt and for every candidate with
       the SAME seed (common random numbers), embed them, and compute
       cos_centroid against a fixed set of original-answer embeddings.
    4. Accept the best candidate if it beats the current prompt's cosine under
       that seed by more than --min_gain. Otherwise try the next-ranked
       position (up to --max_positions per step); if none improve, stop.
    5. Repeat for --n_steps tokens.

KL is used only to choose where to edit, never to choose what goes there.
Cosine is the expensive step (K sampled answers per candidate), so
--max_cos_candidates can cap how many LLM proposals reach it; the cap keeps
the ones with the most negative linear gradient estimate grad[pos, cand],
which costs nothing extra. The default 0 scores every proposal.

The seed changes every step, and the final report re-scores the initial and
final prompts on --n_eval_seeds fresh seeds, since the max over noisy
candidate scores is biased upwards.
"""
import argparse

import torch

# Importing these puts utility/metrics/ and this directory on sys.path.
from kl_mask_llm_rewrite import masked_text, llm_candidates, words_to_single_tokens
from cos_gcg_rewrite import cos_of, final_report, original_embeddings
from kl_gcg_rewrite import DEFAULT_MODEL, build_ids_with_answer, kl_of
from kl_token_grad import token_gradients
from kl_grad_attribution import content_span
from grad_based_utility_score import load


def get_args():
    p = argparse.ArgumentParser(
        description="Greedily edit the REWRITTEN prompt: KL gradient picks the "
                    "position, the LLM proposes fillers for the masked slot, "
                    "centroid cosine over sampled answers picks the winner")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--original", default="I am a 45-year-old male doctor living in New York. "
                                         "What are the best ways to manage stress?")
    p.add_argument("--rewritten", default="What are some good general strategies for "
                                          "managing everyday stress?")
    p.add_argument("--n_steps", type=int, default=10, help="tokens to edit, one per step")
    p.add_argument("--max_positions", type=int, default=3,
                   help="positions tried per step (by rank) before giving up on the step")
    # LLM candidate proposal, as in kl_mask_llm_rewrite.py
    p.add_argument("--n_candidates", type=int, default=20,
                   help="fillers the LLM is asked for per mask prompt")
    p.add_argument("--cand_rounds", type=int, default=2,
                   help="independent sampled generations pooled per mask (1 = greedy only)")
    p.add_argument("--cand_temperature", type=float, default=0.9)
    p.add_argument("--cand_max_new_tokens", type=int, default=160)
    p.add_argument("--max_cos_candidates", type=int, default=0,
                   help="cap on proposals scored by cosine, keeping the most negative "
                        "grad[pos, cand]; 0 = score all")
    # cosine scoring, as in cos_gcg_rewrite.py
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
    p.add_argument("--batch_size", type=int, default=8, help="embedding batch size")
    p.add_argument("--dtype", choices=["float16", "float32"], default="float16")
    p.add_argument("--seed", type=int, default=23)
    return p.parse_args()


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

    current_D = kl_of(model, live_ids, ref_logits_p, loss_slice)
    print(f"step 000  D={current_D:.4f} (initial)  {tok.decode(live_ids[content_slice])!r}")

    for step in range(1, args.n_steps + 1):
        grad = token_gradients(
            model, live_ids, ref_ids,
            content_slice, target_slice, ref_target_slice,
            loss_slice, ref_loss_slice,
        )  # [content_len, vocab]

        content_ids = live_ids[content_slice]
        grad_x_input = grad[torch.arange(grad.shape[0], device=grad.device), content_ids]
        ranked = grad_x_input.abs().argsort(descending=True).tolist()

        # One seed per step, shared by the current prompt and every candidate
        # at every position tried this step, so all comparisons are paired.
        seed = args.seed + step
        current_cos = cos_of(model, tok, live_ids[:P], E_orig, seed, args)
        print(f"step {step:03d}  current cos={current_cos:.4f} (seed {seed})")

        accepted = False
        for i in ranked[:args.max_positions]:
            position = s + i
            old_tok_id = live_ids[position].item()
            old_tok = tok.convert_ids_to_tokens([old_tok_id])[0]

            text = masked_text(tok, live_ids, content_slice, position)
            words = llm_candidates(model, tok, text, args.n_candidates, args.cand_rounds,
                                   args.cand_temperature, args.cand_max_new_tokens)
            cand_ids, cand_words, dropped = words_to_single_tokens(tok, words, old_tok_id)
            print(f"  pos={position} {old_tok!r} g*x={grad_x_input[i].item():+.4f}  "
                  f"mask={text!r}\n    LLM proposed {len(words)} words, "
                  f"{len(cand_ids)} single-token, {dropped} multi-token dropped")
            if not cand_ids:
                continue

            cand_toks = torch.tensor(cand_ids, device=device)
            grad_score = grad[i, cand_toks]  # linear estimate of dD for each swap
            keep = grad_score.argsort().tolist()
            if args.max_cos_candidates > 0:
                keep = keep[:args.max_cos_candidates]

            cand_cos = {}
            for j in keep:
                cand_prompt = live_ids[:P].clone()
                cand_prompt[position] = cand_toks[j]
                cand_cos[j] = cos_of(model, tok, cand_prompt, E_orig, seed, args)
            order = sorted(cand_cos, key=cand_cos.get, reverse=True)
            best = order[0]

            shown = ", ".join(f"{cand_words[j]}(cos={cand_cos[j]:.4f}, g={grad_score[j].item():+.3f})"
                              for j in order[:8])
            print(f"    top candidates: {shown}\n")

            gain = cand_cos[best] - current_cos
            if gain <= args.min_gain:
                print(f"    no candidate beats the current cos ({cand_cos[best]:.4f}, "
                      f"{gain:+.4f}); next position\n")
                continue

            new_tok_id = cand_toks[best].item()
            live_ids[position] = new_tok_id
            current_D = kl_of(model, live_ids, ref_logits_p, loss_slice)
            new_tok = tok.convert_ids_to_tokens([new_tok_id])[0]
            print(f"step {step:03d}  ACCEPT  cos {current_cos:.4f} -> {cand_cos[best]:.4f} "
                  f"(+{gain:.4f})  D={current_D:.4f}  pos={position}  {old_tok!r} -> {new_tok!r}  "
                  f"content={tok.decode(live_ids[content_slice])!r}\n")
            accepted = True
            break

        if not accepted:
            print(f"step {step:03d}  no improving swap in the top {args.max_positions} positions; stopping\n")
            break

    final_report(model, tok, initial_prompt, live_ids[:P], E_orig, args)
    return tok.decode(live_ids[content_slice])


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
