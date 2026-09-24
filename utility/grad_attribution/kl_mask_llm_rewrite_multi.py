"""
kl_mask_llm_rewrite.py, but with the objective averaged over K teacher-forced
answers instead of a single one.

With one answer, D and its gradient describe how the rewrite changes the
probability of THAT answer only, so the rewrite can overfit to it. Here K
answers a_1..a_K are sampled from P_original (or passed with --targets) and

    D = (1/K) * sum_k KL(P_original(. | a_k) || P_rewritten(. | a_k))

each term teacher-forced exactly as in the single-answer script.

Each step:
    1. For every answer, backprop its KL term into the one-hot of every
       CONTENT token of the rewritten prompt (token_gradients). The answer is
       appended after the prompt, so the content positions are identical for
       every answer and the K gradients line up position by position.
    2. Average the K gradients. grad*input is the one_hot.grad entry at a
       position's own token, which is linear in the gradient, so this is
       exactly the average of the K per-answer grad*input scores (signed,
       so a token pushing D up for some answers and down for others is
       ranked by its net effect).
    3. Rank content tokens by |avg grad*input|, MASK the top one, ask the LLM
       for single-word fillers (same prompt and parsing as
       kl_mask_llm_rewrite.py; only the masked REWRITTEN sentence is shown).
    4. Select among the single-token candidates by the averaged objective:
           --select forward  real forward pass per candidate per answer,
                             argmin of the mean D
           --select grad     argmin of the averaged gradient at the candidate
       If nothing lowers the mean D, fall through to the next-ranked position
       (up to --max_positions); if none improve, stop.
    5. Repeat for --n_steps tokens.

Cost per step is K backward passes and K forward passes per candidate.
"""
import argparse
import sys
from pathlib import Path

import torch

# Same path setup as kl_mask_llm_rewrite.py: utility/metrics/ for the shared
# sampling and scoring helpers, this directory for the grad_attribution modules.
_HERE = Path(__file__).resolve().parent
for _dir in (_HERE.parent / "metrics", _HERE):
    if str(_dir) not in sys.path:
        sys.path.insert(0, str(_dir))

from topk_sample import topk_answers
from grad_based_utility_score import load
from kl_grad_attribution import content_span

from kl_token_grad import token_gradients
from kl_gcg_rewrite import build_ids_with_answer, kl_of, evaluate_candidates
from kl_mask_llm_rewrite import masked_text, llm_candidates, words_to_single_tokens

DEFAULT_MODEL = "/work/projects/mkilic/models/Qwen2.5-3B-Instruct"


def get_args():
    p = argparse.ArgumentParser(
        description="Greedily edit the highest-impact token of the REWRITTEN prompt, "
                    "with impact and candidate selection averaged over K answers and "
                    "replacement candidates proposed by the LLM for a masked slot")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--original", default="I am a 45-year-old male doctor living in New York. "
                                         "What are the best ways to manage stress?")
    p.add_argument("--rewritten", default="What are some good general strategies for "
                                          "managing everyday stress?")
    p.add_argument("--k", type=int, default=4,
                   help="number of answers sampled from P_orig to average over")
    p.add_argument("--targets", nargs="+", default=None,
                   help="fixed answers to teacher-force (overrides --k sampling)")
    p.add_argument("--answer_max_new_tokens", type=int, default=64)
    p.add_argument("--n_steps", type=int, default=10, help="tokens to edit, one per step")
    p.add_argument("--n_candidates", type=int, default=20,
                   help="fillers the LLM is asked for per mask prompt")
    p.add_argument("--cand_rounds", type=int, default=2,
                   help="independent sampled generations pooled per mask (1 = greedy only)")
    p.add_argument("--cand_temperature", type=float, default=0.9)
    p.add_argument("--cand_max_new_tokens", type=int, default=160)
    p.add_argument("--select", choices=["forward", "grad"], default="forward",
                   help="forward: check every candidate with real forward passes (GCG); "
                        "grad: rank by the averaged linear gradient estimate only")
    p.add_argument("--max_positions", type=int, default=3,
                   help="positions tried per step (by rank) before giving up on the step")
    p.add_argument("--eval_chunk_size", type=int, default=32)
    p.add_argument("--dtype", choices=["float16", "float32"], default="float16")
    p.add_argument("--seed", type=int, default=23)
    return p.parse_args()


# ------------------------------------------------------------ per-answer setup

def prepare_answers(model, tok, original, rewritten, answers, device):
    """Per answer: the fixed reference sequence and its log-probs, plus the
    answer ids and slices for the live sequence. The live PROMPT ids are the
    same for every answer and are returned once, so edits apply to all."""
    refs, live_prompt, P = [], None, None
    for answer in answers:
        ref_ids, ref_target_slice, ref_loss_slice = build_ids_with_answer(tok, original, answer)
        ref_ids = torch.tensor(ref_ids, device=device)
        with torch.no_grad():
            ref_logits_p = torch.log_softmax(
                model(input_ids=ref_ids.unsqueeze(0)).logits[0, ref_loss_slice, :].float(),
                dim=-1,
            )
        live_ids, target_slice, loss_slice = build_ids_with_answer(tok, rewritten, answer)
        if live_prompt is None:
            P = target_slice.start
            live_prompt = torch.tensor(live_ids[:P], device=device)
        refs.append(dict(
            ref_ids=ref_ids, ref_target_slice=ref_target_slice, ref_loss_slice=ref_loss_slice,
            ref_logits_p=ref_logits_p,
            answer_ids=torch.tensor(live_ids[P:], device=device),
            target_slice=target_slice, loss_slice=loss_slice,
        ))
    return live_prompt, refs


def with_answer(live_prompt, r):
    return torch.cat([live_prompt, r["answer_ids"]])


def mean_kl(model, live_prompt, refs):
    return sum(kl_of(model, with_answer(live_prompt, r), r["ref_logits_p"], r["loss_slice"])
               for r in refs) / len(refs)


def mean_gradients(model, live_prompt, refs, content_slice):
    """Average of the per-answer one-hot gradients, [content_len, vocab]."""
    total = None
    for r in refs:
        g = token_gradients(
            model, with_answer(live_prompt, r), r["ref_ids"],
            content_slice, r["target_slice"], r["ref_target_slice"],
            r["loss_slice"], r["ref_loss_slice"],
        ).float()
        total = g if total is None else total + g
    return total / len(refs)


def mean_candidate_kl(model, live_prompt, refs, position, cand_toks, eval_chunk_size):
    """Mean over answers of D after substituting each candidate at `position`."""
    total = torch.zeros(len(cand_toks), device=cand_toks.device)
    for r in refs:
        total += evaluate_candidates(model, with_answer(live_prompt, r), r["ref_logits_p"],
                                     position, r["loss_slice"], cand_toks, eval_chunk_size)
    return total / len(refs)


# ------------------------------------------------------------------ main loop

def run(model, tok, original, rewritten, answers, device, args):
    live_prompt, refs = prepare_answers(model, tok, original, rewritten, answers, device)
    s, e = content_span(tok, rewritten)
    content_slice = slice(s, e)

    current_D = mean_kl(model, live_prompt, refs)
    print(f"step 000  D={current_D:.4f} (initial, mean over {len(refs)} answers)  "
          f"{tok.decode(live_prompt[content_slice])!r}")

    for step in range(1, args.n_steps + 1):
        grad = mean_gradients(model, live_prompt, refs, content_slice)  # [content_len, vocab]

        content_ids = live_prompt[content_slice]
        grad_x_input = grad[torch.arange(grad.shape[0], device=grad.device), content_ids]
        ranked = grad_x_input.abs().argsort(descending=True).tolist()

        accepted = False
        for i in ranked[:args.max_positions]:
            position = s + i
            old_tok_id = live_prompt[position].item()
            old_tok = tok.convert_ids_to_tokens([old_tok_id])[0]

            text = masked_text(tok, live_prompt, content_slice, position)
            words = llm_candidates(model, tok, text, args.n_candidates, args.cand_rounds,
                                   args.cand_temperature, args.cand_max_new_tokens)
            cand_ids, cand_words, dropped = words_to_single_tokens(tok, words, old_tok_id)
            print(f"  pos={position} {old_tok!r} avg g*x={grad_x_input[i].item():+.4f}  "
                  f"mask={text!r}\n    LLM proposed {len(words)} words, "
                  f"{len(cand_ids)} single-token, {dropped} multi-token dropped")
            if not cand_ids:
                continue

            cand_toks = torch.tensor(cand_ids, device=device)
            grad_score = grad[i, cand_toks]  # linear estimate of d(mean D) for each swap
            if args.select == "grad":
                order = grad_score.argsort()
                best = order[0]
                # forward passes on the chosen token only, to know the real mean D
                new_D = mean_candidate_kl(model, live_prompt, refs, position,
                                          cand_toks[best:best + 1], args.eval_chunk_size)[0].item()
            else:
                cand_D = mean_candidate_kl(model, live_prompt, refs, position,
                                           cand_toks, args.eval_chunk_size)
                order = cand_D.argsort()
                best = order[0]
                new_D = cand_D[best].item()

            shown = ", ".join(
                f"{cand_words[j]}(g={grad_score[j].item():+.3f}"
                + (f", D={cand_D[j].item():.3f})" if args.select == "forward" else ")")
                for j in order[:8].tolist())
            print(f"    top candidates: {shown} \n")

            if new_D >= current_D:
                print(f"    no candidate improves D ({new_D:.4f} >= {current_D:.4f}); next position \n")
                continue

            new_tok_id = cand_toks[best].item()
            live_prompt[position] = new_tok_id
            current_D = new_D
            new_tok = tok.convert_ids_to_tokens([new_tok_id])[0]
            print(f"step {step:03d}  D={current_D:.4f}  pos={position}  "
                  f"{old_tok!r} -> {new_tok!r}  content={tok.decode(live_prompt[content_slice])!r} \n")
            accepted = True
            break

        if not accepted:
            print(f"step {step:03d}  no improving swap in the top {args.max_positions} positions; stopping \n")
            break

    return tok.decode(live_prompt[content_slice])


if __name__ == "__main__":
    args = get_args()
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = getattr(torch, args.dtype)

    print(f"Loading {args.model} on {device} ({args.dtype}) ...")
    model, tok = load(args.model, device, dtype)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    if args.targets is not None:
        answers = args.targets
    else:
        answers = [a for a, _, _ in topk_answers(
            args.original, model, tok, k=args.k, mode="sample",
            max_new_tokens=args.answer_max_new_tokens, temperature=1.0, top_p=1.0) if a.strip()]
    if not answers:
        sys.exit("no non-empty answers to teacher-force")

    print(f"\nORIGINAL : {args.original}")
    print(f"REWRITTEN: {args.rewritten}")
    for j, a in enumerate(answers):
        print(f"TARGET ANSWER {j}: {a}")
    print()

    final_rewrite = run(model, tok, args.original, args.rewritten, answers, device, args)
    print(f"\nfinal rewrite: {final_rewrite!r}")
