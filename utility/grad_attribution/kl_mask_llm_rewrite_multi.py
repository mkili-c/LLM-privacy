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
    5. Repeat for --n_steps tokens, or until the mean D is <= --kl_threshold:
       a prompt that already keeps enough utility is not edited (each edit
       risks pulling private details back in), so it goes back to the
       privacy step unchanged.

Cost per step is K backward passes and K forward passes per candidate.

Gaps: every [MASK] in the rewritten prompt marks words the privacy step
deleted. Before the token steps, each gap is filled once, left to right: the
LLM sees the sentence with that gap as [MASK] (and never the deleted words)
and proposes fillers of up to --gap_max_words words, and the one with the
lowest mean D is kept if it beats leaving the gap empty. A filler may be any
number of tokens: candidates are scored as whole texts, so the sequence length
is free here, unlike in the token steps. Once D <= --kl_threshold the
remaining gaps are left empty. Unfilled gaps simply vanish.
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
from kl_mask_llm_rewrite import MASK, masked_text, llm_candidates, words_to_single_tokens

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
    p.add_argument("--gap_max_words", type=int, default=4, help="longest filler for a [MASK] gap")
    p.add_argument("--kl_threshold", type=float, default=55.0,
                   help="mean D at or below which the prompt keeps enough utility and is "
                        "not edited any further (0 = always edit); D is summed over the "
                        "answer tokens, so it scales with --answer_max_new_tokens")
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


# ------------------------------------------------------------------ gaps

def gap_prompt(text, n, max_words):
    return (
        f"In the sentence below, a few words were removed at {MASK}.\n"
        f"Propose {n} different fillers of 1 to {max_words} words for {MASK} so the "
        "sentence is grammatical and natural again. Vary the meaning: include specific "
        "and general options.\n"
        "Output one filler per line, nothing else -- no numbering, no quotes, "
        "no explanations.\n\n"
        f"Sentence: {text}"
    )


def join(left, fill, right):
    """left + fill + right with a space before the filler, and one after it
    when a word follows ("a" + "farmer" + ", who" -> "a farmer, who")."""
    if not fill:
        return left + right
    pre = " " if left and not left[-1].isspace() else ""
    post = " " if right[:1].isalnum() else ""
    return left + pre + fill + post + right


def render(parts, fills):
    """The text between the gaps (parts) with fills[j] in gap j."""
    text = parts[0]
    for f, p in zip(fills, parts[1:]):
        text = join(text, f, p)
    return text


def text_kl(model, tok, text, answers, refs, device):
    """Mean D of a whole prompt text; its length may differ from the reference's."""
    D = 0.0
    for a, r in zip(answers, refs):
        ids, _, loss_slice = build_ids_with_answer(tok, text, a)
        D += kl_of(model, torch.tensor(ids, device=device), r["ref_logits_p"], loss_slice)
    return D / len(refs)


def fill_gaps(model, tok, original, rewritten, answers, device, args):
    parts = rewritten.split(MASK)
    fills = [""] * (len(parts) - 1)
    _, refs = prepare_answers(model, tok, original, render(parts, fills), answers, device)
    D = text_kl(model, tok, render(parts, fills), answers, refs, device)
    print(f"gaps     D={D:.4f} (all {len(fills)} empty)")
    for j in range(len(fills)):
        if D <= args.kl_threshold:
            print(f"  D <= {args.kl_threshold}: remaining gaps left empty")
            break
        shown = render(parts, fills[:j] + [MASK] + fills[j + 1:])
        cands = llm_candidates(model, tok, shown, args.n_candidates, args.cand_rounds,
                               args.cand_temperature, args.cand_max_new_tokens,
                               prompt=lambda t, n: gap_prompt(t, n, args.gap_max_words),
                               max_words=args.gap_max_words)
        cand_D = [text_kl(model, tok, render(parts, fills[:j] + [c] + fills[j + 1:]), answers, refs, device)
                  for c in cands]
        order = sorted(range(len(cands)), key=cand_D.__getitem__)
        print(f"  gap {j}: {shown!r}\n    top fillers: "
              + ", ".join(f"{cands[i]!r}(D={cand_D[i]:.3f})" for i in order[:8]))
        if order and cand_D[order[0]] < D:
            fills[j], D = cands[order[0]], cand_D[order[0]]
            print(f"    filled with {fills[j]!r}  D={D:.4f}")
        else:
            print("    no filler beats the empty gap; left empty")
    return render(parts, fills)


# ------------------------------------------------------------------ main loop

def run(model, tok, original, rewritten, answers, device, args):
    if MASK in rewritten:
        rewritten = fill_gaps(model, tok, original, rewritten, answers, device, args)
    live_prompt, refs = prepare_answers(model, tok, original, rewritten, answers, device)
    s, e = content_span(tok, rewritten)
    content_slice = slice(s, e)

    current_D = mean_kl(model, live_prompt, refs)
    print(f"step 000  D={current_D:.4f} (initial, mean over {len(refs)} answers)  "
          f"{tok.decode(live_prompt[content_slice])!r}")

    for step in range(1, args.n_steps + 1):
        if current_D <= args.kl_threshold:
            print(f"step {step:03d}  D <= {args.kl_threshold}: utility good enough, no edit \n")
            break
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
