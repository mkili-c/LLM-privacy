"""
Greedy rewrite of the REWRITTEN prompt's own tokens, like kl_gcg_rewrite.py,
but with the replacement candidates proposed by the LLM itself instead of by
the gradient's top-k.

Each step:
    1. Backprop D = KL(P_original || P_rewritten) (teacher-forced with the same
       answer on both sides) into the one-hot of every CONTENT token of the
       rewritten prompt -- token_gradients from kl_token_grad.py.
    2. Rank the content tokens by |grad * input| (the one_hot.grad entry at a
       position's own current token, see kl_gcg_rewrite.py for why that is
       exactly grad*input).
    3. MASK the highest-impact token and ask Qwen, through its chat template,
       to propose a set of single-word fillers for the mask. Only the masked
       REWRITTEN sentence is shown to the model -- never the original prompt,
       so nothing private can be pulled back in through this prompt.
    4. Keep the proposals that tokenize to exactly one token (the sequence
       length is fixed, one id replaces one id) and pick the one in the
       steepest descent direction of D, GCG-style:
           --select forward  (default) real forward pass per candidate, argmin D
           --select grad     linear estimate only: argmin of grad[pos, cand]
       If no candidate lowers D, fall through to the next-ranked position
       (up to --max_positions per step); if none of them improve, stop.
    5. Repeat for --n_steps tokens.

The LLM's candidates are fluent by construction (they are what the model
thinks belongs in that slot), so this trades GCG's unconstrained, often
garbled top-k substitutions for grammatical ones, and uses the gradient /
forward pass only to choose among them.

Everything stays in id-space; text is only decoded for the mask prompt and
for reporting.
"""
import argparse
import re
import sys

import torch

sys.path.append("/work/projects/mkilic/privacy-utility/utility/")
sys.path.append("/work/projects/mkilic/privacy-utility/utility/grad_attribution/")

from topk_sample import topk_answers
from grad_based_utility_score import load
from kl_grad_attribution import content_span

from kl_token_grad import token_gradients
from kl_gcg_rewrite import build_ids_with_answer, kl_of, evaluate_candidates

DEFAULT_MODEL = "/work/projects/mkilic/models/Qwen2.5-3B-Instruct"
MASK = "[MASK]"


def get_args():
    p = argparse.ArgumentParser(
        description="Greedily edit the highest-impact token of the REWRITTEN prompt, "
                    "with replacement candidates proposed by the LLM for a masked slot "
                    "and selected by the KL gradient / forward pass")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--original", default="I am a 45-year-old male doctor living in New York. "
                                         "What are the best ways to manage stress?")
    p.add_argument("--rewritten", default="What are some good general strategies for "
                                          "managing everyday stress?")
    p.add_argument("--target", default=None,
                   help="fixed answer to teacher-force; if omitted, one answer is "
                        "sampled from P_orig")
    p.add_argument("--n_steps", type=int, default=10, help="tokens to edit, one per step")
    p.add_argument("--n_candidates", type=int, default=20,
                   help="fillers the LLM is asked for per mask prompt")
    p.add_argument("--cand_rounds", type=int, default=2,
                   help="independent sampled generations pooled per mask (1 = greedy only)")
    p.add_argument("--cand_temperature", type=float, default=0.9)
    p.add_argument("--cand_max_new_tokens", type=int, default=160)
    p.add_argument("--select", choices=["forward", "grad"], default="forward",
                   help="forward: check every candidate with a real forward pass (GCG); "
                        "grad: rank by the linear gradient estimate only")
    p.add_argument("--max_positions", type=int, default=3,
                   help="positions tried per step (by rank) before giving up on the step")
    p.add_argument("--eval_chunk_size", type=int, default=32)
    p.add_argument("--dtype", choices=["float16", "float32"], default="float16")
    p.add_argument("--seed", type=int, default=23)
    return p.parse_args()


# ---------------------------------------------------------------- mask prompt

def masked_text(tok, live_ids, content_slice, position):
    """Decode the content span with the token at `position` replaced by MASK.
    A leading-space token ('Ġ...') keeps its space in front of the mask so
    the sentence reads naturally."""
    s, e = content_slice.start, content_slice.stop
    left = tok.decode(live_ids[s:position])
    right = tok.decode(live_ids[position + 1:e])
    old = tok.convert_ids_to_tokens([live_ids[position].item()])[0]
    sep = " " if old.startswith("Ġ") and left else ""
    return f"{left}{sep}{MASK}{right}"


def fill_prompt(text, n):
    return (
        f"The sentence below has one word replaced by {MASK}.\n"
        f"Propose {n} different single words that could fill {MASK} so the sentence "
        "stays grammatical and natural. Vary the meaning: include specific and general "
        "options, not just synonyms.\n"
        "Output one word per line, nothing else -- no numbering, no punctuation, "
        "no explanations.\n\n"
        f"Sentence: {text}"
    )


_LINE_CLEAN = re.compile(r"^\s*(?:[-*•]|\d+[.)]|\(\d+\))?\s*")
_STRIP_CHARS = " \t\"'`.,;:!?()[]{}"


def parse_words(text):
    """Turn the model's list into words: strip bullets / numbering / quotes,
    keep only single-token-ish entries (no inner whitespace), dedupe in order."""
    words, seen = [], set()
    for line in text.splitlines():
        if "mask" in line.lower():
            continue
        w = _LINE_CLEAN.sub("", line).strip(_STRIP_CHARS)
        if not w or " " in w:
            continue
        if w.lower() not in seen:
            seen.add(w.lower())
            words.append(w)
    return words


@torch.no_grad()
def llm_candidates(model, tok, text, n, rounds, temperature, max_new_tokens):
    """Ask the LLM for fillers for the mask in `text`. One greedy generation
    plus (rounds - 1) sampled ones, pooled and deduped."""
    chat = tok.apply_chat_template(
        [{"role": "user", "content": fill_prompt(text, n)}],
        tokenize=False, add_generation_prompt=True,
    )
    inputs = tok(chat, return_tensors="pt", add_special_tokens=False).to(model.device)
    P = inputs.input_ids.shape[1]

    outs = [model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                           pad_token_id=tok.pad_token_id)]
    if rounds > 1:
        outs.append(model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=True,
                                   temperature=temperature, top_p=1.0,
                                   num_return_sequences=rounds - 1,
                                   pad_token_id=tok.pad_token_id))
    words = []
    for out in outs:
        for row in tok.batch_decode(out[:, P:], skip_special_tokens=True):
            words.extend(parse_words(row))
    return list(dict.fromkeys(words))


def words_to_single_tokens(tok, words, old_tok_id):
    """Map proposed words to token ids, matching the leading-space convention of
    the token being replaced (a 'Ġword' slot gets ' word'). Keep only words that
    are exactly one token and differ from the current one.
    Returns (ids, kept_words, n_dropped_multitoken)."""
    old = tok.convert_ids_to_tokens([old_tok_id])[0]
    prefix = " " if old.startswith("Ġ") else ""
    ids, kept, dropped = [], [], 0
    for w in words:
        enc = tok(prefix + w, add_special_tokens=False).input_ids
        if len(enc) != 1:
            dropped += 1
            continue
        if enc[0] == old_tok_id or enc[0] in ids:
            continue
        ids.append(enc[0])
        kept.append(w)
    return ids, kept, dropped


# ------------------------------------------------------------------ main loop

def run(model, tok, original, rewritten, answer, device, args):
    ref_ids, ref_target_slice, ref_loss_slice = build_ids_with_answer(tok, original, answer)
    ref_ids = torch.tensor(ref_ids, device=device)
    with torch.no_grad():
        ref_logits_p = torch.log_softmax(
            model(input_ids=ref_ids.unsqueeze(0)).logits[0, ref_loss_slice, :].float(),
            dim=-1,
        )

    live_ids, target_slice, loss_slice = build_ids_with_answer(tok, rewritten, answer)
    live_ids = torch.tensor(live_ids, device=device)
    s, e = content_span(tok, rewritten)
    content_slice = slice(s, e)

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
            if args.select == "grad":
                order = grad_score.argsort()
                best = order[0]
                # one forward pass on the chosen token to know the real D
                new_D = evaluate_candidates(model, live_ids, ref_logits_p, position, loss_slice,
                                            cand_toks[best:best + 1], args.eval_chunk_size)[0].item()
            else:
                cand_D = evaluate_candidates(model, live_ids, ref_logits_p, position, loss_slice,
                                             cand_toks, args.eval_chunk_size)
                order = cand_D.argsort()
                best = order[0]
                new_D = cand_D[best].item()

            shown = ", ".join(
                f"{cand_words[j]}(g={grad_score[j].item():+.3f}"
                + (f", D={cand_D[j].item():.3f})" if args.select == "forward" else ")")
                for j in order[:8].tolist())
            print(f"    top candidates: {shown}")

            if new_D >= current_D:
                print(f"    no candidate improves D ({new_D:.4f} >= {current_D:.4f}); next position")
                continue

            new_tok_id = cand_toks[best].item()
            live_ids[position] = new_tok_id
            current_D = new_D
            new_tok = tok.convert_ids_to_tokens([new_tok_id])[0]
            print(f"step {step:03d}  D={current_D:.4f}  pos={position}  "
                  f"{old_tok!r} -> {new_tok!r}  content={tok.decode(live_ids[content_slice])!r}")
            accepted = True
            break

        if not accepted:
            print(f"step {step:03d}  no improving swap in the top {args.max_positions} positions; stopping")
            break

    return tok.decode(live_ids[content_slice])


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
        answer = args.target
    else:
        answer = next(a for a, _, _ in topk_answers(
            args.original, model, tok, k=1, mode="sample",
            max_new_tokens=64, temperature=1.0, top_p=1.0) if a.strip())

    print(f"\nORIGINAL : {args.original}")
    print(f"REWRITTEN: {args.rewritten}")
    print(f"TARGET ANSWER: {answer}\n")

    final_rewrite = run(model, tok, args.original, args.rewritten, answer, device, args)
    print(f"\nfinal rewrite: {final_rewrite!r}")
