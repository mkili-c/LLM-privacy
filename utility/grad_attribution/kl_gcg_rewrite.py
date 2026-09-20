"""
Greedy, gradient-guided rewrite of the REWRITTEN prompt's own tokens to
recover utility -- no appended suffix, no search over auxiliary positions.
Reuses the one-hot / candidate-checking machinery from gcg.py and
kl_token_grad.py (backprop-ing the KL loss from kl_grad_attribution.py)
but applies it to the rewritten prompt's existing content tokens.

Each step:
    1. Backprop D = KL(P_original || P_rewritten), teacher-forced with the
       same answer on both, into the one-hot of every CONTENT token of the
       rewritten prompt (token_gradients from kl_token_grad.py; content span
       from kl_grad_attribution.content_span, so template/special tokens are
       never touched).
    2. Rank those tokens by |grad * input| -- the same attribution score
       kl_grad_attribution.py reports. One_hot.grad at a position's OWN
       current-token column is exactly grad*input: input_embeds = one_hot @
       embed_weights, so d(one_hot)[i, v] = d(embed[i]) . embed_weights[v],
       and at v == the current token, embed_weights[v] == embed[i] itself,
       giving d(embed[i]) . embed[i] -- the grad*input definition, for free
       from the same backward pass.
    3. Take the single highest-impact position. Use the gradient row at that
       position to shortlist top-k replacement tokens (GCG's usual
       "most negative gradient" candidates), check each with a real forward
       pass, and swap in whichever actually minimizes D.
    4. Repeat for --n_steps tokens.

Minimizing D pulls the rewritten prompt's answer distribution toward the
original's -- i.e. this repairs utility the rewrite cost, one token at a
time, always touching the token the gradient says matters most.

The sequence length never changes (one token id replaces another at a fixed
position), so this stays in id-space throughout and only decodes back to
text for reporting and the final result -- no text round-tripping / slice
re-location needed once the prompt is first tokenized.
"""
import argparse
import sys
from pathlib import Path

import torch

# utility/metrics/ holds the shared sampling and scoring helpers, which import
# each other by bare name, so that directory goes on sys.path as a whole.
# Resolved from __file__ so the script runs from any working directory.
_METRICS = Path(__file__).resolve().parents[1] / "metrics"
if str(_METRICS) not in sys.path:
    sys.path.insert(0, str(_METRICS))

from topk_sample import topk_answers
from kl_utility_scoring import build_prompt_ids
from grad_based_utility_score import load
from kl_grad_attribution import content_span

from kl_token_grad import token_gradients

DEFAULT_MODEL = "/work/projects/mkilic/models/Qwen2.5-3B-Instruct"


def get_args():
    p = argparse.ArgumentParser(
        description="Greedily edit the highest-impact token of the REWRITTEN "
                    "prompt, each step minimizing KL(P_original || P_rewritten)")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--original", default="I am a 45-year-old male doctor living in New York. "
                                         "What are the best ways to manage stress?")
    p.add_argument("--rewritten", default="What are some good general strategies for "
                                          "managing everyday stress?")
    p.add_argument("--target", default=None,
                   help="fixed answer to teacher-force; if omitted, one answer is "
                        "sampled from P_orig")
    p.add_argument("--n_steps", type=int, default=10, help="tokens to edit, one per step")
    p.add_argument("--topk", type=int, default=64, help="replacement candidates checked per edited token")
    p.add_argument("--eval_chunk_size", type=int, default=32)
    p.add_argument("--dtype", choices=["float16", "float32"], default="float16")
    p.add_argument("--seed", type=int, default=23)
    return p.parse_args()


def build_ids_with_answer(tok, prompt, answer):
    """[chat-templated prompt] + [teacher-forced answer], as token ids, plus
    the slices marking the answer and the logits that predict it."""
    prompt_ids = build_prompt_ids(tok, prompt)
    answer_ids = tok(answer, add_special_tokens=False).input_ids
    P = len(prompt_ids)
    target_slice = slice(P, P + len(answer_ids))
    loss_slice = slice(P - 1, P + len(answer_ids) - 1)
    return prompt_ids + answer_ids, target_slice, loss_slice


def kl_of(model, ids, ref_logits_p, loss_slice):
    with torch.no_grad():
        logits = model(input_ids=ids.unsqueeze(0)).logits[0, loss_slice, :]
        logq = torch.log_softmax(logits.float(), dim=-1)
        return (ref_logits_p.exp() * (ref_logits_p - logq)).sum(dim=-1).sum().item()


def evaluate_candidates(model, live_ids, ref_logits_p, position, loss_slice, candidate_toks, eval_chunk_size):
    """Substitute each candidate token id at `position` and score
    D = KL(P_ref || P_live) via a chunked forward pass (a full-vocab logits
    tensor for the whole candidate batch at once is what runs the GPU out of
    memory -- see kl_gcg_attack.py's history)."""
    losses = []
    with torch.no_grad():
        for start in range(0, len(candidate_toks), eval_chunk_size):
            chunk = candidate_toks[start:start + eval_chunk_size]
            batch = live_ids.unsqueeze(0).repeat(len(chunk), 1)
            batch[:, position] = chunk
            logits_live = model(input_ids=batch).logits[:, loss_slice, :]
            logq = torch.log_softmax(logits_live.float(), dim=-1)
            kl = (ref_logits_p.exp() * (ref_logits_p - logq)).sum(dim=-1)
            losses.append(kl.sum(dim=-1))
    return torch.cat(losses)


def run(model, tok, original, rewritten, answer, device, n_steps, topk, eval_chunk_size):
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

    for step in range(1, n_steps + 1):
        grad = token_gradients(
            model, live_ids, ref_ids,
            content_slice, target_slice, ref_target_slice,
            loss_slice, ref_loss_slice,
        )  # [content_len, vocab]

        content_ids = live_ids[content_slice]
        grad_x_input = grad[torch.arange(grad.shape[0], device=grad.device), content_ids]
        i = grad_x_input.abs().argmax().item()
        position = s + i

        candidate_toks = (-grad[i]).topk(topk).indices
        cand_D = evaluate_candidates(model, live_ids, ref_logits_p, position, loss_slice, candidate_toks, eval_chunk_size)
        best = cand_D.argmin()

        old_tok = tok.convert_ids_to_tokens([live_ids[position].item()])[0]
        new_tok_id = candidate_toks[best].item()
        live_ids[position] = new_tok_id
        current_D = cand_D[best].item()

        new_tok = tok.convert_ids_to_tokens([new_tok_id])[0]
        print(f"step {step:03d}  D={current_D:.4f}  pos={position}  "
              f"{old_tok!r} -> {new_tok!r}  content={tok.decode(live_ids[content_slice])!r}")

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

    final_rewrite = run(
        model, tok, args.original, args.rewritten, answer, device,
        n_steps=args.n_steps, topk=args.topk, eval_chunk_size=args.eval_chunk_size,
    )
    print(f"\nfinal rewrite: {final_rewrite!r}")
