"""
Utility rewriting: D's gradient finds what the message lost from the original, an LLM puts it back.

Utility is how close the answers to the message y stay to the answers to the user's message x_0.
K answers a_1..a_K are sampled once from x_0 (or passed with --targets) and teacher-forced on both:

    D(y) = (1/K) Σ_k (1/|a_k|) Σ_j KL( P(·|x_0, a_k,<j) ‖ P(·|y, a_k,<j) )      [nats per answer token]

(j runs over answer tokens; below, t runs over the tokens of the original x_0).

The answers are --answer_max_new_tokens = 256 tokens long: the persona echo ("As a 34-year-old
midwife, ...") is a roughly fixed cost at the answer opening, so in short answers it dominates D.

Each iteration, while D > --kl_threshold (a message that keeps enough utility is not edited):
    1. Gradient on the original's side: with y held fixed, backprop D into the embeddings e_t of
       the tokens of x_0 (kl_grad_attribution.kl_attribution, attribute "original"), averaged
       over the K answers. Its grad x input
           r_t = ⟨∂D/∂e_t, e_t⟩
       is, to first order, how much D drops when token t of x_0 is weakened, i.e. when x_0 moves
       toward y: scaling e_t by (1 - ε) changes D by -ε·r_t.
           r_t > 0   y lacks what token t says, and the answers depend on it
           r_t < 0   y has it too: weakening it in x_0 alone would open a gap
       Content the privacy step deleted has no token left in y, but its token is still here.
    2. A word of x_0 scores the sum of r_t over its tokens (the first-order effect of weakening
       the whole word). The --top_m best content words (function words, as in privacy_rewrite.py,
       are skipped) with a positive score are marked in x_0; marked words that only spaces
       separate form one span.
    3. One greedy LLM call sees x_0 with the marks and y, and puts the marked parts back into y,
       changing the words around them only where the sentence needs it.
    4. D decides: the rewrite is kept if it lowers D. If not, the step ends, since greedy decoding
       would return the same rewrite again.
The step ends when D <= --kl_threshold, when no word of x_0 scores above 0, when a rewrite is not
kept or changes nothing, or after --max_iters.

The first version scored the tokens of y by GCG's swap gain g[t, y_t] - min_v g[t, v]. In job 556
it marked words y shares verbatim with x_0 ("certifications", "security work"): the gain measures
how sensitive D is to a token, not whether a replacement restores anything, and a token the
privacy step introduced could score high through its own term g[t, y_t] alone. With marks that
did not point at the loss, the LLM brought back everything it saw in x_0 at once.

It mirrors privacy_rewrite.py (blame, mark, one LLM rewrite) and shares its marks and greedy
generation. Bringing a private detail back is allowed: the request could not afford to lose it,
and the privacy step sees the message again.

Cost per iteration: K backward passes, about 2K forward passes and one generation.
"""
import argparse
import sys
from pathlib import Path

import torch

# utility/metrics/ for the shared sampling and scoring helpers, this directory for the
# grad_attribution modules, privacy/ for the marks and generation of privacy_rewrite.py.
_HERE = Path(__file__).resolve().parent
for _dir in (_HERE.parent / "metrics", _HERE, _HERE.parents[1] / "privacy"):
    if str(_dir) not in sys.path:
        sys.path.insert(0, str(_dir))

from topk_sample import topk_answers
from grad_based_utility_score import load
from kl_grad_attribution import kl_attribution

from kl_gcg_rewrite import build_ids_with_answer, kl_of
from privacy_rewrite import OPEN, CLOSE, WORD, STOPWORDS, generate_message

DEFAULT_MODEL = "/work/projects/mkilic/models/Qwen2.5-3B-Instruct"


def get_args():
    p = argparse.ArgumentParser(
        description="Bring back into the REWRITTEN prompt, with an LLM, the words of the original "
                    "that the gradient of D (averaged over K answers) says it lost; keep a "
                    "rewrite only if it lowers D")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--original", default="I am a 45-year-old male doctor living in New York. "
                                         "What are the best ways to manage stress?")
    p.add_argument("--rewritten", default="What are some good general strategies for "
                                          "managing everyday stress?")
    p.add_argument("--k", type=int, default=10,
                   help="number of answers sampled from P_orig to average over")
    p.add_argument("--targets", nargs="+", default=None,
                   help="fixed answers to teacher-force (overrides --k sampling)")
    p.add_argument("--answer_max_new_tokens", type=int, default=256)
    p.add_argument("--top_m", type=int, default=3,
                   help="content words of the original marked for the LLM per iteration")
    p.add_argument("--max_iters", type=int, default=4, help="LLM rewrites at most")
    # 0.47 separates the privacy rewrites of data/prompt_rewrites_100.csv (93% left
    # alone) from its off-topic ones (90% edited): Qwen2.5-3B, K = 5, 256-token answers
    p.add_argument("--kl_threshold", type=float, default=0.47,
                   help="mean D (nats per answer token) at or below which the prompt keeps "
                        "enough utility and is not edited any further (0 = always edit)")
    p.add_argument("--dtype", choices=["float16", "float32"], default="float16")
    p.add_argument("--seed", type=int, default=23)
    return p.parse_args()


# ------------------------------------------------------------ D

def prepare_answers(model, tok, original, rewritten, answers, device):
    """Per answer: the fixed reference sequence and its log-probs, plus the
    answer ids and slices for the live sequence. The live PROMPT ids are the
    same for every answer and are returned once."""
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


def answer_kls(model, tok, text, answers, refs, device):
    """[K] D of a whole prompt text on each answer; its length may differ from the reference's."""
    kls = []
    for a, r in zip(answers, refs):
        ids, _, loss_slice = build_ids_with_answer(tok, text, a)
        kls.append(kl_of(model, torch.tensor(ids, device=device), r["ref_logits_p"], loss_slice))
    return kls


def text_kl(model, tok, text, answers, refs, device):
    """Mean D of a whole prompt text over the answers."""
    kls = answer_kls(model, tok, text, answers, refs, device)
    return sum(kls) / len(kls)


# ------------------------------------------------------------ what to bring back

def restore_scores(model, tok, original, text, answers, device):
    """[L0] r_t = ⟨∂D/∂e_t, e_t⟩ for every token of the original's chat-templated prompt,
    with D per answer token and averaged over the answers, as in run()."""
    r = None
    for a in answers:
        res = kl_attribution(model, tok, original, text, a, device, attribute="original")
        g = res["grad_x_input"] / res["n_answer_tokens"]     # kl_attribution sums D over the answer
        r = g if r is None else r + g
    return r / len(answers)


def token_offsets(tok, text):
    """(start, end) in text of every token of its chat-templated prompt, i.e. of the ids
    build_prompt_ids returns; the template's own tokens fall outside [0, len(text))."""
    chat = text
    if getattr(tok, "chat_template", None):
        chat = tok.apply_chat_template([{"role": "user", "content": text}],
                                       tokenize=False, add_generation_prompt=True)
    base = chat.rfind(text)          # the user's text comes last, before the assistant header
    enc = tok(chat, add_special_tokens=False, return_offsets_mapping=True)
    return [(a - base, b - base) for a, b in enc.offset_mapping]


def marked_spans(tok, text, scores, top_m):
    """The top_m content words of text with a positive score, where a word scores the sum
    of its tokens' scores (scores [L] over text's chat-templated prompt), merged where only
    spaces separate them. -> ([(start, end)] left to right, [(word, score)])."""
    offsets = token_offsets(tok, text)
    scores = scores.tolist()
    words = [w for w in WORD.finditer(text) if w.group().lower() not in STOPWORDS]
    score = [sum(r for (a, b), r in zip(offsets, scores) if a < w.end() and w.start() < b)
             for w in words]
    best = sorted(range(len(words)), key=score.__getitem__, reverse=True)[:top_m]
    top = sorted(i for i in best if score[i] > 0)
    spans = []
    for i in top:
        w = words[i]
        if spans and not text[spans[-1][1]:w.start()].strip():
            spans[-1] = (spans[-1][0], w.end())
        else:
            spans.append((w.start(), w.end()))
    return spans, [(words[i].group(), score[i]) for i in top]


def mark(text, spans):
    """text with OPEN ... CLOSE around each span; spans left to right, not overlapping."""
    out, last = "", 0
    for a, b in spans:
        out += text[last:a] + OPEN + text[a:b] + CLOSE
        last = b
    return out + text[last:]


# ------------------------------------------------------------ rewriting

def rewrite_prompt(marked_original, text):
    return (
        "A user wrote the original message below to an AI assistant. The rewritten message leaves "
        "out or generalizes some of its details, and so it loses part of what the original asks "
        f"for. The parts of the original marked {OPEN} {CLOSE} are what it lost that matters most.\n"
        "Rewrite the rewritten message so that it contains the marked parts again, where they fit. "
        "Bring back only the marked parts, change the words around them only where the sentence "
        "needs it, and keep the rest of the rewritten message word for word.\n"
        "Output only the rewritten message, without the marks, and nothing else.\n\n"
        f"Original message: {marked_original}\n\n"
        f"Rewritten message: {text}"
    )


def run(model, tok, original, rewritten, answers, device, args):
    """-> the message after at most --max_iters LLM rewrites, each kept only if it lowers D."""
    text = rewritten
    _, refs = prepare_answers(model, tok, original, original, answers, device)   # x_0's log-probs, once
    D = text_kl(model, tok, text, answers, refs, device)
    for it in range(args.max_iters + 1):
        print(f"\n=== utility iteration {it}: D={D:.4f} (mean over {len(refs)} answers)  {text!r}")
        if D <= args.kl_threshold:
            print(f"  D <= {args.kl_threshold}: utility good enough, no edit")
            return text
        if it == args.max_iters:
            print("  max_iters reached")
            return text

        scores = restore_scores(model, tok, original, text, answers, device)      # [L0]
        spans, top = marked_spans(tok, original, scores, args.top_m)
        if not spans:
            print("  no word of the original scores above 0: nothing to bring back, stop")
            return text
        marked = mark(original, spans)
        print("  marked in the original: " + ", ".join(f"{w!r} ({r:+.4f})" for w, r in top)
              + f"\n    {marked!r}")
        # the prompt shows the message after this label, so the LLM may repeat it
        new = generate_message(model, tok, rewrite_prompt(marked, text))
        new = new.removeprefix("Rewritten message:").strip()
        if not new or new == text:
            print("  rewriter " + ("gave an empty answer" if not new else "changed nothing") + ": stop")
            return text
        D_new = text_kl(model, tok, new, answers, refs, device)
        print(f"  D {D:.4f} -> {D_new:.4f}: {'kept' if D_new < D else 'not kept, stop'}\n    -> {new!r}")
        if D_new >= D:
            return text
        text, D = new, D_new
    return text


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
