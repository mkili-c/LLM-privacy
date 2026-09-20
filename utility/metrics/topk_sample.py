"""
Produces top k answers for a given prompt.

  --mode beam    the K *most likely* answers via beam search. Deterministic and
                 mode-seeking; the K beams are often near-duplicates.

  --mode sample  K independent draws from P(answer | prompt) with temperature /
                 nucleus. Better spread for a distributional similarity metric;
                 duplicates possible.
"""
import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def get_args():
    p = argparse.ArgumentParser(description="Top-K answers for one prompt")
    p.add_argument("--model_path", default="/work/projects/mkilic/models/Qwen2.5-3B-Instruct")
    p.add_argument("--prompt", default="I am a 45-years-old doctor living in New York. Give me one advice for coping with stress.")
    p.add_argument("--k", type=int, default=15)
    p.add_argument("--mode", choices=["beam", "sample"], default="sample")
    p.add_argument("--max_new_tokens", type=int, default=64)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top_p", type=float, default=0.95)
    p.add_argument("--seed", type=int, default=23)
    return p.parse_args()


def load(model_path, device):
    tok = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.float16 if device != "cpu" else torch.float32, 
    ).to(device).eval()

    if tok.pad_token is None:
        tok.pad_token = tok.eos_token   #for batches that are not the same size
    return model, tok


def topk_answers(prompt, model, tok, k=15, mode="sample", max_new_tokens=256,
                 temperature=0.8, top_p=0.95, return_logprobs=False):
    """
    Return a list of (answer_text, total_logprob, per_token_logprob).

    return_logprobs=True: list is sorted by per-token log-prob
    (length-normalized, so long answers aren't penalized; with the default
    length_penalty=1.0 this matches beam search's own ranking).

    With return_logprobs=False (the default) the scores are never requested,
    both log-prob fields are nan, and sequences come back in generation order.
    """
    if getattr(tok, "chat_template", None):
        text = tok.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False, add_generation_prompt=True,
        )
    else:
        text = prompt
    
    inputs = tok(text, return_tensors="pt").to(model.device)

    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        num_return_sequences=k,
        return_dict_in_generate=True,       #returns a specific modelOutput object necessary for output scores 
        output_scores=return_logprobs,
        pad_token_id=tok.pad_token_id,
    )
    if mode == "beam":
        # More beams than returned sequences gives a better approximation of the
        # true top-k; sampling params would be ignored here, so don't pass them.
        gen_kwargs.update(do_sample=False, num_beams=max(2 * k, k + 1))
    else:
        gen_kwargs.update(do_sample=True, temperature=temperature, top_p=top_p)

    with torch.no_grad():
        out = model.generate(**inputs, **gen_kwargs)

    gen_only = out.sequences[:, inputs.input_ids.shape[1]:]
    texts = tok.batch_decode(gen_only, skip_special_tokens=True)

    if not return_logprobs:
        nan = float("nan")
        return [(t, nan, nan) for t in texts]

    # Log-prob of each chosen token, shape [k, gen_len]. Works for both modes
    # (beam_indices only exists on beam outputs and is required there).
    transition = model.compute_transition_scores(
        out.sequences, out.scores,
        beam_indices=getattr(out, "beam_indices", None),
        normalize_logits=True,
    )

    # Finished sequences are right-padded with forced tokens of probability 1
    # (logprob exactly 0.0), while real tokens always have logprob < 0 -- the
    # HF-documented way to mask padding steps.
    valid = transition < 0
    total_logprob = (transition * valid).sum(dim=1)
    n_tokens = valid.sum(dim=1).clamp(min=1)
    per_token = total_logprob / n_tokens

    results = zip(texts, total_logprob.tolist(), per_token.tolist())
    return sorted(results, key=lambda r: r[2], reverse=True)    #sorted by per_token logprobs


if __name__ == "__main__":
    args = get_args()
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading {args.model_path} on {device} ...")
    model, tok = load(args.model_path, device)

    results = topk_answers(
        args.prompt, model, tok, k=args.k, mode=args.mode,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature, top_p=args.top_p,
        return_logprobs=True,
    )

    print(f"\nPrompt: {args.prompt}")
    print(f"Mode: {args.mode}   K={args.k}\n")
    for rank, (answer, total, per_tok) in enumerate(results):
        print(f"[{rank}] total_logprob={total:.2f}  per_token={per_tok:.3f} \n\n{answer} \n\n\n")
