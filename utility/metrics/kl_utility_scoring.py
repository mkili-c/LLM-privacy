"""
Utility scoring with KL divergence

For two prompts T_rew and T_orig, answers define a distribution. 
Estimate KL divergence by sampling answers.

sample answers from each prompt, then cross-score every answer under both prompts:

    KL(P_orig || P_rew) ~= (1/K) sum_i [ log P(a_i|T_orig) - log P(a_i|T_rew) ],
                           a_i ~ P(. | T_orig)

Both directions are estimated (per token, so long answers don't dominate) and symmetrized into 
a Jeffreys divergence J, in nats/token An identical rewrite scores 1.0. The reported score is

    utility = exp(-J)   in (0, 1],  1 = answer distribution unchanged.

-------
Sampling defaults to temperature=1.0, top_p=1.0 (plain ancestral sampling):
the scorer uses the full softmax, so samples must come from the same
distribution or the estimate is biased 
"""

import math
import argparse

import torch
import torch.nn.functional as F

from topk_sample import load, topk_answers

DEFAULT_MODEL = "/work/projects/mkilic/models/Qwen2.5-3B-Instruct"
MESSAGE_SLOT = "\x00MESSAGE\x00"    # placeholder that splits the chat template around the message
# A token above 8 nats of surprisal (P < 1/3000) is a spike: broken grammar ("I am a, who",
# "I sensitivity my eyes") shows up as such spikes, a clean edit mostly does not.
SPIKE_NATS = 8.0


def get_args():
    p = argparse.ArgumentParser(
        description="Utility of a rewritten prompt symmetric per-token MC-KL between the two answer distributions")
    p.add_argument("--model_path", default=DEFAULT_MODEL)
    p.add_argument("--original", default="I am a 45-year-old male doctor living in New York. "
                                         "What are the best ways to manage stress?")
    p.add_argument("--rewritten", default="What are some good general strategies for "
                                          "managing everyday stress?")
    p.add_argument("--k", type=int, default=15)
    p.add_argument("--max_new_tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=1.0,
                   help="keep 1.0 for an unbiased KL estimate")
    p.add_argument("--top_p", type=float, default=1.0,
                   help="keep 1.0 for an unbiased KL estimate")
    p.add_argument("--seed", type=int, default=23)
    p.add_argument("--show_samples", action="store_true",
                   help="print every sampled answer with its cross-scores")
    return p.parse_args()


def build_prompt_ids(tok, prompt):
    """identical to the prefix topk_sample generates after, 
    so scoring conditions on the same context."""

    if getattr(tok, "chat_template", None):
        text = tok.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False, add_generation_prompt=True,
        )
    else:
        text = prompt
    return tok(text, add_special_tokens=False).input_ids    #input ids key store the indices


@torch.no_grad()
def score_answers(answers, prompt, model, tok, batch_size=8):
    """
    Teacher-forced log P(answer | prompt) for each answer text, batched.
    Returns (total_logprobs, n_tokens) as parallel lists. One forward pass per
    batch -- no generation. log_softmax in float32: cross-scoring visits very
    unlikely tokens, exactly where fp16 underflows.
    """
    prompt_ids = build_prompt_ids(tok, prompt)
    P = len(prompt_ids)
    pad_id = tok.pad_token_id

    totals, counts = [], []

    for i in range(0, len(answers), batch_size):

        # Prepare batch:
        # each row of form [prompt ids , answer ids]
        rows = [prompt_ids + tok(a, add_special_tokens=False).input_ids 
                for a in answers[i:i + batch_size]] 

        #Padding
        L = max(len(r) for r in rows)
        input_ids = torch.full((len(rows), L), pad_id, dtype=torch.long)
        attn = torch.zeros((len(rows), L), dtype=torch.long)
        for b, r in enumerate(rows):
            input_ids[b, :len(r)] = torch.tensor(r)
            attn[b, :len(r)] = 1                        #mark the real positions 0 gets converted to -\infty internally
        input_ids = input_ids.to(model.device)
        attn = attn.to(model.device)

        #forward pass
        #logits: [batch, seq_len, vocab_size]
        logits = model(input_ids=input_ids, attention_mask=attn).logits
        #logp has probability distribution for each position in every sequence in the batch
        logp = F.log_softmax(logits.float(), dim=-1)

        # token at position t is predicted by the logits at position t-1, pred[b, t] = log P(x_{t+1} | x_0 … x_t)
        # gather picks the values from logp at the index indicated by the input ids squeeze unsqueeze does the dimension matching 
        pred = logp[:, :-1, :].gather(-1, input_ids[:, 1:].unsqueeze(-1)).squeeze(-1)

        for b, r in enumerate(rows):
            # answer tokens sit at positions [P, len(r)) -> predictions [P-1, len(r)-1)
            # pred[b, t] scores input_ids[b, t+1]. 
            # so a pred index is always one less than the input index of the token it scores.
            span = pred[b, P - 1:len(r) - 1]
            totals.append(span.sum().item())
            counts.append(len(r) - P)
    return totals, counts


@torch.no_grad()
def disfluency(texts, model, tok, batch_size=16):
    """
    Surprisal spikes of each text as a user message, in nats:

        Σ_t max(0, -log P(x_t | x_<t) - SPIKE_NATS)

    over its tokens and the end-of-turn token after them, given the chat
    template before them. 0 = no token less likely than e^-8. The end-of-turn
    token counts because a message cut off mid-phrase is unlikely to end there.
    Not the mean NLL per token: deleting predictable words ("25 years old")
    raises the mean of a perfectly grammatical text by as much as a broken
    edit does, while a spike is what a broken edit adds. Returns a list [N].
    """
    chat = tok.apply_chat_template([{"role": "user", "content": MESSAGE_SLOT}],
                                   tokenize=False, add_generation_prompt=True)
    prefix, suffix = chat.split(MESSAGE_SLOT)
    prefix_ids = tok(prefix, add_special_tokens=False).input_ids
    end_ids = tok(suffix, add_special_tokens=False).input_ids[:1]        # <|im_end|>
    P = len(prefix_ids)

    spikes = []
    for i in range(0, len(texts), batch_size):
        rows = [prefix_ids + tok(t, add_special_tokens=False).input_ids + end_ids
                for t in texts[i:i + batch_size]]
        L = max(len(r) for r in rows)
        input_ids = torch.full((len(rows), L), tok.pad_token_id, dtype=torch.long)
        attn = torch.zeros((len(rows), L), dtype=torch.long)
        for b, r in enumerate(rows):
            input_ids[b, :len(r)] = torch.tensor(r)
            attn[b, :len(r)] = 1
        input_ids, attn = input_ids.to(model.device), attn.to(model.device)

        # message + end token sit at [P, len(r)) and are predicted by the logits at [P - 1, len(r) - 1)
        logits = model(input_ids=input_ids, attention_mask=attn).logits[:, P - 1:-1]   # (batch, L - P, vocab)
        logp = F.log_softmax(logits.float(), dim=-1)
        nll = -logp.gather(-1, input_ids[:, P:].unsqueeze(-1)).squeeze(-1)           # (batch, L - P)
        for b, r in enumerate(rows):
            spikes.append((nll[b, :len(r) - P] - SPIKE_NATS).clamp(min=0).sum().item())
    return spikes


def _mean_stderr(xs):
    m = sum(xs) / len(xs)
    if len(xs) < 2:
        return m, 0.0
    var = sum((x - m) ** 2 for x in xs) / (len(xs) - 1)
    return m, math.sqrt(var / len(xs))


def directional_kl(samples, prompt_own, prompt_other, model, tok, batch_size = 8):
    """
    Per-token MC estimate of KL(P_own || P_other) from answers sampled under
    prompt_own. Returns (kl_mean, kl_stderr, details per sample).
    """
    lp_own, n_tok = score_answers(samples, prompt_own, model, tok, batch_size = batch_size)
    lp_other, _ = score_answers(samples, prompt_other, model, tok, batch_size = batch_size)

    deltas, details = [], []
    for text, lo, lr, n in zip(samples, lp_own, lp_other, n_tok):
        if n == 0:
            continue                       # empty answer (immediate EOS): no evidence
        deltas.append((lo - lr) / n)        #log[P(own)/P(rew)] divided by answer length so this is calculated per token
        details.append({"answer": text, "lp_own": lo, "lp_other": lr,
                        "n_tokens": n, "delta_per_token": (lo - lr) / n})
    if not deltas:
        raise ValueError(f"All sampled answers were empty for prompt: {prompt_own!r}")
    kl, se = _mean_stderr(deltas)
    return kl, se, details


def utility_score(original, rewritten, model, tok, k=5, max_new_tokens=64,
                  temperature=1.0, top_p=1.0):
    """
    Utility of `rewritten` w.r.t. `original` in (0, 1]. Returns a dict with the
    score, both directional KLs (nats/token, with standard errors), and the
    per-sample cross-scores for inspection.
    """
    a_orig = [a for a, _, _ in topk_answers(
        original, model, tok, k=k, mode="sample", max_new_tokens=max_new_tokens,
        temperature=temperature, top_p=top_p) if a.strip()]
    a_rew = [a for a, _, _ in topk_answers(
        rewritten, model, tok, k=k, mode="sample", max_new_tokens=max_new_tokens,
        temperature=temperature, top_p=top_p) if a.strip()]

    kl_o2r, se_o2r, det_o = directional_kl(a_orig, original, rewritten, model, tok)
    kl_r2o, se_r2o, det_r = directional_kl(a_rew, rewritten, original, model, tok)

    J = kl_o2r + kl_r2o

    return {
        "total_divergence_per_token": J,
        "kl_orig_to_rew": kl_o2r, "kl_orig_to_rew_stderr": se_o2r,
        "kl_rew_to_orig": kl_r2o, "kl_rew_to_orig_stderr": se_r2o,
        "samples_orig": det_o,
        "samples_rew": det_r,
    }


if __name__ == "__main__":
    args = get_args()
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading {args.model_path} on {device} ...")
    model, tok = load(args.model_path, device)

    res = utility_score(
        args.original, args.rewritten, model, tok, k=args.k,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature, top_p=args.top_p,
    )

    print(f"\nORIGINAL : {args.original}")
    print(f"REWRITTEN: {args.rewritten}\n")

    if args.show_samples:
        for side, dets in (("orig", res["samples_orig"]), ("rew", res["samples_rew"])):
            for i, d in enumerate(dets):
                print(f"[{side} {i}] lp_own={d['lp_own']:.1f}  lp_other={d['lp_other']:.1f}  "
                      f"n={d['n_tokens']}  delta/token={d['delta_per_token']:+.3f}")
                print(f"  {d['answer']}\n")

    print(f"KL(orig||rew)   = {res['kl_orig_to_rew']:.4f} ± {res['kl_orig_to_rew_stderr']:.4f} nats/token")
    print(f"KL(rew||orig)   = {res['kl_rew_to_orig']:.4f} ± {res['kl_rew_to_orig_stderr']:.4f} nats/token")
    print(f"\nTotal Divergence per Token  = {res['total_divergence_per_token']:.4f} nats/token")    

#What are some good general strategies for Alexander doctors stress?