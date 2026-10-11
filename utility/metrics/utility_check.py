"""
Utility check of the interleaved privacy-utility loop: how far a rewrite y of the user's message x_0
moves the answers, as a share of how far an empty message would move them.

K answers a_1..a_K are sampled once from x_0 and teacher-forced on every text:

    D(y) = (1/K) Σ_k (1/|a_k|) Σ_j KL( P(·|x_0, a_k,<j) ‖ P(·|y, a_k,<j) )      [nats per answer token]
    R(y) = D(y) / D(∅)          D(∅) = D of the empty message: how much x_0 shapes its answers

R >= 100% means y moves the answers as far as sending nothing would: utility is broken. On the
dev set (calibration/analyze_calibration.py) this rule tells the off-topic rewrites from the
privacy and utility rewrites with AUROC 0.993 and accuracy 0.964, and the accuracy-optimal
threshold is 99.5% (bootstrap 95% CI 92.7-105.1%). D's raw scale differs between prompts
(a plain paraphrase costs 0.30 for one prompt, 0.06 for another), R does not.

Cost per prompt: sampling the K answers (the expensive part) and K forward passes for D(∅),
both once; then K forward passes per checked text.

The same D as kl_mask_llm_rewrite_multi.py, the earlier utility rewriter, which is kept as it
was but no longer called by the loop.
"""
import sys
from pathlib import Path

import torch

# this directory for the sampling helper, grad_attribution/ for the teacher-forced KL
_HERE = Path(__file__).resolve().parent
for _dir in (_HERE, _HERE.parent / "grad_attribution"):
    if str(_dir) not in sys.path:
        sys.path.insert(0, str(_dir))

from topk_sample import topk_answers                          # noqa: E402
from kl_gcg_rewrite import build_ids_with_answer, kl_of       # noqa: E402


def sample_answers(model, tok, x0, k, max_new_tokens):
    """k answers sampled from x_0 (temperature 1, top-p 1), empty ones dropped."""
    return [a for a, _, _ in topk_answers(x0, model, tok, k=k, mode="sample",
                                          max_new_tokens=max_new_tokens,
                                          temperature=1.0, top_p=1.0) if a.strip()]


@torch.no_grad()
def reference_logprobs(model, tok, x0, answers, device):
    """[K] log P(·|x_0, a_k,<j) at every position of answer k, each (|a_k|, vocab), float32."""
    out = []
    for a in answers:
        ids, _, loss_slice = build_ids_with_answer(tok, x0, a)
        logits = model(input_ids=torch.tensor([ids], device=device)).logits[0, loss_slice, :]
        out.append(torch.log_softmax(logits.float(), dim=-1))
    return out


def answer_kls(model, tok, text, answers, ref_logp, device):
    """[K] D of text on each answer: (1/|a_k|) Σ_j KL(P(·|x_0, a_k,<j) ‖ P(·|text, a_k,<j))."""
    kls = []
    for a, p in zip(answers, ref_logp):
        ids, _, loss_slice = build_ids_with_answer(tok, text, a)
        kls.append(kl_of(model, torch.tensor(ids, device=device), p, loss_slice))
    return kls


def prepare(model, tok, x0, k, max_new_tokens, device):
    """What the check of one prompt needs, computed once: the answers, x_0's log-probs on them
    and D(∅). -> dict, or None if every sampled answer was empty (then nothing can be checked)."""
    answers = sample_answers(model, tok, x0, k, max_new_tokens)
    if not answers:
        return None
    ref_logp = reference_logprobs(model, tok, x0, answers, device)
    kls = answer_kls(model, tok, "", answers, ref_logp, device)
    return dict(answers=answers, ref_logp=ref_logp, d_empty=sum(kls) / len(kls))


def share(model, tok, text, ref, device):
    """R = D(text) / D(∅) against the prompt of ref (prepare's dict)."""
    kls = answer_kls(model, tok, text, ref["answers"], ref["ref_logp"], device)
    return sum(kls) / len(kls) / ref["d_empty"]
