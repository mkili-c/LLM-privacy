# Gradient Attribution of the Answer-Distribution KL

**File:** `kl_grad_attribution.py`
**Date:** 2026-07-27
**Companion to:** `README.md` (the KL utility metric this differentiates)

## 1. Goal

The KL metric tells you *how much* utility a rewrite costs. This tells you **which prompt tokens are responsible**. Teacher-force the same answer through two forward passes — one conditioned on the original prompt, one on the rewritten prompt — measure the divergence between the two sets of answer logits, and backpropagate it into the input embeddings.

## 2. The objective

For a fixed answer $a$ with $|a|$ tokens:

$$
D(a) \;=\; \sum_{j} D_{\mathrm{KL}}\!\left( P_\text{orig}(\cdot \mid a_{<j}) \;\big\|\; P_\text{rew}(\cdot \mid a_{<j}) \right)
$$

Each term is a **full-vocabulary** KL between the two models' next-token distributions at the same answer prefix — not the log-probability of one sampled token.

## 3. Why this equals the sequence-level KL

By the chain rule for KL on autoregressive sequences:

$$
D_{\mathrm{KL}}\!\left(P_\text{orig}(A) \,\|\, P_\text{rew}(A)\right)
= \mathbb{E}_{a \sim P_\text{orig}}\!\left[ \sum_j D_{\mathrm{KL}}\!\left(P_\text{orig}(\cdot|a_{<j}) \,\|\, P_\text{rew}(\cdot|a_{<j})\right) \right]
$$

*Derivation:* expand $\log[P(a)/Q(a)]$ into a sum of per-token log-ratios, then for each $j$ take the expectation over $a_j \sim P(\cdot|a_{<j})$ first — that inner expectation **is** the per-position KL.

Two consequences:

- **$D$ is an unbiased estimator of the same quantity `utility_scoring.py` computes.** That estimator uses the log-ratio at the single sampled token; this integrates over the whole vocabulary at each position. Replacing a random variable by its conditional expectation preserves the mean and cannot increase variance (**Rao–Blackwell**), so this is the strictly lower-variance version.
- **$a$ must be sampled from $P_\text{orig}$**, because the expectation is over the *first* argument's distribution. Teacher-forcing an answer drawn from the rewrite estimates the reverse direction instead.

## 4. `positionwise_kl` — why `logp.exp()`

KL is an **expectation under $p$**:

$$
D_{\mathrm{KL}}(p \| q) = \sum_v \underbrace{p_v}_{\text{weight}} \cdot \underbrace{(\log p_v - \log q_v)}_{\text{log-ratio}}
$$

The code holds `logp` $= \log p_v$ and `logq` $= \log q_v$, so the log-ratio is directly `logp - logq`. But the expectation's *weight* must be $p_v$ itself, not $\log p_v$ — and `logp.exp()` is simply how you recover it: $\exp(\log p_v) = p_v$.

**Why compute in log-space and exponentiate only the weight?** Because `log(softmax(x))` underflows. Over a 152k vocabulary many tokens have probability $\sim 10^{-30}$; float32 `softmax` flushes them to exactly `0`, and $\log 0 = -\infty$ gives `inf - inf = nan` in the log-ratio. `log_softmax` computes $x_v - \mathrm{logsumexp}(x)$ in one stable step, so log-probabilities stay finite and accurate however small the probability.

Going *back* via `.exp()` is safe in a way the reverse is not: if $p_v$ underflows to `0`, the term is `0 × (finite) = 0`, the correct limit since $p\log p \to 0$. And because `log_softmax` never returns $-\infty$ for finite logits, the `0 × inf → nan` trap never fires.

## 5. Position alignment

For a sequence `[prompt (P tokens), answer (A tokens)]`, position $i$'s logits predict token $i{+}1$. Answer tokens occupy absolute positions $P \ldots P{+}A{-}1$, so they are predicted by logit rows $P{-}1 \ldots P{+}A{-}2$ — hence the slice `logits[0, P-1 : P-1+A]`. Row $j$ of that slice is:

$$
\text{row } j \;=\; P(\cdot \mid T,\, a_{<j})
$$

Row 0 is the distribution over the first answer token given the prompt alone.

**The alignment that matters:** the two branches have different prompt lengths ($P_\text{orig} \neq P_\text{rew}$), but each slices relative to its own $P$, so row $j$ in both means "distribution after the same answer prefix $a_{<j}$." The answer token ids are literally the same tensor in both passes, so the prefixes are identical.

## 6. The gradients

Write $z$ for the live branch's logits. Both formulas verified against autograd to machine precision.

**Live branch = rewritten** ($p$ fixed under `no_grad`, $q$ live):

$$
\frac{\partial D}{\partial z_j} = q_j \cdot \Big(\textstyle\sum_v p_v\Big) - p_j = q_j - p_j
$$

With $p$ constant the entropy term $\sum_v p_v \log p_v$ is a constant that drops out, leaving the familiar softmax cross-entropy gradient.

**Live branch = original** ($q$ fixed, $p$ live) — **not symmetric**, the entropy term now contributes:

$$
\frac{\partial D}{\partial z_j} = p_j \cdot \big(\log p_j - \log q_j - D\big)
$$

Backprop then carries these through the transformer's Jacobian to reach $\partial D / \partial e_t$ at each input embedding.

## 7. The identity test

When $q = p$: the first formula gives $q - p = 0$, the second gives $p_j(0 - 0) = 0$. **Both vanish analytically**, so an unchanged rewrite must produce zero gradients.

In float32, $\sum_v p_v = 1 \pm 10^{-7}$ over a 152k vocabulary rather than exactly 1, so the first formula leaves a residue of order $\varepsilon \cdot q_j \approx 10^{-7}$. Measured: $D = 0$ **exactly**, $\max|\nabla| = 7.17\times10^{-8}$ — the predicted rounding floor, not a bug. Real signal sits orders of magnitude above it (measured $\sim\!10^4\times$ higher).

## 8. Why `.float()` before the softmax

Two reasons, both about the 152k-wide reduction: `logsumexp` in fp16 risks overflow and loses precision on large-magnitude negative log-probs, and summing 152k terms accumulates badly at fp16's ~3 significant digits. Cross-scoring deliberately visits very unlikely tokens — that is where the signal lives — so fp32 is required for both the value of $D$ and a clean gradient.

## 9. Which side to attribute (`--attribute`)

| value | gradient | answers |
|---|---|---|
| `rewritten` | $\partial D / \partial e(T_\text{rew})$ | which surviving rewrite tokens are load-bearing — what not to touch when editing further |
| `original` | $\partial D / \partial e(T_\text{orig})$ | which original details the divergence depends on |
| `both` | both, on the same answers | — |

**The deletion blind spot** is why `original` exists. Gradients only exist for tokens that are *present*, but privatization works by *removing* content. Deleted details have no token in the rewrite to receive a gradient — they do have one in the original.

## 10. Caveats

- $D$ is conditioned on the teacher-forced answer; average over several sampled answers (`--n_answers`) rather than trusting one.
- **Gradients are local.** Token substitution is a discrete jump, so treat rankings as a candidate list to verify with real forward passes. For a signed "what if I swap this token," use the HotFlip-style directional derivative $\langle \nabla_{e_t} D,\, e_\text{alt} - e_t \rangle$ toward a real alternative, **not** grad×input — whose implicit baseline is the zero embedding, which is not a token. Also note grad×input $= \|\nabla\|\,\|e_t\|\cos\theta$ scales with embedding norm, and in Qwen2.5-3B those norms span 0.38–1.63 (p99/p1 $= 3.3\times$), a systematic ranking bias grad-norm does not have.
- Prompts of different length place the same answer token at different RoPE positions, so a small slice of $D$ is positional rather than semantic.
- When the rewrite barely shifts the distribution, $D \approx 0$ and gradients approach the noise floor — check magnitude before interpreting.
- Chat-template tokens collect gradient too; `content_span` finds the raw-text span by diffing against an empty-content render, and ranking covers content only unless `--show_template`.

## 11. Extra diagnostic

Per-position KLs are retained, and the report prints the share of $D$ falling inside the first `--front_tokens` answer positions. A high share means the divergence is concentrated at the answer opening — the **persona echo** (*"As a 34-year-old midwife in Kraków…"*) — rather than permeating the advice, which bears directly on how much of the KL metric's measured cost is surface mirroring versus genuine personalization loss.

## 12. Usage

```bash
srun -A projects --gpus 5060ti:1 -t 00:15:00 \
    python kl_grad_attribution.py --attribute both --n_answers 3

# fixed answer instead of sampling; rank template tokens too
python kl_grad_attribution.py --target "..." --show_template
```

```python
from kl_grad_attribution import kl_attribution, aggregate, front_share
r = kl_attribution(model, tok, original, rewritten, answer, device, attribute="original")
r["kl_total"], r["kl_per_token"]      # D in nats, and per answer token
r["kl_per_pos"]                       # [|a|] per-position KLs
r["grad_norm"], r["grad_x_input"]     # per prompt token
```
