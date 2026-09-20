# Likelihood-Based Utility Scoring for Privacy Rewrites


https://claude.ai/code/artifact/b9590018-8042-4fe8-9842-eb48a0c5e98a


**Files:** `utility_scoring.py` (the metric), `topk_sample.py` (the sampler it builds on)
**Date:** 2026-07-18

## 1. Goal

When a prompt is rewritten for privacy, some of its usefulness may be lost. We define the **utility of a rewrite** as how little it shifts the model's *answer distribution*: if the model would answer $T_\text{rew}$ the same way it would answer $T_\text{orig}$, nothing of use was destroyed. So utility is a similarity between two distributions, $P(A \mid T_\text{orig})$ and $P(A \mid T_\text{rew})$ — not between two single answers.

## 2. The metric

Utility is derived from the **symmetrized, per-token, Monte-Carlo-estimated KL divergence** (Jeffreys divergence) between the two answer distributions.

$$
\mathrm{KL}\!\left(P_\text{orig} \parallel P_\text{rew}\right)
\approx
\frac{1}{K} \sum_{i=1}^{K}
\frac{\log P(a_i \mid T_\text{orig}) - \log P(a_i \mid T_\text{rew})}{|a_i|},
\qquad a_i \sim P(\cdot \mid T_\text{orig})
$$

where $|a_i|$ is the answer length in tokens. The full score combines both directions:

$$
J = \mathrm{KL}\!\left(P_\text{orig} \parallel P_\text{rew}\right) + \mathrm{KL}\!\left(P_\text{rew} \parallel P_\text{orig}\right)
\quad [\text{nats/token}]
$$

$$
\text{UTILITY} = \exp\!\big(\!-\max(J, 0)\big) \in (0, 1]
$$

Each direction is estimated from $K$ sampled answers; per-token normalization (dividing by $|a_i|$) stops long answers from dominating. The $\max(J, 0)$ guards against Monte-Carlo noise pushing the estimate slightly negative (true $J \ge 0$). The standard error of each directional estimate is reported alongside the mean.

**Interpretation:** $J = 0 \Rightarrow$ utility $1.0$ (distribution unchanged); $J = 0.1$ nats/token $\Rightarrow 0.90$; $J = 1.0 \Rightarrow 0.37$. The score is symmetric in the two prompts.

## 3. Why this metric

- **The distributions are actually available.** An autoregressive LM assigns an exact likelihood to *any* candidate answer under *any* prompt via one teacher-forced forward pass: $P(a \mid T) = \prod_{t} P(a_t \mid T, a_{<t})$. Generation is only needed to obtain samples; evaluation is a forward pass. This makes a direct KL estimate possible — no embedding model, no judge LLM.
- **Identity invariance, exactly.** For an unchanged prompt the two conditionals coincide, so every log-prob difference cancels sample-by-sample and utility $= 1.0$ *exactly*, independent of sampling noise. (The previous LLM-judge utility failed precisely this test — it scored identical prompts $0.0$; see `knowledge_graph/REVIEW.md`, P1.)
- **Low variance at small $K$.** Every sample contributes a full-sequence likelihood rather than a single point in embedding space.
- **The signal is the right one.** An answer that says "as a doctor in New York…" is *systematically* less likely under the privatized prompt — that drop is exactly the utility loss being measured, not an artifact.

## 4. How it is computed (pipeline)

1. **Sample** $K$ answers from each prompt (`topk_sample.topk_answers`, `mode="sample"`), with **temperature $= 1.0$ and top_p $= 1.0$** (plain ancestral sampling). Both prompts use identical generation settings and the chat template with `add_generation_prompt=True`.
2. **Cross-score** each answer under *both* prompts (`score_answers`):
	- Build `[chat-templated prompt] + [answer tokens]`, right-padded into a batch with an attention mask; one forward pass, no generation.
	- `log_softmax` of the logits **in float32** — cross-scoring deliberately visits very unlikely tokens, exactly where fp16 underflows to $-\infty$.
	- Index arithmetic: the token at position $t$ is predicted by the logits at position $t-1$, so an answer spanning positions $[P, L)$ is scored by logit positions $[P-1, L-1)$, where $P$ = prompt length in tokens.
	- Sum over the answer span $\Rightarrow$ total $\log P(\text{answer} \mid \text{prompt})$; also return the answer token count for normalization.
3. **Combine** (`directional_kl`, `utility_score`): per-sample per-token deltas $\Rightarrow$ mean and standard error per direction $\Rightarrow J \Rightarrow \exp(-\max(J, 0))$. Empty answers (immediate EOS) are excluded — they carry no evidence, and counting them as zero divergence would bias utility upward.

The result dict retains every sample's cross-scores (`lp_own`, `lp_other`, `delta_per_token`) so you can inspect *which* answers the rewritten prompt finds unlikely (`--show_samples`).

## 5. Assumptions and caveats

- **Sampling and scoring must use the same distribution.** The scorer uses the full softmax, so samples must be plain (temp $1.0$, no top-p). Tempered sampling on both sides measures similarity of the *tempered* distributions — a valid but different notion; top-p is worse: a token outside the other prompt's nucleus would have probability $0 \Rightarrow \log = -\infty$. Hence the defaults.
- **Token-space KL partially counts phrasing.** Two answers with identical content but different wording register as divergence. Complementary embedding-based metrics (mean-cosine / MMD / Wasserstein over sentence embeddings) are paraphrase-invariant and can be added as a second axis.
- **Answers are re-tokenized from text** for scoring rather than reusing the sampled token ids. BPE round-trips of generated text are near-idempotent, and the terminator token is excluded consistently on both sides, so the difference-of-logprobs is unaffected in practice.
- **$K = 5$ is a noisy estimate.** Use the reported standard errors; increase `--k` if they are large relative to $J$ (cost grows linearly and is dominated by generation, not scoring).

## 6. Validation performed

- Identical prompts $\Rightarrow J = 0.000000$, utility $= 1.000000$ exactly (invariance test, asserted).
- Distinct prompts $\Rightarrow$ positive directional KLs with finite standard errors, utility $\in (0, 1)$.
- (Mechanics validated on a tiny CPU model; semantic discrimination requires a trained model — see usage below.)

## 7. Usage

```bash
# CLI (defaults are the doctor/stress pair used in earlier experiments)
srun -A projects --gpus 5060ti:1 -t 00:15:00 \
    python utility_scoring.py --show_samples

# custom pair, more samples
python utility_scoring.py --original "..." --rewritten "..." --k 10
```

```python
# as a library (e.g. from graph_privacy.py)
from utility_scoring import utility_score
res = utility_score(original, rewritten, model, tok, k=5)
res["utility"]              # scalar in (0, 1]
res["jeffreys_per_token"]   # J in nats/token
res["kl_orig_to_rew"], res["kl_orig_to_rew_stderr"]   # + mirror direction
res["samples_orig"], res["samples_rew"]               # per-sample diagnostics
```
