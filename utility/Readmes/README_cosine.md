# Cosine-Similarity Utility Scoring for Privacy Rewrites

**Files:** `cosine_utility.py` (the metric), `topk_sample.py` (the sampler it builds on)
**Date:** 2026-07-22
**Companion to:** the likelihood-based metric in `utility_scoring.py` (see `README.md`) — same sampling front end, different comparison back end.

## 1. Goal

As before, the utility of a privacy rewrite is how little it shifts the model's *answer distribution*: we compare $P(A \mid T_\text{orig})$ and $P(A \mid T_\text{rew})$, not two single answers. Here the comparison happens in **embedding space**: each sampled answer becomes a vector, and the two answer *sets* are compared by cosine similarity. This trades the exactness of the likelihood metric for **phrasing invariance** — two answers with the same content but different wording land close in embedding space, while token-space KL counts them as divergence.

## 2. The metric

Sample $K$ answers per prompt, embed them ($e_i$ for the original, $f_j$ for the rewrite, all L2-normalized), and compute three cosine averages:

$$
\text{cross} = \frac{1}{K_o K_r} \sum_{i,j} \cos(e_i, f_j)
\qquad
\text{self}_o = \frac{1}{K_o(K_o-1)} \sum_{i \ne i'} \cos(e_i, e_{i'})
\qquad
\text{self}_r = \frac{1}{K_r(K_r-1)} \sum_{j \ne j'} \cos(f_j, f_{j'})
$$

These combine into a **maximum mean discrepancy (MMD) two-sample statistic with a cosine kernel**:

$$
\mathrm{MMD}^2 = \text{self}_o + \text{self}_r - 2\,\text{cross}
$$

$$
\text{UTILITY} = \mathrm{clamp}\!\left(1 - \tfrac{\mathrm{MMD}^2}{2},\; 0,\; 1\right) \in [0, 1]
$$

**Why the self-similarity terms matter:** raw cross-set cosine is uninterpretable on its own — answers to the *same* prompt already differ, and same-topic text clusters in a narrow cosine band. The within-set means are exactly the "same-prompt ceiling": if the two distributions are identical, $\mathbb{E}[\text{cross}] = \mathbb{E}[\text{self}]$ and $\mathrm{MMD}^2 \to 0 \Rightarrow$ utility $\to 1$. The calibration that had to be done manually for a plain cosine score is built into the statistic.

**Interpretation:** utility $\approx 1$ — answer distributions indistinguishable; utility $\approx 0$ — the two answer sets are internally coherent but mutually unrelated ($\text{cross} \approx 0$, $\text{self} \approx 1$). A rewrite that collapses answer *diversity* is penalized even if answers stay on topic: $\text{self}_r$ rises above cross.

## 3. Embeddings

Each answer text is embedded **standalone** (no prompt in the context — the embedding must reflect answer content, not which prompt produced it):

1. Forward pass of the answer text through the same causal LM (`output_hidden_states=True`).
2. Mean-pool the final hidden layer over non-pad tokens (attention-mask weighted), in float32.
3. L2-normalize, so dot products are cosine similarities.

This keeps the pipeline dependency-free (no sentence-transformers download on the cluster) at the cost of embedding quality — decoder-LM mean-pooling is serviceable but not trained for semantic similarity. Swapping in a dedicated sentence embedder later is a one-function change (`embed_answers`).

## 4. How it is computed (pipeline)

1. **Sample** $K$ answers from each prompt (`topk_sample.topk_answers`, `mode="sample"`), identical generation settings on both sides. Unlike the likelihood metric, **tempering would be admissible here** (there is no likelihood mismatch to bias — samples only need to be representative), but the defaults are plain sampling (temperature $1.0$, top-p $1.0$) to match `utility_scoring.py`, so both metrics compare the *same* answer distribution and their disagreements are attributable to the metric, not the sampling. Empty answers are dropped; $\ge 2$ per side are required (self-similarity needs pairs).
2. **Embed** both answer sets (`embed_answers`, batched).
3. **Compare** (`cosine_stats`): pairwise cosine matrix $\Rightarrow$ cross mean and the two self means (diagonal excluded — an answer is trivially similar to itself and would inflate the ceiling) $\Rightarrow \mathrm{MMD}^2 \Rightarrow$ utility.

The result dict retains per-sample diagnostics: each answer's mean cross-similarity and its best-matching answer from the other set (`--show_samples`), so you can see *which* original answers have no counterpart under the rewrite.

## 5. Relation to the likelihood metric

| | `utility_scoring.py` (KL) | `cosine_utility.py` (this) |
|---|---|---|
| Compares | exact answer likelihoods | embeddings of samples |
| Phrasing sensitivity | counts wording changes | largely invariant |
| Identical rewrite | $= 1.0$ **exactly** | $\approx 1$ up to sampling noise |
| Tempered sampling | biases the estimate | fine (both sides identical) |
| Diversity collapse | visible in KL | explicitly penalized via $\text{self}_r$ |
| Needs | forward passes only | forward passes only |

They are complementary axes: where both agree, trust the result; where they disagree (KL high, cosine high), the rewrite changed *wording/style* more than *content* — arguably utility preserved.

## 6. Validation performed

- Mechanics validated on a tiny CPU model: identical prompts $\Rightarrow$ utility near $1$; distinct prompts $\Rightarrow$ finite $\mathrm{MMD}^2$, utility $\in [0,1]$; the $\ge 2$-samples guard fires on demand.
- Semantic discrimination (privatized vs. unrelated rewrites separating cleanly) requires a trained model — run on GPU, see below.

## 7. Usage

```bash
# CLI (same defaults as utility_scoring.py for side-by-side comparison)
srun -A projects --gpus 5060ti:1 -t 00:15:00 \
    python cosine_utility.py --show_samples

# custom pair, more samples
python cosine_utility.py --original "..." --rewritten "..." --k 10
```

```python
# as a library
from cosine_utility import cosine_utility_score
res = cosine_utility_score(original, rewritten, model, tok, k=5)
res["utility"]                      # scalar in [0, 1]
res["mmd2"]                         # cosine-kernel MMD^2
res["cross_mean"], res["self_orig"], res["self_rew"]   # components
res["samples_orig"], res["samples_rew"]                # best-match diagnostics
```
