# Privacy–utility loop: roadmap

Status: 2026-10-04. Items are implemented one at a time and may change along the way.

## How the loop works now

1. **Privacy step** (`privacy/privacy_rewrite.py`): delete every run of 1–4 words and measure how much the attackers' (Qwen yes/no probe, Laya) probability `P_a` drops. Rewrite the best spans with an LLM generalization or a deletion, whichever leaks least. A deletion leaves a `[…]` slot.
2. **Utility step** (`utility/grad_attribution/kl_mask_llm_rewrite_multi.py`): D is the teacher-forced KL in nats per answer token, averaged over K = 10 answers of 256 tokens, sampled once from the user's message x₀. If D ≤ δ = 0.47 the prompt is not edited. Otherwise the slots are filled with 0–4-word phrases, then single tokens are swapped, each edit only if it lowers D.
3. **Loop** (`privacy_utility_loop.py`): alternate the two steps until the leak (max over attributes and attackers) is ≤ τ = 0.5. The utility step runs last, so it wins a stalemate. After the last iteration the LLM fixes grammar if the text has surprisal spikes.

Every edit in both steps must pass a fluency guard: it may not add more than ε = 4 nats of surprisal spikes.

Design rules:
- The utility step may put private attributes back. If it does, the question cannot afford to lose them, so there is no privacy check on utility edits.
- No grammar or position heuristics, because prompts are not always natural.

## Phase 0: fixes

- [x] `--utility_ref` defaults to `user` (x₀), so the reference no longer drifts to a garbled earlier rewrite.
- [x] Reference answers are sampled once and reused (both utility backends). `--utility_ref iteration` still resamples every iteration.
- [x] D is in nats per answer token (`kl_of`, `evaluate_candidates`, `token_gradients`). Answers went from 64 to 256 tokens and K from 4 to 10, `--eval_chunk_size` from 32 to 8 (same memory), and δ from 55 (summed) to 0.47 nats/token.
- [x] Early-stop `break` in the privacy step's right-to-left span loop: stop once nothing leaks. In the fireworks run a second span was rewritten after the leak was already 0, on the Σ P_a tie-break.
- [x] Added with it: a privacy edit is kept only if it lowers the hinge Σ_a max(0, P_a − τ), measured against exactly τ (no margin), not just Σ P_a. Spans are picked one per leaking attribute by their blame on *that* attribute, at most `--top_m` per iteration. A second cue for the same attribute waits for the next iteration, which re-blames on the edited text. This stopped "conclusion" → "conclusionary remarks" and "stress" → "anxiously" in the thesis run.
- [ ] Fix the `--question` flag in `privacy/attribute_probe.py`. Its name was corrupted by pasted text; this only affects the standalone CLI.
- [ ] Single-token swaps: decision pending, see below. Since Phase 2 the swaps pass the fluency guard too, so keeping them on is now a real option.

### Open decision: the single-token swaps

With the swaps off (`--n_steps 0`) the utility step still runs, but it can only refill *deleted* spans. Generalized spans ("25 years old" → "individual") and the rest of the prompt become untouchable.

| Option | Fluency | Utility step can repair |
|---|---|---|
| Keep the swaps (current) | breaks when D stays above δ after gap filling | anything |
| `--n_steps 0` | fluent | deletions only |
| Generalized spans become slots too, then drop the swaps | fluent | every privacy edit |

## Phase 1: measure before tuning

- [ ] Run the loop over `utility/data/prompt_rewrites_100.csv` (one new script) and score each result:
  - **privacy:** a held-out Qwen 32B probe on each row's `protected_attrs`, plus the attacker's guess accuracy against the persona in `id` (the protocol of Staab et al.);
  - **utility:** per-token KL and centroid cosine against the original;
  - **fluency:** prompt NLL per token.

  Baselines: the dataset's `priv_preserving` and `priv_targeted` columns.
- [x] Calibrate δ on per-token D from the existing 256-token run: 0.47. Re-check it on loop outputs.
- [ ] Log `front_share`, the share of D in the first answer tokens (the persona echo). It decides whether the dropped scrubbed-answer idea comes back.

## Phase 2: naturalness

- [x] Fluency guard: in both steps (privacy candidates; kl_multi gap fillers and token swaps), reject an edit whose surprisal spikes `Σ_t max(0, NLL_t − 8 nats)` (`disfluency` in `utility/metrics/kl_utility_scoring.py`) grow by more than ε = 4 nats. This also stops privacy "gains" that are only broken grammar fooling the probes. Mean NLL per token was tried first and dropped: deleting predictable words ("25 years old") raises it as much as a broken edit does. On 21 edits from the runs (Qwen 3B), ε = 4 passes all 8 clean ones and blocks 9 of 10 broken ones; the one it misses starts from already-garbled text.
- [x] Final fluency fix after the last iteration (`--fix_candidates`, default 4): the LLM proposes minimal grammar fixes. A fix is kept only if it has fewer spikes, leaks no more than max(leak, τ) and has a D no higher than max(D, δ); the most fluent one wins. kl_multi only.
- [ ] Re-check ε on loop outputs in Phase 1. `cos_mask` has neither the guard nor the fix.

## Phase 3: privacy strength

- [ ] Score span occlusion on the logit or `nll` instead of the saturated `p`. Two cues that each push p to 1.0 are invisible when deleted one at a time.
- [ ] Qwen 7B as the in-loop attacker, and 32B only for evaluation. The 3B model is attacker, rewriter and scorer at once, so its gains are partly circular.
- [ ] The dataset's 15 attributes instead of the loop's 5.
- [ ] Pass the privacy step's generalizations of a deleted span to the gap filler as candidates. D then decides between deleting and generalizing, and a deleted question can come back.

Also open: `cos_mask` drops slots instead of filling them.

## Laya fine-tuning (added 2026-10-05)

Goal: make Laya a real second attacker. Not fine-tuned, it ranks leaks somewhat but its P sits too low (see Evidence). Two claims, two kinds of evidence:
1. Fine-tuned Laya detects leaks better: before/after on held-out test sets (Steps 2 and 4).
2. The loop produces more private rewrites with it: needs the Phase 1 eval script (Step 5).

### Step 1: data
- [ ] One row per pair: `id`, `attributes`, `values`, `leaking`, `safe`, `not_inferable`, `edit_type`, `cue_type`. Generated by a Claude Code cloud session in a separate, empty GitHub repo (it cannot see this one), then copied into `privacy/data/laya_ft/`.
  - `attributes`: **every** attribute the leaking prompt reveals, out of 16: the dataset's 15 without criminal records, immigration status, sexual orientation and disabilities, plus name, residential address, date of birth, credit card number and email address. `values`: the user's value for each, for a later guess-accuracy eval.
  - `not_inferable`: 1–2 attributes the generator vouches cannot be guessed from either message. Sampling them at random would mislabel implied attributes ("cardiologist" also reveals education level).
  - `safe`: a minimal edit of `leaking`. Only the cues change: deleted, generalized, or moved to another person ("my sister is a nurse").
  - Deletions look the way the loop leaves them: the span cut out, no `[…]`. `score()` strips slots before the attackers see the text.
  - Leaking cues both explicit ("I'm 34") and indirect ("I pour drinks all night"), 1–3 attributes per prompt.
- [ ] Fix the granularity rule first: does "physician" leak profession? Does "Poland" leak location? A default table is in the generation prompt; confirm it before generating.
- [ ] Generate with a model other than Qwen 32B, so the judge stays independent.
- [ ] Train: ~2,000 pairs (~130 per attribute). Test: 300–450 pairs (20–30 per attribute), separate personas and seed, near-duplicates of train removed, labels checked by hand (at least a random 100).
- Items per pair: (leaking, a) → yes and (safe, a) → no for a in `attributes`, plus (leaking, b) → no for b in `not_inferable`. The last kind makes Laya read the attribute instead of learning "vague text → no".

### Step 2: before numbers
- [ ] Eval script: `--laya_path` + test CSV → accuracy at P = 0.5, AUROC, pair accuracy P(leaking) > P(safe), ECE and Brier, per-attribute accuracy, bootstrap 95% CIs over pairs.
- [ ] Test sets: the synthetic one, and `prompt_rewrites_100.csv` as out-of-distribution (original vs `priv_targeted` on `protected_attrs`, without the 13 of 200 cells on attributes Laya is not trained on: disabilities 9, sexual orientation 2, criminal records 1, immigration status 1).
- [ ] Rows: Laya; Laya + a bias σ(α·ℓ + β) fitted on train (is the gain more than a threshold shift?); Qwen 3B / 7B / 32B probes as reference.

### Step 3: training (`privacy/laya_finetune.py`, one GPU job)
Follows the creators' fine-tuning notebook (`notebooks/laya_finetune_typed_decisions_2xT4_kaggle.ipynb` in the Laya repo), by the user's choice on 2026-10-05. Code done the same day (`laya_finetune.py`, `LayaProbe.item_scores`, `sbatch_scripts/run_laya_finetune.sbatch`). Smoke-tested on CPU: a reload with `laya.load` reproduces the reported numbers exactly. No GPU run yet.
- [x] Forward through `LayaProbe.item_scores` (raw (z_false, z_true), one row per item): the loop's token layout and question (`laya_probe.QUESTION`).
- [x] Loss as in the notebook: a policy-gradient term on the proper score (G = 4 noisy logit samples, σ 0.4 → 0.1, log + 0.75·spherical) plus 1.0 × cross-entropy.
- [x] AdamW, encoder 2.5e-5 / head 1e-4, weight decay 0.01, cosine to 1e-6, effective batch 64 (8 × 8 accumulation on one GPU), clipping 1.0, 4 epochs, last epoch kept. `--freeze_encoder` for head only.
- [x] Held out: 70 pairs (402 items, the notebook's min(400, 10%)), whole pairs, never trained on. The noul temperature is fit on them by LBFGS, after the fp16 round trip.
- [x] Save to `/scratch/users/mkilic/laya-privacy-n<pairs>` (`/work` is at 13 of 20 GB): fp16 weights (842 MB, as shipped) and the config as the notebook writes it (temperatures [1.2, 1.2, T_noul], no per-bucket temperatures, which also silences the load warning). Copy the one to keep to `models/laya-privacy`.
- Deviations, hardware only: one GPU instead of two, bf16 instead of fp16 with a grad scaler, no gradient checkpointing.
- [ ] Run it, plus the learning curve: `N_PAIRS` = 250 / 500 / 1,000 / all (1,930). Still rising at 1,930 → generate more.

### Step 4: after numbers
- [ ] Same script and test sets on `models/laya-privacy`, plus McNemar's test on before/after correctness of the same items. Claim 1 holds if both test sets improve beyond the fitted-bias row.

### Step 5: in the loop (after Phase 1)
- [ ] Run the loop on the same prompts with old and fine-tuned Laya (`--laya_path`), and score both outputs with the held-out 32B probe.

Rules: never train on `prompt_rewrites_100.csv` (out-of-distribution test and Phase 1 set), PUPA or SynthPAI. Laya's own scores on loop outputs are not evidence, since the loop optimized against them.

## The four original redesign steps

| Step | Status |
|---|---|
| 1. Spans of 1–4 words + deletion | done; Phase 3 changes the scoring |
| 2. Utility relevance U(s) via attribute-scrubbed answers | dropped (KL already measures meaning loss); revisit only if `front_share` shows the echo dominates D |
| 3. Deleted spans handed to the utility step as slots | done |
| 4. Multi-word gap filling | done, with `--kl_threshold`; Phase 2 and Phase 3 extend it |

## Evidence behind the plan

| Measurement | Result |
|---|---|
| Qwen probe: share of `util_broken` flagged as leaking (p > 0.5) | 3B 28%, 7B 1%, 32B 1% |
| Laya: share flagged as leaking | originals 15%, `priv_preserving` 15% |
| Original vs `priv_targeted` on `protected_attrs`, AUROC / accuracy at P = 0.5 | Laya 0.76 / 0.54 · Qwen 3B 0.87 / 0.81 · 7B 0.92 / 0.88 · 32B 0.94 / 0.88 |
| KL(orig ‖ rew) in nats/token, 256-token answers (3B, K = 30) | util_preserving 0.20 · priv_targeted 0.27 · priv_preserving 0.36 · util_broken 0.57 |
| δ = 0.47 (K = 5) | leaves 186/200 privacy rewrites alone, edits 90/100 off-topic rewrites |
| Stress run, old summed D with 64-token answers | garbled "I am programmer scientist a mind, …" scored 33.9, below the sensible "I am a individual, …" at 51.9 |

## Order

Finish Phases 0 and 1 first. Later choices (δ, attacker, echo) should come from Phase 1's numbers, not from single runs.

Laya fine-tuning Steps 1–4 do not depend on the loop and can run alongside it; Step 5 waits for Phase 1.
