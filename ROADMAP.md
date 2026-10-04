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
- [ ] Early-stop `break` in the privacy step's right-to-left span loop: stop once nothing leaks. In the fireworks run a second span was rewritten after the leak was already 0, on the Σ P_a tie-break.
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
| KL(orig ‖ rew) in nats/token, 256-token answers (3B, K = 30) | util_preserving 0.20 · priv_targeted 0.27 · priv_preserving 0.36 · util_broken 0.57 |
| δ = 0.47 (K = 5) | leaves 186/200 privacy rewrites alone, edits 90/100 off-topic rewrites |
| Stress run, old summed D with 64-token answers | garbled "I am programmer scientist a mind, …" scored 33.9, below the sensible "I am a individual, …" at 51.9 |

## Order

Finish Phases 0 and 1 first. Later choices (δ, attacker, echo) should come from Phase 1's numbers, not from single runs.
