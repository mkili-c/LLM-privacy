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
  - **privacy:** a held-out Qwen 32B probe on each row's `protected_attrs`, plus the attacker's guess accuracy against the persona in `id` (the protocol of Staab et al.). Superseded 2026-10-06: the Evaluation pipeline uses an external two-step LLM judge and no 32B;
  - **utility:** per-token KL and centroid cosine against the original;
  - **fluency:** prompt NLL per token.

  Baselines: the dataset's `priv_preserving` and `priv_targeted` columns. The script is built as Steps 2–3 of the Evaluation pipeline below, which also covers real data and published anonymizers.
- [x] Calibrate δ on per-token D from the existing 256-token run: 0.47. Re-check it on loop outputs.
- [ ] Log `front_share`, the share of D in the first answer tokens (the persona echo). It decides whether the dropped scrubbed-answer idea comes back.

## Phase 2: naturalness

- [x] Fluency guard: in both steps (privacy candidates; kl_multi gap fillers and token swaps), reject an edit whose surprisal spikes `Σ_t max(0, NLL_t − 8 nats)` (`disfluency` in `utility/metrics/kl_utility_scoring.py`) grow by more than ε = 4 nats. This also stops privacy "gains" that are only broken grammar fooling the probes. Mean NLL per token was tried first and dropped: deleting predictable words ("25 years old") raises it as much as a broken edit does. On 21 edits from the runs (Qwen 3B), ε = 4 passes all 8 clean ones and blocks 9 of 10 broken ones; the one it misses starts from already-garbled text.
- [x] Final fluency fix after the last iteration (`--fix_candidates`, default 4): the LLM proposes minimal grammar fixes. A fix is kept only if it has fewer spikes, leaks no more than max(leak, τ) and has a D no higher than max(D, δ); the most fluent one wins. kl_multi only.
- [ ] Re-check ε on loop outputs in Phase 1. `cos_mask` has neither the guard nor the fix.

## Phase 3: privacy strength

- [ ] Score span occlusion on the logit or `nll` instead of the saturated `p`. Two cues that each push p to 1.0 are invisible when deleted one at a time.
- [ ] Qwen 7B as the in-loop attacker; evaluation by the Evaluation pipeline's external judge (no 32B, 2026-10-06). The 3B model is attacker, rewriter and scorer at once, so its gains are partly circular.
- [ ] The dataset's 15 attributes instead of the loop's 5. Partly done: 10 since 2026-10-06 (Evaluation pipeline).
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
- [x] Eval script `privacy/laya_eval.py` (+ `sbatch_scripts/run_laya_eval.sbatch`), written 2026-10-05 and smoke-tested on CPU. It covers base and every checkpoint in one run, with three metrics only (the user's choice): accuracy at P = 0.5 (next to the majority baseline), AUROC, and pair accuracy P(leaking) > P(safe). Each comes with a paired bootstrap 95% CI over pairs, per model and for the difference to base, plus accuracy by item kind, cue_type, edit_type and attribute. Brier, ECE, balanced accuracy and McNemar were dropped. Test items are built by `laya_finetune.make_items` itself.
- [x] Test sets: the synthetic one, and `prompt_rewrites_100.csv` as out-of-distribution (original vs `priv_targeted` on `protected_attrs`, without the 13 of 200 cells on attributes Laya is not trained on: disabilities 9, sexual orientation 2, criminal records 1, immigration status 1).
- [x] Rows: Laya; Laya + a bias σ(α·ℓ + β) fitted on train.csv (is the gain more than a threshold shift?).
- [ ] Qwen 3B / 7B probes on the same test items as reference (not in laya_eval.py). No 32B (too slow; 2026-10-06).
- [ ] Run it on the GPU.

### Step 3: training (`privacy/laya_finetune.py`, one GPU job)
Follows the creators' fine-tuning notebook (`notebooks/laya_finetune_typed_decisions_2xT4_kaggle.ipynb` in the Laya repo), by the user's choice on 2026-10-05. Code done the same day (`laya_finetune.py`, `LayaProbe.item_scores`, `sbatch_scripts/run_laya_finetune.sbatch`). Smoke-tested on CPU: a reload with `laya.load` reproduces the reported numbers exactly. No GPU run yet.
- [x] Forward through `LayaProbe.item_scores` (raw (z_false, z_true), one row per item): the loop's token layout and question (`laya_probe.QUESTION`).
- [x] Loss as in the notebook: a policy-gradient term on the proper score (G = 4 noisy logit samples, σ 0.4 → 0.1, log + 0.75·spherical) plus 1.0 × cross-entropy.
- [x] AdamW, encoder 2.5e-5 / head 1e-4, weight decay 0.01, cosine to 1e-6, effective batch 64 (8 × 8 accumulation on one GPU), clipping 1.0, 4 epochs, last epoch kept. `--freeze_encoder` for head only.
- [x] Held out: 70 pairs (402 items, the notebook's min(400, 10%)), whole pairs, never trained on. The noul temperature is fit on them by LBFGS, after the fp16 round trip.
- [x] Save to `/scratch/users/mkilic/laya-privacy-n<pairs>` (`/work` is at 13 of 20 GB): fp16 weights (842 MB, as shipped) and the config as the notebook writes it (temperatures [1.2, 1.2, T_noul], no per-bucket temperatures, which also silences the load warning). Copy the one to keep to `models/laya-privacy`.
- Deviations, hardware only: one GPU instead of two, bf16 instead of fp16 with a grad scaler, no gradient checkpointing.
- [x] Run it, plus the learning curve (jobs 404–407, 2026-10-05; logs in `sbatch_scripts/laya_ft_40*.out`). On the 402 held-out items (accuracy / AUROC): base 0.789 / 0.886; n250 0.950 / 0.968; n500 0.948 / 0.985; n1000 0.970 / 0.993; n1930 0.973 / 0.994. Flat from 1,000 on, within the held-out noise (about ±0.01). The fitted T hits Laya's cap of 5.0 for n250, n500 and n1000 (n1930: 4.74), so those stay over-confident. Checkpoints are on /scratch (auto-deleted): copy the kept one to `models/`.

### Step 4: after numbers
- [x] Done 2026-10-05 (job 408, `results/laya_eval/laya_eval_408*`). Claim 1 holds: both test sets improve far beyond the fitted-bias row.
  - Synthetic, base → n1930: accuracy 0.772 → 0.958 (+18.6 points [17.2, 20.1]; majority baseline 0.675), AUROC 0.833 → 0.981, pair accuracy 0.848 → 0.995. Base + bias: +0.2 points.
  - OOD: accuracy 0.535 → 0.888 (+35.3 [31.5, 38.9]), AUROC 0.749 → 0.947, pair accuracy 0.818 → 0.963. Base + bias: +6.4 points.
  - n1930 ≈ n1000: every difference CI includes or touches 0.
  - Remaining errors (n1930): mostly confident misses (median p 0.025). They are implied attributes (gender from a name, age from a birth date, education from a profession), cues that need world knowledge (symptoms, activities), identifiers belonging to someone else counted as the user's, and origin conflated with residence. Some "errors" are test labels where the cue survived the edit.
  - OOD gaps: political views stays at 0.50 (our data only has party affiliation; the OOD set uses ideology and religion words); addiction and mental health are almost absent from the training data; profession detection depends on phrasing ("How does a 41-year-old teacher…").

### Step 5: in the loop (after Phase 1)
- [ ] Run the loop on the same prompts with old and fine-tuned Laya (`--laya_path`), and score both outputs with the Evaluation pipeline's judge.

Rules: never train on `prompt_rewrites_100.csv` (out-of-distribution test and Phase 1 set), PUPA or SynthPAI. Laya's own scores on loop outputs are not evidence, since the loop optimized against them.

Decision (2026-10-05): no further Laya training now. Inferences that need world knowledge ("1st of August fireworks" → Switzerland, symptoms → condition, wealth from possessions) are the LLM attacker's job (Qwen, or a larger model); Laya covers cues stated in the text, quickly. Future work: a second dataset with implied attributes, customs and holidays, addiction and mental health, ideology for political views, profession described by activity, and others' identifiers. Split it by fact or country to test whether the encoder generalizes beyond memorized facts.

## Evaluation pipeline (added 2026-10-06)

Goal: compare the loop with published anonymizers on real prompts, every method scored by the same script. This section builds Phase 1's eval script and extends it to real data and baselines.

Setup:
- **Test set:** PUPA-SD (real prompts with self-disclosures, i.e. our attribute threat model). `utility/data/prompt_rewrites_100.csv` is the dev set: harness checks and any tuning happen there, never on the test set. PUPA-TNB is dropped for now (2026-10-06, see Step 1).
- **Rewriters:** every method runs on Qwen2.5-7B-Instruct. No method ever sees the judge: a stronger LLM (Claude, run as a cloud Claude Code session). Nothing in the evaluation uses the Qwen 32B (too slow; decided 2026-10-06).
- **Inputs:** the anonymizers get only the prompt (`user_query`), never the row's labels. The loop and Staab et al. take an attribute list, and both get the same fixed 10 for every row. The other methods take none.
- **Attributes (decided 2026-10-06):** the loop's 5 plus education level, socioeconomic status, relationship status, family situation and nationality. The loop and the scorer use the same 10 (`ATTRIBUTES` in `privacy/attribute_probe.py`). This brings forward part of Phase 3's attribute item.
- **Baselines only produce rewrites**, all in one format (`id`, `method`, `rewrite`), and `score_rewrites.py` scores everything.

### Step 1: data
PUPA-SD: `data/pupa_sd/PUPA_SD.csv`, 166 rows, commit `a282782` of github.com/siyan-sylvia-li/Probabilistic_PAPILLON, sha256 `4a21bf51…`.
- [x] Downloaded (2026-10-06).
Built by `data/pupa_sd/build.py` (2026-10-06). **Labels are never edited (decided 2026-10-06): rows are only selected.**
- [x] Parse `pii_units` (`category: span; …`) and keep the disclosures in the categories that correspond to our attributes: occupation (and the typo `occupaion`) ↔ profession, location ↔ geographic location, health / mental_health / reproductive_health ↔ health condition, education ↔ education level, finance ↔ socioeconomic status, family ↔ family situation, race/nationality ↔ nationality, and age, gender, relationship_status directly. Category names and values stay verbatim; nothing reads them except the judge, and the report can group by our attribute names. Disclosures in other categories (events, appearance, PII, …) are left out.
- [x] Keep rows with at least one such disclosure and at most `MAX_LAYA_TOKENS` tokens. Write `data/pupa_sd/pupa_sd_eval.csv`: `id` (= `index`), `original` (= `user_query`), `truth` (JSON: the dataset's category → its values). Result: 141 rows, 207 (row, category) pairs, 311 values (occupation 62 incl. the typo, location 39, education 25, health 15 + reproductive 8 + mental 6, age 20, gender 16, family 6, finance 5, race/nationality 3, relationship_status 2).
- [x] Length cap (2026-10-06): `MAX_LAYA_TOKENS` = 424, counted with Laya's tokenizer. Laya reads 512 tokens; the longest of our 10 attribute questions takes 37 and [SEP] 1, leaving 474 for the message, and `laya_probe.py` raises above that. The 50 tokens of headroom (~10%) are for loop edits that lengthen the text. No PUPA-SD prompt lies between 424 and 474 tokens, so the headroom costs no rows. The cap drops 18 of 166 rows; the longest kept prompt is 408 tokens (325 words). It replaced a 300-word cap, which had no justification and kept 2 prompts over Laya's limit. Drop, never truncate: truncating would change the prompt.
- Some labels describe someone other than the user (row 90 is a patient's transcript), and some are loose ("one year of work experience" as profession). That is fine for the judge: it asks whether the disclosed value can still be inferred, not whose it is. A first-pass relabeling (135 hand fixes) was discarded by this decision.

PUPA-TNB: dropped for now (2026-10-06). Its 237 turns from 197 WildChat conversations (PAPILLON's evaluation split: applications, finance, quoted emails) are labeled with PII strings, not attributes. In a sample of 40 units: ~18 organizations, 11 places, 7 person names, mostly not the user's. That is what our method deliberately does not remove, and running the loop on its 201 rows within the cap would cost about a day on GB10. If revived: score it by PII leak rate, or by location only, using the WildChat country of each conversation (185 of 197) as an independent label.

SynthPAI, only for the Staab et al. reproduction check (comments, not prompts, so the loop does not run on it):
- [ ] The 250 profiles in `data/base_inferences/synthpai/inference_0.jsonl` of github.com/eth-sri/llm-anonymization.

### Step 2: scoring script (`score_rewrites.py`)
Metrics are Phase 1's list, with the judge in place of the probe.
- **Privacy (decided 2026-10-06): an inferability judge on the dataset's labels.**
  - A stronger LLM than any method (Claude, run as a cloud Claude Code session) works in two steps per labeled (row, attribute) pair, as in Staab et al. (decided 2026-10-06):
    1. **Guess, blind:** it sees only the text and the attribute name, and gives one guess or says the attribute cannot be inferred.
    2. **Compare:** a separate call sees the guess and the dataset's values and decides whether the guess matches any of them.

    It judges the originals and every rewrite, with method names hidden and items shuffled. Record the model ID and both prompts, and judge each item once. The blind first step keeps the judge from finding the answer just because it knows it.
  - Leak rate = (1/|P|) Σ_{(i,a)∈P} match(guess(x̃ᵢ, a), truthᵢₐ), where P = {(i, a) : match(guess(xᵢ, a), truthᵢₐ) = 1} is the set of labeled pairs guessed correctly from the original (the leak filter). P is the same for every method, so the original scores 1 by construction. PUPA-SD has 207 (row, category) pairs before the filter: a 95% CI of roughly ±7–10 points overall, and per-attribute numbers (categories grouped by our attribute names) only for profession, location, education and health.
  - [ ] Validate the judge: check ~50 of its decisions by hand and report the agreement, since every privacy number rests on it.
  - Checked 2026-10-06: none of the baselines replaces attributes with fake values by design. Staab et al. tell their anonymizer "do not invent new information"; PAPILLON's published Qwen-7B rewrites abstract ("[Manager's Name]", "Example County") and change some numbers and times, but no personal attributes.
- **Utility:** D (per-token KL against Qwen 7B's answers to the original, K = 10 answers of 256 tokens) and the centroid cosine.
- **Fluency:** prompt NLL per token.
- Each metric gets a bootstrap 95% CI over rows, per method and as a difference to the loop, broken down by attribute.
- [ ] Load the 7B once, for D and NLL. Reuse the D code in `utility/metrics/kl_utility_scoring.py`. The judge runs outside the script: the script writes the judge items (item id, text, attribute; method hidden, shuffled), and after the judge's two steps it reads back the guesses and matches.
- [ ] Sanity check on the dev set with the same judge: the originals' `protected_attrs` should be guessed correctly (values from the persona in `id`) and `priv_targeted`'s mostly not, and `util_broken` should be worst on D.

### Step 3: batch run of the loop
- [ ] Add `--input_csv` / `--output_csv` to `privacy_utility_loop.py`. `main()` loads the models once and calls `privacy_utility_loop()` per row, writing each row as it finishes and skipping rows already in the output, so a run resumes after the 24 h limit.
- [x] Extend `ATTRIBUTES` in `privacy/attribute_probe.py` to the 10 (2026-10-06). It is the `--attributes` default of the loop, `privacy_rewrite.py`, `laya_probe.py` and `privacy_token_grad.py`. All 10 names match Laya's training data.
- [ ] Runtime: ~7 min per short prompt on GB10 (7B) with 5 attributes. 10 attributes add attacker passes, so re-measure on a few rows first. 141 PUPA-SD rows at ≥ 7 min each is ≥ 16 h, so expect to resume across two jobs. Long prompts cost more (PUPA-SD's median is 57 Laya tokens, the longest kept 408).

### Step 4: baselines (one sweep script; target ½ day of coding, 2026-10-06)
Four baselines: Presidio, Loiseau et al., PAPILLON and Staab et al. Their code is used as released; only the plumbing around it is ours.

**Sweep script** `run_baselines.py --method <name> --data <eval csv> --out <rewrites csv>`:
- [x] Reads `id`, `original`; calls one small adapter per method; writes `id`, `method`, `rewrite` row by row to `results/rewrites/<data>_<method>.csv`; skips ids already written, so a run resumes; `--limit N` for smoke tests. Written 2026-10-06 with the Presidio adapter; the other three adapters go into `load_adapter`.
- [ ] Runs inside the method's own venv. The repos pin conflicting packages: Staab et al. need `openai==0.28.1`, while DSPy (PAPILLON, Loiseau et al.) needs openai ≥ 1. So there are three venvs.
- **Reproducibility** (2026-10-06): venvs live on permanent storage in `/work/projects/mkilic/venvs/<method>` and are built only from frozen requirements, `baselines/requirements-<method>.txt` (a `pip freeze` of the venv the results come from), by `bash baselines/setup_venvs.sh <method>`. Also pinned:
  - repo commits: eth-sri/llm-anonymization `5dafa6d`, siyan-sylvia-li/Probabilistic_PAPILLON `a282782`, gabrielloiseau/adaptive-text-anonymization `315ed31`;
  - the model: Qwen/Qwen2.5-7B-Instruct revision `a09a35458c702b33eeacc393d103063234e8bc28` (the copy in /scratch; re-download it with `hf download --revision` if scratch is cleaned);
  - each method's own sampling settings, as released, recorded in the deviations table.
  - [x] `presidio` (x86, Python 3.12.3, 789 MB): presidio-analyzer / -anonymizer 2.2.364, spaCy 3.8.16, `en_core_web_lg` 3.8.0. A rebuild from the requirements file matches the original exactly. No GPU needed.
  - [ ] `dspy` (PAPILLON, Loiseau et al.) and `staab`: without torch (the server holds the model), so each stays small. Built on GB10 (aarch64), where the sweep runs.
- [x] GPU (2026-10-06): **GB10**. A test on the 5060 Ti failed: the bf16 weights alone filled 15.2 of 16 GiB, some layers were offloaded to the CPU, there was no room for the KV cache, and the request returned a 500. The GB10 venv (`.llmvenv-gb10`) already has transformers 5.8 + fastapi + uvicorn for `transformers serve`. Quantization is not an option: it would change the model.
- **Decoding settings:** the server fills any setting a client doesn't send from Qwen's `generation_config.json` (sampling, temperature 0.7, top_p 0.8, top_k 20, repetition penalty 1.05), and continuous batching drops the repetition penalty (its log: "RepetitionPenaltyLogitsProcessor isn't supported by CB"). **Decision 2026-10-06: no continuous batching** (time is not a constraint). Qwen then decodes exactly as its `generation_config.json` says, repetition penalty included. With `--default-seed 0` and requests sent one at a time in a fixed order (Staab et al.'s `max_workers: 1`, sequential adapters), a rerun on the same software reproduces the rewrites. Each method's own explicit settings still override the defaults (e.g. Staab et al.'s temperature 0.1); record them in the deviations table. The saved rewrites CSVs remain the artifact the judge scores.
- [ ] One Slurm job on GB10: start `transformers serve` with Qwen2.5-7B-Instruct and `--default-seed 0` in the background, wait for `/health`, run the sweeps one request at a time, stop the server. Presidio's spaCy NER is model inference too, so it ran in its own CPU job.
- **Same backbone for everyone, including our loop.** The loop's `DEFAULT_MODEL` is still the 3B (`privacy/attribute_probe.py`), so its evaluation run must pass `--model_path` for the 7B at the same revision.
- [ ] GPU time: Staab et al. make 10 calls per prompt. Their main loop runs `max_num_iterations` = 3 times: infer attributes, anonymize, score utility. Then comes one final inference. Only infer → anonymize shapes the output; the utility and final-inference calls are measurements, but they stay because their code runs unchanged. That is about 1,410 calls for 141 rows, plus about 140 each for PAPILLON and Loiseau et al. Without batching GB10 decodes about 18 tokens/s (15.2 GB of weights / ~273 GB/s), so expect roughly 10 h in total. Time 5 rows first and extrapolate. Their utility model is GPT-4 in their configs; point it at the same Qwen server (a config change, logged as a deviation).

| Baseline | Source | Adapter |
|---|---|---|
| Presidio | presidio-analyzer / -anonymizer + spaCy `en_core_web_lg` | analyze + anonymize, default `replace` operator |
| Adaptive Text Anonymization (Loiseau et al. 2026) | github.com/gabrielloiseau/adaptive-text-anonymization | only the prompt text is released (`prompts/<task>/qwen_2_5_7b.md`), not the compiled program, so build the DSPy `text → anonymized_text` signature with that instruction |
| PAPILLON (Li et al. 2025) | github.com/siyan-sylvia-li/Probabilistic_PAPILLON | prompt level only (decided 2026-10-06): `PAPILLON(...).prompt_creater(userQuery=q).createdPrompt` is the rewrite, with no remote model and no aggregation. Zero-shot (PCD-k dropped 2026-10-06: it was optimized on PUPA-SD, our test set) |
| Adversarial anonymization (Staab et al. 2025) | github.com/eth-sri/llm-anonymization | the most plumbing, since their loop lives in their `main.py`: export the prompts as one-comment profiles plus a config yaml, run their `main.py` unchanged, read the anonymized comment back. Give every row the same 10 attribute types (their inference step otherwise takes them from each profile's labels), adding type descriptions their code lacks (e.g. health condition). Hardness/certainty filter off. `model_factory.py` imports every provider at the top (torch, ollama, together, anthropic), so either install those or make the imports lazy, and log that as a deviation. Qwen via openai 0.28's `api_base` |

**Models (2026-10-06):** every LLM baseline runs on Qwen2.5-7B-Instruct, and each choice has a published counterpart:
- Loiseau et al.: their released `qwen_2_5_7b` prompts were optimized for exactly this model (their appendix C); the main paper used 24–30B models.
- PAPILLON zero-shot: its prompts are not tied to a model, and Li et al. 2026 report zero-shot PAPILLON with Qwen2.5-7B as the local model.
- Staab et al.: hand-written prompts, run by the authors on many backbones (GPT-4 in the main results; their repo has configs for Qwen1.5 7B–72B, Llama-3.1 8B/70B, Mistral-7B, Gemma-2-27B). Qwen2.5-7B is a backbone swap within that practice: log it as a deviation and expect weaker anonymization than their GPT-4 numbers. As in their configs, the attacker inside their loop uses the same model.
- Presidio: no LLM.

- [ ] Each row of the table. Presidio done 2026-10-06: `results/rewrites/pupa_sd_presidio.csv`, 141 rows, looks as expected at first sight.
- [ ] One deviations table: what changed per baseline and why (plumbing only).
- Dropped: RUPTA (its utility is occupation classification; Loiseau et al. ran it only on DB-Bio for that reason) and DP-Prompt (2026-10-06: it would have to be reimplemented, and four baselines suffice).

### Step 5: reproduction checks
- [ ] Staab et al. on SynthPAI with a backbone from their configs, against their reported numbers.
- [ ] Optional, needs PUPA-TNB: PAPILLON zero-shot with Qwen2.5-7B, against the published leakage of 35.1% (measured on `createdPrompt`, i.e. at prompt level like our runs). Needs their judge model (GPT-4o-mini, an OpenAI key).
- [ ] Loiseau et al. with Qwen2.5-7B on PUPA: published privacy / utility 88.4 / 79.2 (Gemini 2.5 Flash judge via OpenRouter).
- Without an API budget, report the ports unchecked and say so.

### Step 6: runs and report
- [ ] All methods on PUPA-SD: one table, a breakdown by attribute, and one plot of the judge's leak rate against D.
- Dropped 2026-10-06: a second Staab et al. run with the 32B as anonymizer (no 32B runs). Without it, a reviewer may ask whether the 7B backbone handicapped them; their repo has configs for 7B-class anonymizers (Qwen1.5-7B, Mistral-7B, Llama-3.1-8B), so check whether the paper reports those runs and cite them.

Rules:
- No tuning on PUPA-SD. δ, τ and ε stay as calibrated on the dev set.
- Contamination: PUPA-SD is clean for all four baselines. Loiseau et al. and the original PAPILLON were optimized on PUPA-New, which shares no prompt with PUPA-SD (checked 2026-10-06).
- Change the plumbing (data loader, model server, output parsing), never the algorithm (prompts, loop, thresholds, iterations).

Open decisions:
- [x] Length cap for PUPA-SD (2026-10-06): 424 Laya tokens, from Laya's input limit with 50 tokens of headroom (Step 1).
- [x] Privacy metric (2026-10-06, revised the same day): the dataset's labels stay verbatim, and a stronger LLM judge decides whether each labeled value can still be inferred. This replaces the 32B open-guess attacker with a verifier and the hand-fixed labels.
- [x] Judge design (2026-10-06): two steps, a blind guess and then a comparison with the values.
- [x] No 32B probe, and no 32B anywhere in the evaluation (2026-10-06, too slow). The judge only covers labeled values, so a leak a method *adds* (an attribute the original did not reveal) goes unmeasured. State this as a limitation.
- [ ] Matching rule for the compare step: does a less precise guess count as a match ("Canada" for "Vancouver")? Loiseau et al.'s verifier says yes; Staab et al.'s keeps it as a separate answer. If yes, generalizing a city to its country earns no privacy credit for location.
- [x] The 5 added attributes (2026-10-06): education level, socioeconomic status, relationship status, family situation, nationality. All five are in the dev set's 15-attribute list and in Laya's training data, and four roughly match SynthPAI's (education, income, relationship status, place of birth). PUPA-SD labels them in 37, 8, 4, 7 and 4 rows. Below education, accuracy CIs are too wide to read; the probe still covers those attributes on every row.
- [ ] Which Loiseau et al. prompt on PUPA-SD: PUPA (same input type), SynthPAI (same threat), or both?
- [x] PAPILLON at prompt level only (2026-10-06): no answer-quality judge and no system-level table.

Effort: PUPA-SD done; baselines with the sweep script ~½ day of coding plus a few GPU hours; scorer and batch run as above; reproduction checks optional.

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

Evaluation pipeline: Steps 1–2 first (data and scorer, checked on the dev set), then Step 3 and the two cheapest baselines (Presidio, Loiseau et al.) to test the harness end to end, then the remaining baselines, Step 5 and the full runs.
