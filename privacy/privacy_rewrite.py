"""
Privacy rewriting by word occlusion + LLM replacement, against two attackers.

Each iteration:
    1. Score the message with both probes, for every attribute a:
           Qwen  P_a = P(yes) of attribute_probe.py (chat LLM, Yes/No logits)
           Laya  P_a = P(true) of laya_probe.py     (calibrated classifier)
       and combine them as P_a = max(Qwen, Laya): an attribute leaks if EITHER
       attacker says so. Stop once every P_a <= --threshold.
    2. Span occlusion: delete every run of 1..--max_n consecutive words that
       holds at least one content word (words are letter/digit runs, so
       "15-year-old" is one word; function words such as "from" or "can" may
       sit inside a span but never make one up alone), re-score every variant
       with both probes and blame each span s of n words for each attribute a by
           blame_s,a = max_probe (P_a(text) - P_a(text without s)) - --span_penalty * (n - 1)
       Deleting more never leaks more, so without the penalty the longest
       spans would always win; with it a longer span must earn its extra words.
    3. Pick one span per leaking attribute, at most --top_m (5): from the
       most to the least leaking attribute, its
       best-blamed span if the blame is positive and the span overlaps none
       picked so far (else that attribute waits for the next iteration). Spans
       blamed for different attributes are rewritten in the same iteration. A
       second cue for the same attribute waits for the next iteration, which
       blames again on the edited text: with the first cue gone, the second
       may no longer matter. A span already tried for an attribute without
       success is not picked for it again, so the next-best span gets its turn;
       often that is the same cue cut differently ("a 25 years old" where
       "am a 25 years" left "I old ..." whatever replaced it).
    4. From right to left (so the spans left of it stay valid), mark the span
       in the message and ask Qwen for more general replacements of it. The
       candidates are these replacements plus deleting the span. Drop every
       candidate whose surprisal spikes (disfluency in kl_utility_scoring.py)
       exceed min(spikes of the current text, spikes of the user's message
       x_0) + --fluency_eps nats: the probes are fooled by broken sentences,
       so a broken candidate would win for the wrong reason. Both bounds hold
       at once. The x_0 bound stops edits that each add a little from adding
       up. The current-text bound stops the spikes freed by removing a rare
       word ("Kraków", "midwife") from paying for broken grammar later. Score
       the rest with both probes, take the best by
           (sum_a max(0, P_a - threshold),  sum_a P_a)     compared in order
       and keep it only if it lowers the hinge sum_a max(0, P_a - threshold)
       by at least min(--min_hinge_drop, the hinge left). Pushing a P that is
       already below the threshold further down buys no privacy and costs
       fluency. A smaller gain is within the attackers' fp16 noise (about
       ±0.03 between batched and single scoring); in job 452 such edits were
       the most damaging ones ("follow on days off" -> "work" for 0.476 ->
       0.474). An edit that clears all of what is left always counts, so an
       attribute just above the threshold can still be fixed. A span is skipped once no attribute it is
       blamed for leaks any more (an earlier edit of the iteration fixed it).
       A span whose candidates all break fluency, or none of which lowers the
       hinge, goes on the tried list. The loop passes one tried list to every
       privacy step of a prompt, so a cut that failed is not tried again in a
       later loop iteration. The step ends when nothing leaks, when no untried
       span lowers the leakage, or after --max_iters.

The LLM sees the word it replaces: generalizing needs it ("Zurich" -> "the
city", "25" -> "an adult"). A first version hid it behind [MASK]; the LLM then
guessed blindly, and the probes were fooled by broken sentences ("a human year
old woman from Mother") instead of the information being removed. The prompt
asks for generalizations only, never a different specific value ("Zurich" ->
"Mexico City" still leaks a location, just a wrong one), and candidates that
still contain the word are dropped. It gives no concrete examples: Qwen copied
them verbatim ("Adult" proposed for "Zurich", "7th" and "Can").

Utility is deliberately not measured here; a separate utility loop handles it.
A candidate is chosen for privacy only, so deletion usually wins: the utility
loop is the one that puts back what turns out to matter. A deleted span
leaves SLOT in the text so the utility step knows where something was; the
attackers and the fill LLM only ever see the text without it (visible).
"""
import argparse
import re

import torch

import laya
from attribute_probe import (ATTRIBUTES, DEFAULT_MODEL, QUESTION as QWEN_QUESTION,
                             AttributeProbe, load)
from laya_probe import PRIVACY_LAYA, QUESTION as LAYA_QUESTION, LayaProbe
from kl_utility_scoring import disfluency        # utility/metrics, put on sys.path by attribute_probe

OPEN, CLOSE = "<<", ">>"     # marks the word to generalize in the fill prompt
SLOT = "[…]"                 # left where a span was deleted, for the utility step to refill
# A leaking attribute whose best span blame is below this looks saturated: its P is pinned near 1
# by redundant cues, so no single deletion moves it (ROADMAP Phase 3, occlusion on -log(1 - P)).
STALL_BLAME = 0.01
WORD = re.compile(r"[\w'’-]+")
STOPWORDS = set("""
a an the and or but if so as of at by for from in into on onto to with without about
over under near after before than then there here this that these those
i me my mine myself we us our you your he him his she her it its they them their
am is are was were be been being do does did have has had will would can could
shall should may might must not no yes just very too also only what which who whom
whose when where why how all any some each every up down out off again
""".split())


# ---------------------------------------------------------------- attackers

class QwenAttacker:
    name = "qwen"

    def __init__(self, model, tok, attributes):
        self.probes = [AttributeProbe(model, tok, QWEN_QUESTION.format(attribute=a)) for a in attributes]
        self.tok = tok

    def p(self, texts):
        """[N, K] P(yes) per text and attribute."""
        ids = [self.tok(t, add_special_tokens=False).input_ids for t in texts]
        return torch.stack([torch.sigmoid(pr.logits_batch(ids)) for pr in self.probes], dim=1).cpu()


class LayaAttacker:
    name = "laya"

    def __init__(self, agent, attributes):
        self.probe = LayaProbe(agent, attributes, LAYA_QUESTION)

    def p(self, texts):
        return torch.sigmoid(self.probe.logits([self.probe.encode(t) for t in texts])).cpu()


def visible(text):
    """The text without slot marks: what the attackers and every LLM see."""
    return text.replace(SLOT, "")


def score(attackers, texts):
    """{attacker: [N, K]} and the combined max over attackers [N, K]."""
    texts = [visible(t) for t in texts]
    per = {a.name: a.p(texts) for a in attackers}
    return per, torch.stack(list(per.values())).amax(0)


def objective(P, threshold):
    """Per row: (sum_a hinge above threshold, sum_a P) -- smaller is more private."""
    return list(zip((P - threshold).clamp(min=0).sum(1).tolist(), P.sum(1).tolist()))


# ---------------------------------------------------------------- occlusion

def spans(text, max_n):
    """(start, end, n words) of every run of 1..max_n consecutive words with at
    least one content word. A span of function words alone is skipped:
    deleting one ("from", "Can") breaks the sentence, and the probes react to
    the broken grammar, not to a leak, so they used to rank at the top."""
    ws = list(WORD.finditer(text))
    content = [w.group().lower() not in STOPWORDS for w in ws]
    return [(ws[i].start(), ws[j].end(), j - i + 1)
            for i in range(len(ws)) for j in range(i, min(i + max_n, len(ws)))
            if any(content[i:j + 1])]


def delete(text, s, e, mark=""):
    """Remove text[s:e] and one adjacent space, leaving `mark` in their place."""
    if s > 0 and text[s - 1] == " ":
        s -= 1
    elif e < len(text) and text[e] == " ":
        e += 1
    return text[:s] + mark + text[e:]


def span_occlusion(attackers, text, P_text, max_n, penalty):
    """-> (spans, blame [U, K]): how much deleting each span lowers each
    attribute's P under either attacker, minus the length penalty."""
    sp = spans(text, max_n)
    per, _ = score(attackers, [delete(text, s, e) for s, e, _ in sp])
    drops = torch.stack([P_text[n] - v for n, v in per.items()]).amax(0)   # [U, K], either attacker
    n_words = torch.tensor([n for _, _, n in sp], dtype=drops.dtype)
    return sp, drops - penalty * (n_words[:, None] - 1)


def pick(sp, blame, hinge, m):
    """One span per leaking attribute, at most m: for each attribute from the
    most to the least leaking (hinge [K] = max(0, P_a - threshold)), its
    best-blamed span if the blame is positive and the span overlaps none
    picked so far. A span picked for one attribute that is also the best for
    another counts for both. -> [(span index, attribute index)]."""
    taken = []
    for a in hinge.argsort(descending=True).tolist():
        if hinge[a] <= 0 or len(taken) == m:
            break
        u = blame[:, a].argmax().item()
        overlaps = any(sp[u][0] < sp[t][1] and sp[t][0] < sp[u][1] for t, _ in taken)
        if blame[u, a] > 0 and not overlaps:
            taken.append((u, a))
    return taken


# ---------------------------------------------------------------- replacements

def fill_prompt(marked, word, n, attributes):
    return (
        "The message below was written by a user to an AI assistant. The part "
        f"{OPEN}{word}{CLOSE} in it may reveal the writer's {', '.join(attributes)}.\n"
        f"Propose {n} different, more general replacements for \"{word}\" (a single "
        "word or a short phrase) that fit exactly in its place, keep the message "
        "grammatical, natural and its request intact, but no longer pin down those "
        f"attributes. Each replacement must be a broader expression for what \"{word}\" "
        "means in this message, never a different specific value.\n"
        "Output one replacement per line, nothing else -- no numbering, no quotes, "
        "no explanations.\n\n"
        f"Message: {marked}"
    )


_LINE_CLEAN = re.compile(r"^\s*(?:[-*•]|\d+[.)]|\(\d+\))?\s*")


def parse_fills(text, max_words=4):
    fills, seen = [], set()
    for line in text.splitlines():
        f = _LINE_CLEAN.sub("", line).strip(" \t\"'`.,;:!?()[]{}")
        if not f or "mask" in f.lower() or len(f.split()) > max_words or f.lower() in seen:
            continue
        seen.add(f.lower())
        fills.append(f)
    return fills


@torch.no_grad()
def llm_fills(model, tok, marked, word, attributes, n, rounds, temperature, max_new_tokens=120):
    """One greedy + (rounds - 1) sampled generations, pooled and deduped."""
    chat = tok.apply_chat_template([{"role": "user", "content": fill_prompt(marked, word, n, attributes)}],
                                   tokenize=False, add_generation_prompt=True)
    inputs = tok(chat, return_tensors="pt", add_special_tokens=False).to(model.device)
    P = inputs.input_ids.shape[1]
    outs = [model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                           pad_token_id=tok.pad_token_id)]
    if rounds > 1:
        outs.append(model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=True,
                                   temperature=temperature, num_return_sequences=rounds - 1,
                                   pad_token_id=tok.pad_token_id))
    fills = []
    for out in outs:
        for row in tok.batch_decode(out[:, P:], skip_special_tokens=True):
            fills.extend(parse_fills(row))
    return list(dict.fromkeys(fills))


# ---------------------------------------------------------------- main loop

def show(attackers, attributes, per, P):
    for n in [a.name for a in attackers]:
        print(f"    {n:<5}" + "".join(f"  {a[:8]:>8} {per[n][0, k]:.3f}" for k, a in enumerate(attributes)))
    print(f"    {'max':<5}" + "".join(f"  {a[:8]:>8} {P[0, k]:.3f}" for k, a in enumerate(attributes)))


def rewrite(model, tok, attackers, text, attributes, args, x0_spikes=None, tried=None):
    """x0_spikes: disfluency of the user's message x_0; by default text is x_0.
    tried: span text -> attribute indices it was tried for without success. Pass the
    same dict to every call for one prompt; it is updated in place."""
    dis = disfluency([visible(text)], model, tok)[0]      # of the current text, updated on every edit
    x0 = dis if x0_spikes is None else x0_spikes
    tried = {} if tried is None else tried
    print(f"  fluency budget: spikes <= min(current, x_0 = {x0:.1f}) + {args.fluency_eps}")
    for it in range(args.max_iters + 1):
        per, P = score(attackers, [text])
        print(f"\n=== iteration {it}: {text!r}")
        show(attackers, attributes, per, P)
        leaking = P[0] > args.threshold
        if not leaking.any():
            print(f"  all attributes <= {args.threshold}: done")
            return text
        if it == args.max_iters:
            print("  max_iters reached")
            return text

        sp, blame = span_occlusion(attackers, text, {n: v[0] for n, v in per.items()},
                                   args.max_n, args.span_penalty)
        # saturation check, count with: grep -c "\[stall\]" <log>
        for a in leaking.nonzero().flatten().tolist():
            if blame[:, a].max() < STALL_BLAME:
                print(f"  [stall] {attributes[a]}: best span blame {blame[:, a].max().item():+.4f} "
                      f"at P {P[0, a].item():.4f}")
        # a span tried for an attribute without success would fail again: hide it from pick
        for u, (s, e, _) in enumerate(sp):
            for a in tried.get(visible(text[s:e]), ()):
                blame[u, a] = float("-inf")
        order = pick(sp, blame, (P[0] - args.threshold).clamp(min=0), args.top_m)
        print("  leaking: " + ", ".join(a for a, l in zip(attributes, leaking) if l))
        print("  picked spans: " + ", ".join(f"{text[sp[u][0]:sp[u][1]]!r} ({attributes[a]} {blame[u, a]:+.3f})"
                                             for u, a in order))
        if not order:
            print("  no untried span lowers the leakage when deleted: stop")
            return text

        for u, a in sorted(order, key=lambda ua: sp[ua[0]][0], reverse=True):   # right to left
            s, e, _ = sp[u]
            if not ((P[0] > args.threshold) & (blame[u] > 0)).any():
                print(f"    {visible(text[s:e])!r}: nothing it is blamed for leaks any more, skipped")
                continue
            old = visible(text[s:e])
            marked = visible(text[:s] + OPEN + text[s:e] + CLOSE + text[e:])
            gone = {w.lower() for w in WORD.findall(old)} - STOPWORDS
            fills = [f for f in llm_fills(model, tok, marked, old, attributes,
                                          args.n_candidates, args.cand_rounds, args.cand_temperature)
                     if not gone & {w.lower() for w in WORD.findall(f)}]   # "Zurich area" still names Zurich
            fills.append("")                                                 # delete the span
            cands = [text[:s] + f + text[e:] if f else delete(text, s, e, SLOT) for f in fills]
            # fluency guard: at most min(current, x_0) + --fluency_eps spikes
            dis_c = disfluency([visible(c) for c in cands], model, tok)
            keep = [i for i, d in enumerate(dis_c) if d <= min(dis, x0) + args.fluency_eps]
            n_broken = len(cands) - len(keep)
            fills, cands, dis_c = [fills[i] for i in keep], [cands[i] for i in keep], [dis_c[i] for i in keep]
            if not cands:
                print(f"    {old!r}: all {n_broken} candidates break fluency")
                tried.setdefault(old, set()).add(a)
                continue
            _, Pc = score(attackers, cands)
            obj, cur = objective(Pc, args.threshold), objective(P, args.threshold)[0]
            b = min(range(len(cands)), key=lambda i: obj[i])
            gain = cur[0] - obj[b][0]                   # drop of the hinge, not of sum_a P_a
            if gain > 0 and gain >= min(args.min_hinge_drop, cur[0]):
                print(f"    {old!r} -> {fills[b] or '<deleted>'!r}  (hinge {cur[0]:.3f} -> {obj[b][0]:.3f}, "
                      f"spikes {dis:.1f} -> {dis_c[b]:.1f}, {len(fills)} candidates, {n_broken} broke fluency)")
                text, P, dis = cands[b], Pc[b:b + 1], dis_c[b]
            else:
                print(f"    {old!r}: none of {len(fills)} candidates lowers the hinge enough "
                      f"(best {fills[b] or '<deleted>'!r}: {cur[0]:.3f} -> {obj[b][0]:.3f})")
                tried.setdefault(old, set()).add(a)
    return text


def main():
    p = argparse.ArgumentParser(description="Rewrite a message until neither Qwen nor Laya "
                                            "infers any attribute with P > threshold")
    p.add_argument("--text", required=True)
    p.add_argument("--attributes", nargs="+", default=ATTRIBUTES)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--top_m", type=int, default=5,
                   help="cap on spans rewritten per iteration (one per leaking attribute)")
    p.add_argument("--max_n", type=int, default=4, help="longest span, in words")
    p.add_argument("--span_penalty", type=float, default=0.05,
                   help="score a span pays per word beyond its first")
    p.add_argument("--max_iters", type=int, default=8)
    p.add_argument("--n_candidates", type=int, default=10, help="replacements asked for per mask")
    p.add_argument("--cand_rounds", type=int, default=2, help="1 greedy + (rounds-1) sampled generations")
    p.add_argument("--cand_temperature", type=float, default=0.9)
    p.add_argument("--min_hinge_drop", type=float, default=0.05,
                   help="an edit must lower the hinge by at least min(this, the hinge left); "
                        "above the attackers' fp16 noise of about ±0.03")
    p.add_argument("--fluency_eps", type=float, default=4.0,
                   help="fluency guard: a candidate may have at most min(spikes of the current text, "
                        "spikes of the input) + this many nats (disfluency)")
    p.add_argument("--model_path", default=DEFAULT_MODEL)
    p.add_argument("--laya_path", default=PRIVACY_LAYA)
    p.add_argument("--seed", type=int, default=23)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tok = load(args.model_path, device)
    model.requires_grad_(False)
    agent = laya.load(args.laya_path)
    attackers = [QwenAttacker(model, tok, args.attributes), LayaAttacker(agent, args.attributes)]

    final = rewrite(model, tok, attackers, args.text, args.attributes, args)
    print(f"\noriginal: {args.text!r}\nfinal   : {final!r}")


if __name__ == "__main__":
    main()
