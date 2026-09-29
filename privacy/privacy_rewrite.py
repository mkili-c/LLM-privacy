"""
Privacy rewriting by word occlusion + LLM replacement, against two attackers.

Each iteration:
    1. Score the message with both probes, for every attribute a:
           Qwen  P_a = P(yes) of attribute_probe.py (chat LLM, Yes/No logits)
           Laya  P_a = P(true) of laya_probe.py     (calibrated classifier)
       and combine them as P_a = max(Qwen, Laya): an attribute leaks if EITHER
       attacker says so. Stop once every P_a <= --threshold.
    2. Word occlusion: delete each content word from the text (words are
       letter/digit runs, so "15-year-old" is one unit and punctuation is never
       touched; function words such as "from" or "can" are skipped),
       re-score every variant with both probes and give each word
           score_w = sum over leaking a of max_probe (P_a(text) - P_a(text without w))
    3. Take the --top_m words with a positive score. From right to left (so
       the spans of the words left of it stay valid), mark the word in the
       message and ask Qwen for more general replacements of it, score every
       candidate text with both probes and keep the best one if it lowers
           (sum_a max(0, P_a - threshold),  sum_a P_a)     compared in order.

The LLM sees the word it replaces: generalizing needs it ("Zurich" -> "the
city", "25" -> "an adult"). A first version hid it behind [MASK]; the LLM then
guessed blindly, and the probes were fooled by broken sentences ("a human year
old woman from Mother") instead of the information being removed. The prompt
asks for generalizations only, never a different specific value ("Zurich" ->
"Mexico City" still leaks a location, just a wrong one), and candidates that
still contain the word are dropped. It gives no concrete examples: Qwen copied
them verbatim ("Adult" proposed for "Zurich", "7th" and "Can").

Utility is deliberately not measured here; a separate utility loop handles it.
A candidate is chosen for privacy only.
"""
import argparse
import re

import torch

import laya
from attribute_probe import (ATTRIBUTES, DEFAULT_MODEL, QUESTION as QWEN_QUESTION,
                             AttributeProbe, load)
from laya_probe import DEFAULT_LAYA, QUESTION as LAYA_QUESTION, LayaProbe

OPEN, CLOSE = "<<", ">>"     # marks the word to generalize in the fill prompt
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


def score(attackers, texts):
    """{attacker: [N, K]} and the combined max over attackers [N, K]."""
    per = {a.name: a.p(texts) for a in attackers}
    return per, torch.stack(list(per.values())).amax(0)


def objective(P, threshold):
    """Per row: (sum_a hinge above threshold, sum_a P) -- smaller is more private."""
    return list(zip((P - threshold).clamp(min=0).sum(1).tolist(), P.sum(1).tolist()))


# ---------------------------------------------------------------- occlusion

def words(text):
    """Content-word spans. Function words are skipped: deleting one ("from",
    "Can") breaks the sentence, and the probes react to the broken grammar,
    not to a leak, so they used to rank at the top."""
    return [(m.start(), m.end()) for m in WORD.finditer(text) if m.group().lower() not in STOPWORDS]


def delete(text, s, e):
    """Remove text[s:e] and one adjacent space."""
    if s > 0 and text[s - 1] == " ":
        s -= 1
    elif e < len(text) and text[e] == " ":
        e += 1
    return text[:s] + text[e:]


def word_occlusion(attackers, text, P_text, leaking):
    """-> (spans, score [U], per-attacker drops {name: [U, K]})."""
    spans = words(text)
    per, _ = score(attackers, [delete(text, s, e) for s, e in spans])
    drops = {n: P_text[n] - v for n, v in per.items()}           # [U, K]
    best = torch.stack(list(drops.values())).amax(0)              # either attacker
    return spans, (best * leaking.float()).sum(1), drops


# ---------------------------------------------------------------- replacements

def fill_prompt(marked, word, n, attributes):
    return (
        "The message below was written by a user to an AI assistant. The word "
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


def rewrite(model, tok, attackers, text, attributes, args):
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

        spans, wscore, _ = word_occlusion(attackers, text, {n: v[0] for n, v in per.items()}, leaking)
        order = [u for u in wscore.argsort(descending=True).tolist() if wscore[u] > 0][:args.top_m]
        print("  leaking: " + ", ".join(a for a, l in zip(attributes, leaking) if l))
        print("  top words: " + ", ".join(f"{text[slice(*spans[u])]!r} {wscore[u]:+.3f}" for u in order))
        if not order:
            print("  no word lowers the leakage when deleted: stop")
            return text

        changed = False
        for u in sorted(order, key=lambda u: spans[u][0], reverse=True):   # right to left
            s, e = spans[u]
            old = text[s:e]
            marked = text[:s] + OPEN + old + CLOSE + text[e:]
            fills = [f for f in llm_fills(model, tok, marked, old, attributes,
                                          args.n_candidates, args.cand_rounds, args.cand_temperature)
                     if old.lower() not in f.lower()]    # "Zurich area" still names Zurich
            if not fills:
                print(f"    {old!r}: no candidates")
                continue
            cands = [text[:s] + f + text[e:] for f in fills]
            _, Pc = score(attackers, cands)
            obj, cur = objective(Pc, args.threshold), objective(P, args.threshold)[0]
            b = min(range(len(cands)), key=lambda i: obj[i])
            if obj[b] < cur:
                print(f"    {old!r} -> {fills[b]!r}  (hinge {cur[0]:.3f} -> {obj[b][0]:.3f}, "
                      f"{len(fills)} candidates)")
                text, P, changed = cands[b], Pc[b:b + 1], True
            else:
                print(f"    {old!r}: none of {len(fills)} candidates helps (best {fills[b]!r})")
        if not changed:
            print("  no replacement helped: stop")
            return text
    return text


def main():
    p = argparse.ArgumentParser(description="Rewrite a message until neither Qwen nor Laya "
                                            "infers any attribute with P > threshold")
    p.add_argument("--text", required=True)
    p.add_argument("--attributes", nargs="+", default=ATTRIBUTES)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--top_m", type=int, default=2, help="words masked per iteration")
    p.add_argument("--max_iters", type=int, default=8)
    p.add_argument("--n_candidates", type=int, default=10, help="replacements asked for per mask")
    p.add_argument("--cand_rounds", type=int, default=2, help="1 greedy + (rounds-1) sampled generations")
    p.add_argument("--cand_temperature", type=float, default=0.9)
    p.add_argument("--model_path", default=DEFAULT_MODEL)
    p.add_argument("--laya_path", default=DEFAULT_LAYA)
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
