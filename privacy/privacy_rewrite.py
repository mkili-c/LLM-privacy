"""
Privacy rewriting: span occlusion finds what leaks, an LLM rewrites it, against two attackers.

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
    3. Pick one span per leaking attribute, at most --top_m: from the most to
       the least leaking attribute, its best-blamed span if the blame is
       positive and the span overlaps none picked so far (else that attribute
       waits for the next iteration). A second cue for the same attribute
       waits for the next iteration, which blames again on the edited text.
    4. One after another, the LLM rewrites the message around each picked
       span. With the utility check (5.), the least utility-bearing span goes
       first: each picked span's cost is R of the message with the span
       deleted (exact, K forward passes each; the gradient of D is zero at the
       start, where the message is still the original). The most
       utility-bearing span comes last, so a cheaper edit may already resolve
       its attributes, and if the check refuses anything, it refuses that one.
       Without the check: right to left. The LLM sees the message with the span marked and the leaking
       attributes the span is the best-blamed span for (a blame > 0 alone is
       too loose: noise gives almost every span a little of every attribute,
       and the first run told the LLM that "Kraków" reveals the writer's
       gender and education), and returns the whole message with
       the marked part made more general or left out. It may reword the words
       right around the span so that the result reads naturally, and keeps the
       rest word for word. There are no candidates to choose from, so no fluency
       guard; the only test a rewrite has to pass is the utility check below.
       Each span is found again by its text in the current message (the
       occurrence nearest its old position), since earlier rewrites shift it;
       if an earlier rewrite changed its words, it is blamed again in the next
       iteration. After each rewrite the message is scored again, and a span is
       skipped once nothing it is blamed for leaks any more.
    5. Utility check (when the loop passes utility_share): after each rewrite,
           R = D(rewrite) / D(∅)          (utility/metrics/utility_check.py)
       R >= --max_share means the rewrite moves the answers as far as an empty
       message would: the request is lost. The LLM is then asked once more for
       the same span (generalize_prompt): it is shown its refused rewrite as
       the one that lost the request, and asked to keep the marked part but
       make it less specific, a broader term of the same kind. This retry does
       not demand that the attribute can no longer be inferred: where the
       request depends on the attribute, a less specific mention is the
       compromise. Asked for both, the LLM repeated its deletion (job 649:
       "a midwife like me" -> "I" three times out of four). The retry goes in
       if its R < --max_share; a retry identical to the refused rewrite is not
       checked again. Otherwise the span stays as it was
       and is protected: its text is kept in a list, and no span that overlaps
       it is picked again for this message, so its attributes fall to their
       next-best spans. Without the list, the next iteration would blame the
       same span, greedy decoding would repeat the same rewrite, and the check
       would refuse it again every iteration. Utility wins: a protected span
       stays even if it leaks.
The step ends when nothing leaks, when no unprotected span lowers the leakage,
when an iteration neither changes the message nor protects a span, or after
--max_iters.

The LLM sees the part it rewrites: generalizing needs it ("Zurich" -> "the
city", "25" -> "an adult"). The prompt forbids a different specific value
("Zurich" -> "Mexico City" still leaks a location, just a wrong one) and new
details, and gives no concrete examples: Qwen copied examples verbatim in an
earlier version. A greedy rewrite of the whole region replaced picking among
word-level candidates: the candidates broke the sentence around the span, and
choosing among them needed a fluency guard, a tried list and an acceptance
test that together blamed the wrong spans and cost up to 117 LLM calls per
prompt (job 463).

The utility check sits inside this step since 2026-10-11, interleaved with the
rewrites: checking each rewrite as it is made knows exactly what was lost (this
span), where the separate utility step afterwards had to guess it and cycled
with this step (jobs 556, 561). That earlier utility rewriter,
kl_mask_llm_rewrite_multi.py, is kept as it was and still imports the marks and
generate_message from here.
"""
import argparse
import re
import time

import torch

import laya
from attribute_probe import (ATTRIBUTES, DEFAULT_MODEL, QUESTION as QWEN_QUESTION,
                             AttributeProbe, load, logits_shared)
from laya_probe import PRIVACY_LAYA, QUESTION as LAYA_QUESTION, LayaProbe

OPEN, CLOSE = "<<", ">>"         # marks the span in the rewrite prompt
REWRITE_MAX_NEW_TOKENS = 1024    # the whole message comes back; greedy decoding stops at its end
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
        """[N, K] P(yes) per text and attribute; each text is encoded once for all K questions."""
        ids = [self.tok(t, add_special_tokens=False).input_ids for t in texts]
        return torch.sigmoid(logits_shared(self.probes, ids)).cpu()


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


def hinge(P, threshold):
    """sum_a max(0, P_a - threshold) of the first row: how far the message is from private."""
    return (P[0] - threshold).clamp(min=0).sum().item()


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


def delete(text, s, e):
    """Remove text[s:e] and one adjacent space."""
    if s > 0 and text[s - 1] == " ":
        s -= 1
    elif e < len(text) and text[e] == " ":
        e += 1
    return text[:s] + text[e:]


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


# ---------------------------------------------------------------- rewriting

def rewrite_prompt(marked, attributes):
    attrs = ", ".join(attributes)
    return (
        "The message below was written by a user to an AI assistant. The part marked "
        f"{OPEN} {CLOSE} reveals the writer's {attrs}.\n"
        f"Rewrite the message so that the writer's {attrs} can no longer be inferred from it. "
        "Make the marked part more general or leave it out; never replace it with a different "
        "specific value, and do not add new details. Change the words right around it only "
        "where the sentence needs it to read naturally, and keep the rest of the message word "
        "for word. The request of the message must stay intact.\n"
        "Output only the rewritten message, without the marks, and nothing else.\n\n"
        f"Message: {marked}"
    )


def generalize_prompt(marked, attributes, refused):
    """The retry after the utility check refused a rewrite: keep the part, less specific."""
    attrs = ", ".join(attributes)
    return (
        "The message below was written by a user to an AI assistant. The part marked "
        f"{OPEN} {CLOSE} reveals the writer's {attrs}, and the request of the message depends "
        "on it. An earlier rewrite left it out, and the request lost what it needs:\n"
        f"Earlier rewrite: {refused}\n"
        "Rewrite the message again. Keep the marked part, but make it less specific: replace it "
        "with a broader term of the same kind, so that the request still gets what it needs. Do "
        "not leave it out, never replace it with a different specific value, and do not add new "
        "details. Change the words right around it only where the sentence needs it to read "
        "naturally, and keep the rest of the message word for word.\n"
        "Output only the rewritten message, without the marks, and nothing else.\n\n"
        f"Message: {marked}"
    )


@torch.no_grad()
def generate_message(model, tok, prompt):
    """The LLM's greedy answer to prompt, without leftover marks. Asked for the rewritten
    message only, the answer is that message (job 544). The utility rewriter uses it too."""
    chat = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                   tokenize=False, add_generation_prompt=True)
    inputs = tok(chat, return_tensors="pt", add_special_tokens=False).to(model.device)
    out = model.generate(**inputs, max_new_tokens=REWRITE_MAX_NEW_TOKENS, do_sample=False,
                         pad_token_id=tok.pad_token_id)
    answer = tok.decode(out[0, inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return answer.replace(OPEN, "").replace(CLOSE, "").strip()


def llm_rewrite(model, tok, text, s, e, attributes):
    """The message rewritten around text[s:e] by one greedy generation."""
    marked = text[:s] + OPEN + text[s:e] + CLOSE + text[e:]
    return generate_message(model, tok, rewrite_prompt(marked, attributes))


def llm_generalize(model, tok, text, s, e, attributes, refused):
    """The retry: text rewritten around text[s:e] with the part kept but less specific."""
    marked = text[:s] + OPEN + text[s:e] + CLOSE + text[e:]
    return generate_message(model, tok, generalize_prompt(marked, attributes, refused))


# ---------------------------------------------------------------- utility check

def utility_vetted(model, tok, text, s, e, attributes, new, utility_share, max_share):
    """new if the utility check passes it (R < max_share); else the LLM's retry of text
    around text[s:e], shown new as the refused rewrite and asked to keep the part less
    specific, if that passes; else None: the span stays as it was."""
    R = utility_share(new)
    if R < max_share:
        print(f"      utility R {R:.0%}: in")
        return new
    general = llm_generalize(model, tok, text, s, e, attributes, new)
    failed = None
    if not general:
        failed = "gave an empty answer"
    elif general == text:
        failed = "changed nothing"
    elif general == new:
        failed = "repeated the refused rewrite"
    if failed:
        print(f"      utility R {R:.0%} >= {max_share:.0%}: request lost; the retry to generalize {failed}")
        return None
    R_general = utility_share(general)
    print(f"      utility R {R:.0%} >= {max_share:.0%}: request lost; more general:\n"
          f"      -> {general!r}\n      utility R {R_general:.0%}: "
          + ("in" if R_general < max_share else "still lost"))
    return general if R_general < max_share else None


def locate(text, old, near):
    """Start of the occurrence of old in text nearest to position near, or None if it is gone."""
    starts = [m.start() for m in re.finditer(re.escape(old), text)]
    return min(starts, key=lambda p: abs(p - near)) if starts else None


def utility_order(text, sp, order, utility_share):
    """The picked spans, least utility-bearing first: by R of text with the span deleted."""
    cost = {}
    for u, _ in order:
        cost[u] = utility_share(delete(text, sp[u][0], sp[u][1]))
    queue = sorted(order, key=lambda ua: cost[ua[0]])
    print("  utility order (R with the span deleted): "
          + ", ".join(f"{text[sp[u][0]:sp[u][1]]!r} {cost[u]:.0%}" for u, _ in queue))
    return queue


def protected_spans(text, sp, protected):
    """[U] bool: the spans that overlap an occurrence of a protected span text."""
    kept = [(m.start(), m.end()) for p in protected for m in re.finditer(re.escape(p), text)]
    return torch.tensor([any(s < b and a < e for a, b in kept) for s, e, _ in sp], dtype=torch.bool)


# ---------------------------------------------------------------- main loop

def show(attackers, attributes, per, P):
    for n in [a.name for a in attackers]:
        print(f"    {n:<5}" + "".join(f"  {a[:8]:>8} {per[n][0, k]:.3f}" for k, a in enumerate(attributes)))
    print(f"    {'max':<5}" + "".join(f"  {a[:8]:>8} {P[0, k]:.3f}" for k, a in enumerate(attributes)))


def rewrite(model, tok, attackers, text, attributes, args, utility_share=None):
    """-> the rewritten message. utility_share(text) -> R = D(text) / D(∅) is the utility
    check, against args.max_share; None (standalone use): no check."""
    protected = []                   # span texts the utility check keeps in the message
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

        start_time = time.perf_counter()
        sp, blame = span_occlusion(attackers, text, {n: v[0] for n, v in per.items()},
                                   args.max_n, args.span_penalty)
        print(f"  occlusion: {len(sp)} spans scored in {time.perf_counter() - start_time:.0f} s")
        # saturation check, count with: grep -c "\[stall\]" <log>
        for a in leaking.nonzero().flatten().tolist():
            if blame[:, a].max() < STALL_BLAME:
                print(f"  [stall] {attributes[a]}: best span blame {blame[:, a].max().item():+.4f} "
                      f"at P {P[0, a].item():.4f}")
        # protected spans compete no more: their attributes fall to their next-best spans
        blame = blame.masked_fill(protected_spans(text, sp, protected)[:, None], float("-inf"))
        order = pick(sp, blame, (P[0] - args.threshold).clamp(min=0), args.top_m)
        print("  leaking: " + ", ".join(a for a, l in zip(attributes, leaking) if l))
        print("  picked spans: " + ", ".join(f"{text[sp[u][0]:sp[u][1]]!r} ({attributes[a]} {blame[u, a]:+.3f})"
                                             for u, a in order))
        if not order:
            print("  no unprotected span lowers the leakage when deleted: stop")
            return text

        if utility_share is None:
            queue = sorted(order, key=lambda ua: sp[ua[0]][0], reverse=True)          # right to left
        else:
            queue = utility_order(text, sp, order, utility_share)
        best = blame.argmax(0)                        # [K] each attribute's best-blamed span
        start, changed, n_protected = text, False, len(protected)
        for u, _ in queue:
            old = start[sp[u][0]:sp[u][1]]
            s = locate(text, old, sp[u][0])
            if s is None:
                print(f"    {old!r}: an earlier rewrite changed it, blamed again next iteration")
                continue
            e = s + len(old)
            # the attributes this span is the best cue for and that still leak (a blame > 0
            # alone is too loose: noise gives almost every span a little of every attribute)
            blamed = [attributes[b] for b in range(len(attributes)) if P[0, b] > args.threshold and best[b] == u]
            if not blamed:
                print(f"    {old!r}: nothing it is blamed for leaks any more, skipped")
                continue
            new = llm_rewrite(model, tok, text, s, e, blamed)
            if not new or new == text:
                print(f"    {old!r} ({', '.join(blamed)}): rewriter "
                      + ("gave an empty answer" if not new else "changed nothing"))
                continue
            print(f"    {old!r} ({', '.join(blamed)})\n      -> {new!r}")
            if utility_share is not None:
                new = utility_vetted(model, tok, text, s, e, blamed, new, utility_share, args.max_share)
                if new is None:
                    protected.append(old)
                    print(f"      span kept and protected: {old!r}")
                    continue
            _, P_new = score(attackers, [new])
            print(f"      hinge {hinge(P, args.threshold):.3f} -> {hinge(P_new, args.threshold):.3f}")
            text, P, changed = new, P_new, True
        if not changed and len(protected) == n_protected:
            print("  no rewrite changed the message: stop")
            return text
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
