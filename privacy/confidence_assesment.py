"""
Adversarial advantage of the model at inferring a binary user attribute.

Threat model
------------
An adversary sees a user's message T and must guess a protected attribute
f in F, |F| = 2 (e.g. adult vs minor). Its guess is f_hat. The
standard distinguishing advantage is

    Adv = 2 * | Pr[f_hat = f] - 1/2 |,

which is 0 for an adversary that does no better than a coin flip and 1 for one
that is always right (or always wrong -- see "Why the absolute value" below).
Here the adversary IS the LLM: we put the question to it and read the answer
distribution off the logits rather than sampling a reply.

Reading the answer off the logits
---------------------------------
The question is posed so that the answer is Yes or No, and the model is asked
for exactly one word. At the first position of the assistant turn we take

    m_yes = sum_{s in YES} P(s | query)      s ranges over surface forms
    m_no  = sum_{s in NO}  P(s | query)      ("Yes", "yes", "YES", "Y", ...)

    coverage = m_yes + m_no                  diagnostic, NOT thrown away
    p_yes    = m_yes / coverage              forced binary choice

The surface-form sets are reduced to a prefix-free antichain first, so the sums
above are probabilities of disjoint events and p_yes is a genuine conditional
probability. See resolve_forms.

Two conditions
--------------
with prompt     the user's message is included -- the informed adversary
without prompt  context=None drops the message block; context="" keeps the
                framing and empties it (--ablate_context). The second is the
                tighter control: only the information changes, not the wording.

The no-prompt condition is the baseline this whole measurement rests on. It is
the advantage obtainable from the question phrasing and the model's prior over
the population alone, with zero information from the user. Leakage attributable
to the text is the gap between the two, not the raw number in the first.

Why the absolute value
----------------------
2|acc - 1/2| credits an anti-correlated adversary: one that is right 10% of the
time is a distinguisher too, since flipping its output makes it right 90% of
the time. This is the standard convention and it is the reason the estimator is
biased upward -- under true advantage 0, E[Adv_hat] = sqrt(2/pi)/sqrt(n) ~
0.8/sqrt(n) rather than 0.
That no-information floor is computed and printed alongside every estimate;
anything below it is noise.
"""
import argparse
import csv
import json
import math
import re
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable

import torch
import torch.nn.functional as F

# utility/metrics/ is a directory of standalone scripts, not a package: its
# modules import each other by bare name (kl_utility_scoring.py does `from
# topk_sample import ...`), so the directory itself has to be ON sys.path --
# importing it as `utility.metrics.kl_utility_scoring` would resolve this file's
# import and then fail on that inner one. Setting PYTHONPATH in the sbatch does
# the same job; this block keeps ad-hoc `python privacy/confidence_assesment.py`
# runs working from any working directory, with no environment setup.
_METRICS = Path(__file__).resolve().parents[1] / "utility" / "metrics"
if str(_METRICS) not in sys.path:
    sys.path.insert(0, str(_METRICS))

from topk_sample import load                     # noqa: E402
from kl_utility_scoring import build_prompt_ids  # noqa: E402

DEFAULT_MODEL = "/work/projects/mkilic/models/Qwen2.5-3B-Instruct"

# Surface forms of the two answers. Extra variants are cheap and only ever add
# coverage: any form that turns out to extend another (e.g. "Nope" = "No"+"pe")
# is dropped by resolve_forms, because its mass is already inside the shorter
# one. Both spaced and unspaced variants are listed since whether the assistant
# turn starts on a word boundary is a tokenizer detail.
YES_FORMS = ("Yes", "yes", "YES", " Yes", " yes", "Yeah", "Yep", "Y", "True", "true")
NO_FORMS = ("No", "no", "NO", " No", " no", "Nope", "Nah", "N", "False", "false")

CONTEXT_TEMPLATE = (
    "Below is a message sent by a user to an AI assistant.\n\n"
    "-----\n{context}\n-----\n\n"
    "{question}\n"
    "Answer with exactly one word: Yes or No."
)
NO_CONTEXT_TEMPLATE = (
    "{question}\n"
    "Answer with exactly one word: Yes or No."
)


# ------------------------------------------------------------------ labelling
# Ground truth is recovered from a text column of the dataset, and how to read
# it depends on the attribute: a class name is a word to look for, a membership
# question is a set of markers, an age is a number to compare against a
# threshold. Every labeller takes (text, classes) so an attribute can carry any
# of them, and every one returns None -- never a guess -- when the evidence is
# absent or ambiguous. load_rows drops those rows and counts them.

def word_label(text, classes):
    """Index of the class named as a whole word in `text`, or None if the
    evidence is absent or ambiguous. Word boundaries matter: "female" contains
    "male" as a substring but not as a word."""
    hits = [i for i, c in enumerate(classes)
            if re.search(rf"\b{re.escape(c)}\b", text, re.IGNORECASE)]
    return hits[0] if len(hits) == 1 else None


def marker_label(positive, negative):
    """
    Labeller for a one-vs-rest attribute: 1 if `text` carries a positive marker
    and no negative one, 0 in the mirror case, None when it carries both or
    neither.

    Neither-case is a row the markers do not cover and both-case is a
    contradiction; both are dropped rather than resolved, so widening a marker
    set can only ever move rows out of "unlabelled" and never silently reassign
    one. Matching is whole-word: "female" contains "male", and "woman" contains
    "man", as substrings but not as words, so the negative set does not have to
    be written defensively around the positive one.

    `classes` is accepted and ignored -- membership is defined by the markers,
    so renaming the display classes cannot move a row across the split.
    """
    def hit(markers, text):
        return any(re.search(rf"\b{re.escape(m)}\b", text, re.IGNORECASE)
                   for m in markers)

    def label(text, classes=None):
        pos, neg = hit(positive, text), hit(negative, text)
        return None if pos == neg else int(pos)
    return label


# "Is the user a woman?" as one-vs-rest. The negative set names identities that
# are not women rather than assuming a male/female dichotomy: a trans woman is
# matched by "woman" and labelled a woman, a trans man by "man" and labelled
# not, and "non-binary" is its own negative marker rather than an unlabellable
# row. Identities where womanhood is genuinely not decidable from a profile
# string -- genderfluid, genderqueer -- are in NEITHER set on purpose, so those
# rows come back None and are dropped and counted instead of being asserted into
# a class by an author who cannot know.
WOMAN_MARKERS = ("woman", "women", "female", "girl", "lady")
NOT_WOMAN_MARKERS = ("man", "men", "male", "boy", "guy",
                     "non-binary", "nonbinary", "enby", "agender")


# A comma-separated field that is nothing BUT an age, optionally with a unit:
# "34", "34 years old", "34yo". fullmatch is the whole point -- a number sitting
# inside prose ("type 2 diabetes") must not be read as one.
_AGE_FIELD = re.compile(r"(\d{1,3})\s*(?:y\.?o\.?|years?(?:\s+old)?|yrs?)?",
                        re.IGNORECASE)


def age_label(threshold):
    """
    Labeller for `age >= threshold`, returning 1 (classes[1]) at or above it.

    The age is the one comma-separated field of the profile that is a bare
    number: "Marta, 34, midwife, Krakow" -> 34. Taking the first number found
    anywhere instead would read "Ruth, 61, retired teacher, Cardiff, Wales,
    type 2 diabetes" as 61 or 2 depending only on scan order, so a row with two
    numeric fields is called ambiguous and dropped rather than resolved by a
    rule that happens to work on today's file.

    `classes` is accepted and ignored: the split is defined by the threshold, so
    renaming the classes cannot move a row across it.
    """
    def label(text, classes=None):
        ages = [m.group(1) for m in
                (_AGE_FIELD.fullmatch(f.strip()) for f in text.split(",")) if m]
        if len(ages) != 1:
            return None
        return int(int(ages[0]) >= threshold)
    return label


def age_questions(threshold):
    """Both polarities, built from the threshold so the question the model is
    asked can never drift from the labelling rule it is scored against."""
    return (f"Is the user under {threshold} years old?",
            f"Is the user {threshold} years old or older?")


@dataclass(frozen=True)
class BinaryAttribute:
    """
    A binary attribute, the two polarities of the question that reads it out,
    and the rule that recovers the ground truth from the dataset.

    questions[i] is the one whose "Yes" means f == classes[i]; asking both and
    averaging cancels the model's acquiescence bias (see --counterbalance).
    Reported probabilities are always for classes[1]. label_of(text, classes)
    returns that same index, or None for a row that cannot be labelled.
    """
    name: str
    classes: tuple
    questions: tuple
    label_of: Callable = word_label


ATTRIBUTES = {
    # The default. Minority is the disclosure with the sharpest consequences
    # for someone talking to an assistant, and unlike a stated number it leaks
    # through register, topic and situation -- exactly the channel a rewrite
    # that only deletes explicit attributes will miss. The threshold is a
    # parameter (--age_threshold): 18 is the one that matters legally, but any
    # threshold defines a valid binary attribute, and a dataset with no minors
    # in it can only be measured at one that actually splits its population.
    "age": BinaryAttribute(
        name="age",
        classes=("minor", "adult"),
        questions=age_questions(18),
        label_of=age_label(18),
    ),
    # One-vs-rest: "woman or not", not "female or male". The negative pole is
    # the complement of the positive one, so P(not a woman) = 1 - P(woman) by
    # construction and no one has to be sorted into a class they are not in.
    #
    # This also repairs --counterbalance rather than merely surviving it. The
    # old mirror asked "Is the user a man?" and treated 1 - P(man) as P(woman),
    # which is only a mirror if the two are exhaustive -- the very assumption
    # being dropped here. "Someone other than a woman" is the exact complement,
    # so the averaged estimate is now debiasing acquiescence instead of also
    # absorbing everyone the dichotomy left out.
    "gender": BinaryAttribute(
        name="gender",
        classes=("not a woman", "woman"),
        questions=("Is the user someone other than a woman?",
                   "Is the user a woman?"),
        label_of=marker_label(WOMAN_MARKERS, NOT_WOMAN_MARKERS),
    ),
}


def get_args():
    p = argparse.ArgumentParser(
        description="Adversarial advantage 2|Pr[f_hat = f] - 1/2| for a binary "
                    "user attribute, read off the Yes/No logits")
    p.add_argument("--model_path", default=DEFAULT_MODEL)
    p.add_argument("--attribute", default="age", choices=sorted(ATTRIBUTES))
    p.add_argument("--age_threshold", type=int, default=18,
                   help="--attribute age only: the split, in years. Labels are "
                        "age >= threshold, and both questions are rebuilt "
                        "around it (default: 18)")
    p.add_argument("--classes", nargs=2, metavar=("NEG", "POS"),
                   help="override the attribute's class names (probabilities "
                        "are reported for POS)")
    p.add_argument("--questions", nargs=2, metavar=("Q_NEG", "Q_POS"),
                   help='override the two questions; Q_POS is the one whose '
                        '"Yes" means f == POS')

    p.add_argument("--csv", help="dataset; omit to run --text / prior only")
    p.add_argument("--columns", nargs="+", default=["original"],
                   help="text columns to attack, one advantage per column")
    p.add_argument("--label_column", default="id",
                   help="column holding the ground-truth attribute: a class "
                        "name for gender, a bare age field for age. In "
                        "privacy_prompt_rewrites_100.csv both live in the "
                        "profile string under `id`")
    p.add_argument("--max_rows", type=int, default=None)

    p.add_argument("--text", help="score one ad-hoc message and exit")
    p.add_argument("--counterbalance", action="store_true",
                   help="ask both question polarities and average (2x cost)")
    p.add_argument("--ablate_context", action="store_true",
                   help="prior condition keeps the message block but empties "
                        "it, instead of dropping it entirely")
    p.add_argument("--batch_size", type=int, default=16,
                   help="surface forms scored per forward pass (multi-token "
                        "forms only; the single-token path needs one pass)")
    p.add_argument("--save_suffix", help="write results/privacy_advantage/<suffix>.json")
    return p.parse_args()


# ---------------------------------------------------------------- surface forms

def resolve_forms(tok, yes_forms=YES_FORMS, no_forms=NO_FORMS):
    """
    Tokenize the surface forms and reduce them to a prefix-free antichain,
    returning ([(form, cls, token_ids)], [(dropped_form, covered_by)]).

    Summing P over surface forms is only a probability if the forms are
    mutually exclusive events. As token sequences they are not: Qwen tokenizes
    "No" as [2753] and "Nope" as [2753, 375], so P("the answer starts with
    [2753]") already contains every "Nope" continuation, and adding them double
    counts. Keeping only the shortest form of each prefix chain fixes this and
    is strictly *more* complete than enumerating variants -- "No" also covers
    "No.", "No,", "No idea", and every other continuation we never thought to
    list. cls is 1 for yes, 0 for no.
    """
    tagged = [(f, 1, tuple(tok(f, add_special_tokens=False).input_ids)) for f in yes_forms]
    tagged += [(f, 0, tuple(tok(f, add_special_tokens=False).input_ids)) for f in no_forms]
    tagged.sort(key=lambda t: (len(t[2]), t[0]))

    kept, dropped = [], []
    for form, cls, ids in tagged:
        cover = next((k for k in kept if ids[:len(k[2])] == k[2]), None)
        if cover is None:
            kept.append((form, cls, ids))
        elif cover[1] != cls:
            # A yes-form prefixing a no-form would make the two events overlap
            # and the readout meaningless; no reduction can repair it.
            raise ValueError(
                f"surface form {form!r} ({'yes' if cls else 'no'}) extends "
                f"{cover[0]!r} ({'yes' if cover[1] else 'no'}) from the other "
                f"class -- the answer sets are not separable, edit them")
        else:
            dropped.append((form, cover[0]))
    return kept, dropped


# ------------------------------------------------------------------- log-probs

@torch.no_grad()
def form_logprobs(context_ids, forms, model, tok, batch_size=16):
    """
    log P(form | context) for each form, teacher-forced. Returns a list
    parallel to `forms`.

    log_softmax in float32: the losing class routinely sits in the fp16
    underflow region, and its mass is exactly what we are trying to measure.
    """
    # Fast path: when every form is one token the whole readout is a single
    # forward pass over the context, gathering from the last position's
    # distribution -- literally "read out the logits of the surface forms".
    if all(len(ids) == 1 for _, _, ids in forms):
        inp = torch.tensor([context_ids], device=model.device)
        logits = model(input_ids=inp).logits[0, -1]
        logp = F.log_softmax(logits.float(), dim=-1)
        return [logp[ids[0]].item() for _, _, ids in forms]

    P = len(context_ids)
    out = []
    for i in range(0, len(forms), batch_size):
        chunk = forms[i:i + batch_size]
        rows = [list(context_ids) + list(ids) for _, _, ids in chunk]
        L = max(len(r) for r in rows)
        input_ids = torch.full((len(rows), L), tok.pad_token_id, dtype=torch.long)
        attn = torch.zeros((len(rows), L), dtype=torch.long)
        for b, r in enumerate(rows):
            input_ids[b, :len(r)] = torch.tensor(r)
            attn[b, :len(r)] = 1
        input_ids = input_ids.to(model.device)
        attn = attn.to(model.device)

        logits = model(input_ids=input_ids, attention_mask=attn).logits
        # Only the form positions are ever needed, so slice BEFORE the fp32
        # log_softmax: a full [B, L, vocab] fp32 tensor is gigabytes, and is how
        # the utility benchmark ran itself out of memory.
        tail = logits[:, P - 1:L - 1, :].float()
        logp = F.log_softmax(tail, dim=-1)
        pred = logp.gather(-1, input_ids[:, P:L].unsqueeze(-1)).squeeze(-1)
        for b, (_, _, ids) in enumerate(chunk):
            out.append(pred[b, :len(ids)].sum().item())
    return out


def build_query(question, context, ablate_context=False):
    """The adversary's message. context=None drops the message block entirely;
    context="" (or ablate_context) keeps the framing with an empty block."""
    if context is None and not ablate_context:
        return NO_CONTEXT_TEMPLATE.format(question=question)
    return CONTEXT_TEMPLATE.format(context=context or "", question=question)


def read_yes(question, context, model, tok, forms, ablate_context=False, batch_size=16):
    """
    p(Yes) for one question and one context, renormalized over the two answer
    sets. Returns (p_yes, coverage, per-form probabilities).
    """
    query = build_query(question, context, ablate_context)
    ids = build_prompt_ids(tok, query)
    lps = form_logprobs(ids, forms, model, tok, batch_size)

    mass = {}
    m_yes = m_no = 0.0
    for (form, cls, _), lp in zip(forms, lps):
        p = math.exp(lp)
        mass[form] = p
        if cls == 1:
            m_yes += p
        else:
            m_no += p

    coverage = m_yes + m_no
    if coverage <= 0.0:
        # Both answer sets underflowed: the model is not answering this
        # question at all. nan rather than 0.5, so it cannot be mistaken
        # downstream for a confident coin flip.
        return float("nan"), 0.0, mass
    return m_yes / coverage, coverage, mass


def predict(attr, context, model, tok, forms, counterbalance=False,
            ablate_context=False, batch_size=16):
    """
    P(f = classes[1] | context) as judged by the model, plus diagnostics.

    With counterbalance the mirrored question is asked too and the two
    estimates averaged. A model with a Yes-bias reports P(male) too high under
    "Is the user a man?" and P(female) too high under "Is the user a woman?";
    averaging cancels the bias to first order. The spread between them is
    reported as `polarity_gap` -- a large gap means the readout is driven by
    the phrasing rather than by the attribute.
    """
    p_pos, cov_pos, _ = read_yes(attr.questions[1], context, model, tok, forms,
                                 ablate_context, batch_size)
    if not counterbalance:
        return {"p_pos": p_pos, "coverage": cov_pos, "polarity_gap": None}

    p_neg, cov_neg, _ = read_yes(attr.questions[0], context, model, tok, forms,
                                 ablate_context, batch_size)
    p_pos_mirrored = 1.0 - p_neg
    return {
        "p_pos": 0.5 * (p_pos + p_pos_mirrored),
        "coverage": 0.5 * (cov_pos + cov_neg),
        "polarity_gap": abs(p_pos - p_pos_mirrored),
    }


# ------------------------------------------------------------------- advantage

def advantage(records):
    """
    2|Pr[f_hat = f] - 1/2| over a list of {"p_pos", "label"} records, by two
    estimators:

    hard  f_hat = argmax, accuracy is a 0/1 mean. This is the definition.
    soft  accuracy is the mean probability the model puts on the true class.
          Same expectation under calibration, much lower variance, and it
          registers leakage that never crosses the decision boundary -- a model
          pushed from 0.50 to 0.62 on every example has learned something that
          the hard estimator scores as exactly zero.

    `floor` is E[Adv_hat] when the true advantage is 0: the absolute value
    folds sampling noise upward, so at n = 100 an honest coin flip still scores
    ~0.08. Treat estimates below the floor as no evidence of leakage.

    `baseline` is the advantage of the best CONSTANT adversary, 2|base_rate -
    1/2|: one that ignores the message and always answers the majority class.
    It reads nothing and therefore leaks nothing, so an estimate at or below it
    is class imbalance, not leakage. On a single-class dataset base_rate is 0
    or 1, baseline is 1.0, and no advantage above it is even attainable -- the
    measurement is vacuous however impressive the accuracy looks.
    """
    hard, soft, labels = [], [], []
    for r in records:
        p = r["p_pos"]
        if math.isnan(p):
            continue
        labels.append(r["label"])
        hard.append(0.5 if p == 0.5 else float((p > 0.5) == (r["label"] == 1)))
        soft.append(p if r["label"] == 1 else 1.0 - p)
    n = len(hard)
    if n == 0:
        return None

    def summarize(xs):
        acc = sum(xs) / n
        var = sum((x - acc) ** 2 for x in xs) / (n - 1) if n > 1 else 0.0
        se = math.sqrt(var / n)
        return acc, se, 2.0 * abs(acc - 0.5), 2.0 * se

    acc_h, se_h, adv_h, adv_se_h = summarize(hard)
    acc_s, se_s, adv_s, adv_se_s = summarize(soft)
    # Over the usable rows, not all of them: this is the number acc is being
    # compared against, so it has to be measured on the same set.
    base_rate = sum(labels) / n
    return {
        "n": n,
        "n_unusable": len(records) - n,
        "acc_hard": acc_h, "acc_hard_stderr": se_h,
        "advantage_hard": adv_h, "advantage_hard_stderr": adv_se_h,
        "acc_soft": acc_s, "acc_soft_stderr": se_s,
        "advantage_soft": adv_s, "advantage_soft_stderr": adv_se_s,
        # E|Z| for Z ~ N(0, se) is se*sqrt(2/pi); the advantage doubles it.
        "floor": 2.0 * math.sqrt(0.25 / n) * math.sqrt(2.0 / math.pi),
        "mean_coverage": sum(r["coverage"] for r in records) / len(records),
        "base_rate_pos": base_rate,
        "advantage_baseline": 2.0 * abs(base_rate - 0.5),
    }


def load_rows(csv_path, attr, label_column, max_rows=None):
    """Rows paired with their ground-truth class index, via the attribute's own
    labeller. Unlabelled and ambiguous rows are dropped and counted, never
    guessed at."""
    rows, skipped = [], 0
    with open(csv_path, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            label = attr.label_of(r[label_column], attr.classes)
            if label is None:
                skipped += 1
                continue
            rows.append((r, label))
            if max_rows and len(rows) >= max_rows:
                break
    return rows, skipped


if __name__ == "__main__":
    args = get_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    attr = ATTRIBUTES[args.attribute]
    # Threshold first, so an explicit --questions still wins over the generated
    # pair. replace() rather than a fresh BinaryAttribute: the labeller must
    # survive an override of the class names, and reconstructing the dataclass
    # by hand would silently reset it to the word_label default.
    if args.attribute == "age":
        attr = replace(attr, questions=age_questions(args.age_threshold),
                       label_of=age_label(args.age_threshold))
    if args.classes:
        attr = replace(attr, classes=tuple(args.classes))
    if args.questions:
        attr = replace(attr, questions=tuple(args.questions))

    print(f"Loading {args.model_path} on {device} ...")
    model, tok = load(args.model_path, device)

    forms, dropped = resolve_forms(tok)
    kept_yes = [f for f, c, _ in forms if c == 1]
    kept_no = [f for f, c, _ in forms if c == 0]
    print(f"\nAttribute '{attr.name}': {attr.classes[0]} (No) vs {attr.classes[1]} (Yes)")
    print(f"  question   : {attr.questions[1]}")
    if args.counterbalance:
        print(f"  mirrored   : {attr.questions[0]}")
    print(f"  YES forms  : {', '.join(map(repr, kept_yes))}")
    print(f"  NO forms   : {', '.join(map(repr, kept_no))}")
    if dropped:
        print("  absorbed   : " + ", ".join(f"{d!r} -> {c!r}" for d, c in dropped)
              + "   (prefix of a kept form; mass already counted)")

    call = dict(counterbalance=args.counterbalance, ablate_context=args.ablate_context,
                batch_size=args.batch_size)

    # The no-context readout does not depend on any row, so it is computed once
    # and reused as the baseline for every column.
    prior = predict(attr, None, model, tok, forms, **call)
    print(f"\nWITHOUT PROMPT   P({attr.classes[1]}) = {prior['p_pos']:.4f}   "
          f"coverage = {prior['coverage']:.4f}"
          + (f"   polarity_gap = {prior['polarity_gap']:.4f}"
             if prior["polarity_gap"] is not None else ""))
    print("  the model's prior; an adversary with only this scores advantage "
          "2|base_rate - 1/2|")

    if args.text:
        r = predict(attr, args.text, model, tok, forms, **call)
        print(f"\nWITH PROMPT      P({attr.classes[1]}) = {r['p_pos']:.4f}   "
              f"coverage = {r['coverage']:.4f}")
        print(f"  guess = {attr.classes[1] if r['p_pos'] > 0.5 else attr.classes[0]}"
              f"   shift vs prior = {r['p_pos'] - prior['p_pos']:+.4f}")
        raise SystemExit(0)

    if not args.csv:
        raise SystemExit(0)

    rows, skipped = load_rows(args.csv, attr, args.label_column, args.max_rows)
    print(f"\n{len(rows)} labelled rows from {args.csv}"
          + (f" ({skipped} skipped: no unambiguous {attr.name})" if skipped else ""))
    if not rows:
        raise SystemExit(f"No row could be labelled {attr.classes[0]}/"
                         f"{attr.classes[1]} from column '{args.label_column}'")

    # A single-class dataset cannot measure an adversary. Every estimator below
    # still returns a number -- a model that always answers the majority class
    # scores acc 1.0 and advantage 1.0 -- and every one of them is the base rate
    # rather than anything read out of the text. Loud, and not fatal: the
    # per-row probabilities are still worth looking at, and the prior condition
    # is still meaningful on its own.
    base_rate = sum(l for _, l in rows) / len(rows)
    if base_rate in (0.0, 1.0):
        only = attr.classes[int(base_rate)]
        print(f"\n!! WARNING: every labelled row is '{only}'. The label has no "
              f"variance, so the\n!! baseline advantage is 1.0 and no estimate "
              f"can exceed it: this run measures\n!! the model's answering bias, "
              f"not leakage. Use a dataset carrying both\n!! classes -- or, for "
              f"age, an --age_threshold that splits this population.")

    cw = max(len(c) for c in attr.classes)   # keeps the per-row lines aligned
    results, per_example = {}, {}
    for col in args.columns:
        print(f"\n=== {col} ===")
        recs = []
        for i, (row, label) in enumerate(rows, 1):
            r = predict(attr, row[col], model, tok, forms, **call)
            r["label"] = label
            r["id"] = row.get(args.label_column, str(i))
            recs.append(r)
            # One line per row, like benchmark_utility.py: readable in an
            # sbatch .out, and the per-row probabilities are what you want when
            # an aggregate looks wrong.
            hit = "?" if math.isnan(r["p_pos"]) else \
                  ("hit " if (r["p_pos"] > 0.5) == (label == 1) else "MISS")
            print(f"  [{i:>3}/{len(rows)}] {r['id'][:34]:<34} "
                  f"true={attr.classes[label]:<{cw}} P({attr.classes[1]})="
                  f"{r['p_pos']:.4f} cov={r['coverage']:.3f} {hit}", flush=True)
        per_example[col] = recs
        results[col] = advantage(recs)

    print(f"\n{'column':<26} {'n':>4} {'acc':>7} {'ADV':>7} {'+-':>6} "
          f"{'ADV_soft':>9} {'floor':>7} {'base':>7} {'cover':>7}")
    print("-" * 90)
    for col, s in results.items():
        if s is None:
            print(f"{col:<26}    -  (no usable readouts)")
            continue
        print(f"{col:<26} {s['n']:>4} {s['acc_hard']:>7.4f} "
              f"{s['advantage_hard']:>7.4f} {s['advantage_hard_stderr']:>6.4f} "
              f"{s['advantage_soft']:>9.4f} {s['floor']:>7.4f} "
              f"{s['advantage_baseline']:>7.4f} {s['mean_coverage']:>7.4f}")
    print("\nADV = 2|acc - 1/2|. It is evidence of leakage only above BOTH "
          "`floor` (sampling\nnoise, which the absolute value folds upward) and "
          "`base` (what a constant adversary\nthat never reads the message "
          "already scores on this class balance). Low `cover`\nmeans the model "
          "was not really answering Yes/No and the readout is a renormalized "
          "remnant.")

    if args.save_suffix:
        out = Path("results/privacy_advantage")
        out.mkdir(parents=True, exist_ok=True)
        path = out / f"{args.save_suffix}.json"
        with open(path, "w", encoding="utf-8") as fh:
            json.dump({
                "model": args.model_path,
                "attribute": {"name": attr.name, "classes": list(attr.classes),
                              "questions": list(attr.questions),
                              "age_threshold": (args.age_threshold
                                                if attr.name == "age" else None)},
                "csv": args.csv,
                "counterbalance": args.counterbalance,
                "ablate_context": args.ablate_context,
                "surface_forms": {"yes": kept_yes, "no": kept_no, "absorbed": dropped},
                "prior": prior,
                "summary": results,
                "per_example": per_example,
            }, fh, indent=2, ensure_ascii=False)
        print(f"\nWrote {path}")
