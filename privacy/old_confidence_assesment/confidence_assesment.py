"""
How well can the LLM guess a binary user attribute (age / gender) from a message?

For every message we ask the model a Yes/No question ("Is the user a woman?")
and read P(Yes) straight off the logits instead of sampling an answer:

    p_yes = P(yes-words) / (P(yes-words) + P(no-words))

The adversarial advantage over a dataset is

    Adv = 2 * |accuracy - 1/2|

0 means coin flip, 1 means always right (or always wrong). We also compute the
same readout WITHOUT the message (the "prior") as a baseline.

Two numbers help interpreting Adv:
    floor : what pure noise scores at this n (about 0.8 / sqrt(n))
    base  : what "always guess the majority class" scores
Only an Adv above both is evidence of leakage.
"""
import argparse
import math
import re
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

# the helper scripts live in ../utility/metrics
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "utility" / "metrics"))
from topk_sample import load                     # noqa: E402
from kl_utility_scoring import build_prompt_ids  # noqa: E402

DEFAULT_MODEL = "/work/projects/mkilic/models/Qwen2.5-3B-Instruct"

YES_FORMS = ["Yes", "yes", "YES", " Yes", " yes", "Yeah", "Yep", "Y", "True", "true"]
NO_FORMS = ["No", "no", "NO", " No", " no", "Nope", "Nah", "N", "False", "false"]

WITH_CONTEXT = ("Below is a message sent by a user to an AI assistant.\n\n"
                "-----\n{context}\n-----\n\n"
                "{question}\nAnswer with exactly one word: Yes or No.")
WITHOUT_CONTEXT = "{question}\nAnswer with exactly one word: Yes or No."

WOMAN_WORDS = ["woman", "women", "female", "girl", "lady"]
NOT_WOMAN_WORDS = ["man", "men", "male", "boy", "guy",
                   "non-binary", "nonbinary", "enby", "agender"]


# ------------------------------------------------------------------ labels
# Each labeller returns 1 (positive class), 0 (negative class) or None (unclear).

def has_word(words, text):
    return any(re.search(rf"\b{re.escape(w)}\b", text, re.IGNORECASE) for w in words)


def gender_label(text):
    is_woman = has_word(WOMAN_WORDS, text)
    is_not_woman = has_word(NOT_WOMAN_WORDS, text)
    if is_woman == is_not_woman:        # both or neither -> unclear
        return None
    return int(is_woman)


def age_label(text, threshold):
    # the age is the one comma-separated field that is just a number, e.g. "34" or "34 years old"
    ages = []
    for field in text.split(","):
        m = re.fullmatch(r"(\d{1,3})\s*(?:y\.?o\.?|years?(?:\s+old)?|yrs?)?",
                         field.strip(), re.IGNORECASE)
        if m:
            ages.append(int(m.group(1)))
    if len(ages) != 1:
        return None
    return int(ages[0] >= threshold)


def make_attribute(args):
    """classes[1] is the positive class; questions[1] is the question whose Yes means positive."""
    if args.attribute == "age":
        t = args.age_threshold
        attr = {"name": "age",
                "classes": ["minor", "adult"],
                "questions": [f"Is the user under {t} years old?",
                              f"Is the user {t} years old or older?"],
                "label": lambda text: age_label(text, t)}
    else:
        attr = {"name": "gender",
                "classes": ["not a woman", "woman"],
                "questions": ["Is the user someone other than a woman?",
                              "Is the user a woman?"],
                "label": gender_label}
    if args.classes:
        attr["classes"] = list(args.classes)
    if args.questions:
        attr["questions"] = list(args.questions)
    return attr


# ------------------------------------------------------------------ answer words

def resolve_forms(tok):
    """
    Tokenize the Yes/No words and drop any word whose tokens start with another
    kept word (e.g. "Nope" = "No" + "pe"): its probability is already inside
    the shorter one, so summing both would double count.
    Returns kept [(word, is_yes, token_ids)] and dropped [(word, covered_by)].
    """
    forms = [(w, 1, tok(w, add_special_tokens=False).input_ids) for w in YES_FORMS]
    forms += [(w, 0, tok(w, add_special_tokens=False).input_ids) for w in NO_FORMS]
    forms.sort(key=lambda f: (len(f[2]), f[0]))

    kept, dropped = [], []
    for word, is_yes, ids in forms:
        cover = next((k for k in kept if ids[:len(k[2])] == k[2]), None)
        if cover is None:
            kept.append((word, is_yes, ids))
        elif cover[1] != is_yes:
            raise ValueError(f"{word!r} starts with {cover[0]!r} from the other class")
        else:
            dropped.append((word, cover[0]))
    return kept, dropped


# ------------------------------------------------------------------ model readout

@torch.no_grad()
def log_probs_after(ids, model, start):
    """Next-token log-probabilities for positions start.. of `ids`, in float32 so
    tiny probabilities don't underflow."""
    inp = torch.tensor([ids], device=model.device)
    return F.log_softmax(model(input_ids=inp).logits[0, start:].float(), dim=-1)


def word_prob(context_ids, next_logp, ids, model):
    """P(ids | context). One-token words reuse the context's next-token distribution;
    longer words need their own forward pass."""
    if len(ids) == 1:
        return math.exp(next_logp[ids[0]].item())
    logp = log_probs_after(context_ids + ids, model, len(context_ids) - 1)
    return math.exp(sum(logp[i, t].item() for i, t in enumerate(ids)))


def ask(question, context, model, tok, forms, ablate_context):
    """Return (p_yes, coverage) for one question. context=None means no message."""
    if context is None and not ablate_context:
        query = WITHOUT_CONTEXT.format(question=question)
    else:
        query = WITH_CONTEXT.format(context=context or "", question=question)
    ids = build_prompt_ids(tok, query)
    next_logp = log_probs_after(ids, model, len(ids) - 1)[0]

    p_yes = p_no = 0.0
    for _, is_yes, form_ids in forms:
        p = word_prob(ids, next_logp, form_ids, model)
        if is_yes:
            p_yes += p
        else:
            p_no += p

    coverage = p_yes + p_no
    if coverage == 0:
        return float("nan"), 0.0
    return p_yes / coverage, coverage


def predict(attr, context, model, tok, forms, counterbalance, ablate_context):
    """P(positive class). With counterbalance also ask the opposite question and average,
    which cancels the model's tendency to just say Yes."""
    p_pos, cov_pos = ask(attr["questions"][1], context, model, tok, forms, ablate_context)
    if not counterbalance:
        return {"p_pos": p_pos, "coverage": cov_pos, "polarity_gap": None}

    p_neg, cov_neg = ask(attr["questions"][0], context, model, tok, forms, ablate_context)
    p_mirror = 1.0 - p_neg
    return {"p_pos": (p_pos + p_mirror) / 2,
            "coverage": (cov_pos + cov_neg) / 2,
            "polarity_gap": abs(p_pos - p_mirror)}


# ------------------------------------------------------------------ advantage

def mean_and_stderr(xs):
    n = len(xs)
    mean = sum(xs) / n
    var = sum((x - mean) ** 2 for x in xs) / (n - 1) if n > 1 else 0.0
    return mean, math.sqrt(var / n)


def advantage(records):
    """
    hard: accuracy of the argmax guess (the textbook definition)
    soft: mean probability put on the true class (less noisy)
    """
    good = [r for r in records if not math.isnan(r["p_pos"])]
    n = len(good)
    if n == 0:
        return None

    hard, soft = [], []
    for r in good:
        p, y = r["p_pos"], r["label"]
        hard.append(0.5 if p == 0.5 else float((p > 0.5) == (y == 1)))
        soft.append(p if y == 1 else 1 - p)

    acc_h, se_h = mean_and_stderr(hard)
    acc_s, se_s = mean_and_stderr(soft)
    base_rate = sum(r["label"] for r in good) / n
    return {
        "n": n,
        "n_unusable": len(records) - n,
        "acc_hard": acc_h, "acc_hard_stderr": se_h,
        "advantage_hard": 2 * abs(acc_h - 0.5), "advantage_hard_stderr": 2 * se_h,
        "acc_soft": acc_s, "acc_soft_stderr": se_s,
        "advantage_soft": 2 * abs(acc_s - 0.5), "advantage_soft_stderr": 2 * se_s,
        "floor": 2 * math.sqrt(0.25 / n) * math.sqrt(2 / math.pi),
        "mean_coverage": sum(r["coverage"] for r in records) / len(records),
        "base_rate_pos": base_rate,
        "advantage_baseline": 2 * abs(base_rate - 0.5),
    }


# ------------------------------------------------------------------ setup (shared with confidence_sweep.py)

def add_common_args(p):
    p.add_argument("--model_path", default=DEFAULT_MODEL)
    p.add_argument("--attribute", default="age", choices=["age", "gender"])
    p.add_argument("--age_threshold", type=int, default=18)
    p.add_argument("--classes", nargs=2, metavar=("NEG", "POS"))
    p.add_argument("--questions", nargs=2, metavar=("Q_NEG", "Q_POS"))
    p.add_argument("--counterbalance", action="store_true")
    p.add_argument("--ablate_context", action="store_true",
                   help="prior keeps the message block but leaves it empty")


def setup(args):
    """Load the model, resolve the answer words and compute the no-message prior.
    Returns (attr, run, prior, forms_info); run(text) scores one message."""
    attr = make_attribute(args)
    pos, neg = attr["classes"][1], attr["classes"][0]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {args.model_path} on {device} ...")
    model, tok = load(args.model_path, device)

    forms, dropped = resolve_forms(tok)
    forms_info = {"yes": [w for w, y, _ in forms if y],
                  "no": [w for w, y, _ in forms if not y],
                  "absorbed": dropped}
    print(f"\nAttribute '{attr['name']}': {neg} (No) vs {pos} (Yes)")
    print(f"  question : {attr['questions'][1]}")
    if args.counterbalance:
        print(f"  mirrored : {attr['questions'][0]}")
    print(f"  YES words: {forms_info['yes']}")
    print(f"  NO words : {forms_info['no']}")
    if dropped:
        print(f"  absorbed : {dropped}")

    def run(context):
        return predict(attr, context, model, tok, forms, args.counterbalance, args.ablate_context)

    # baseline: the model's answer without seeing any message
    prior = run(None)
    print(f"\nWITHOUT PROMPT  P({pos}) = {prior['p_pos']:.4f}  coverage = {prior['coverage']:.4f}")
    if prior["polarity_gap"] is not None:
        print(f"  polarity_gap = {prior['polarity_gap']:.4f}")
    return attr, run, prior, forms_info


# ------------------------------------------------------------------ main: score one message

def main():
    p = argparse.ArgumentParser(description="P(attribute) the LLM infers from one message")
    add_common_args(p)
    p.add_argument("--text", help="message to score; omit to print only the prior")
    args = p.parse_args()

    attr, run, prior, _ = setup(args)
    if args.text:
        pos, neg = attr["classes"][1], attr["classes"][0]
        r = run(args.text)
        guess = pos if r["p_pos"] > 0.5 else neg
        print(f"\nWITH PROMPT     P({pos}) = {r['p_pos']:.4f}  coverage = {r['coverage']:.4f}")
        print(f"  guess = {guess}  shift vs prior = {r['p_pos'] - prior['p_pos']:+.4f}")


if __name__ == "__main__":
    main()
