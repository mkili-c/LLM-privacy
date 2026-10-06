"""Build pupa_sd_eval.csv, the evaluation copy of PUPA-SD.

Input:  PUPA_SD.csv (166 rows; github.com/siyan-sylvia-li/Probabilistic_PAPILLON,
        commit a282782, data/PUPA_SD.csv).
Output: pupa_sd_eval.csv with
          id        the row's `index`
          original  the user's prompt (`user_query`); the only column the anonymizers read
          truth     JSON {category: [values]}: the dataset's own disclosures, categories and
                    values verbatim, for the judge that decides whether each is still inferable

Only rows are selected; labels are never edited. A row is kept if its prompt has at most
MAX_LAYA_TOKENS tokens and it has at least one disclosure in a category that corresponds
to one of our attributes (ATTRIBUTES in privacy/attribute_probe.py); disclosures in other
categories are left out. Longer prompts are dropped, not truncated.

Run from this folder with ~/.llmvenv active (it needs Laya's tokenizer): python build.py
"""
import csv
import json
from collections import Counter

from transformers import AutoTokenizer

RAW = "PUPA_SD.csv"
OUT = "pupa_sd_eval.csv"
LAYA_TOKENIZER = "/work/projects/mkilic/models/laya/tokenizer"
# Laya, the loop's second attacker, reads 512 tokens: the longest of our 10 attribute
# questions takes 37 and [SEP] 1, leaving 474 for the message (laya_probe.py raises above).
# 50 tokens of headroom because the loop's edits can lengthen the text.
MAX_LAYA_TOKENS = 474 - 50

# PUPA-SD categories that correspond to our attributes: occupation (and its typo) ->
# profession, location -> geographic location, the three health ones -> health condition,
# education -> education level, finance -> socioeconomic status, family -> family situation,
# race/nationality -> nationality; age, gender and relationship_status match directly
OUR_CATEGORIES = {
    "occupation", "occupaion", "location", "health", "mental_health", "reproductive_health",
    "age", "gender", "education", "finance", "relationship_status", "family", "race/nationality",
}


def read_rows():
    with open(RAW, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def parse_labels(pii_units):
    """'category: span; ...' -> {category: [span, ...]} for our categories, both verbatim."""
    labels = {}
    for part in pii_units.split(";"):
        if ":" not in part:
            continue
        category, span = (x.strip() for x in part.split(":", 1))
        if category in OUR_CATEGORIES and span not in labels.get(category, []):
            labels.setdefault(category, []).append(span)
    return labels


def count_tokens(raw):
    """{index: number of Laya tokens of the prompt}, as laya_probe.LayaProbe.encode counts them."""
    tok = AutoTokenizer.from_pretrained(LAYA_TOKENIZER)
    return {r["index"]: len(tok(r["user_query"], add_special_tokens=False).input_ids) for r in raw}


def select_rows(raw, n_tokens):
    """Rows within MAX_LAYA_TOKENS that disclose at least one of our attributes, with their labels."""
    selected = []
    for row in raw:
        labels = parse_labels(row["pii_units"])
        if labels and n_tokens[row["index"]] <= MAX_LAYA_TOKENS:
            selected.append((row, labels))
    return selected


def write_eval(selected):
    with open(OUT, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "original", "truth"])
        for row, labels in selected:
            writer.writerow([row["index"], row["user_query"], json.dumps(labels, ensure_ascii=False)])


def print_statistics(raw, n_tokens, selected):
    too_long = sum(n > MAX_LAYA_TOKENS for n in n_tokens.values())
    per_category = Counter(c for _, labels in selected for c in labels)
    values = sum(len(v) for _, labels in selected for v in labels.values())
    print(f"{len(raw)} rows read, {too_long} longer than {MAX_LAYA_TOKENS} Laya tokens, "
          f"{len(selected)} written to {OUT}")
    print(f"(row, category) pairs: {sum(per_category.values())}, values: {values}")
    for category, count in per_category.most_common():
        print(f"  {category}: {count}")


def main():
    raw = read_rows()
    n_tokens = count_tokens(raw)
    selected = select_rows(raw, n_tokens)
    write_eval(selected)
    print_statistics(raw, n_tokens, selected)


if __name__ == "__main__":
    main()
