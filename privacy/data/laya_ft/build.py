"""Merge the batch files, validate them, print statistics and write the outputs.

Reads batches/{test,train}_NN.jsonl (one pair per line, the fields of the CSVs),
checks every pair and the dataset-wide rules, then writes test.csv, train.csv,
test_values.json and the "Statistics" section of README.md.
If any check fails it prints the problems and exits with status 1, writing nothing.
Unmet quotas are only marked in the statistics, so partial datasets still build.
"""
import csv
import glob
import json
import re
import statistics
import sys
from collections import Counter, defaultdict

ATTRIBUTES = [
    "profession", "geographic location", "health condition", "relationship status",
    "socioeconomic status", "age", "political views", "nationality", "family situation",
    "education level", "gender", "name", "residential address", "date of birth",
    "credit card number", "email address",
]
FIELDS = ["id", "attributes", "values", "leaking", "safe", "not_inferable", "edit_type", "cue_type"]
EDIT_SHARE = {"deletion": 0.40, "generalization": 0.40, "other_person": 0.20}
CUE_TYPES = ["explicit", "indirect", "mixed"]
COUNT_SHARE = {"1": 0.40, "2": 0.40, "3-4": 0.20}
SHARE_TOLERANCE = 0.05
SIZE = {"test": 400, "train": 2000}
MIN_ATTRIBUTE = {"test": 25, "train": 130}
MIN_NOT_INFERABLE = {"test": 20, "train": 100}
WORDS = (15, 90)
MAX_JACCARD = 0.5
IDENTIFIERS = ["name", "residential address", "date of birth", "credit card number", "email address"]
TEST_ONLY_VALUES = [
    "profession", "geographic location", "health condition", "name",
    "residential address", "email address", "credit card number",
]
UNMET = "  <-- NOT MET"


def load_pairs():
    """Return {split: [pair, ...]} read from the batch files, plus read problems."""
    pairs, problems = {}, []
    for split in SIZE:
        pairs[split] = []
        for path in sorted(glob.glob(f"batches/{split}_*.jsonl")):
            with open(path, encoding="utf-8") as f:
                for number, line in enumerate(f, 1):
                    if not line.strip():
                        continue
                    try:
                        pairs[split].append(json.loads(line))
                    except json.JSONDecodeError as error:
                        problems.append(f"{path}:{number}: invalid JSON ({error})")
    return pairs, problems


def split_list(text):
    return [part for part in text.split("; ")] if text else []


def tokens(text):
    return re.findall(r"\w+|[^\w\s]", text)


def is_subsequence(small, big):
    remaining = iter(big)
    return all(token in remaining for token in small)


def digits(text):
    return re.sub(r"\D", "", text)


def in_text(value, attribute, text):
    """Does an identifier value (or any value, for non-cards) occur in a message?"""
    if attribute == "credit card number":
        return digits(value) in digits(text)
    return value.lower() in text.lower()


def check_pair(split, pair):
    """Return the problems of one pair."""
    pid = pair.get("id", "?")
    if set(pair) != set(FIELDS):
        return [f"{pid}: fields are {sorted(pair)}"]
    if any(not isinstance(pair[f], str) or not pair[f].strip() for f in FIELDS):
        return [f"{pid}: empty or non-string field"]
    problems = []
    if not re.fullmatch(rf"{split}_\d{{4}}", pair["id"]):
        problems.append(f"{pid}: id does not match {split}_NNNN")
    attributes, not_inferable = split_list(pair["attributes"]), split_list(pair["not_inferable"])
    if not 1 <= len(attributes) <= 4 or len(set(attributes)) != len(attributes):
        problems.append(f"{pid}: attributes must be 1-4 distinct names")
    if not 1 <= len(not_inferable) <= 2 or len(set(not_inferable)) != len(not_inferable):
        problems.append(f"{pid}: not_inferable must be 1-2 distinct names")
    for name in attributes + not_inferable:
        if name not in ATTRIBUTES:
            problems.append(f"{pid}: unknown attribute '{name}'")
    if set(attributes) & set(not_inferable):
        problems.append(f"{pid}: attributes and not_inferable overlap")
    if pair["edit_type"] not in EDIT_SHARE:
        problems.append(f"{pid}: bad edit_type")
    if pair["cue_type"] not in CUE_TYPES:
        problems.append(f"{pid}: bad cue_type")
    if pair["cue_type"] == "mixed" and len(attributes) < 2:
        problems.append(f"{pid}: a single attribute cannot have mixed cues")
    leaking, safe = pair["leaking"], pair["safe"]
    if leaking == safe:
        problems.append(f"{pid}: leaking equals safe")
    if re.search(r"\[[^\]]*\]", leaking + safe):
        problems.append(f"{pid}: slot mark in a message")
    if not WORDS[0] <= len(leaking.split()) <= WORDS[1]:
        problems.append(f"{pid}: leaking has {len(leaking.split())} words")
    if "\n" in leaking + safe:
        problems.append(f"{pid}: newline in a message")
    if pair["edit_type"] == "deletion" and not is_subsequence(tokens(safe), tokens(leaking)):
        problems.append(f"{pid}: deletion but safe is not a token subsequence of leaking")
    values = parse_values(pair["values"])
    if [a for a, _ in values] != attributes:
        problems.append(f"{pid}: values do not match attributes in order")
    for attribute, value in values:
        if attribute in IDENTIFIERS and not in_text(value, attribute, leaking):
            problems.append(f"{pid}: {attribute} value not found verbatim in leaking")
        if pair["edit_type"] != "other_person" and attribute in IDENTIFIERS + [
            "profession", "geographic location", "health condition"
        ] and in_text(value, attribute, safe):
            problems.append(f"{pid}: {attribute} value still in safe")
    return problems


def parse_values(text):
    values = []
    for part in split_list(text):
        attribute, _, value = part.partition(": ")
        values.append((attribute, value))
    return values


def trigrams(text):
    words = re.findall(r"\w+", text.lower())
    return {tuple(words[i:i + 3]) for i in range(len(words) - 2)}


def check_near_duplicates(all_pairs):
    grams = [trigrams(p["leaking"]) for p in all_pairs]
    index = defaultdict(list)
    problems = []
    for i, current in enumerate(grams):
        seen = Counter(j for gram in current for j in index[gram])
        for j in seen:
            jaccard = len(current & grams[j]) / len(current | grams[j])
            if jaccard >= MAX_JACCARD:
                problems.append(f"{all_pairs[j]['id']} and {all_pairs[i]['id']}: trigram Jaccard {jaccard:.2f}")
        for gram in current:
            index[gram].append(i)
    return problems


def check_unique_identifiers(all_pairs):
    """Names (and their parts), addresses, emails and card numbers are used once."""
    seen, problems = {}, []
    for pair in all_pairs:
        for attribute, value in parse_values(pair["values"]):
            if attribute not in IDENTIFIERS or attribute == "date of birth":
                continue
            keys = [value.lower()]
            if attribute == "credit card number":
                keys = [digits(value)]
            if attribute == "name":
                keys += [part.lower() for part in value.split() if len(part) > 2]
            for key in dict.fromkeys(keys):
                if (attribute, key) in seen:
                    problems.append(f"{pair['id']} and {seen[(attribute, key)]}: same {attribute} '{key}'")
                seen[(attribute, key)] = pair["id"]
    return problems


def collect_test_values(test_pairs):
    values = {attribute: set() for attribute in TEST_ONLY_VALUES}
    for pair in test_pairs:
        for attribute, value in parse_values(pair["values"]):
            if attribute in values:
                values[attribute].add(value)
    return {attribute: sorted(found) for attribute, found in values.items()}


def check_test_values_absent(test_values, train_pairs):
    problems = []
    for pair in train_pairs:
        text = pair["leaking"] + " " + pair["safe"]
        for attribute, values in test_values.items():
            for value in values:
                hit = in_text(value, attribute, text)
                if attribute == "name":
                    hit = hit or any(
                        re.search(rf"\b{re.escape(part)}\b", text, re.IGNORECASE)
                        for part in value.split() if len(part) > 2
                    )
                if hit:
                    problems.append(f"{pair['id']}: test {attribute} value '{value}' appears in train")
    return problems


def check_flagged(all_pairs):
    try:
        with open("flagged.csv", encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
    except FileNotFoundError:
        return []
    ids = {p["id"] for p in all_pairs}
    return [f"flagged.csv: unknown id {r['id']}" for r in rows if r["id"] not in ids]


def share_mark(share, target):
    return UNMET if abs(share - target) > SHARE_TOLERANCE else ""


def statistics_lines(split, pairs):
    """Counts and quota marks of one split."""
    n = len(pairs)
    lines = [f"== {split}: {n} pairs (target {SIZE[split]}){UNMET if n != SIZE[split] else ''}"]
    in_attributes = Counter(a for p in pairs for a in split_list(p["attributes"]))
    in_not_inferable = Counter(a for p in pairs for a in split_list(p["not_inferable"]))
    lines.append(f"{'attribute':24}{'attributes':>12}{'not_inferable':>15}")
    for attribute in ATTRIBUTES:
        a, b = in_attributes[attribute], in_not_inferable[attribute]
        marks = [name for name, count, minimum in (
            ("attributes", a, MIN_ATTRIBUTE[split]), ("not_inferable", b, MIN_NOT_INFERABLE[split])
        ) if count < minimum]
        lines.append(f"{attribute:24}{a:>12}{b:>15}{UNMET + ' (' + ', '.join(marks) + ')' if marks else ''}")
    lines.append(f"(minimum per attribute: {MIN_ATTRIBUTE[split]} in attributes, "
                 f"{MIN_NOT_INFERABLE[split]} in not_inferable)")
    if n == 0:
        return lines
    edits = Counter(p["edit_type"] for p in pairs)
    lines.append("edit_type: " + ", ".join(
        f"{e} {edits[e]} ({edits[e] / n:.0%}){share_mark(edits[e] / n, t)}" for e, t in EDIT_SHARE.items()))
    cues = Counter(p["cue_type"] for p in pairs)
    explicit_share = (cues["explicit"] + cues["mixed"] / 2) / n
    lines.append("cue_type: " + ", ".join(f"{c} {cues[c]} ({cues[c] / n:.0%})" for c in CUE_TYPES)
                 + f"; explicit cues about {explicit_share:.0%}{share_mark(explicit_share, 0.5)}")
    sizes = Counter(len(split_list(p["attributes"])) for p in pairs)
    groups = {"1": sizes[1], "2": sizes[2], "3-4": sizes[3] + sizes[4]}
    lines.append("attributes per pair: " + ", ".join(
        f"{g} {c} ({c / n:.0%}){share_mark(c / n, COUNT_SHARE[g])}" for g, c in groups.items())
        + f" (3: {sizes[3]}, 4: {sizes[4]})")
    for field in ("leaking", "safe"):
        counts = [len(p[field].split()) for p in pairs]
        lines.append(f"words in {field}: min {min(counts)}, mean {statistics.mean(counts):.1f}, "
                     f"median {statistics.median(counts)}, max {max(counts)}")
    return lines


def write_csv(path, pairs):
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(sorted(pairs, key=lambda p: p["id"]))


def update_readme(lines):
    with open("README.md", encoding="utf-8") as f:
        head = f.read().split("\n## Statistics")[0].rstrip("\n")
    with open("README.md", "w", encoding="utf-8") as f:
        f.write(head + "\n\n## Statistics\n\n```\n" + "\n".join(lines) + "\n```\n")


def main():
    pairs, problems = load_pairs()
    all_pairs = pairs["test"] + pairs["train"]
    for split, split_pairs in pairs.items():
        for pair in split_pairs:
            problems += check_pair(split, pair)
    ids = Counter(p.get("id") for p in all_pairs)
    problems += [f"{i}: duplicate id" for i, c in ids.items() if c > 1]
    if not problems:
        test_values = collect_test_values(pairs["test"])
        problems += check_near_duplicates(all_pairs)
        problems += check_unique_identifiers(all_pairs)
        problems += check_test_values_absent(test_values, pairs["train"])
        problems += check_flagged(all_pairs)
    if problems:
        print("\n".join(problems[:200]))
        print(f"{len(problems)} problem(s); nothing written")
        sys.exit(1)
    for split, split_pairs in pairs.items():
        write_csv(f"{split}.csv", split_pairs)
    with open("test_values.json", "w", encoding="utf-8") as f:
        json.dump(test_values, f, ensure_ascii=False, indent=1)
    lines = statistics_lines("test", pairs["test"]) + [""] + statistics_lines("train", pairs["train"])
    print("\n".join(lines))
    update_readme(lines)


if __name__ == "__main__":
    main()
