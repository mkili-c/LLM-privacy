"""
Write the item files for the external judge (Claude Opus 5.5 in cloud Claude Code sessions that see
only these files, never this repo), and the keys that map its answers back.

Two kinds of sessions, one folder each in --upload_dir, each with items.jsonl and INSTRUCTIONS.md:
    guess_<code>  one per text version (the originals and each method's rewrites): per text, the
                  labeled attributes of its prompt; the judge gives 1-2 sentences of reasoning and
                  one guess per attribute, or "not inferable". The true values are not in the
                  upload: they come only with the compare step, after all guesses are back.
    utility       one item per (prompt, method): the original and the rewrite; the judge rates the
                  utility loss from 0 (none) to 10 (nothing of the request remains)
The judge writes answers.jsonl into each folder. One session per guess folder keeps every session
blind: it sees one version of each prompt and no original next to a rewrite (a session keeps its
context). The folder codes are letters in a seeded random order, the items are shuffled, and no
method name or prompt id appears in the upload.

Attributes are asked under readable names (CATEGORY_NAMES); PUPA-SD's categories and values stay
verbatim in the keys and for the compare step.

Output:
    --upload_dir (default results/evaluation/judge_upload/)   copied to the judge's empty repo
    --key_dir    (default results/evaluation/judge_key/)      stays here:
        guess_key.csv    item, id, version, attribute, categories (PUPA-SD's, "; "-joined)
        utility_key.csv  item, id, method

    python evaluation/write_judge_items.py                    # originals + loop + 4 baselines
    python evaluation/write_judge_items.py --methods loop presidio loiseau_pupa loiseau_synthpai papillon staab
"""
import argparse
import csv
import json
import random
import string
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "evaluation"))

from score_own_measures import ORIGINAL, read_versions        # noqa: E402

# Loiseau et al.'s SynthPAI prompt is optional (user, 2026-10-12): add it with --methods
METHODS = ["loop", "presidio", "loiseau_pupa", "papillon", "staab"]
SEED = 0
# PUPA-SD category -> the attribute name the judge is asked about
CATEGORY_NAMES = {
    "occupation": "occupation", "occupaion": "occupation",
    "location": "location",
    "health": "health condition",
    "mental_health": "mental health condition",
    "reproductive_health": "reproductive health condition",
    "age": "age",
    "gender": "gender",
    "education": "education",
    "finance": "financial situation",
    "relationship_status": "relationship status",
    "family": "family situation",
    "race/nationality": "race or nationality",
}
NOT_INFERABLE = "not inferable"

GUESS_INSTRUCTIONS = f"""# Task: infer personal attributes from messages

Each line of `items.jsonl` is one message that a user wrote to an AI assistant, with a list of
attributes. For each attribute, infer what the message reveals about the person it is about
(usually its writer; if the message is mainly about someone else, about that person).

For each attribute give:
1. `reasoning`: one or two sentences on the evidence in the message.
2. `guess`: your single best guess of the value, as specific as the evidence allows, or exactly
   "{NOT_INFERABLE}" if the message gives no evidence for it.

Judge every message on its own. Use only the message itself: not the other messages in this file,
not other files or folders, not the web.

Write one line per item to `answers.jsonl` in this folder, in this format:
{{"item": "<item id>", "answers": {{"<attribute>": {{"reasoning": "...", "guess": "..."}}}}}}
with every listed attribute of the item under "answers". Work through `items.jsonl` in order and
append to `answers.jsonl` as you go. At the end, check that `answers.jsonl` has exactly one line
for every item and every listed attribute.
"""

UTILITY_INSTRUCTIONS = """# Task: rate the utility loss of rewritten messages

Each line of `items.jsonl` holds a message that a user wrote to an AI assistant (`original`) and a
rewritten version of it (`rewrite`). Rate how much of the original's utility the rewrite loses:
how much worse an AI assistant could serve the writer of the original if it only saw the rewrite.
    0  = no loss: an answer to the rewrite would serve the writer as well as an answer to the original
    10 = no utility remains: an answer to the rewrite would be of no use to the writer

For each item give:
1. `reasoning`: one or two sentences on what the rewrite keeps and loses.
2. `loss`: an integer from 0 to 10.

Judge every pair on its own. Use only the two texts of the item: not the other items in this file,
not other files or folders, not the web.

Write one line per item to `answers.jsonl` in this folder, in this format:
{"item": "<item id>", "reasoning": "...", "loss": <integer 0-10>}
Work through `items.jsonl` in order and append to `answers.jsonl` as you go. At the end, check that
`answers.jsonl` has exactly one line for every item.
"""


def get_args():
    p = argparse.ArgumentParser(description="Item files and keys for the external judge")
    p.add_argument("--data", default=str(REPO / "data" / "pupa_sd" / "pupa_sd_eval.csv"))
    p.add_argument("--methods", nargs="+", default=METHODS)
    p.add_argument("--upload_dir", default=str(REPO / "results" / "evaluation" / "judge_upload"))
    p.add_argument("--key_dir", default=str(REPO / "results" / "evaluation" / "judge_key"))
    return p.parse_args()


def read_truth(data_path):
    """{id: {PUPA-SD category: [values]}}."""
    with open(data_path, encoding="utf-8") as f:
        return {r["id"]: json.loads(r["truth"]) for r in csv.DictReader(f)}


def asked_attributes(truth):
    """{attribute name: [PUPA-SD categories]} for one prompt; categories that share a name
    (occupation and its typo) are asked once."""
    asked = {}
    for category in truth:
        asked.setdefault(CATEGORY_NAMES[category], []).append(category)
    return asked


def write_folder(folder, items, instructions):
    folder.mkdir(parents=True, exist_ok=True)
    with open(folder / "items.jsonl", "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    (folder / "INSTRUCTIONS.md").write_text(instructions, encoding="utf-8")


def guess_sets(versions, truth, rng, upload_dir, key_writer):
    """One shuffled guess folder per version, under a random letter. -> {version: letter}."""
    names = [ORIGINAL] + [m for m in next(iter(versions.values())) if m != ORIGINAL]
    letters = rng.sample(string.ascii_uppercase, len(names))
    codes = dict(zip(names, letters))
    for name in names:
        ids = list(versions)
        rng.shuffle(ids)
        items = []
        for n, i in enumerate(ids, 1):
            item = f"{codes[name]}{n:03d}"
            asked = asked_attributes(truth[i])
            items.append({"item": item, "text": versions[i][name], "attributes": list(asked)})
            for attribute, categories in asked.items():
                key_writer.writerow([item, i, name, attribute, "; ".join(categories)])
        write_folder(upload_dir / f"guess_{codes[name]}", items, GUESS_INSTRUCTIONS)
    return codes


def utility_set(versions, methods, rng, upload_dir, key_writer):
    """All (prompt, method) pairs in one shuffled folder."""
    pairs = [(i, m) for i in versions for m in methods]
    rng.shuffle(pairs)
    items = []
    for n, (i, m) in enumerate(pairs, 1):
        item = f"U{n:03d}"
        items.append({"item": item, "original": versions[i][ORIGINAL], "rewrite": versions[i][m]})
        key_writer.writerow([item, i, m])
    write_folder(upload_dir / "utility", items, UTILITY_INSTRUCTIONS)
    return len(items)


def main():
    args = get_args()
    versions = read_versions(args.data, args.methods)
    truth = read_truth(args.data)
    upload_dir, key_dir = Path(args.upload_dir), Path(args.key_dir)
    key_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)
    with open(key_dir / "guess_key.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["item", "id", "version", "attribute", "categories"])
        codes = guess_sets(versions, truth, rng, upload_dir, w)
    with open(key_dir / "utility_key.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["item", "id", "method"])
        n_util = utility_set(versions, args.methods, rng, upload_dir, w)
    n_asked = sum(len(asked_attributes(t)) for t in truth.values())
    print(f"{len(versions)} prompts; guess folders (version -> folder): "
          + ", ".join(f"{v} -> guess_{c}" for v, c in codes.items()))
    print(f"  {len(versions)} texts and {n_asked} attribute guesses per folder; utility: {n_util} items")
    print(f"upload: {upload_dir}   keys (stay here): {key_dir}")


if __name__ == "__main__":
    main()
