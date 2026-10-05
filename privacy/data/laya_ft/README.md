# laya-ft-data

Synthetic fine-tuning and test data for Laya, a small encoder classifier that detects privacy leakage in prompts ("Can the user's {attribute} be inferred from this message?").

Every pair is a message that reveals some personal attributes of the user and a minimally edited version that reveals none of them. All messages were written by hand (no templates); `build.py` only merges, validates and counts. The batch files in `batches/` are the source of truth; `build.py` writes `test.csv`, `train.csv`, `test_values.json` and the statistics below. `flagged.csv` lists pairs whose label was doubtful.

## Generation prompt

````
# Task: generate a fine-tuning dataset for a privacy leak detector

You are working in an empty git repository created for this dataset. Create a synthetic dataset of user messages to an AI assistant, in pairs: a message that reveals some personal attributes of the user, and a minimally edited version that reveals none of them. It will be used to fine-tune Laya, a small encoder classifier that answers "Can the user's {attribute} be inferred from this message?", and to measure Laya before and after fine-tuning on a held-out test split.

Write every message yourself. Do not generate text with templates, string substitution or code; scripts are only for validation, statistics and assembling the CSVs.

## Output

Everything goes in the repository root:
- `test.csv` (400 pairs) and `train.csv` (2,000 pairs), UTF-8, written with Python's csv module.
- `batches/`: one JSONL file per batch of 50 pairs (`test_00.jsonl`, …, `train_00.jsonl`, …), same fields as the CSVs.
- `build.py`: merges the batches, validates, prints statistics, writes the CSVs. Keep it short and top-down: docstring, config constants, functions in the order they run, `main()` last.
- `test_values.json`: the test split's values (see Quotas).
- `flagged.csv`: pairs whose label you were unsure about, with a one-line reason.
- `README.md`: replace the existing one with this task text verbatim under "Generation prompt", followed by the statistics from build.py.

Columns (multi-valued fields separated by "; "):

| column | content |
|---|---|
| `id` | `test_0001`, `train_0001`, … |
| `attributes` | every attribute that can be inferred from `leaking`: 1–4 of the 16 names below |
| `values` | the user's value for each, e.g. `profession: paramedic; age: 41` |
| `leaking` | the message; it reveals exactly `attributes` |
| `safe` | a minimal edit of `leaking` that reveals none of them |
| `not_inferable` | 1–2 attributes from the list that can be inferred from neither message |
| `edit_type` | `deletion`, `generalization` or `other_person` |
| `cue_type` | `explicit`, `indirect` or `mixed` |

The 16 attribute names, spelled exactly: profession, geographic location, health condition, relationship status, socioeconomic status, age, political views, nationality, family situation, education level, gender, name, residential address, date of birth, credit card number, email address.

## What counts as a leak

An attribute leaks if a careful reader could make an informed, specific guess about the user's own value. Use these levels. If a case falls between them, rewrite it until it is clearly on one side.

| attribute | leaks | does not leak |
|---|---|---|
| profession | the occupation, or a narrow group that pins it down: "paramedic", "I teach year 5", "physician", "I fix lifts for a living" | "my job", "at work", "my shift", "my boss" |
| geographic location | city, region or country where the user lives or is: "here in Lyon", "our Bavarian winters", a local institution | "my city", "here", "abroad", a continent |
| age | an age, a decade, or a life stage that places it within about ten years: "41", "in my sixties", "just retired", "starting high school" | "as an adult", "not young anymore" |
| health condition | a specific condition of the user: "my asthma", "my chemo", "my ADHD meds" | "feeling run down", "a doctor's appointment" |
| relationship status | single, dating, married, divorced, widowed: "my wife", "since the divorce", "my boyfriend" | "a friend", "my flatmates" |
| socioeconomic status | income or class: "living paycheck to paycheck", "our second home", "on benefits" | "on a budget", "want to save" |
| political views | party or ideology, through an affiliation: "as a lifelong Labour voter", "canvassing for the Greens" | "I vote every year", "a town hall meeting" |
| nationality | citizenship or national origin: "as a Brazilian", "back home in Nigeria" | "abroad", "my home country" |
| family situation | children, dependants, caring duties: "my two kids", "caring for my mum" | "a family dinner" |
| education level | highest degree or current stage: "my PhD", "left school at 16", "my master's thesis" | "learning", "a course" |
| gender | gendered self-reference: "as a woman", "us dads" | first-person text without gendered words |
| name | the user's first name, surname or full name: "I'm Jonas", "sign it Amira Haddad" | someone else's name, "my name" |
| residential address | where the user lives, at street level: street and house number, or a full postal address | the city alone (that is geographic location), "my street", a shop's or office's address |
| date of birth | day and month of the user's birth, with or without the year: "born 14 March 1990", "my birthday is the 1st of August" | the year alone (that is age), "my birthday is next week" |
| credit card number | the user's full card number, in any format | the last four digits, "my Visa", an order or tracking number |
| email address | the user's full email address | "my Gmail", a company's support address |

Cues imply each other. Watch for these:
- partner words reveal relationship status;
- "mom" and "dad" reveal family situation and gender;
- a job title can reveal education level or socioeconomic status (surgeon, cleaner);
- a hometown abroad reveals nationality;
- "my pension" reveals age;
- a name usually reveals gender and can reveal nationality;
- an email address often contains the name, and its domain can reveal profession, education level or location (an employer's or university's domain);
- a residential address reveals geographic location;
- a date of birth with the year reveals age.

If a cue implies another attribute, either add that attribute to `attributes` and `values` and remove it in `safe`, or choose a different cue. Include the user's name, address, date of birth, card number or email only when that attribute is in `attributes`.

## The leaking message

- A realistic first-person message to an AI assistant with a real task: a question, a request for advice, a writing or planning task. 15–90 words, in English.
- Personal details appear naturally as context, never as a bio list ("I'm 41, female, a paramedic").
- Cues are explicit ("I'm 41", "as a paramedic") or indirect ("I spend my nights in the back of an ambulance", "since my pension started"). Aim for about half of each across the dataset.
- Sensitive attributes (health condition, political views) come from self-disclosure or concrete facts, never from stereotypes. Keep them respectful and realistic, with no real people.
- No controversial or recent topics, in any message: no political issues, current events, elections, wars or conflicts, socially divisive debates or named public figures, and nothing similar even if it is not listed here. Choose everyday tasks instead. When political views is an attribute, it comes only from an affiliation mentioned in passing (a party, a voting habit, volunteering for a campaign), never from a position on an issue, and the task itself stays non-political.
- Identifiers (name, residential address, date of birth, credit card number, email address) appear where users really paste them: a letter or email to draft, a form to fill in, a declined payment, a delivery problem, a signup. Invent every value. Card numbers are made up, not well-known test numbers, and come in varied formats (spaces, dashes, none). Every name, address, email and card number is unique across the dataset.
- Use look-alikes that are not the user's, in `safe` messages or with `not_inferable`: an order or tracking number, a company's support email, a shop's address, someone else's name.
- Vary everything:
  - places: countries and cities in Europe and America;
  - ages: 16–85;
  - jobs: across all sectors and income levels;
  - topics: health, money, work, law, housing, travel, family, study, tech, hobbies, and more;
  - register: formal, casual, terse, all lowercase, the occasional typo;
  - length.

## The safe message

Same message, same task. Only the cues change, so that none of `attributes` leaks.
- `deletion` (~40%): cut the cue words out and change nothing else. No rewording, and no fixing of capitals, punctuation or grammar. The result may read slightly awkwardly; that is what the target system produces. Example: "Since moving to Lyon I can't fall asleep before 2am" → "Since moving I can't fall asleep before 2am".
- `generalization` (~40%): replace each cue with a vaguer phrase below the leak level ("a paramedic" → "someone in my line of work", "Lyon" → "my city"). Vary the wording so no stock phrase ("a certain", "somewhere") becomes a giveaway.
- `other_person` (~20%): the cue now describes someone else, so it says nothing about the user ("as a paramedic" → "my friend, a paramedic, says"). Make sure the new person doesn't leak something else: a colleague who is a paramedic suggests the user works in emergency care.

## Quotas (check with build.py)

- Each attribute appears in `attributes` at least 130 times in train and 25 times in test. It appears in `not_inferable` at least 100 times in train and 20 times in test.
- Attributes per pair: 1 in ~40% of pairs, 2 in ~40%, 3–4 in ~20%.
- For `not_inferable`, prefer attributes the topic comes close to but the message does not reveal.
- The test split gets its own values for profession, geographic location (city or region), health condition, name, residential address, email address and credit card number: none of these values may appear in train. Write the test split first and record its values in `test_values.json`.

## Validation in build.py

- Every field is filled and uses the allowed names; `attributes` and `not_inferable` are disjoint; `leaking` ≠ `safe`; neither message contains a slot mark such as `[…]` or `[...]`.
- `deletion` pairs: split both messages into words and punctuation with `re.findall(r"\w+|[^\w\s]", text)` and check that the tokens of `safe` are a subsequence of the tokens of `leaking`.
- Near-duplicates: no two `leaking` messages, within or across splits, with word-trigram Jaccard similarity ≥ 0.5.
- No value from `test_values.json` appears in any train message (case-insensitive).
- Print, per split: counts per attribute in `attributes` and in `not_inferable`; counts per `edit_type`, per `cue_type` and per number of attributes; word-count statistics of `leaking` and `safe`. Mark every quota that is not met.

## Process

1. Test split first, then train, in batches of 50 pairs. Before each batch, choose its attribute mix from the remaining quotas.
2. After each batch, re-read every pair and check:
   - `leaking` reveals exactly `attributes`, and nothing else from the 16;
   - `safe` reveals none of them, and its task is still clear;
   - each `not_inferable` attribute really cannot be guessed from either message;
   - neither message touches a controversial or recent topic.

   Fix or drop pairs that fail, and add doubtful ones to `flagged.csv`. Then run build.py, commit the batch and push it, so no batch is lost if the session stops.
3. If you cannot finish in this session, stop after a pushed batch and report how many pairs each split has. A later session started from the same branch continues from the batches already there.

## When done

Report: the branch you pushed to, pairs per split, the quota table from build.py, how many pairs you dropped or flagged and why, and any rule in this prompt you found ambiguous.
````

## Statistics

```
== test: 400 pairs (target 400)
attribute                 attributes  not_inferable
profession                        54            107
geographic location               63             32
health condition                  49             30
relationship status               44             24
socioeconomic status              40             25
age                               50            139
political views                   35             35
nationality                       37             48
family situation                  55             37
education level                   38             28
gender                            50             65
name                              67             83
residential address               46             27
date of birth                     40             42
credit card number                36             30
email address                     41             48
(minimum per attribute: 25 in attributes, 20 in not_inferable)
edit_type: deletion 159 (40%), generalization 160 (40%), other_person 81 (20%)
cue_type: explicit 225 (56%), indirect 72 (18%), mixed 103 (26%); explicit cues about 69%  <-- NOT MET
attributes per pair: 1 162 (40%), 2 151 (38%), 3-4 87 (22%) (3: 67, 4: 20)
words in leaking: min 16, mean 31.9, median 32.0, max 55
words in safe: min 13, mean 28.6, median 29.0, max 56

== train: 2000 pairs (target 2000)
attribute                 attributes  not_inferable
profession                       305            703
geographic location              237            115
health condition                 249            162
relationship status              230            165
socioeconomic status             251            108
age                              281            357
political views                  180            800
nationality                      228            268
family situation                 287            141
education level                  250            182
gender                           255            128
name                             223            239
residential address              184            107
date of birth                    179            246
credit card number               177            155
email address                    182            124
(minimum per attribute: 130 in attributes, 100 in not_inferable)
edit_type: deletion 800 (40%), generalization 800 (40%), other_person 400 (20%)
cue_type: explicit 1172 (59%), indirect 495 (25%), mixed 333 (17%); explicit cues about 67%  <-- NOT MET
attributes per pair: 1 800 (40%), 2 800 (40%), 3-4 400 (20%) (3: 302, 4: 98)
words in leaking: min 15, mean 25.5, median 25.0, max 59
words in safe: min 4, mean 21.3, median 21.0, max 52
```
