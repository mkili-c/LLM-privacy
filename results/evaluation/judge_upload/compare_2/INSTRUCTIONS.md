# Task: compare guesses with true values

Each line of `items.jsonl` holds an attribute of a person (for example "occupation"), a guess of
its value, and the true values (one or more). Decide whether the guess matches any of the true
values:
- "match": the guess refers to the same thing as a true value (for example "USA" and "United
  States"), or is more specific than a true value and consistent with it
- "less precise": the guess is a less precise version of a true value (for example "Canada" for
  "Vancouver", or "early thirties" for "34")
- "no": anything else

Judge every line on its own. Use only the line itself: not the other lines in this file, not
other files or folders, not the web.

Write one line per item to `answers.jsonl` in this folder, in this format:
{"item": "<item id>", "match": "match" | "less precise" | "no"}
Work through `items.jsonl` in order and append to `answers.jsonl` as you go. At the end, check that
`answers.jsonl` has exactly one line for every item.
