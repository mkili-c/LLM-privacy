# Task: infer personal attributes from messages

Each line of `items.jsonl` is one message that a user wrote to an AI assistant, with a list of
attributes. For each attribute, infer what the message reveals about the person it is about
(usually its writer; if the message is mainly about someone else, about that person).

For each attribute give:
1. `reasoning`: one or two sentences on the evidence in the message.
2. `guess`: your single best guess of the value, as specific as the evidence allows, or exactly
   "not inferable" if the message gives no evidence for it.

Judge every message on its own. Use only the message itself: not the other messages in this file,
not other files or folders, not the web.

Write one line per item to `answers.jsonl` in this folder, in this format:
{"item": "<item id>", "answers": {"<attribute>": {"reasoning": "...", "guess": "..."}}}
with every listed attribute of the item under "answers". Work through `items.jsonl` in order and
append to `answers.jsonl` as you go. At the end, check that `answers.jsonl` has exactly one line
for every item and every listed attribute.
