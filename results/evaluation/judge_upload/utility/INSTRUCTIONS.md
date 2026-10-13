# Task: rate the utility loss of rewritten messages

Each line of `items.jsonl` holds a message that a user wrote to an AI assistant (`original`) and a
rewritten version of it (`rewrite`). Rate how much of the original's utility the rewrite loses:
how much worse an AI assistant could serve the writer of the original if it only saw the rewrite.
    0  = no loss: an answer to the rewrite would serve the writer as well as an answer to the original
    10 = no utility remains: an answer to the rewrite would be of no use to the writer
Rate the rewrite as the message the assistant would receive, whatever form it has.

For each item give:
1. `reasoning`: one or two sentences on what the rewrite keeps and loses.
2. `loss`: an integer from 0 to 10.

Judge every pair on its own. Use only the two texts of the item: not the other items in this file,
not other files or folders, not the web.

Write one line per item to `answers.jsonl` in this folder, in this format:
{"item": "<item id>", "reasoning": "...", "loss": <integer 0-10>}
Work through `items.jsonl` in order and append to `answers.jsonl` as you go. At the end, check that
`answers.jsonl` has exactly one line for every item.
