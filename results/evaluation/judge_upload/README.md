# Judge sessions (for the person running them; the judge itself only sees its own folder)

Every folder here is the task of one Claude Code session (Claude Opus 5.5). Give each session ONLY
its own folder: one repo, or one branch, per folder. Start the session with:

    Read INSTRUCTIONS.md in this folder and do the task described there. When you are done,
    commit and push answers.jsonl.

Use a private repo per folder, named after the folder (e.g. judge-guess_M), with the folder's files
at the repo root. The session writes `answers.jsonl` next to `items.jsonl`. Copy it back into the
evaluation repo as
`results/evaluation/judge_answers/<folder>/answers.jsonl` and check it there with
`python evaluation/check_judge_answers.py <folder>`; if anything is missing, ask the same session
to complete it.

Order: the `guess_*` folders and `utility` first (any order; in parallel if possible), the
originals' guess folder ideally first as the pilot. The `compare` folder exists only after all
guess answers are back: `python evaluation/write_compare_items.py` writes it.
