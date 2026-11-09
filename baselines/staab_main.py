"""
Staab et al.'s main.py with one plumbing fix around their model call; their code stays unchanged.

Their attacker prompt asks for plain "Type: / Inference: / Guess: / Certainty:" lines. Qwen2.5-7B
sometimes answers in Markdown instead ("### Type: age", "**Guess:**" with the value on the next
line). Their parse_answer (src/reddit/reddit.py) does not recognize "### Type" as the label "type"
(Jaro-Winkler 0), so everything after it is dropped and all attributes come back as
{'inference': 'MISSING', 'guess': []}: the anonymizer then works without any inference for that
round. This wrapper puts the attacker's answer into the plain layout before their code sees it;
the anonymizer's and the utility scorer's answers pass through untouched.

Run from their repo, in place of main.py (run_baselines.py, method staab_md):
    python <this file> --config_path <config.yaml>
"""
import os
import re
import runpy
import sys

ATTACKER_SYSTEM_PROMPT = "You are an expert investigator"     # the start of their attacker's system prompt
LABELS = "type|inference|guess|certainty"
# a label with Markdown around it, at the start of a line: "### Type:", "**Guess:**", "- **Guess**:"
MARKED_LABEL = re.compile(rf"^[ \t]*[#>*_-]*[ \t]*[*_]*({LABELS})[*_]*[ \t]*:[ \t]*[*_]*[ \t]*", re.I | re.M)
# a label alone on its line, whose value is on the next line (unless that line is another label)
LONE_LABEL = re.compile(rf"^({LABELS}): *\n\s*(?!(?:{LABELS}):)", re.I | re.M)


def plain_layout(answer):
    """The attacker's answer with its labels as their parser expects them."""
    answer = MARKED_LABEL.sub(lambda m: m.group(1).capitalize() + ": ", answer)
    return LONE_LABEL.sub(lambda m: m.group(1) + ": ", answer)


def main():
    sys.path.insert(0, os.getcwd())                           # their repo, as for `python main.py`
    from src.models.open_ai import OpenAIGPT
    predict_call = OpenAIGPT._predict_call

    def predict_call_plain(self, messages):
        answer = predict_call(self, messages)
        is_attacker = messages[0]["role"] == "system" and messages[0]["content"].startswith(ATTACKER_SYSTEM_PROMPT)
        return plain_layout(answer) if is_attacker else answer

    OpenAIGPT._predict_call = predict_call_plain
    runpy.run_path("main.py", run_name="__main__")            # their main.py reads --config_path from sys.argv


if __name__ == "__main__":
    main()
