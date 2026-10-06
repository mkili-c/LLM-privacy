"""

Ask llm:

    "Could someone reading this message make an informed guess about the
     user's {attribute}?"

The answer is read off the logits of the first answer token:

    logit = logsumexp(logp[yes ids]) - logsumexp(logp[no ids])
    p_pos = sigmoid(logit) = P(yes forms) / (P(yes forms) + P(no forms))


Logit is relative leakage score. For attributes the model is trained not to guess, 
it stays biased towards No.

The query is built by concatenating the token ids of three pieces,
[chat-templated prefix] + [message] + [suffix with the question], so the
message's token span is known exactly and can later be edited in place, one
token id for another, as in the utility rewriting scripts. `p_pos` accepts
inputs_embeds, so the one-hot gradient trick of kl_token_grad.py applies.

"""
import argparse
import sys
from pathlib import Path

import torch

# the helper scripts live in ../utility/metrics
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "utility" / "metrics"))
from topk_sample import load 

DEFAULT_MODEL = "/work/projects/mkilic/models/Qwen2.5-3B-Instruct"

# No bare "Y" / "N": as a first token they also collect non-answers such as
# "N/A" (N + /A) or "Yikes" (Y + ikes).
YES_FORMS = ["Yes", "yes", "YES", " Yes", " yes", "Yeah", "Yep", "True", "true", "correct", "Correct"]
NO_FORMS = ["No", "no", "NO", " No", " no", "Nope", " Nope", "Nah", " Nah", "False", "false", "wrong", "Wrong"]

# Any free-text attribute works; these are the ones we use. The last 5 were added
# for the evaluation pipeline; their names match Laya's training data.
ATTRIBUTES = ["gender", "age", "geographic location", "profession", "health condition",
              "education level", "socioeconomic status", "relationship status",
              "family situation", "nationality"]

QUESTION = ("Could someone reading this message make an informed guess "
            "about the user's {attribute}?")

#newline at the boundaries of message slot so that tokenization difference 
#between tokenizing everything at once or in pieces is harmless

TEMPLATE = ("Below is a message sent by a user to an AI assistant.\n\n"
            "-----\n{message}\n-----\n\n"
            "{question}\nAnswer with exactly one word: Yes or No.")

_SLOT = "\x00MESSAGE\x00"   # placeholder the chat template cannot contain


class AttributeProbe:
    """P(positive answer) to one Yes/No question about a message."""

    def __init__(self, model, tok, question):
        self.model, self.tok, self.question = model, tok, question

        # Split the chat-templated query at the message slot into fixed
        # prefix / suffix ids; the message ids go in between.
        text = TEMPLATE.format(message=_SLOT, question=question)
        if getattr(tok, "chat_template", None):
            text = tok.apply_chat_template([{"role": "user", "content": text}],
                                           tokenize=False, add_generation_prompt=True)
        #instead of hardcoding the message position to qwen chat template 
        #this part fixes the message slot and then splits from there after 
        #applying templates 
    
        prefix, suffix = text.split(_SLOT)
        self.prefix_ids = tok(prefix, add_special_tokens=False).input_ids
        self.suffix_ids = tok(suffix, add_special_tokens=False).input_ids
        
        #do this here to avoid retokenizing everything for every candidate
        self.yes_ids, self.no_ids, self.dropped = self._single_token_forms()
        device = model.device
        self._yes = torch.tensor(self.yes_ids, device=device)
        self._no = torch.tensor(self.no_ids, device=device)

    def _single_token_forms(self):
        """Distinct single-token ids for each class; multi-token forms dropped."""
        yes, no, dropped = [], [], []
        for forms, ids in ((YES_FORMS, yes), (NO_FORMS, no)):
            for w in forms:
                enc = self.tok(w, add_special_tokens=False).input_ids
                if len(enc) != 1:
                    dropped.append(w)
                elif enc[0] not in ids:
                    ids.append(enc[0])
        if set(yes) & set(no):
            raise ValueError(f"yes and no forms share token ids {set(yes) & set(no)}")
        return yes, no, dropped

    def build(self, message):
        """-> (input_ids [L] on the model's device, slice of the message tokens)."""
        return self.build_ids(self.tok(message, add_special_tokens=False).input_ids)

    def build_ids(self, msg_ids):
        """Same as build, for a message given as token ids."""
        msg_ids = list(msg_ids)
        s = len(self.prefix_ids)
        ids = torch.tensor(self.prefix_ids + msg_ids + self.suffix_ids, device=self.model.device)
        return ids, slice(s, s + len(msg_ids))

    def p_pos(self, input_ids=None, inputs_embeds=None):
        """Differentiable readout from one sequence ([L] ids or [1, L, H] embeds).
        Returns (p_pos, logit, coverage) as torch scalars; coverage is the total
        probability on the kept yes/no forms, a check that the model answers
        in the expected format at all."""\
    #unsqueeze turns [L] into [1,L] a batch with a single input
        if input_ids is not None:
            out = self.model(input_ids=input_ids.unsqueeze(0))
        else:
            out = self.model(inputs_embeds=inputs_embeds)
        p, logit, coverage = self._readout(out.logits[:, -1])
        return p[0], logit[0], coverage[0]

    def _readout(self, last_logits):
        """[B, V] next-token logits -> (p_pos, logit, coverage), each [B]."""
        logp = torch.log_softmax(last_logits.float(), dim=-1)
        lse_yes = torch.logsumexp(logp[:, self._yes], dim=-1)
        lse_no = torch.logsumexp(logp[:, self._no], dim=-1)
        logit = lse_yes - lse_no
        coverage = torch.logaddexp(lse_yes, lse_no).exp()
        return torch.sigmoid(logit), logit, coverage

    @torch.no_grad()
    def logits_batch(self, msgs_ids, batch_size=16):
        """Yes-vs-no logits [N] for messages given as token-id lists of any
        lengths. Rows are right-padded, so every real token keeps its position
        and the answer is read at each row's own last token."""
        out = []
        for i in range(0, len(msgs_ids), batch_size):
            rows = [self.build_ids(m)[0] for m in msgs_ids[i:i + batch_size]]
            lens = torch.tensor([len(r) for r in rows], device=self.model.device)
            ids = torch.full((len(rows), int(lens.max())), self.tok.pad_token_id, device=self.model.device)
            att = torch.zeros_like(ids)
            for j, r in enumerate(rows):
                ids[j, :len(r)], att[j, :len(r)] = r, 1
            logits = self.model(input_ids=ids, attention_mask=att).logits
            last = logits[torch.arange(len(rows), device=ids.device), lens - 1]
            out.append(self._readout(last)[1])
        return torch.cat(out)

    @torch.no_grad()
    def score(self, message):
        """Plain-float readout for one message."""
        ids, _ = self.build(message)
        p, logit, cov = self.p_pos(input_ids=ids)
        return {"p_pos": p.item(), "logit": logit.item(), "coverage": cov.item()}


def main():
    p = argparse.ArgumentParser(description="P(a user attribute can be inferred) from one message")
    p.add_argument("--model_path", default=DEFAULT_MODEL)
    p.add_argument("--attribute", default="gender",
                   help=f"any free-text attribute, e.g. {', '.join(map(repr, ATTRIBUTES))}")
    p.add_argument("--quessigmoid of a log-odds value turns it back into a probability. Here that probability is renormalised over just the two classes: it's P(Yes) given that the model answerstion", default=None, help="override the Yes/No question")
    p.add_argument("--text", nargs="+", required=True, help="message(s) to score")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {args.model_path} on {device} ...")
    model, tok = load(args.model_path, device)
    model.requires_grad_(False)     # gradients are only ever needed w.r.t. the input

    probe = AttributeProbe(model, tok, args.question or QUESTION.format(attribute=args.attribute))
    print(f"question : {probe.question}")
    print(f"YES ids  : {tok.convert_ids_to_tokens(probe.yes_ids)}")
    print(f"NO ids   : {tok.convert_ids_to_tokens(probe.no_ids)}")
    if probe.dropped:
        print(f"dropped (multi-token): {probe.dropped}")

    for text in args.text:
        r = probe.score(text)
        print(f"\nP(yes) = {r['p_pos']:.4f}  logit = {r['logit']:+.3f}  "
              f"coverage = {r['coverage']:.4f}\n  {text!r}")


if __name__ == "__main__":
    main()
