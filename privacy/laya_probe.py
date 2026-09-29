"""
Laya as the privacy adversary: P(an attribute of the user can be inferred from
a message), differentiable in the message tokens.

Laya (github.com/NandhaKishorM/laya) is a ModernBERT encoder with a typed
decision head. A `noul` question is one sequence

    [CLS] noul question: <instructions> [SEP] [MASK] false: .. [MASK] true: .. [SEP] message [SEP]

and the scorer reads a logit off each [MASK]; P(true) = softmax(z / T)[1] with
the checkpoint's fitted noul temperature T. No text is generated, so all
attributes are answered in one batched forward pass of a 421M encoder instead
of one pass of a 3B LLM each, and the answer is calibrated rather than read
off the Yes / No tokens of a chat model.

One row per attribute. The message is the last segment of every row, so its
token span is known exactly (as in attribute_probe.py), and the rows only
differ in the question. `logit_one_hot` embeds the message as one_hot @ W,
shared by all rows, and runs Laya's forward on inputs_embeds in fp32 (the
stock Agent autocasts to bf16, which moves P by up to ~0.07); the act head is
skipped, it does not affect the answer.

    logit_a = (z_true - z_false) / T,   P_a = sigmoid(logit_a)

Attribution: `gradients` backprops sum_a loss(logit_a) (losses as in
privacy_token_grad.py) into the one-hot, `occlusion` scores every one-word or
one-token deletion in batched forward passes. Words are grouped from the
tokenizer's leading-space markers, so "Zurich," is one unit and "15" is not
split into "1" + "5" as with token-level occlusion.

Laya's vocabulary (ModernBERT BPE, 50k) is not Qwen's. One-hot gradients
propose swaps in Laya token space; the text they decode to must be re-scored,
by Laya and by the utility side, before it is accepted.
"""
import argparse

import torch
import torch.nn.functional as F

import laya
from laya.common import build_sequence

from attribute_probe import ATTRIBUTES
from privacy_token_grad import LOSSES

DEFAULT_LAYA = "/work/projects/mkilic/models/laya"
QUESTION = "Can the user's {attribute} be inferred from this message?"


class LayaProbe:
    """P(attribute inferable) for several attributes at once, one Laya row each."""

    def __init__(self, agent, attributes, template=QUESTION):
        self.agent, self.tok, self.model = agent, agent.tok, agent.model
        self.attributes = list(attributes)
        self.device = agent.device
        self.max_len = agent.cfg.get("max_len", 512)
        head_max_len = agent.cfg.get("head_max_len", 192)
        self.T = agent.temperature_by_options.get("noul:2", agent.temperature[2])

        # Each row is prefix + message + [SEP]; build it around an empty
        # message to get the fixed prefix and the two [MASK] positions.
        self.prefixes, self.markers = [], []
        for a in self.attributes:
            q = {"t": "noul", "ins": template.format(attribute=a), "crit": None}
            ids, markers = build_sequence(self.tok, "", q, self.max_len, head_max_len, state_ids=[])
            assert ids[-1] == self.tok.sep_token_id and len(markers) == 2
            self.prefixes.append(ids[:-1])
            self.markers.append(markers)
        self._markers = torch.tensor(self.markers, device=self.device)
        self._qtype = torch.full((len(self.attributes),), laya.common.QTYPES["noul"], device=self.device)
        self._embed = self.model.encoder.get_input_embeddings()

    def encode(self, message):
        return self.tok(message.replace(self.tok.mask_token, " "), add_special_tokens=False).input_ids

    def _rows(self, msg_ids):
        """Token ids of each attribute's row for one message."""
        msg_ids = list(msg_ids)
        longest = max(len(p) for p in self.prefixes) + len(msg_ids) + 1
        if longest > self.max_len:
            raise ValueError(f"message too long for Laya: {longest} > max_len={self.max_len} tokens")
        return [p + msg_ids + [self.tok.sep_token_id] for p in self.prefixes]

    def _head(self, embeds, att, markers, qtype):
        """DecisionModel.forward on inputs_embeds, answer logits only -> [B]."""
        m = self.model
        h = m.encoder(inputs_embeds=embeds, attention_mask=att).last_hidden_state
        h = h + m.type_emb(qtype)[:, None, :]
        pad = ~att.bool()
        for layer in m.head.layers:
            h = layer(h, src_key_padding_mask=pad)
        z = m.scorer(torch.gather(h, 1, markers[:, :, None].expand(-1, -1, h.size(-1))))
        z = z.squeeze(-1).float()
        return (z[:, 1] - z[:, 0]) / self.T

    def _pad(self, rows):
        L = max(len(r) for r in rows)
        ids = torch.full((len(rows), L), self.tok.pad_token_id, device=self.device)
        att = torch.zeros((len(rows), L), dtype=torch.long, device=self.device)
        for i, r in enumerate(rows):
            ids[i, :len(r)] = torch.tensor(r, device=self.device)
            att[i, :len(r)] = 1
        return ids, att

    @torch.no_grad()
    def logits(self, msgs_ids, batch_size=64):
        """[N, K] logits for N messages (token-id lists, any lengths)."""
        rows = [r for m in msgs_ids for r in self._rows(m)]
        K, out = len(self.attributes), []
        for i in range(0, len(rows), batch_size * K):
            chunk = rows[i:i + batch_size * K]
            ids, att = self._pad(chunk)
            n = len(chunk) // K
            out.append(self._head(self._embed(ids), att, self._markers.repeat(n, 1),
                                  self._qtype.repeat(n)))
        return torch.cat(out).view(len(msgs_ids), K)

    def score(self, message):
        """{attribute: {"p", "logit"}} for one message."""
        z = self.logits([self.encode(message)])[0]
        return {a: {"p": torch.sigmoid(v).item(), "logit": v.item()} for a, v in zip(self.attributes, z)}

    def logit_one_hot(self, msg_ids):
        """Differentiable logits [K], the one-hot [n, V] of the message they depend
        on, and its embedding x = one_hot @ W [n, H] (grad retained)."""
        W = self._embed.weight
        msg = torch.tensor(list(msg_ids), device=self.device)
        one_hot = torch.zeros(len(msg), W.shape[0], device=self.device, dtype=W.dtype)
        one_hot.scatter_(1, msg.unsqueeze(1), 1.0)
        one_hot.requires_grad_()
        x = one_hot @ W                                         # [n, H], shared by all rows
        x.retain_grad()

        rows = self._rows(msg_ids)
        ids, att = self._pad(rows)
        fixed = self._embed(ids).detach()
        embeds = torch.stack([
            torch.cat([fixed[k, :len(p)], x, fixed[k, len(p) + len(msg):]])
            for k, p in enumerate(self.prefixes)])
        return self._head(embeds, att, self._markers, self._qtype), one_hot, x

    @torch.enable_grad()
    def gradients(self, msg_ids, loss="nll"):
        """Backprop sum_a loss(logit_a). Returns (one_hot grad [n, V], stats)."""
        z, one_hot, x = self.logit_one_hot(msg_ids)
        per = LOSSES[loss](z)
        per.sum().backward()
        g = one_hot.grad.detach()
        msg = torch.tensor(list(msg_ids), device=self.device)
        return g, {
            "loss": per.sum().item(),
            "per_attribute": {a: {"p": torch.sigmoid(v).item(), "logit": v.item(), "loss": l.item()}
                              for a, v, l in zip(self.attributes, z, per)},
            "grad_norm": x.grad.float().norm(dim=-1).cpu(),
            "grad_x_input": g.gather(1, msg.unsqueeze(1)).squeeze(1).float().cpu(),
        }

    def words(self, msg_ids):
        """Word units [(start, stop)] over the message tokens: a word starts at a
        token with a leading space (BPE "Ġ") or at the first token."""
        toks = self.tok.convert_ids_to_tokens(list(msg_ids))
        starts = [i for i, t in enumerate(toks) if i == 0 or t.startswith("Ġ")]
        return list(zip(starts, starts[1:] + [len(toks)]))

    def occlusion(self, msg_ids, unit="word", loss="nll", batch_size=64):
        """Deletion effects. Returns (units [(start, stop)], delta [U, K]) with
        delta = loss(message) - loss(message without the unit); > 0 = the unit leaks."""
        msg_ids = list(msg_ids)
        units = self.words(msg_ids) if unit == "word" else [(i, i + 1) for i in range(len(msg_ids))]
        variants = [msg_ids] + [msg_ids[:s] + msg_ids[e:] for s, e in units]
        L = LOSSES[loss](self.logits(variants, batch_size))
        return units, (L[0:1] - L[1:]).cpu()


def main():
    p = argparse.ArgumentParser(description="Laya leakage probe + attribution for one message")
    p.add_argument("--laya_path", default=DEFAULT_LAYA)
    p.add_argument("--text", required=True)
    p.add_argument("--attributes", nargs="+", default=ATTRIBUTES)
    p.add_argument("--question", default=QUESTION, help="template with {attribute}")
    p.add_argument("--loss", choices=sorted(LOSSES), default="nll")
    p.add_argument("--unit", choices=["word", "token"], default="word")
    p.add_argument("--top_k", type=int, default=12)
    p.add_argument("--check", action="store_true",
                   help="compare P with laya's own Agent.predict (fp32, no autocast)")
    args = p.parse_args()

    agent = laya.load(args.laya_path)
    agent.model.requires_grad_(False)   # gradients are only ever needed w.r.t. the input
    probe = LayaProbe(agent, args.attributes, args.question)
    msg_ids = probe.encode(args.text)

    _, stats = probe.gradients(msg_ids, loss=args.loss)
    units, occ = probe.occlusion(msg_ids, unit=args.unit, loss=args.loss)

    print(f"\nmessage: {args.text!r}  ({len(msg_ids)} Laya tokens, T = {probe.T:.3f})")
    for a, r in stats["per_attribute"].items():
        print(f"  {a:<22} P = {r['p']:.4f}  logit = {r['logit']:+7.3f}  loss = {r['loss']:7.3f}")
    print(f"  total loss ({args.loss}) = {stats['loss']:.4f}")

    if args.check:
        agent.amp_enabled = False
        qs = {a: {"type": "noul", "instructions": args.question.format(attribute=a)}
              for a in args.attributes}
        ref = agent.predict(args.text, qs)["answers"]
        diff = max(abs(ref[a]["noul"] - stats["per_attribute"][a]["p"]) for a in args.attributes)
        print(f"  check vs Agent.predict: max |dP| = {diff:.4f} (Agent rounds to 4 decimals)")

    total = occ.sum(1)
    gxi = stats["grad_x_input"]
    order = sorted(range(len(units)), key=lambda u: total[u], reverse=True)
    short = [a.split()[-1][:8] for a in args.attributes]
    head = "".join(f" {s:>9}" for s in short)
    print(f"\n  {'rank':>4}  {args.unit:<18} {'occl':>8} |{head} | {'grad*in':>9}")
    print("  " + "-" * (48 + 10 * len(short)))
    for rank, u in enumerate(order[:args.top_k]):
        s, e = units[u]
        text = probe.tok.decode(msg_ids[s:e], clean_up_tokenization_spaces=False).strip()
        cols = "".join(f" {occ[u, k]:>+9.3f}" for k in range(len(short)))
        print(f"  {rank:>4}  {text[:18]:<18} {total[u]:>+8.3f} |{cols} | {gxi[s:e].sum():>+9.4f}")
    print(f"  occl > 0: deleting the {args.unit} lowers the loss, i.e. it leaks.")


if __name__ == "__main__":
    main()
