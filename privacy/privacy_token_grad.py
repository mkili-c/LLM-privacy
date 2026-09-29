"""
Attribution of the privacy loss onto the message tokens.

    L = sum_a loss(P_a(yes))      one AttributeProbe per attribute a

Loss (--loss)
-------------
nll    -log(1 - P(yes)) = softplus(logit)  (default)
p      P(yes)
logit  the yes-vs-no logit

All three are minimized by P(yes) -> 0, but P(yes) saturates:
dP/dlogit = p(1 - p) vanishes at p ~ 1, so a fully leaked attribute (location
at P = 1.0000 for "... from Zurich ...") gets no gradient and the sum is
driven by whichever attribute sits near 0.5. -log(1 - p) has d/dlogit = p:
~1 while the attribute leaks, fading to 0 once it is private. An attribute the
model already answers No to (gender at p ~ 1e-3 even when leaked, see
attribute_probe.py) gets no gradient under nll or p; only --loss logit
keeps pushing on it.

Two attributions
----------------
occlusion   L(message) - L(message without token i), one exact forward pass
            per token and attribute. Positive = removing the token lowers the
            loss, i.e. the token carries the leak. This is the ranking to
            trust. Token-level: a word split over several tokens ("15" = "1" +
            "5") keeps leaking through the rest when one piece is removed.
gradients   dL/d one_hot [n_msg, V] through the one-hot trick of
            kl_token_grad.py: the GCG candidate signal, whose most negative
            entries in row i are the substitutions at position i that lower L
            the most to first order. Also reported per token: grad_norm
            (||dL/d embedding||) and grad*input (the one-hot gradient at the
            current id). These are local and follow sensitivity rather than
            content: on the example above, "could" / "I" / "be" next to the
            question outrank "Zurich" / "girl". Use them to propose swaps, and
            occlusion (or a real forward pass) to judge them.

Every probe wraps the same message ids in its own query (only the question in
the suffix differs), so one shared one-hot matrix is stitched between each
probe's fixed prefix / suffix embeddings. Each attribute gets its own forward
+ backward; the one-hot gradients accumulate, so peak memory is one query.
"""
import argparse

import torch
import torch.nn.functional as F

from attribute_probe import ATTRIBUTES, DEFAULT_MODEL, QUESTION, AttributeProbe, load

LOSSES = {
    "nll": F.softplus,          # -log(1 - sigmoid(z)) = softplus(z), stable for large z
    "p": torch.sigmoid,
    "logit": lambda z: z,
}


@torch.enable_grad()
def privacy_token_gradients(model, probes, msg_ids, loss="nll"):
    """
    probes: {attribute: AttributeProbe}, msg_ids: list of message token ids.
    Returns (one_hot_grad [n, V], stats) where stats holds the loss, the
    per-token grad_norm / grad_x_input and, per attribute, p / logit /
    coverage / loss.
    """
    W = model.get_input_embeddings().weight
    msg = torch.tensor(msg_ids, device=model.device)
    one_hot = torch.zeros(len(msg_ids), W.shape[0], device=model.device, dtype=W.dtype)
    one_hot.scatter_(1, msg.unsqueeze(1), 1.0)
    one_hot.requires_grad_()

    emb_grad = torch.zeros(len(msg_ids), W.shape[1], device=model.device)
    per_attr, total = {}, 0.0
    for name, probe in probes.items():
        ids, sl = probe.build_ids(msg_ids)
        x = one_hot @ W                     # [n, H], differentiable in one_hot
        x.retain_grad()
        embeds = model.get_input_embeddings()(ids).detach()
        full = torch.cat([embeds[:sl.start], x, embeds[sl.stop:]]).unsqueeze(0)

        p, logit, cov = probe.p_pos(inputs_embeds=full)
        target = LOSSES[loss](logit)
        target.backward()                   # accumulates into one_hot.grad

        emb_grad += x.grad.float()
        total += target.item()
        per_attr[name] = {"p": p.item(), "logit": logit.item(),
                          "coverage": cov.item(), "loss": target.item()}

    grad = one_hot.grad.detach()
    return grad, {
        "loss": total,
        "grad_norm": emb_grad.norm(dim=-1).cpu(),
        "grad_x_input": grad.gather(1, msg.unsqueeze(1)).squeeze(1).float().cpu(),
        "per_attribute": per_attr,
    }


@torch.no_grad()
def occlusion(probes, msg_ids, loss="nll"):
    """{attribute: [n] loss(message) - loss(message without token i)}."""
    msg_ids = list(msg_ids)
    variants = [msg_ids] + [msg_ids[:i] + msg_ids[i + 1:] for i in range(len(msg_ids))]
    out = {}
    for name, probe in probes.items():
        full = LOSSES[loss](probe.logits_batch(variants[:1]))
        drop = LOSSES[loss](probe.logits_batch(variants[1:]))
        out[name] = (full - drop).cpu()
    return out


def report(tok, msg_ids, occ, stats, top_k):
    tokens = tok.convert_ids_to_tokens(msg_ids)
    total = sum(occ.values())
    gn, gxi = stats["grad_norm"], stats["grad_x_input"]
    order = sorted(range(len(tokens)), key=lambda i: total[i], reverse=True)

    short = {a: a.split()[-1][:8] for a in occ}   # "geographic location" -> "location"
    head = "".join(f" {short[a]:>9}" for a in occ)
    print(f"  {'rank':>4}  {'idx':>4}  {'token':<16} {'occl':>8} |{head} | "
          f"{'grad_norm':>10} {'grad*in':>9}")
    print("  " + "-" * (64 + 10 * len(occ)))
    for rank, i in enumerate(order[:top_k]):
        tk = tokens[i].replace("Ġ", "·").replace("Ċ", "\\n")
        cols = "".join(f" {occ[a][i]:>+9.3f}" for a in occ)
        print(f"  {rank:>4}  {i:>4}  {tk:<16} {total[i]:>+8.3f} |{cols} | "
              f"{gn[i]:>10.4f} {gxi[i]:>+9.4f}")
    print("  occl > 0: removing the token lowers the loss, i.e. the token leaks.")


def main():
    p = argparse.ArgumentParser(description="Attribute sum_a loss(P_a(yes)) to the message tokens")
    p.add_argument("--model_path", default=DEFAULT_MODEL)
    p.add_argument("--text", required=True, help="message to attribute")
    p.add_argument("--attributes", nargs="+", default=ATTRIBUTES,
                   help="free-text attributes, one probe each")
    p.add_argument("--loss", choices=sorted(LOSSES), default="nll",
                   help="nll = -log(1 - P(yes)); p = P(yes); logit = yes-vs-no logit")
    p.add_argument("--top_k", type=int, default=15)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {args.model_path} on {device} ...")
    model, tok = load(args.model_path, device)
    model.requires_grad_(False)     # gradients are only ever needed w.r.t. the input

    probes = {a: AttributeProbe(model, tok, QUESTION.format(attribute=a)) for a in args.attributes}
    msg_ids = tok(args.text, add_special_tokens=False).input_ids
    _, stats = privacy_token_gradients(model, probes, msg_ids, loss=args.loss)
    occ = occlusion(probes, msg_ids, loss=args.loss)

    print(f"\nmessage: {args.text!r}  ({len(msg_ids)} tokens)")
    for a, r in stats["per_attribute"].items():
        print(f"  {a:<22} P(yes) = {r['p']:.4f}  logit = {r['logit']:+7.3f}  "
              f"loss = {r['loss']:7.3f}  coverage = {r['coverage']:.4f}")
    print(f"  total loss ({args.loss}) = {stats['loss']:.4f}\n")
    report(tok, msg_ids, occ, stats, args.top_k)


if __name__ == "__main__":
    main()
