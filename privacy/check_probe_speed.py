"""
Check of the two probe speed-ups against the probe as it was, on the GPU the loop runs on.

    old      AttributeProbe.logits_batch before 2026-10-11: the full (rows, length, vocab)
             logits of every row, then each row's last position; 16 rows per pass
    batch    logits_batch now: the output layer on each row's last position only
    shared   logits_shared: prefix + message encoded once per batch, its KV cache reused for
             each attribute's question

For the occlusion variants of a message (every 1-4-word span deleted, as privacy_rewrite.py
scores them), each method's P(yes) is compared with old's and timed. The P differences should
be at bf16 rounding level (well under 0.01); a NaN or a large difference means a bug. The
seconds are those of one occlusion round's Qwen part for that message.

Messages: a dev message (all 10 attributes) and PUPA-SD id 331 (119 words; the first
--long_attributes attributes, since old takes minutes there). PUPA-SD is only timed here; no
result of this check feeds back into the algorithm.

Besides |P - P_old|, each line reports what the privacy step decides from these P: how many
(variant, attribute) cells land on the other side of the 0.5 threshold, and for how many
attributes the best-blamed span changes (blame as in privacy_rewrite.py, length penalty included).
With --dtype float32 both paths should agree almost exactly; if they do there while bfloat16
differs, the bfloat16 differences are rounding (different kernels for a cached pass), not a bug.

Job 640 (2026-10-11) already showed that batch gives the same P as old at 16, 64 and 128 rows
(max difference 0.0000) and only x1.1 speed at any batch size: the probe is compute-bound
(~500 tokens/s on the GB10), so only fewer tokens help. Hence batch runs at 64 only here.

    sbatch privacy/sbatch_scripts/run_check_probe_speed_gb10.sbatch
"""
import argparse
import csv
import sys
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
for _dir in (REPO / "privacy", REPO / "utility" / "metrics", REPO / "utility" / "grad_attribution"):
    if str(_dir) not in sys.path:
        sys.path.insert(0, str(_dir))

from attribute_probe import ATTRIBUTES, QUESTION, AttributeProbe, logits_shared   # noqa: E402
from grad_based_utility_score import load                                         # noqa: E402
from privacy_rewrite import delete, spans                                         # noqa: E402

DEV_MESSAGE = ("The night shifts at the maternity ward here in Kraków are wrecking my sleep. "
               "At 34, what recovery routine should a midwife like me follow on days off?")
OLD_BATCH = 16
BATCH = 64                       # for batch (16, 64 and 128 were equally fast in job 640)
SHARED_BATCHES = [64, 128]       # tried for shared
MAX_N = 4                        # span length of privacy_rewrite.py
SPAN_PENALTY = 0.05              # and its length penalty, for the best-blamed span per attribute
THRESHOLD = 0.5                  # tau of the loop


def get_args():
    p = argparse.ArgumentParser(description="Check the probe speed-ups against the old probe")
    p.add_argument("--model_path", default="/scratch/users/mkilic/Qwen2.5-7B-Instruct")
    p.add_argument("--pupa", default=str(REPO / "data" / "pupa_sd" / "pupa_sd_eval.csv"))
    p.add_argument("--pupa_id", default="331")
    p.add_argument("--long_attributes", type=int, default=2)
    p.add_argument("--dtype", choices=["bfloat16", "float32"], default="bfloat16",
                   help="float32: both paths should agree almost exactly (rounding vs bug)")
    p.add_argument("--only_dev", action="store_true", help="skip the long PUPA-SD message")
    return p.parse_args()


@torch.no_grad()
def old_logits_batch(probe, msgs_ids, batch_size=OLD_BATCH):
    """The probe before the change: full logits, then each row's last position."""
    model, tok = probe.model, probe.tok
    out = []
    for i in range(0, len(msgs_ids), batch_size):
        rows = [probe.build_ids(m)[0] for m in msgs_ids[i:i + batch_size]]
        lens = torch.tensor([len(r) for r in rows], device=model.device)
        ids = torch.full((len(rows), int(lens.max())), tok.pad_token_id, device=model.device)
        att = torch.zeros_like(ids)
        for j, r in enumerate(rows):
            ids[j, :len(r)], att[j, :len(r)] = r, 1
        logits = model(input_ids=ids, attention_mask=att).logits          # (rows, length, vocab)
        out.append(probe._readout(logits[torch.arange(len(rows), device=ids.device), lens - 1])[1])
    return torch.cat(out)


def timed(fn):
    """-> (fn(), seconds), with the GPU work finished on both ends."""
    torch.cuda.synchronize()
    start = time.perf_counter()
    out = fn()
    torch.cuda.synchronize()
    return out, time.perf_counter() - start


def best_spans(P, n_words):
    """[K] index of each attribute's best-blamed variant: blame = P(text) - P(variant) - penalty,
    with row 0 of P the full text and rows 1.. the variants."""
    blame = P[:1] - P[1:] - SPAN_PENALTY * (n_words[:, None] - 1)
    return blame.argmax(0)


def compare(name, text, attributes, model, tok):
    """Time old, batch and shared on the occlusion variants of text; print the differences."""
    sp = spans(text, MAX_N)
    n_words = torch.tensor([n for _, _, n in sp], device=model.device, dtype=torch.float32)
    variants = [text] + [delete(text, s, e) for s, e, _ in sp]          # row 0: the full text
    ids = [tok(v, add_special_tokens=False).input_ids for v in variants]
    probes = [AttributeProbe(model, tok, QUESTION.format(attribute=a)) for a in attributes]
    n_tokens = sum(len(p.prefix_ids) + len(m) + len(p.suffix_ids) for p in probes for m in ids)
    print(f"\n{name}: {len(text.split())} words, {len(variants)} span variants x {len(attributes)} "
          f"attributes ({n_tokens} probe tokens the old way)")

    old, t_old = timed(lambda: torch.stack([torch.sigmoid(old_logits_batch(p, ids)) for p in probes], dim=1))
    print(f"  {'old, batch 16':<18} {t_old:7.1f} s")
    runs = [(f"batch, batch {BATCH}",
             lambda: torch.stack([torch.sigmoid(p.logits_batch(ids, BATCH)) for p in probes], dim=1))]
    runs += [(f"shared, batch {b}", lambda b=b: torch.sigmoid(logits_shared(probes, ids, b))) for b in SHARED_BATCHES]
    best_old = best_spans(old, n_words)
    for label, fn in runs:
        P, t = timed(fn)                                            # (variants, attributes)
        diff = (P - old).abs()
        flips = ((P > THRESHOLD) != (old > THRESHOLD)).sum().item()
        moved = (best_spans(P, n_words) != best_old).sum().item()
        v, a = divmod(diff.argmax().item(), diff.shape[1])
        print(f"  {label:<18} {t:7.1f} s   x{t_old / t:5.1f}   |P - P_old| max {diff.max().item():.4f}, "
              f"mean {diff.mean().item():.5f}; {flips} cells cross {THRESHOLD}, best span changes for "
              f"{moved}/{len(attributes)} attributes; worst: {attributes[a]} P_old {old[v, a].item():.3f} "
              f"-> {P[v, a].item():.3f}" + ("   NaN!" if torch.isnan(P).any() else ""))


def main():
    args = get_args()
    print(f"Loading {args.model_path} ({args.dtype}) ...")
    model, tok = load(args.model_path, "cuda", getattr(torch, args.dtype))
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    with open(args.pupa, encoding="utf-8") as f:
        long_text = next(r["original"] for r in csv.DictReader(f) if r["id"] == args.pupa_id)
    compare("dev message", DEV_MESSAGE, ATTRIBUTES, model, tok)
    if not args.only_dev:
        compare(f"PUPA-SD id {args.pupa_id}", long_text, ATTRIBUTES[:args.long_attributes], model, tok)


if __name__ == "__main__":
    main()
