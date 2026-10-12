"""
Diagnostics of the attacker probe's speed on the GPU it runs on. Nothing here changes a result.

Job 660 measured ~444 probe tokens/s for Qwen2.5-7B in bf16 on the GB10, roughly 6 TFLOPS. This
script finds where an occlusion round's time goes and whether a change that keeps P identical
could make it faster:
    1. setup: GPU, versions, transformers' attention implementation, the SDPA backends enabled
    2. GPU reference: a bf16 matmul with the model's MLP shape on its own, in TFLOPS
    3. one occlusion round split into building the inputs (CPU), the Qwen probe (logits_shared)
       and Laya, with the probe's achieved TFLOPS = 2 * (non-embedding parameters) * tokens / s
    4. torch.profiler on one batch of logits_shared: the CUDA ops that take the most time
    5. attention variants on the same variants: transformers' "eager" next to the default, and
       each SDPA backend forced (flash, efficient, cudnn, math); seconds and |P - P_default|
    6. padding: the share of pad tokens in the batches, and the probe on variants sorted by length
How to read it: if 2. is far above 3., the time goes to overhead (attention path, small ops,
Python), which a P-identical change may cut; if they are close, the GPU stack is the limit.

Messages: the dev message of check_probe_speed.py (all sections) and PUPA-SD id 331 (119 words;
sections 3 and 6 only, timing only).

    sbatch privacy/sbatch_scripts/run_profile_probe_gb10.sbatch
"""
import argparse
import csv
import sys
import time
from pathlib import Path

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.profiler import ProfilerActivity, profile

REPO = Path(__file__).resolve().parents[1]
for _dir in (REPO / "privacy", REPO / "utility" / "metrics", REPO / "utility" / "grad_attribution"):
    if str(_dir) not in sys.path:
        sys.path.insert(0, str(_dir))

import transformers                                                                 # noqa: E402
import laya                                                                         # noqa: E402
from attribute_probe import ATTRIBUTES, BATCH_SIZE, QUESTION, AttributeProbe, logits_shared   # noqa: E402
from grad_based_utility_score import load                                           # noqa: E402
from laya_probe import PRIVACY_LAYA                                                 # noqa: E402
from privacy_rewrite import LayaAttacker, delete, spans                             # noqa: E402
from check_probe_speed import DEV_MESSAGE, MAX_N, timed                             # noqa: E402

MATMUL_TOKENS = 8192             # rows of the reference matmul (tokens of a large probe batch)
MATMUL_REPS = 10
PROFILE_ROWS = 64                # one batch of logits_shared under torch.profiler
TOP_OPS = 15
BACKENDS = [SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION,
            SDPBackend.CUDNN_ATTENTION, SDPBackend.MATH]


def get_args():
    p = argparse.ArgumentParser(description="Where the attacker probe's time goes")
    p.add_argument("--model_path", default="/scratch/users/mkilic/Qwen2.5-7B-Instruct")
    p.add_argument("--pupa", default=str(REPO / "data" / "pupa_sd" / "pupa_sd_eval.csv"))
    p.add_argument("--pupa_id", default="331")
    return p.parse_args()


def setup(model):
    """1. What the probe runs on."""
    print("=== 1. setup")
    print(f"  GPU {torch.cuda.get_device_name()} (capability {torch.cuda.get_device_capability()}), "
          f"torch {torch.__version__}, transformers {transformers.__version__}")
    print(f"  attention implementation: {model.config._attn_implementation}, dtype {model.dtype}")
    print(f"  SDPA backends enabled: flash {torch.backends.cuda.flash_sdp_enabled()}, "
          f"efficient {torch.backends.cuda.mem_efficient_sdp_enabled()}, "
          f"cudnn {torch.backends.cuda.cudnn_sdp_enabled()}, math {torch.backends.cuda.math_sdp_enabled()}")


def matmul_reference(model):
    """2. TFLOPS of the MLP's up projection, (tokens, hidden) x (hidden, intermediate), alone."""
    h, i = model.config.hidden_size, model.config.intermediate_size
    x = torch.randn(MATMUL_TOKENS, h, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(h, i, device="cuda", dtype=torch.bfloat16)
    for _ in range(3):
        x @ w
    _, t = timed(lambda: [x @ w for _ in range(MATMUL_REPS)])
    print(f"\n=== 2. reference bf16 matmul ({MATMUL_TOKENS} x {h}) @ ({h} x {i}): "
          f"{2 * MATMUL_TOKENS * h * i * MATMUL_REPS / t / 1e12:.1f} TFLOPS")


def flops_per_token(model):
    """2 * the parameters every token passes through: all but the embedding and output tables
    (untied in Qwen2.5-7B; the probe applies the output table to one position per row only)."""
    n = sum(p.numel() for p in model.parameters())
    tables = 2 * model.config.vocab_size * model.config.hidden_size
    return 2 * (n - tables)


def breakdown(name, text, model, tok, probes, attacker):
    """3. One occlusion round of text, split into its parts. -> (ids, P of the default probe)."""
    start = time.perf_counter()
    variants = [delete(text, s, e) for s, e, _ in spans(text, MAX_N)]
    ids = [tok(v, add_special_tokens=False).input_ids for v in variants]
    t_build = time.perf_counter() - start
    P, t_qwen = timed(lambda: torch.sigmoid(logits_shared(probes, ids)))
    _, t_laya = timed(lambda: attacker.p(variants))
    n_tok = sum(len(probes[0].prefix_ids) + len(m) + sum(len(p.suffix_ids) for p in probes) for m in ids)
    print(f"\n=== 3. {name}: {len(text.split())} words, {len(variants)} span variants, "
          f"{len(probes)} attributes, {n_tok} probe tokens")
    print(f"  building the inputs {t_build:6.2f} s | Qwen probe {t_qwen:6.1f} s "
          f"({n_tok / t_qwen:.0f} tokens/s, {flops_per_token(model) * n_tok / t_qwen / 1e12:.1f} TFLOPS) "
          f"| Laya {t_laya:6.1f} s")
    return ids, P


def profiled_batch(probes, ids):
    """4. The CUDA ops of one logits_shared batch, by their own GPU time."""
    batch = ids[:PROFILE_ROWS]
    logits_shared(probes, batch)                                   # warm-up
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        logits_shared(probes, batch)
        torch.cuda.synchronize()
    print(f"\n=== 4. torch.profiler, one batch of {len(batch)} variants x {len(probes)} attributes")
    for key in ["self_device_time_total", "self_cuda_time_total", "cpu_time_total"]:
        try:
            print(prof.key_averages().table(sort_by=key, row_limit=TOP_OPS))
            return
        except (AttributeError, KeyError, RuntimeError):
            continue


def attention_variants(model, probes, ids, P_default):
    """5. The probe with each attention path on the same variants: seconds and |P - P_default|."""
    print("\n=== 5. attention variants (dev message)")
    default = model.config._attn_implementation
    for impl in ["eager", default]:
        model.set_attn_implementation(impl)
        P, t = timed(lambda: torch.sigmoid(logits_shared(probes, ids)))
        print(f"  {impl:<22} {t:6.1f} s   |P - P_default| max {(P - P_default).abs().max().item():.4f}")
    model.set_attn_implementation("sdpa")                           # forcing a backend needs SDPA
    for backend in BACKENDS:
        try:
            with sdpa_kernel(backend):
                P, t = timed(lambda: torch.sigmoid(logits_shared(probes, ids)))
            print(f"  sdpa forced {backend.name:<10} {t:6.1f} s   "
                  f"|P - P_default| max {(P - P_default).abs().max().item():.4f}")
        except RuntimeError as err:
            print(f"  sdpa forced {backend.name:<10} not available: {str(err).splitlines()[0][:90]}")
    model.set_attn_implementation(default)


def padding(name, probes, ids, P_default):
    """6. Pad share of the prefix + message batches, and the probe on length-sorted variants."""
    prefix = len(probes[0].prefix_ids)
    pads = total = 0
    for i in range(0, len(ids), BATCH_SIZE):
        lens = [prefix + len(m) for m in ids[i:i + BATCH_SIZE]]
        pads += sum(max(lens) - n for n in lens)
        total += max(lens) * len(lens)
    order = sorted(range(len(ids)), key=lambda j: len(ids[j]))
    P_sorted, t = timed(lambda: torch.sigmoid(logits_shared(probes, [ids[j] for j in order])))
    back = torch.empty_like(P_sorted)
    back[order] = P_sorted                                          # rows back in the original order
    print(f"\n=== 6. {name}: {pads / total:.1%} of the prefix + message batch tokens are padding; "
          f"sorted by length: {t:6.1f} s, |P - P_default| max {(back - P_default).abs().max().item():.4f}")


def main():
    args = get_args()
    print(f"Loading {args.model_path} (bfloat16) and Laya ...")
    model, tok = load(args.model_path, "cuda", torch.bfloat16)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    probes = [AttributeProbe(model, tok, QUESTION.format(attribute=a)) for a in ATTRIBUTES]
    attacker = LayaAttacker(laya.load(PRIVACY_LAYA), ATTRIBUTES)
    with open(args.pupa, encoding="utf-8") as f:
        long_text = next(r["original"] for r in csv.DictReader(f) if r["id"] == args.pupa_id)

    setup(model)
    matmul_reference(model)
    ids, P = breakdown("dev message", DEV_MESSAGE, model, tok, probes, attacker)
    profiled_batch(probes, ids)
    attention_variants(model, probes, ids, P)
    padding("dev message", probes, ids, P)
    ids_long, P_long = breakdown(f"PUPA-SD id {args.pupa_id}", long_text, model, tok, probes, attacker)
    padding(f"PUPA-SD id {args.pupa_id}", probes, ids_long, P_long)


if __name__ == "__main__":
    main()
