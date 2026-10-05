"""
Fine-tune Laya on the leaking/safe pairs in data/laya_ft/train.csv with the
recipe of its creators' fine-tuning notebook (github.com/NandhaKishorM/laya,
notebooks/laya_finetune_typed_decisions_2xT4_kaggle.ipynb), and save a
checkpoint that laya.load reads (pass it to the loop as --laya_path).

Labeled items per pair, a in `attributes`, b in `not_inferable`:

    (leaking, a) -> 1      (safe, a) -> 0      (leaking, b) -> 0

so one text appears under both labels and the model has to read which
attribute is asked. Each item is one LayaProbe row (question, then message),
the token layout the loop scores, with target t = (1 − y, y) over the options
(false, true).

Loss, the notebook's RLCD step on the raw scores z = (z_false, z_true):

    z̃_g = z + ε_g,   ε_g ~ N(0, σ²) made zero-mean over the options,   g = 1..G
    q_g = softmax(z̃_g),   r_g = t·log q_g + w_sph·(t·q_g)/‖q_g‖          (proper score)
    A_g = (r_g − mean_g r) / std(r),   log π_g = −‖z̃_g − z‖² / 2σ²
    L   = −mean(A_g · log π_g)  −  t·log softmax(z)

σ goes linearly from 0.4 to 0.1 over the epochs. AdamW (encoder 2.5e-5, head
1e-4), cosine decay, effective batch 64, 4 epochs; the last epoch is kept.

Calibration: min(400, 10% of the items) are held out and never enter a
training batch, taken as whole pairs so no text is in both splits. After
training, the weights are saved in fp16 (as the notebook does) and the noul
temperature is fit on the held-out items with those saved weights, by LBFGS:

    T = argmin_T  −Σ t·log softmax(z / T)

Deviations, all for hardware: one GPU instead of two (8 micro-batches of 8 are
accumulated, so the effective batch stays 64); bf16 autocast instead of fp16
with a grad scaler (the notebook's T4 has no bf16, and the checkpoint's own
amp_dtype is bf16); no gradient checkpointing (our messages are short).
"""
import argparse
import json
import math
import random
import shutil
import time
from itertools import accumulate
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from safetensors.torch import save_file

import laya
from laya.common import QTYPES, TEMP_MAX, TEMP_MIN, proper_reward

from laya_probe import DEFAULT_LAYA, QUESTION, LayaProbe

DATA = Path(__file__).resolve().parent / "data" / "laya_ft" / "train.csv"
OUT_ROOT = "/scratch/users/mkilic"      # 842 MB per checkpoint; /work is capped at 20 GB

# the notebook's recipe
EPOCHS = 4
MICRO_BATCH = 8
GRAD_ACCUM = 8                # 8 x 8 = 64 = the notebook's effective batch (8 x 2 GPUs x 4)
GROUP_SIZE = 4                # noisy samples per item, G
LR_ENCODER = 2.5e-5
LR_HEAD = 1.0e-4
WEIGHT_DECAY = 0.01
LR_MIN = 1e-6                 # cosine floor
SIGMA_START, SIGMA_END = 0.4, 0.1
W_SPH, W_RPS = 0.75, 1.0      # spherical and ranked-probability weights in proper_reward
CE_WEIGHT = 1.0
GRAD_CLIP = 1.0
CALIB_MAX, CALIB_FRAC = 400, 0.1
LBFGS_LR, LBFGS_ITERS = 0.1, 100
MIN_FIT_ITEMS = 10            # fewer held-out items -> T = 1
TEMP_UNFIT = 1.2              # temperature for question types with no held-out items (choice, score)

EVAL_BATCH = 64               # items per forward pass on the calibration set


def load_pairs(path, seed):
    """All pairs of the CSV in a seeded random order, the two attribute fields as lists."""
    df = pd.read_csv(path)
    pairs = [{"leaking": r.leaking, "safe": r.safe,
              "attributes": [a.strip() for a in r.attributes.split(";")],
              "not_inferable": [a.strip() for a in r.not_inferable.split(";")]}
             for r in df.itertuples()]
    random.Random(seed).shuffle(pairs)
    return pairs


def split_pairs(pairs, n_pairs):
    """Calibration = pairs from the front until they hold min(CALIB_MAX, CALIB_FRAC of all)
    items; training = the first n_pairs of the rest, nested across n_pairs (learning curve)."""
    sizes = [2 * len(p["attributes"]) + len(p["not_inferable"]) for p in pairs]
    n_calib = min(CALIB_MAX, int(CALIB_FRAC * sum(sizes)))
    cut = next(i + 1 for i, c in enumerate(accumulate(sizes)) if c >= n_calib)
    train = pairs[cut:]
    return (train[:n_pairs] if n_pairs else train), pairs[:cut]


def make_items(pairs, probe):
    """Labeled items (message ids, attribute index, y) of the pairs."""
    k = {a: i for i, a in enumerate(probe.attributes)}
    items = []
    for p in pairs:
        leak, safe = probe.encode(p["leaking"]), probe.encode(p["safe"])
        items += [(leak, k[a], 1.0) for a in p["attributes"]]
        items += [(safe, k[a], 0.0) for a in p["attributes"]]
        items += [(leak, k[b], 0.0) for b in p["not_inferable"]]
    return items


def batches(items, batch_size, rng=None):
    """(message ids, attribute indices, labels [B]) in chunks; shuffled if rng is given."""
    order = list(range(len(items)))
    if rng is not None:
        rng.shuffle(order)
    for i in range(0, len(order), batch_size):
        ids, ks, ys = zip(*(items[j] for j in order[i:i + batch_size]))
        yield list(ids), list(ks), torch.tensor(ys)


def auroc(scores, labels):
    """P(score of a random positive > score of a random negative), ties count half."""
    pos, neg = scores[labels == 1], scores[labels == 0]
    greater = (pos[:, None] > neg[None]).float().mean()
    ties = (pos[:, None] == neg[None]).float().mean()
    return (greater + 0.5 * ties).item()


@torch.no_grad()
def predict(probe, items):
    """Raw scores [N, 2] and labels [N], fp32 in eval mode as the loop computes them."""
    probe.model.eval()
    zs, ys = [], []
    for ids, ks, y in batches(items, EVAL_BATCH):
        zs.append(probe.item_scores(ids, ks).cpu())
        ys.append(y)
    return torch.cat(zs), torch.cat(ys)


def metrics(z, y, T):
    """BCE of ℓ = (z_true − z_false)/T, accuracy at P = 0.5 (ℓ > 0) and AUROC."""
    logit = (z[:, 1] - z[:, 0]) / T
    return {"loss": F.binary_cross_entropy_with_logits(logit, y).item(),
            "acc": ((logit > 0).float() == y).float().mean().item(),
            "auroc": auroc(logit, y)}


def make_optimizer(model, n_items):
    """AdamW, LR_ENCODER on the encoder and LR_HEAD on the rest; cosine decay to LR_MIN
    over the notebook's update count."""
    enc = [p for n, p in model.named_parameters() if "encoder." in n and p.requires_grad]
    head = [p for n, p in model.named_parameters() if "encoder." not in n]
    groups = [{"params": head, "lr": LR_HEAD}] + ([{"params": enc, "lr": LR_ENCODER}] if enc else [])
    opt = torch.optim.AdamW(groups, weight_decay=WEIGHT_DECAY)
    total_updates = (n_items // (MICRO_BATCH * GRAD_ACCUM)) * EPOCHS
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, total_updates), eta_min=LR_MIN)
    return opt, sched


def rlcd_loss(z, y, sigma):
    """Policy gradient on the proper score plus cross-entropy (formulas in the docstring).
    Returns the loss and the mean reward."""
    t = torch.stack([1 - y, y], -1)                                      # [B, 2]
    mask = torch.ones_like(t, dtype=torch.bool)
    qtype = torch.full((len(y),), QTYPES["noul"], device=z.device)
    eps = torch.randn((GROUP_SIZE,) + z.shape, device=z.device) * sigma  # [G, B, 2]
    eps = eps - eps.mean(-1, keepdim=True)
    z_noisy = z.detach()[None] + eps
    with torch.no_grad():
        r = proper_reward(torch.softmax(z_noisy, -1), t[None], qtype, mask, w_sph=W_SPH, w_rps=W_RPS)  # [G, B]
        adv = r - r.mean(0, keepdim=True)
        adv = adv / (adv.std() + 1e-6)
    logp = -((z_noisy - z[None]) ** 2).sum(-1) / (2 * sigma ** 2)       # [G, B]
    loss_rl = -(adv * logp).mean()
    loss_ce = -(t * torch.log_softmax(z, -1)).sum(-1).mean()
    return loss_rl + CE_WEIGHT * loss_ce, r.mean().item()


def train_epoch(probe, items, opt, sched, sigma, rng):
    """One shuffled pass in micro-batches, bf16 autocast on GPU; an optimizer step every
    GRAD_ACCUM micro-batches and after the last. Returns the mean loss and mean reward."""
    probe.model.train()
    amp = probe.device.type == "cuda"
    n_micro = math.ceil(len(items) / MICRO_BATCH)
    total_loss = total_r = 0.0
    opt.zero_grad(set_to_none=True)
    for i, (ids, ks, y) in enumerate(batches(items, MICRO_BATCH, rng), 1):
        with torch.autocast(probe.device.type, dtype=torch.bfloat16, enabled=amp):
            z = probe.item_scores(ids, ks)                               # [B, 2]
        loss, r = rlcd_loss(z, y.to(z.device), sigma)
        (loss / GRAD_ACCUM).backward()
        if i % GRAD_ACCUM == 0 or i == n_micro:
            torch.nn.utils.clip_grad_norm_(probe.model.parameters(), GRAD_CLIP)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
        total_loss += loss.item()
        total_r += r
    return total_loss / n_micro, total_r / n_micro


def save_weights(model, base_dir, out_dir):
    """Copy the base checkpoint (tokenizer, encoder config, ...) with the model's weights
    in fp16, as the shipped checkpoint and the notebook store them, and load the rounded
    weights back, so the temperature is fit on exactly what was saved."""
    shutil.copytree(base_dir, out_dir, dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns("model.safetensors", ".cache"))
    # copy=True: no two keys share storage, which save_file refuses
    weights = {k: v.detach().to("cpu", dtype=torch.float16 if v.is_floating_point() else v.dtype, copy=True)
               for k, v in model.state_dict().items()}
    save_file(weights, str(Path(out_dir) / "model.safetensors"), metadata={"format": "pt"})
    model.load_state_dict(weights)


def fit_temperature(z, y):
    """T = argmin −Σ t·log softmax(z/T) by LBFGS on log T from T = 1, clamped to Laya's
    [TEMP_MIN, TEMP_MAX]."""
    if len(y) < MIN_FIT_ITEMS:
        return 1.0
    t = torch.stack([1 - y, y], -1)
    log_t = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=LBFGS_LR, max_iter=LBFGS_ITERS)

    def closure():
        opt.zero_grad()
        loss = -(t * torch.log_softmax(z / log_t.exp(), -1)).sum(-1).mean()
        loss.backward()
        return loss

    opt.step(closure)
    return torch.clamp(log_t.exp(), TEMP_MIN, TEMP_MAX).item()


def write_config(base_dir, out_dir, T, info):
    """rl_agent_config.json as the notebook writes it: one temperature per question type
    (TEMP_UNFIT for choice and score, which had no held-out items) and no per-bucket
    temperatures, since inherited ones would override the new noul value. This also
    drops choice:11+ = 0.10 and with it Laya's load warning."""
    cfg = json.loads((Path(base_dir) / "rl_agent_config.json").read_text())
    cfg["fine_tuned"] = True
    cfg["model_name"] = "laya-privacy"
    cfg["temperature"] = [TEMP_UNFIT] * len(QTYPES)
    cfg["temperature"][QTYPES["noul"]] = T
    cfg.pop("temperature_by_options", None)
    cfg["privacy_finetune"] = info
    (Path(out_dir) / "rl_agent_config.json").write_text(json.dumps(cfg, indent=2))


def main():
    p = argparse.ArgumentParser(description="Fine-tune Laya on leaking/safe pairs (creators' recipe)")
    p.add_argument("--laya_path", default=DEFAULT_LAYA)
    p.add_argument("--data", default=str(DATA))
    p.add_argument("--out", default=None, help=f"default: {OUT_ROOT}/laya-privacy-n<pairs>[-head]")
    p.add_argument("--n_pairs", type=int, default=None, help="train on the first n pairs only (learning curve)")
    p.add_argument("--freeze_encoder", action="store_true", help="train the decision head only")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    pairs = load_pairs(args.data, args.seed)
    train_pairs, calib_pairs = split_pairs(pairs, args.n_pairs)
    out = args.out or f"{OUT_ROOT}/laya-privacy-n{len(train_pairs)}" + ("-head" if args.freeze_encoder else "")

    agent = laya.load(args.laya_path)
    if args.freeze_encoder:
        agent.model.encoder.requires_grad_(False)
    attributes = sorted({a for q in pairs for a in q["attributes"] + q["not_inferable"]})
    probe = LayaProbe(agent, attributes, QUESTION)
    train_items, calib_items = make_items(train_pairs, probe), make_items(calib_pairs, probe)
    print(f"pairs: train {len(train_pairs)}, calibration {len(calib_pairs)}  ->  items: train {len(train_items)} "
          f"({sum(y for *_, y in train_items) / len(train_items):.0%} yes), calibration {len(calib_items)}")
    print(f"{len(attributes)} attributes, base T = {probe.T:.3f}, device {probe.device}, out -> {out}")

    z, y = predict(probe, calib_items)
    base = metrics(z, y, probe.T)
    print(f"epoch 0  calib acc {base['acc']:.3f}  auroc {base['auroc']:.3f}  loss {base['loss']:.4f}   (base Laya)")

    opt, sched = make_optimizer(agent.model, len(train_items))
    for epoch in range(EPOCHS):
        sigma = SIGMA_START + (SIGMA_END - SIGMA_START) * epoch / max(1, EPOCHS - 1)
        t0 = time.time()
        loss, reward = train_epoch(probe, train_items, opt, sched, sigma, rng)
        z, y = predict(probe, calib_items)
        m = metrics(z, y, probe.T)
        print(f"epoch {epoch + 1}  calib acc {m['acc']:.3f}  auroc {m['auroc']:.3f}   (monitor only)   "
              f"train loss {loss:.4f}  reward {reward:.3f}  σ {sigma:.2f}  ({time.time() - t0:.0f}s)")

    save_weights(agent.model, args.laya_path, out)
    z, y = predict(probe, calib_items)                  # on the saved (fp16-rounded) weights
    T = fit_temperature(z, y)
    final = metrics(z, y, T)
    print(f"T {probe.T:.3f} -> {T:.3f}: calib loss {final['loss']:.4f}  acc {final['acc']:.3f}  auroc {final['auroc']:.3f}")
    info = {"recipe": "laya_finetune_typed_decisions_2xT4_kaggle.ipynb", "data": str(args.data),
            "n_train_pairs": len(train_pairs), "n_calib_pairs": len(calib_pairs), "n_calib_items": len(calib_items),
            "attributes": attributes, "question": QUESTION, "T_base": probe.T,
            "calib_base": base, "calib_final": final, "args": vars(args)}
    write_config(args.laya_path, out, T, info)
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
