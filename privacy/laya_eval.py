"""
Before/after evaluation of the Laya fine-tuning (ROADMAP "Laya fine-tuning",
Steps 2 and 4): base Laya and each fine-tuned checkpoint on the same items.

Test sets
  synthetic  data/laya_ft/test.csv, labeled as in training (laya_finetune.make_items):
             (leaking, a) -> 1, (safe, a) -> 0 for a in attributes, (leaking, b) -> 0 for b in not_inferable
  ood        utility/data/prompt_rewrites_100.csv, a different generator and style:
             (original, a) -> 1, (priv_targeted, a) -> 0 for a in protected_attrs that
             Laya was trained on (13 of the 200 cells are dropped)

Every model answers the loop's question in the loop's token layout (LayaProbe),
fp32 in eval mode, with its own temperature:  ℓ = (z_true − z_false)/T,  p = σ(ℓ).

Metrics
  acc       accuracy at p = 0.5 (the loop's threshold), printed next to the
            majority-class baseline
  auroc     P(ℓ of a random yes item > ℓ of a random no item): the ranking at any threshold
  pair_acc  P(p(leaking) > p(safe)) for the same pair and attribute: the drop the
            loop's span occlusion relies on

95% CIs come from a bootstrap over pairs (the items of a pair share text), with the
same resamples for every model, so each checkpoint's difference to base gets its own
CI. Accuracy is also broken down by item kind, cue_type, edit_type and attribute.

"base + bias" is base Laya with p = σ(α·ℓ + β), α and β fit on train.csv by logistic
regression. If fine-tuning only moved Laya's decision threshold, this row would match
its accuracy. AUROC and pair_acc cannot change under it (α > 0 keeps the order).
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

import laya

import laya_finetune as lf
from laya_probe import DEFAULT_LAYA, QUESTION, LayaProbe

HERE = Path(__file__).resolve().parent
TEST = HERE / "data" / "laya_ft" / "test.csv"
TRAIN = HERE / "data" / "laya_ft" / "train.csv"
OOD = HERE.parent / "utility" / "data" / "prompt_rewrites_100.csv"
OUT_DIR = HERE / "results" / "laya_eval"
N_BOOT = 1000
BOOT_SEED = 0
BIAS_ITERS = 100              # LBFGS iterations for the logistic fit of α, β
# accuracy breakdowns per test set: (column, item kind or None for all items)
GROUPS = {"synthetic": [("kind", None), ("cue_type", "leaking"), ("edit_type", "safe"), ("attribute", None)],
          "ood": [("attribute", None)]}


def split(field):
    return [a.strip() for a in field.split(";")]


def synthetic_pairs(path):
    """Pairs of a laya_ft CSV in file order: laya_finetune's pair format plus id, edit_type, cue_type."""
    return [{"id": r.id, "leaking": r.leaking, "safe": r.safe, "attributes": split(r.attributes),
             "not_inferable": split(r.not_inferable), "edit_type": r.edit_type, "cue_type": r.cue_type}
            for r in pd.read_csv(path).itertuples()]


def ood_pairs(path, attributes):
    """prompt_rewrites rows as pairs: original (leaking) vs priv_targeted (safe), on the
    protected attributes Laya was trained on."""
    pairs = []
    for r in pd.read_csv(path).itertuples():
        attrs = [a for a in split(r.protected_attrs) if a in attributes]
        if attrs:
            pairs.append({"id": r.id, "leaking": r.original, "safe": r.priv_targeted_rewrite, "attributes": attrs,
                          "not_inferable": [], "edit_type": "priv_targeted", "cue_type": "-"})
    return pairs


def item_frame(pairs):
    """One row per labeled item, in laya_finetune.make_items' order and with its labels."""
    rows = []
    for i, p in enumerate(pairs):
        meta = {"pair": i, "id": p["id"], "edit_type": p["edit_type"], "cue_type": p["cue_type"]}
        rows += [dict(meta, kind="leaking", attribute=a, y=1.0, text=p["leaking"]) for a in p["attributes"]]
        rows += [dict(meta, kind="safe", attribute=a, y=0.0, text=p["safe"]) for a in p["attributes"]]
        rows += [dict(meta, kind="not_inferable", attribute=b, y=0.0, text=p["leaking"]) for b in p["not_inferable"]]
    return pd.DataFrame(rows)


def score(probe, pairs, df):
    """ℓ [N] of the items, built by laya_finetune.make_items (the training construction),
    which must agree with df item by item."""
    items = lf.make_items(pairs, probe)
    assert [probe.attributes[k] for _, k, _ in items] == df.attribute.tolist()
    assert [y for *_, y in items] == df.y.tolist()
    z, _ = lf.predict(probe, items)
    return ((z[:, 1] - z[:, 0]) / probe.T).double().numpy()


def fit_bias(logit, y):
    """α, β of p = σ(α·ℓ + β) by logistic regression (LBFGS on the BCE)."""
    l, y = torch.tensor(logit), torch.tensor(y)
    ab = torch.tensor([1.0, 0.0], dtype=l.dtype, requires_grad=True)
    opt = torch.optim.LBFGS([ab], max_iter=BIAS_ITERS)

    def closure():
        opt.zero_grad()
        loss = F.binary_cross_entropy_with_logits(ab[0] * l + ab[1], y)
        loss.backward()
        return loss

    opt.step(closure)
    return ab.detach().tolist()


def score_models(paths, attributes, sets, frames, train_pairs):
    """{model: {test set: ℓ}} for base (paths[0]), "base + bias" and each checkpoint, one
    model on the GPU at a time; also α, β of the bias fit."""
    logits, bias = {}, None
    for i, path in enumerate(paths):
        agent = laya.load(path)
        probe = LayaProbe(agent, attributes, QUESTION)
        name = "base" if i == 0 else Path(path).name
        logits[name] = {s: score(probe, pairs, frames[s]) for s, pairs in sets.items()}
        print(f"{name}: T = {probe.T:.3f}")
        if i == 0:
            train_df = item_frame(train_pairs)
            a, b = fit_bias(score(probe, train_pairs, train_df), train_df.y.to_numpy())
            bias = {"alpha": a, "beta": b, "n_items": len(train_df)}
            print(f"base + bias: α = {a:.3f}, β = {b:+.3f} (fit on {len(train_df)} train items)")
            logits["base + bias"] = {s: a * l + b for s, l in logits["base"].items()}
        del agent, probe
        torch.cuda.empty_cache()
    return logits, bias


def pair_index(df):
    """Item indices (leaking, safe) of every (pair, attribute) that has both, and their pair."""
    idx = df.reset_index().pivot_table(index=["pair", "attribute"], columns="kind", values="index", aggfunc="first")
    idx = idx[["leaking", "safe"]].dropna().astype(int)
    return idx.leaking.to_numpy(), idx.safe.to_numpy(), idx.index.get_level_values("pair").to_numpy()


def metrics(l, y, l_leak, l_safe):
    """acc, auroc, pair_acc of logits l [N] with labels y [N], and the paired leaking/safe logits."""
    return {"acc": ((l > 0) == (y == 1)).mean(),
            "auroc": lf.auroc(torch.tensor(l), torch.tensor(y)),
            "pair_acc": ((l_leak > l_safe) + 0.5 * (l_leak == l_safe)).mean()}


def bootstrap(logits, df, i_leak, i_safe, pair_of_link, rng):
    """95% CIs of every metric per model and of each model's difference to base, from
    N_BOOT resamples of whole pairs (the same resamples for all models)."""
    y = df.y.to_numpy()
    items_of = df.groupby("pair").indices
    links_of = pd.Series(np.arange(len(i_leak))).groupby(pair_of_link).indices
    pairs = np.array(sorted(items_of))
    none = np.empty(0, dtype=int)
    draws = {m: [] for m in logits}
    for _ in range(N_BOOT):
        s = rng.choice(pairs, len(pairs))
        idx = np.concatenate([items_of[q] for q in s])
        jdx = np.concatenate([links_of.get(q, none) for q in s])
        for m, l in logits.items():
            draws[m].append(metrics(l[idx], y[idx], l[i_leak[jdx]], l[i_safe[jdx]]))
    draws = {m: pd.DataFrame(d) for m, d in draws.items()}
    ci = {m: {k: np.percentile(d[k], [2.5, 97.5]).tolist() for k in d} for m, d in draws.items()}
    diff = {m: {k: np.percentile(d[k] - draws["base"][k], [2.5, 97.5]).tolist() for k in d}
            for m, d in draws.items() if m != "base"}
    return ci, diff


def group_accuracy(df, logits, column, kind):
    """Accuracy at p = 0.5 per value of a column (only items of one kind, if given), one column per model."""
    sel = (df.kind == kind).to_numpy() if kind else np.ones(len(df), dtype=bool)
    sub = df[sel]
    table = {"n": sub.groupby(column).size()}
    for m, l in logits.items():
        table[m] = pd.Series((l[sel] > 0) == (sub.y.to_numpy() == 1), index=sub.index).groupby(sub[column]).mean()
    return pd.DataFrame(table)


def report(name, df, logits, rng):
    """Print and return one test set's metrics with CIs, differences to base and accuracy breakdowns."""
    y = df.y.to_numpy()
    i_leak, i_safe, pair_of_link = pair_index(df)
    point = {m: metrics(l, y, l[i_leak], l[i_safe]) for m, l in logits.items()}
    ci, diff = bootstrap(logits, df, i_leak, i_safe, pair_of_link, rng)

    print(f"\n=== {name}: {df.pair.nunique()} pairs, {len(df)} items ({y.mean():.0%} yes), "
          f"{len(i_leak)} leaking/safe links; majority-class accuracy {max(y.mean(), 1 - y.mean()):.3f}")
    rows = [{"model": m, **{k: f"{v:.3f} [{ci[m][k][0]:.3f}, {ci[m][k][1]:.3f}]" for k, v in point[m].items()}}
            for m in logits]
    print(pd.DataFrame(rows).to_string(index=False))
    print("\ndifference to base [95% CI]")
    rows = [{"model": m, **{k: f"{point[m][k] - point['base'][k]:+.3f} [{lo:+.3f}, {hi:+.3f}]" for k, (lo, hi) in d.items()}}
            for m, d in diff.items()]
    print(pd.DataFrame(rows).to_string(index=False))
    for column, kind in GROUPS[name]:
        print(f"\naccuracy by {column}" + (f" ({kind} items)" if kind else ""))
        print(group_accuracy(df, logits, column, kind).to_string(float_format="%.3f"))
    return {"n_pairs": int(df.pair.nunique()), "n_items": len(df), "share_yes": float(y.mean()),
            "metrics": point, "ci": ci, "diff_ci": diff}


def main():
    p = argparse.ArgumentParser(description="Base vs fine-tuned Laya on the test sets")
    p.add_argument("--checkpoints", nargs="+", required=True, help="fine-tuned checkpoint dirs (laya_finetune.py)")
    p.add_argument("--laya_path", default=DEFAULT_LAYA, help="base Laya")
    p.add_argument("--test", default=str(TEST))
    p.add_argument("--train", default=str(TRAIN), help="for the base + bias fit")
    p.add_argument("--ood", default=str(OOD))
    p.add_argument("--save_suffix", default="laya_eval")
    args = p.parse_args()

    test, train = synthetic_pairs(args.test), synthetic_pairs(args.train)
    attributes = sorted({a for q in test + train for a in q["attributes"] + q["not_inferable"]})
    sets = {"synthetic": test, "ood": ood_pairs(args.ood, attributes)}
    frames = {s: item_frame(pairs) for s, pairs in sets.items()}
    logits, bias = score_models([args.laya_path] + args.checkpoints, attributes, sets, frames, train)

    rng = np.random.default_rng(BOOT_SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = {"base": args.laya_path, "checkpoints": args.checkpoints, "question": QUESTION, "base_bias": bias}
    for s, df in frames.items():
        L = {m: l[s] for m, l in logits.items()}
        summary[s] = report(s, df, L, rng)
        df.assign(**{f"logit {m}": l for m, l in L.items()}).to_csv(OUT_DIR / f"{args.save_suffix}_{s}.csv", index=False)
    (OUT_DIR / f"{args.save_suffix}.json").write_text(json.dumps(summary, indent=2))
    print(f"\nresults -> {OUT_DIR}/{args.save_suffix}*")


if __name__ == "__main__":
    main()
