"""
Which hidden layer makes the best answer embedding?

What is measured, per layer
---------------------------
For each persona the answers of the original prompt are compared against those
of its paraphrase (details kept) and of its privatized rewrite (details
removed). A layer is good if it separates those two, so we report:
 
    util_diff              mean(utility paraphrase) - mean(utility privatized)
    util_cohen_d           that gap in standard deviations of the paired per-persona
                           difference (scale-free -- the number to rank layers by)
    util_rank              personas where paraphrase scored above privatized, out of
                           n_personas
    mean_pairwise_cos_all  mean pairwise cosine over all embeddings at that layer
                           -- a direct anisotropy readout: near 1.0 means the cone
                           effect has collapsed the space and no kernel can recover it.
                           All three sets are pooled on purpose: this is a property of
                           the layer, not a comparison between prompt types.

Every column is a mean over personas at that layer; the pairing is per persona,
so util_cohen_d and util_rank are computed from the per-persona differences
rather than from the two column means.

    cross_para   mean of E_orig @ E_para.T -- all K x K cross-set pairs
    cross_priv   mean of E_orig @ E_priv.T -- likewise
    self_orig    mean off-diagonal cosine within the original's own answers
    self_para    likewise within the paraphrase's, self_priv within the
                 privatized rewrite's

u_mean_d: returns unbiased difference of the mean distances of answers created by para and priv calculated as
priv - para since priv is expected to have a larger distance

cos_centroid_para / cos_centroid_priv
    cross / sqrt(self_orig * self_rew): cosine between the two cloud centroids,
    i.e. the angle alone with the concentration divided out. Higher = more
    utility, so its gap / cohen_d / rank are taken as para - priv.
"""
import os
import csv
import time
import numpy as np
import argparse

import torch
import torch.nn.functional as F

from topk_sample import load, topk_answers
from cosine_utility import cosine_stats


def get_args():
    p = argparse.ArgumentParser(description="Sweep the hidden layer used for answer embeddings")
    p.add_argument("--model_path", default="/work/projects/mkilic/models/Qwen2.5-3B-Instruct")
    p.add_argument("--csv", default="data/privacy_prompt_rewrites.csv")
    p.add_argument("--k", type=int, default=15, help="answers sampled per prompt")
    p.add_argument("--max_new_tokens", type=int, default=256)
    p.add_argument("--pooling", choices=["mean", "last"], default="mean",
                   help="mean pools over all tokens; last takes the final real token, "
                        "which under causal attention is the only position that has "
                        "seen the whole answer")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top_p", type=float, default=1.0)
    p.add_argument("--seed", type=int, default = 34)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--out_dir", default="./results/layer_sweep")
    p.add_argument("--save_suffix", default="sweep")
    return p.parse_args()


def sample_answers(prompt, model, tok, k, max_new_tokens, temperature, top_p):
    """K non-empty sampled answers for one prompt."""
    return [a for a, _, _ in topk_answers(
        prompt, model, tok, k=k, mode="sample", max_new_tokens=max_new_tokens,
        temperature=temperature, top_p=top_p) if a.strip()]


@torch.no_grad()
def embed_all_layers(answers, model, tok, batch_size=8, pooling="mean"):
    """
    Pool every hidden layer in one forward pass.
    Returns [n_layers, n_answers, H], L2-normalized so dot products are cosines.
    """
    per_batch = []
    for i in range(0, len(answers), batch_size):
        enc = tok(answers[i:i + batch_size], return_tensors="pt",
                  padding=True).to(model.device)
        hs = model(**enc, output_hidden_states=True).hidden_states   # tuple of length n_layers with each element [B, T, H]
        mask = enc.attention_mask
        if pooling == "mean":
            m = mask.unsqueeze(-1).float()
            pooled = [(h.float() * m).sum(1) / m.sum(1).clamp(min=1) for h in hs]
        
        else:
            # index of the last real token, honouring the tokenizer's padding side
            if getattr(tok, "padding_side", "right") == "left":
                idx = torch.full((mask.shape[0],), mask.shape[1] - 1, device=mask.device) #L-1 ie the last elem is the last real token
            else:
                idx = mask.sum(1) - 1
            rows = torch.arange(mask.shape[0], device=mask.device)
            pooled = [h.float()[rows, idx] for h in hs] #one h vector per row remains at h[row_i, idx[i]]

        per_batch.append(torch.stack([F.normalize(p, dim=-1) for p in pooled])) #[n_layers,B,H]
    return torch.cat(per_batch, dim=1)                               # [n_layers, N, H]


def mean_pairwise_cos(E):
    """Mean off-diagonal cosine over a set of unit vectors."""
    S = E @ E.T
    n = S.shape[0]
    return ((S.sum() - S.diagonal().sum()) / (n * (n - 1))).item()


def stats_at_layer(E_ref, E_cmp):
    """
    Cosine-dist between two embedding sets, full component dict. `cross_mean` is
    the mean over the Ko x Kr cross-set pairs (15 x 15 = 225 here) --
    E_ref @ E_cmp.T, the same matrix the cosine utility is built from -- and
    `self_orig` / `self_rew` are the off-diagonal within-set means (105 pairs
    each). Those three are what make a layer's cross-set similarity
    interpretable; the utility alone hides them.
    """
    cos, _ = cosine_stats(E_ref, E_cmp) #just use utility
    return cos


def mean_sd(xs):
    #returns mean and standard deviation
    m = sum(xs) / len(xs)
    if len(xs) < 2:
        return m, 0.0
    return m, (sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5



if __name__ == "__main__":

    args = get_args()
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    with open(args.csv, newline="", encoding="utf-8-sig") as f:
        personas = list(csv.DictReader(f))

    print(f"Loading {args.model_path} on {device} ...", flush=True)
    model, tok = load(args.model_path, device)

    n_layers = model.config.num_hidden_layers + 1        # + embedding layer
    layers = list(range(n_layers)) 

    print(f"{len(personas)} personas x K={args.k} answers, pooling={args.pooling}")
    print(f"sweeping {n_layers} hidden_states indices "
          f"(0 = embedding layer, {n_layers - 1} = final block)\n")

    rows = []                      # one rec per persona; each value is a per-layer list
    t0 = time.time()

    for n, row in enumerate(personas, 1):
        orig = row["original"].strip()
        para = row["util_preserving_rewrite"].strip()
        priv = row["priv_preserving_rewrite"].strip()

        # one sample set per prompt; the original's is shared by both comparisons
        answers = {}
        for name, prompt in (("orig", orig), ("para", para), ("priv", priv)):
            answers[name] = sample_answers(prompt, model, tok, args.k,
                                        args.max_new_tokens, args.temperature, args.top_p)
            if len(answers[name]) < 2:
                break

        if any(len(v) < 2 for v in answers.values()):
            print(f"[{n}/{len(personas)}] {row['id']}: too few non-empty answers, skipped")
            continue

        # 4 dimensional: 3(rewrite types) x Embedding matrix [n_layers, n_answers, H]
        emb = {k: embed_all_layers(v, model, tok, args.batch_size, args.pooling)
               for k, v in answers.items()}

        rec = {"id": row["id"], "mean_pairwise_cos_all": [], "udist2_para" :[], "udist2_priv": [], 
               "cross_para": [], "cross_priv": [],
               "self_orig": [], "self_para": [], "self_priv": [], "dist2_para":[], "dist2_priv":[],
               "cos_centroid_para": [], "cos_centroid_priv": []}

        for layer in layers:

            #embed for all layers and calculate the utility metric
            Embedorig, Embedpara, Embedpriv = emb["orig"][layer], emb["para"][layer], emb["priv"][layer]
            statpara = stats_at_layer(Embedorig, Embedpara)
            statpriv = stats_at_layer(Embedorig, Embedpriv)

            rec["udist2_para"].append(statpara["udist2"])
            rec["udist2_priv"].append(statpriv["udist2"])

            # Raw cross-set cosine: the K x K pairing the utility is built on,
            # kept per layer so "how similar are answers to these two prompts"

            rec["cross_para"].append(statpara["cross_mean"])   # orig x para
            rec["cross_priv"].append(statpriv["cross_mean"])   # orig x priv
            
            # Within-set means, the baseline a cross-set mean has to be read
            # against: in a collapsed layer every cosine is ~0.99, cross-set
            # pairs included, so cross alone says nothing about separation.

            rec["self_orig"].append(statpara["self_orig"])     # statpara and statpriv agree here
            rec["self_para"].append(statpara["self_rew"])
            rec["self_priv"].append(statpriv["self_rew"])

            rec["dist2_para"].append(statpara["squared_diff_of_means"])
            rec["dist2_priv"].append(statpriv["squared_diff_of_means"])

            # Centroid cosine: cross with the centroid norms divided out, so
            # only the angle between the two clouds remains (higher = closer).
            rec["cos_centroid_para"].append(statpara["cos_centroid"])
            rec["cos_centroid_priv"].append(statpriv["cos_centroid"])

            # Anisotropy readout over all three sets pooled -- a property of the
            # layer, deliberately not a comparison between prompt types.
            rec["mean_pairwise_cos_all"].append(mean_pairwise_cos(torch.cat([Embedorig, Embedpara, Embedpriv], 0)))

        # rec is complete for THIS persona; rows carries it out of the persona
        # loop, which rebinds rec on the next iteration.
        rows.append(rec)

        print(f"[{n}/{len(personas)}] {rec['id'][:38]:<38} done ({time.time()-t0:.0f}s)", flush=True)

    if not rows:
        raise SystemExit("no personas produced usable samples")

    total = round(time.time() - t0, 1)

    os.makedirs(args.out_dir, exist_ok=True)
    stem = os.path.join(args.out_dir, f"{args.save_suffix}_{args.pooling}")

    with open(stem + ".csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["layer", "mean_pairwise_cos_all",
                    "udist2_para", "udist2_priv",
                    "umean_diff (gap priv-para)", "udist2_cohen_d", "udist2_rank",
                    "self_orig", "self_para", "self_priv",
                    "cross_para", "cross_priv", 
                    "dist2_para", "dist2_priv",
                    "mean_diff (gap priv-para)", "dist2_cohen_d", "dist2_rank",
                    "cos_centroid_para", "cos_centroid_priv",
                    "cos_centroid_diff (gap para-priv)", "cos_centroid_cohen_d", "cos_centroid_rank"])

        for li, layer in enumerate(layers):
            # column mean over personas at this layer
            def col(key, i=li):
                return float(np.mean([row[key][i] for row in rows]))

            # paired per-persona difference: the pairing is what makes the
            # effect size meaningful, so it is taken before averaging
            #UNBIASED NOT INCLUDING THE DIAGONAL
            u_diffs = [row["udist2_priv"][li] - row["udist2_para"][li] for row in rows]
            u_mean_diff, u_sd_d = mean_sd(u_diffs)
            u_cohen_dist_para_priv = u_mean_diff / u_sd_d if u_sd_d > 0 else 0.0
            u_rank = sum(1 for d in u_diffs if d > 0)

            # BIASED INCLUDING THE DIAGONAL
            diffs = [row["dist2_priv"][li] - row["dist2_para"][li] for row in rows]
            mean_diff, sd_d = mean_sd(diffs)
            cohen_dist_para_priv = mean_diff / sd_d if sd_d > 0 else 0.0
            rank = sum(1 for d in diffs if d > 0)

            # CENTROID COSINE: higher = more utility, so the gap is para - priv
            c_diffs = [row["cos_centroid_para"][li] - row["cos_centroid_priv"][li] for row in rows]
            c_mean_diff, c_sd_d = mean_sd(c_diffs)
            c_cohen_para_priv = c_mean_diff / c_sd_d if c_sd_d > 0 else 0.0
            c_rank = sum(1 for d in c_diffs if d > 0)

            w.writerow([layer, f"{col('mean_pairwise_cos_all'):.6f}",
                        f"{col('udist2_para'):.6f}", f"{col('udist2_priv'):.6f}",
                        f"{u_mean_diff:.6f}", f"{u_cohen_dist_para_priv:.4f}", u_rank,
                        f"{col('self_orig'):.6f}", f"{col('self_para'):.6f}",
                        f"{col('self_priv'):.6f}",
                        f"{col('cross_para'):.6f}", f"{col('cross_priv'):.6f}",
                        f"{col('dist2_para'):.6f}", f"{col('dist2_priv'):.6f}",
                        f"{mean_diff:.6f}", f"{cohen_dist_para_priv:.4f}", rank,
                        f"{col('cos_centroid_para'):.6f}", f"{col('cos_centroid_priv'):.6f}",
                        f"{c_mean_diff:.6f}", f"{c_cohen_para_priv:.4f}", c_rank])

    print(f"\nDone in {total}s over {len(rows)} personas. CSV -> {stem}.csv")

  