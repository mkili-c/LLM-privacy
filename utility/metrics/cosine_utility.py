"""
Cosine-similarity-based utility score for a privacy rewrite.

We have two prompts Original (Orig) and Rewritten (Rew). The samples from
those answers follow a probability distribution: P(answer|prompt). We sample
from both. We embed the answers (using the average of hidden states at a chosen 
layer for each token in an answer) Take means of the two clouds of samples and 
calculate their difference

    ||mu_orig-mu_rew||^2

Below it is done as mu_orig T mu_orig + mu_rew T mu_rew - 2cross

    cross   = mean_{i,j}    cos(e_i, f_j)      e_i: embeddings of answers to T_orig
    self_o  = mean_{i != i'} cos(e_i, e_i')    f_j: embeddings of answers to T_rew
    self_r  = mean_{j != j'} cos(f_j, f_j')
    udist^2   = self_o + self_r - 2 * cross      (0 = identical distributions)
    dist^2 = ||mean_e - mean_f||^2

Note that cross is NOT itself a cosine: by bilinearity

    cross = <mean_e, mean_f> = ||mean_e|| ||mean_f|| cos(mean_e, mean_f)

and the centroids of unit vectors are not unit vectors: ||mean_e||^2 ~ self_o < 1,
shrinking as the answer cloud spreads. So cross ~ self_o (not 1) for identical
prompts. Dividing out the centroid norms gives the angle alone:

    cos_centroid = cross / sqrt(self_o * self_r)       (1 = same direction)
    rel_udist2   = udist^2 / sqrt(self_o * self_r)
                 = 2 (1 - cos_centroid) + (r - 1)^2 / r,   r = ||mean_e|| / ||mean_f||

i.e. an angle term plus a concentration-mismatch penalty. Both use the
off-diagonal self terms, so they are the centroid cosine / distance with the
1/n diagonal bias removed. The plain squared difference of sample means is
biased upwards by exactly (1 - self_o)/n + (1 - self_r)/m and is kept only
as a diagnostic.

Compared to the likelihood metric (utility_scoring.py) this one is largely
phrasing-invariant. Tempered sampling would be admissible here (there is no
likelihood mismatch to bias, as long as both sides use identical settings), but
the defaults are plain sampling (temp 1.0 / top_p 1.0) to match
utility_scoring.py, so both metrics compare the same answer distribution.
"""
import argparse
import torch
import torch.nn.functional as F

from topk_sample import load, topk_answers

DEFAULT_MODEL = "/work/projects/mkilic/models/Qwen2.5-3B-Instruct"


def get_args():
    p = argparse.ArgumentParser(
        description="Utility of a rewritten prompt via cosine distance "
                    "between the two sets of sampled answers")
    p.add_argument("--model_path", default=DEFAULT_MODEL)
    p.add_argument("--original", default="I am a 45-year-old male doctor living in New York. "
                                         "What are the best ways to manage stress?")
    p.add_argument("--rewritten", default="What are some good general strategies for "
                                          "managing everyday stress?")
    p.add_argument("--k", type=int, default=5, help="samples per prompt (needs >= 2)")
    p.add_argument("--max_new_tokens", type=int, default=64)
    p.add_argument("--temperature", type=float, default=1.0,
                   help="plain sampling matches utility_scoring.py; tempering is "
                        "admissible for this metric if both sides stay identical")
    p.add_argument("--top_p", type=float, default=1.0,
                   help="plain sampling matches utility_scoring.py; tempering is "
                        "admissible for this metric if both sides stay identical")
    p.add_argument("--seed", type=int, default=23)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--show_samples", action="store_true",
                   help="print every sampled answer with its cross-set similarities")
    p.add_argument("--layer", type=int, default=-1)
    p.add_argument("--layers", type=int, nargs="+", default=None,
                   help="if given, report the metrics for each of these layers "
                        "individually plus their sum (e.g. --layers -1 -2 -3)")
    return p.parse_args()


@torch.no_grad()
def embed_answers(answers, model, tok, batch_size=8, layer=-1):
    """
    Embed each answer text standalone: mean-pool the model's final hidden states
    over non-pad tokens, in float32, L2-normalized. Returns [N, H].
    """
    embs = []
    for i in range(0, len(answers), batch_size):
        enc = tok(answers[i:i + batch_size], return_tensors="pt",
                  padding=True).to(model.device)
        hidden = model(**enc, output_hidden_states=True).hidden_states[layer].float() 
        #final hidden states B batch L length in number of tokens hidden: [B,L,H]
        mask = enc.attention_mask.unsqueeze(-1).float() #match dimensions
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1) # real tokens / no of real tokens
        embs.append(F.normalize(pooled, dim=-1)) #embs [B,H]
    return torch.cat(embs, dim=0)


@torch.no_grad()
def embed_answers_layers(answers, model, tok, batch_size=8, layers=(-1,)):
    """
    Like embed_answers, but returns one embedding set per requested layer:
    {layer: [N, H]}. One forward pass per batch for all layers.
    """
    embs = {l: [] for l in layers}
    for i in range(0, len(answers), batch_size):
        enc = tok(answers[i:i + batch_size], return_tensors="pt",
                  padding=True).to(model.device)
        hs = model(**enc, output_hidden_states=True).hidden_states
        mask = enc.attention_mask.unsqueeze(-1).float()
        for l in layers:
            hidden = hs[l].float()
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            embs[l].append(F.normalize(pooled, dim=-1))
    return {l: torch.cat(v, dim=0) for l, v in embs.items()}


def cosine_stats(E_orig, E_rew):
    """
    Cosine-dist between the two embedding sets plus its components.
    Self-similarities exclude the diagonal (an answer is trivially similar to
    itself and would inflate the ceiling). Returns (stats dict, cross matrix).
    """
    C = E_orig @ E_rew.T                       # i,j = [(ho_i)T * hr_j] equal to cosine since normalized
    cross = C.mean().item()                    # mean of pairwise cosines == <centroid_o, centroid_r> (bilinearity); NOT itself a cosine

    def self_mean(E):
        #take only off diag because including diagonal elements would 
        #inflate the result and make it biased especially for such small size
        n = E.shape[0]
        S = E @ E.T
        return ((S.sum() - S.diagonal().sum()) / (n * (n - 1))).item() 
     

    self_o = self_mean(E_orig)
    self_r = self_mean(E_rew)
    udist2 = self_o + self_r - 2.0 * cross

    # sqrt(self_o * self_r) estimates ||centroid_o|| * ||centroid_r|| with the
    # 1/n diagonal bias removed. Dividing it out leaves only the angle between
    # the centroids (cos_centroid) and makes rel_udist2 symmetric in orig/rew:
    #   rel_udist2 = 2 (1 - cos_centroid) + (r - 1)^2 / r,  r = ratio of centroid norms
    # The clamp guards the degenerate case of a near-zero/negative U-statistic
    # (only possible for extremely diffuse clouds at tiny k).
    norm_prod = max(self_o * self_r, 1e-12) ** 0.5
    cos_centroid = cross / norm_prod
    rel_udist2 = udist2 / norm_prod

    mu_orig = E_orig.mean(dim=0)
    mu_rew = E_rew.mean(dim=0)
    d = mu_orig - mu_rew
    squared_diff_of_means = (d @ d).item() 
    mean_orig = (mu_orig@mu_orig).item()
    rel_squared_diff_of_means = squared_diff_of_means / mean_orig


    return {
        "udist2": udist2,
        "rel_udist2" : rel_udist2,
        "cos_centroid": cos_centroid,
        "cross_mean": cross,
        "self_orig": self_o,
        "self_rew": self_r,
        "squared_diff_of_means": squared_diff_of_means,
        "rel_squared_diff_of_means" : rel_squared_diff_of_means
    }, C



def cosine_utility_score(original, rewritten, model, tok, k=5, max_new_tokens=64,
                         temperature=1.0, top_p=1.0, batch_size=8, layer = -1):
    """
    Distance of rewritten answers' mean to original's answers' mean. 
    Returns a dict with the score, its components, and
    per-sample best-match diagnostics.
    """
    a_orig = [a for a, _, _ in topk_answers(
        original, model, tok, k=k, mode="sample", max_new_tokens=max_new_tokens,
        temperature=temperature, top_p=top_p) if a.strip()]
    a_rew = [a for a, _, _ in topk_answers(
        rewritten, model, tok, k=k, mode="sample", max_new_tokens=max_new_tokens,
        temperature=temperature, top_p=top_p) if a.strip()]

    if len(a_orig) < 2 or len(a_rew) < 2:
        raise ValueError(
            f"Need >= 2 non-empty samples per side for self-similarity "
            f"(got {len(a_orig)} orig / {len(a_rew)} rew); raise --k or --max_new_tokens")

    E_orig = embed_answers(a_orig, model, tok, batch_size,layer)
    E_rew = embed_answers(a_rew, model, tok, batch_size,layer)
    stats, C = cosine_stats(E_orig, E_rew)

    def diagnostics(answers, others, M):
        best = M.max(dim=1)
        return [{
            "answer": a,
            "mean_cross": M[i].mean().item(),
            "best_cross": best.values[i].item(),
            "best_match": others[best.indices[i]],
        } for i, a in enumerate(answers)]

    stats["samples_orig"] = diagnostics(a_orig, a_rew, C)
    stats["samples_rew"] = diagnostics(a_rew, a_orig, C.T)
    return stats


def cosine_utility_score_layers(original, rewritten, model, tok, k=5, max_new_tokens=64,
                                temperature=1.0, top_p=1.0, batch_size=8, layers=(-1,)):
    """
    Same metrics as cosine_utility_score, but computed separately for each layer
    in `layers` (one set of samples shared by all layers).
    Returns {"per_layer": {layer: stats}, "sum": stats summed over the layers}.
    """
    a_orig = [a for a, _, _ in topk_answers(
        original, model, tok, k=k, mode="sample", max_new_tokens=max_new_tokens,
        temperature=temperature, top_p=top_p) if a.strip()]
    a_rew = [a for a, _, _ in topk_answers(
        rewritten, model, tok, k=k, mode="sample", max_new_tokens=max_new_tokens,
        temperature=temperature, top_p=top_p) if a.strip()]

    if len(a_orig) < 2 or len(a_rew) < 2:
        raise ValueError(
            f"Need >= 2 non-empty samples per side for self-similarity "
            f"(got {len(a_orig)} orig / {len(a_rew)} rew); raise --k or --max_new_tokens")

    Eo = embed_answers_layers(a_orig, model, tok, batch_size, layers)
    Er = embed_answers_layers(a_rew, model, tok, batch_size, layers)

    per_layer = {l: cosine_stats(Eo[l], Er[l])[0] for l in layers}
    total = {key: sum(s[key] for s in per_layer.values())
             for key in next(iter(per_layer.values()))}
    return {"per_layer": per_layer, "sum": total,
            "answers_orig": a_orig, "answers_rew": a_rew}


if __name__ == "__main__":
    args = get_args()
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading {args.model_path} on {device} ...")
    model, tok = load(args.model_path, device)

    if args.layers:
        out = cosine_utility_score_layers(
            args.original, args.rewritten, model, tok, k=args.k,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature, top_p=args.top_p,
            batch_size=args.batch_size, layers=tuple(args.layers)
        )
        print(f"\nORIGINAL : {args.original}")
        print(f"REWRITTEN: {args.rewritten}\n")
        for l, s in out["per_layer"].items():
            print(f"[layer {l:>4}] udist2={s['udist2']:.4f} rel_udist2={s['rel_udist2']:.4f} "
                  f"cos_centroid={s['cos_centroid']:.4f} "
                  f"sq_diff_means={s['squared_diff_of_means']:.4f} "
                  f"rel_sq_diff_means={s['rel_squared_diff_of_means']:.4f}")
        t = out["sum"]
        print(f"[sum over {list(out['per_layer'])}] udist2={t['udist2']:.4f} "
              f"rel_udist2={t['rel_udist2']:.4f} "
              f"sq_diff_means={t['squared_diff_of_means']:.4f} "
              f"rel_sq_diff_means={t['rel_squared_diff_of_means']:.4f}")
        raise SystemExit

    res = cosine_utility_score(
        args.original, args.rewritten, model, tok, k=args.k,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature, top_p=args.top_p,
        batch_size=args.batch_size, layer = args.layer
    )

    print(f"\nORIGINAL : {args.original}")
    print(f"REWRITTEN: {args.rewritten}\n")

    if args.show_samples:
        for side, dets in (("orig", res["samples_orig"]), ("rew", res["samples_rew"])):
            for i, d in enumerate(dets):
                print(f"[{side} {i}] mean_cross={d['mean_cross']:.3f}  "
                      f"best_cross={d['best_cross']:.3f}")
                print(f"  {d['answer']}\n")

    print(f"\n---------layer-----------          {args.layer}\n")
    print(f"self-similarity   orig={res['self_orig']:.4f}  rew={res['self_rew']:.4f}   (within-set ceilings)")
    print(f"cross-similarity  {res['cross_mean']:.4f}   (= <centroid_o, centroid_r>, ~self for identical prompts)")
    print(f"cos_centroid      {res['cos_centroid']:.4f}   (angle between centroids only; 1 = same direction)")
    print(f"Squared difference of means: {res['squared_diff_of_means']:.4f} Relative squared diff of means: {res['rel_squared_diff_of_means']:.4f}")
    print(f"Unbiased distance^2 (cosine)    {res['udist2']:.4f} Relative unbiased dist2 (cosine) {res['rel_udist2']:.4f}")


