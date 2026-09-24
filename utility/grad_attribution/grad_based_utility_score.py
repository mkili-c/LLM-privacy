"""
Gradient-based utility scoring of input tokens.

Idea
----
Token IDs are discrete, so we can't differentiate w.r.t. them directly. Instead
we differentiate w.r.t. the *input embeddings* (a continuous relaxation): feed
`inputs_embeds` instead of `input_ids`, define a scalar "utility" objective on
the model's output, backprop, and read the gradient sitting on each input
token's embedding. The size of that gradient tells us how much each input token
influences the objective.


Two attribution signals are reported per token:
  * grad-norm   : ||d(D)/d(embed_t)||_2         -> unsigned importance
  * grad-x-input: <d(D)/d(embed_t), embed_t>     -> signed contribution

"""

import argparse
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_MODEL = "/work/projects/mkilic/models/Qwen2.5-3B-Instruct"


def load(model_path, device, dtype):
    tok = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=dtype).to(device).eval()
    # We only need gradients w.r.t. the inputs, not the weights. Freezing the
    # parameters avoids allocating gradient buffers for ~3B weights.
    model.requires_grad_(False)
    return model, tok


def build_inputs(tok, prompt, target, device):
    """
    Returns (input_ids, prompt_len). The sequence is:
        [ chat-templated prompt ] + [ target answer ]
    `prompt_len` marks where the target begins, so we can (a) score the target's
    likelihood and (b) attribute back to the prompt tokens.
    """
    prompt_text = tok.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False, add_generation_prompt=True,
    )
    prompt_ids = tok(prompt_text, add_special_tokens=False).input_ids
    target_ids = tok(target, add_special_tokens=False).input_ids

    #glue prompt and answer together, prep for teacher forced forward pass

    input_ids = torch.tensor([prompt_ids + target_ids], device=device)
    return input_ids, len(prompt_ids)


def utility_objective(logits, input_ids, prompt_len):
    """
    Scalar utility = total log-likelihood the model assigns to the target tokens
    (the answer that occupies positions [prompt_len, L)).

    Position t's logits predict token t+1, so the target tokens are predicted by
    positions [prompt_len-1, L-1).
    """
    log_probs = F.log_softmax(logits.float(), dim=-1)          # [1, L, V] where 1 is the batch size
    shift_logp = log_probs[:, :-1, :]                          # predict t+1 from t
    shift_labels = input_ids[:, 1:]                            # [1, L-1]
    token_logp = shift_logp.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)  # [1, L-1] pick logprobs of true tokens in target answer

    # keep only positions that predict a target token
    keep = torch.zeros_like(token_logp, dtype=torch.bool)
    keep[:, prompt_len - 1:] = True
    return token_logp[keep].sum()


@torch.enable_grad()
def token_attributions(model, tok, prompt, target, device):
    input_ids, prompt_len = build_inputs(tok, prompt, target, device)

    # Continuous relaxation: embed the tokens and require grad on the embeddings.
    embed_layer = model.get_input_embeddings()
    inputs_embeds = embed_layer(input_ids).detach().clone().requires_grad_(True)

    logits = model(inputs_embeds=inputs_embeds).logits
    obj = utility_objective(logits, input_ids, prompt_len)

    model.zero_grad(set_to_none=True)
    obj.backward()

    grad = inputs_embeds.grad[0].float()        # [L, H]
    embeds = inputs_embeds[0].detach().float()  # [L, H]

    grad_norm = grad.norm(dim=-1)                       # [L] unsigned saliency
    grad_x_input = (grad * embeds).sum(dim=-1)          # [L] signed contribution

    tokens = tok.convert_ids_to_tokens(input_ids[0].tolist())
    return {
        "tokens": tokens,
        "prompt_len": prompt_len,
        "objective": obj.item(),
        "grad_norm": grad_norm.cpu(),
        "grad_x_input": grad_x_input.cpu(),
    }

def chat_generate(model, tok, prompt):

    #model = AutoModelForCausalLM.from_pretrained(
    #    model, torch_dtype=torch.float16
    #).to("cuda").eval()

    text = tok.apply_chat_template(
        [{"role": "user", "content" : prompt}],
        tokenize=False,
        add_generation_prompt = True
    )

    inputs = tok(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens = 128,
            do_sample = False,
            pad_token_id = tok.eos_token_id,
        )
    answer = tok.decode(out[0, inputs.input_ids.shape[1]:], skip_special_tokens = True)

    return answer


def report(attr, top_k=15):
    tokens = attr["tokens"]
    P = attr["prompt_len"]
    gn = attr["grad_norm"]
    gxi = attr["grad_x_input"]

    # Attribute only over the prompt (input) tokens.
    prompt_gn = gn[:P]
    prompt_gxi = gxi[:P]

    share = prompt_gn / (prompt_gn.sum() + 1e-9)   # fraction of total saliency

    print(f"\nTarget log-likelihood (utility objective): {attr['objective']:.3f}")
    print(f"Prompt tokens: {P}  |  showing top {min(top_k, P)} by grad-norm\n")
    print(f"{'rank':>4}  {'idx':>4}  {'token':<16}  {'grad_norm':>10}  {'share':>7}  {'grad*input':>11}")
    print("-" * 64)
    order = torch.argsort(prompt_gxi, descending=True)
    for rank, i in enumerate(order[:top_k].tolist()):
        tk = tokens[i].replace("Ġ", "·").replace("Ċ", "\\n")
        print(f"{rank:>4}  {i:>4}  {tk:<16}  {gn[i]:>10.4f}  {share[i]:>6.1%}  {gxi[i]:>11.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--prompt", default="As I was in celebrations on August the 1 I hurt my eyes. What should I do?")
    ap.add_argument("--target", default=None,
                    help="Answer to attribute; if omitted, the model generates it greedily")
    ap.add_argument("--dtype", choices=["float16", "float32"], default="float16",
                    help="float32 gives cleaner gradients but needs more VRAM")
    ap.add_argument("--top_k", type=int, default=10)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = getattr(torch, args.dtype)
    model, tok = load(args.model, device, dtype)

    if args.target is None:
        answer = chat_generate(model, tok, args.prompt)
    else:
        answer = args.target

    attr = token_attributions(model, tok, args.prompt, answer, device)

    print(f"\nPROMPT : {args.prompt}")
    print(f"TARGET : {answer}")
    report(attr, top_k=args.top_k)


if __name__ == "__main__":
    main()
