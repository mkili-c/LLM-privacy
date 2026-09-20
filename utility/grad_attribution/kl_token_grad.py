"""
The llm_attacks-free core of gcg.py: Qwen embedding lookup and the KL-loss
token_gradients used for the one-hot / coordinate-gradient search. Split out
so it can be imported without llm_attacks, which GCGAttackPrompt and friends
in gcg.py depend on but this does not.
"""
import torch


def get_embedding_matrix(model):
    """Qwen2-style models keep the token embedding table at model.model.embed_tokens."""
    return model.model.embed_tokens.weight


def get_embeddings(model, input_ids):
    """Qwen2-style models keep the token embedding table at model.model.embed_tokens."""
    return model.model.embed_tokens(input_ids)


def token_gradients(model, input_ids, ref_ids, input_slice, target_slice, ref_target_slice, loss_slice, ref_loss_slice):

    """
    Computes gradients of the answer-distribution KL divergence with respect
    to the one-hot coordinates of the control tokens.

    The live sequence (input_ids) carries the control tokens being optimized:
    its control slice is one-hot encoded and its embedding is stitched back
    into the sequence so gradients can flow into the one-hot. The reference
    sequence (ref_ids) is the fixed, unmodified prompt teacher-forced with the
    same answer, run under no_grad. The loss is D = KL(P_ref || P_live),
    summed over the answer positions -- see kl_grad_attribution.py for the
    same construction without the one-hot/replacement machinery.

    Parameters
    ----------
    model : Transformer Model
        The transformer model to be used.
    input_ids : torch.Tensor
        The live (control-bearing) sequence, as token ids.
    ref_ids : torch.Tensor
        The fixed reference sequence, as token ids.
    input_slice : slice
        The slice of input_ids holding the control tokens to compute gradients for.
    target_slice : slice
        The slice of input_ids holding the teacher-forced answer tokens.
    ref_target_slice : slice
        The slice of ref_ids holding the same teacher-forced answer tokens.
    loss_slice : slice
        The slice of input_ids' logits that predict target_slice.
    ref_loss_slice : slice
        The slice of ref_ids' logits that predict ref_target_slice.

    Returns
    -------
    torch.Tensor
        The gradients of each token in the input_slice with respect to the KL loss.
    """

    embed_weights = get_embedding_matrix(model)
    one_hot = torch.zeros(
        input_ids[input_slice].shape[0],
        embed_weights.shape[0],
        device=model.device,
        dtype=embed_weights.dtype
    )
    one_hot.scatter_(
        1,
        input_ids[input_slice].unsqueeze(1),
        torch.ones(one_hot.shape[0], 1, device=model.device, dtype=embed_weights.dtype)
    )
    one_hot.requires_grad_()
    input_embeds = (one_hot @ embed_weights).unsqueeze(0)

    # now stitch it together with the rest of the embeddings
    embeds = get_embeddings(model, input_ids.unsqueeze(0)).detach()
    full_embeds = torch.cat(
        [
            embeds[:,:input_slice.start,:],
            input_embeds,
            embeds[:,input_slice.stop:,:]
        ],
        dim=1)

    logits_live = model(inputs_embeds=full_embeds).logits[0, loss_slice, :]

    with torch.no_grad():
        logits_ref = model(input_ids=ref_ids.unsqueeze(0)).logits[0, ref_loss_slice, :]

    # KL(P_ref || P_live), softmax in float32 for numerical stability; the
    # reference branch ran under no_grad, so gradients only flow into logits_live.
    logp = torch.log_softmax(logits_ref.float(), dim=-1)
    logq = torch.log_softmax(logits_live.float(), dim=-1)
    loss = (logp.exp() * (logp - logq)).sum(dim=-1).sum()

    loss.backward()

    return one_hot.grad.clone()
