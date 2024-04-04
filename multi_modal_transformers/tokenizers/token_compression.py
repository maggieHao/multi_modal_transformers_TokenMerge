"""
Methods for compressing tokens. 
"""

from functools import partial
from typing import Tuple, Callable

import math
import jax
import jax.numpy as jnp
import einops as e

# token pruning methods
@partial(jax.jit, static_argnums=(2,3))
def compute_top_k_tokens(embeddings, importance_scores, tokenset_idx, tokenset_k):
    """
    Compute top-k tokens per modality based on importance scores.

    Args:
        embeddings: token embeddings
        importance_scores: token importance scores (mean across attention heads + keys)
        tokenset_idx: (start_idx, num_tokens) for each modality in the sequence
        tokenset_k: the number of tokens to compress to for each modality
    """
    
    def top_k(importance_scores, k, seq_start_idx):
        """Compute top-k tokens based on importance scores for a tokenset from sequence."""

        # compute indices of top-k
        _, idx = jax.lax.top_k(importance_scores, k)

        # adjust top-k indices to account for token set starting index in sequence
        idx += seq_start_idx
        
        return idx

    # get top-k tokens for each tokenset
    ids = []
    for k, slice_id in zip(tokenset_k, tokenset_idx):
        subset = jax.lax.dynamic_slice_in_dim(importance_scores, slice_id[0], slice_id[1], axis=0)
        idx = top_k(subset, k, slice_id[0])
        jax.debug.print("subset shape: {}", subset.shape)
        jax.debug.print("idx shape: {}", idx.shape)
        ids.append(idx)

    ids = jnp.concatenate(ids, axis=-1)
    #jax.debug.print("ids shape: {}", ids.shape)

    # finally assemble compressed sequence
    compressed_embeddings = jnp.take(embeddings, ids, axis=0)    

    return compressed_embeddings



# token merging methods
def do_nothing(x, mode=None):
    return x

def bipartite_soft_matching(
    metric: jnp.array,
    r: int,
    class_token: bool = False,
    distill_token: bool = False,
) -> Tuple[Callable, Callable]:
    protected = 0
    if class_token:
        protected += 1
    if distill_token:
        protected += 1

    t = metric.shape[1]
    r = min(r, (t - protected) // 2)

    if r <= 0:
        return do_nothing, do_nothing

    metric = metric / jnp.linalg.norm(metric, axis=-1, keepdims=True)
    a, b = metric[..., ::2, :], metric[..., 1::2, :]

    scores = jnp.matmul(a, jnp.swapaxes(b, -1, -2))

    if class_token:
        scores = scores.at[..., 0, :].set(-jnp.inf)
    if distill_token:
        scores = scores.at[..., :, 0].set(-jnp.inf)

    node_max = scores.max(axis=-1)
    node_idx = scores.argmax(axis=-1)
    edge_idx = jnp.argsort(node_max, axis=-1)[:,::-1][..., None]

    unm_idx = edge_idx[..., r:, :]  # Unmerged Tokens
    src_idx = edge_idx[..., :r, :]  # Merged Tokens
    dst_idx = jnp.take_along_axis(node_idx[..., None], src_idx, axis=-2)

    def merge(x: jnp.array, mode="sum") -> jnp.array:
        n, t, c = x.shape
        t1 = t // 2

        # Simulating gather operation in JAX
        unm = jnp.take_along_axis(x[..., ::2, :], unm_idx, axis=1)
        src = jnp.take_along_axis(x[..., ::2, :], src_idx, axis=1)
        dst = jnp.asarray(x[..., 1::2, :])

        if mode == "sum":
            for i in range(dst_idx.shape[1]):
                dst = dst.at[jnp.arange(n), dst_idx[:, i, 0], :].add(src[:, i, :])

        if distill_token:
            # Concatenate considering distillation token
            result = jnp.concatenate([unm[:, :1], dst[:, :1], unm[:, 1:], dst[:, 1:]], axis=1)
        else:
            # Simple concatenation if no distillation token
            result = jnp.concatenate([unm, dst], axis=1)
        return result


    return merge

def merge_wavg(
    merge: Callable, x: jnp.ndarray, size: jnp.ndarray = None
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Applies the merge function by taking a weighted average based on token size.
    Returns the merged tensor and the new token sizes.
    """
    if size is None:
        size = jnp.ones_like(x[..., 0, None])

    # Assuming merge can take JAX arrays and has a compatible signature.
    x = merge(x * size, mode="sum")
    size = merge(size, mode="sum")
    x = x / size

    return x, size


if __name__=="__main__":
    from token_sequencer import TokenSequence
    import jax
    from jax import random

    batch_size = 10
    seq_len = 40
    embed_dim = 128

    # dummy token sequence 
    multi_modal_seq = "[TaskDescriptionPrefix{20}] [Image{10};Readout{10}]"
    multi_modal_compressed_seq = "[TaskDescriptionPrefix{2}] [Image{2};Readout{0}]"
    seq = TokenSequence(multi_modal_seq, multi_modal_compressed_seq)
    compressed_seq = seq.generate_layer_token_sequence(layer=1)
    top_k = tuple([tokenset.num_tokens for tokenset in compressed_seq])
    jax.debug.print("top_k: {}", top_k)

    # dummy embeddings
    task_description_embeddings = jnp.ones((batch_size, 20, embed_dim))
    image_embeddings = jnp.ones((batch_size, 10, embed_dim))
    readout_embeddings = jnp.ones((batch_size, 10, embed_dim))
    input_embeddings = jnp.concatenate([task_description_embeddings, image_embeddings, readout_embeddings], axis=1)

    # dummy importance scores
    key = random.PRNGKey(0)
    random_importance_scores = random.normal(key, (batch_size, seq_len))
    
    # generate token slices for modality
    slices = seq.tokenset_slices
    jax.debug.print("slices: {}", slices)

    # compress token sequence

    jax.debug.print("input_embeddings shape: {}", input_embeddings.shape)
    #output_embeddings = jax.vmap(compute_top_k_tokens, in_axes=(0, 1, None, None), out_axes=0)(input_embeddings, random_importance_scores, slices, top_k)
    #output_embeddings = compute_top_k_tokens(input_embeddings[0], random_importance_scores[0], slices, top_k)
    output_embeddings = jax.vmap(compute_top_k_tokens, in_axes=(0, 0, None, None), out_axes=0)(input_embeddings, random_importance_scores, slices, top_k)
    jax.debug.print("output_embeddings shape: {}", output_embeddings.shape)

