"""
Methods for compressing tokens. 
"""

from typing import Tuple, Callable

import math
import jax.numpy as jnp


# token pruning methods
def compute_top_k_tokens(importance_scores, tokenset_idx, tokenset_k):
    """Compute top-k tokens based on importance scores."""
    
    def top_k(importance_scores, k, start_idx):
        """Compute top-k tokens based on importance scores for a tokenset from sequence."""
        
        # compute indices of top-k
        _, idx = jax.lax.top_k(importance_scores, k)

        # adjust top-k indices to account for token set starting index in sequence
        idx += start_idx
        
        return idx

    # get top-k tokens for each modality
    modality_subsets = jax.vmap(jax.lax.dynamic_slice_in_dim, (None, 0, 0, None))(importance_scores, tokenset_idx[0], tokenset_idx[1], axis=-1)
    idx = jax.vmap(top_k, in_axes=(0, 0))(modality_subsets, tokenset_merge_k)
    idx = e.rearrange(idx, 'num_modalities num_tokens idx -> (num_modalities num_tokens) idx')

    return idx



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

