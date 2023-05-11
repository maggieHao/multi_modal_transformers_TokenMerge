"""
Implementation of a tokenizer for point cloud data.
"""

# base python imports
from typing import Callable

# deep learning framework
import jax.numpy as jnp
from jax import pmap, random, vmap


### Distance Metrics ###

def euclidean_distance(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    """
    Implements euclidean distance between two vectors.
    """
    return jnp.linalg.norm(x - y, axis=-1)

def cosine_distance(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    """
    Implements cosine distance between two vectors.
    """
    return jnp.dot(x, y) / (jnp.linalg.norm(x) * jnp.linalg.norm(y))

### Sampling Methods ###

def farthest_point_sampling(points: jnp.ndarray, 
                            num_samples: int,
                            distance_metric: Callable) -> jnp.ndarray:
    """
    Implements farthest point sampling.

    Args:
        points: a point cloud.
        num_samples: the number of points to sample.
        distance_metric: a vectorise distance metric to use for sampling.

    Returns: a downsampled point cloud

    Reference: https://arxiv.org/pdf/2208.08795.pdf
    """
    NUM_POINTS = points.shape[0]

    # initialize infinite distance values
    distances = jnp.ones(NUM_POINTS) * jnp.inf
    sampled_pt_ids = []

    # draw random sample
    # TODO: check how to pass PRNGKey
    sampled_pt_id = random.choice(random.PRNGKey(0), NUM_POINTS, replace=False)
    sampled_pt_val = points[sampled_pt_id, :]

    # add sampled point to list of sampled points
    sampled_pt_ids.append(sampled_pt_id)

    for itr in range(num_samples - 1):
        # calculate distance between sampled point and all (unsampled) points
        # TODO: ensure this is vectorized
        distance_to_sampled_pt = distance_metric(
                                                sampled_pt_val, 
                                                points[~sampled_pt_ids,:]
                                                )

        # update distance value when a smaller distance value is found
        distances = jnp.where(distance_to_sampled_pt < distances, 
                            distance_to_sampled_pt, 
                            distances)
        
        # sample the point with the largest distance value
        sampled_pt_id = jnp.argmax(distances)
        sampled_pt_val = points[sampled_pt, :]
        sampled_pt_ids.append(sampled_pt_id)

    return sampled_pt_ids


### Grouping Methods ###

def ball_query(points: jnp.ndarray, 
                radius: float,
                distance_metric: Callable) -> jnp.ndarray:
    """
    Implements ball query.
    """
    pass

def knn(points, centroids, k, distance_metric="euclidean"):
    """
    Implements k-nearest neighbors.
    """
    if distance_metric=="euclidean":
        distance = (centroids**2) + (points**2) - (2 * centroids * points.T)
    else:
        raise NotImplementedError

    return jax.lax.top_k(-distance, k)

    

def aggregate_features():
    """
    Implements feature aggregation over a local region of point cloud.
    """
    pass


if __name__=="__main__":
    pass
