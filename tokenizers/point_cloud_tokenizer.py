"""
Implementation of a tokenizer for point cloud data.
"""

# base python imports
from typing import Callable

# deep learning framework
import chex
import jax
import jax.numpy as jnp
from jax import pmap, random, vmap

# choose to enable/disable chex asserts
#chex.disable_asserts()

### Distance Metrics ###

def euclidean_distance(point: jnp.ndarray, point_set: jnp.ndarray) -> jnp.ndarray:
    """
    Calculates the Euclidean distance between a point and set of points.

    Recall: (x_1 - x_2)^2 = x_1^2 + x_2^2 - 2 x_1 x_2
    """
    # repeat sampled point for vectorized calculation
    point_repeated = jnp.tile(point, (point_set.shape[0], 1))
    chex.assert_shape(point_repeated, (point_set.shape[0], point_set.shape[1]))
    
    # calculate terms in distance sum
    sq_term_1 = (point_repeated**2).sum(-1) 
    sq_term_2 = (point_set**2).sum(-1)
    sq_term_3 = (2 * jnp.matmul(point, point_set.T).T)
    chex.assert_equal_shape([sq_term_1, sq_term_2, sq_term_3])    

    # compute the distance
    distances = sq_term_1 + sq_term_2 - sq_term_3
    chex.assert_shape(distances, (point_set.shape[0],))

    return distances

#def euclidean_distance(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
#    """
#    Implements euclidean distance between two vectors.
#    """
#    return jnp.linalg.norm(x - y, axis=-1)

def cosine_distance(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    """
    Implements cosine distance between two vectors.
    """
    return jnp.dot(x, y) / (jnp.linalg.norm(x) * jnp.linalg.norm(y))

### Sampling Methods ###

def farthest_point_sampling(points: jnp.ndarray, 
                            num_samples: int,
                            distance_metric: Callable,
                            random_key) -> jnp.ndarray:
    """
    Implements farthest point sampling.

    Args:
        points: a point cloud.
        num_samples: the number of points to sample.
        distance_metric: a vectorise distance metric to use for sampling.
        random_key: required as we sample using the random module.
    Returns: a downsampled point cloud

    Reference: https://arxiv.org/pdf/2208.08795.pdf
    """
    NUM_POINTS = points.shape[0]

    # initialize infinite distance values
    distances = jnp.ones(NUM_POINTS) * jnp.inf
    sampled_pt_ids = jnp.array([])

    # draw random sample
    sampled_pt_id = random.choice(random_key, NUM_POINTS, replace=False)
    sampled_pt_val = points[sampled_pt_id, :]
    sampled_pt_ids = jnp.append(sampled_pt_ids, sampled_pt_id)

    for itr in range(num_samples - 1):
        # calculate distance between sampled point and all points
        # TODO: consider computational benefit of filtering for unsampled points
        unsampled_pt_ids = jnp.setdiff1d(jnp.arange(NUM_POINTS), sampled_pt_ids)
        distance_to_sampled_pt = distance_metric(
                                                sampled_pt_val, 
                                                points
                                                )

        # update distance value when a smaller distance value is found
        distances = jnp.where(distance_to_sampled_pt < distances, 
                            distance_to_sampled_pt, 
                            distances)

        # set the distance to sampled points to -inf
        distances = jnp.where(jnp.isin(jnp.arange(NUM_POINTS), sampled_pt_ids),
                            -jnp.inf,
                            distances)

        # sample the point with the largest distance value
        sampled_pt_id = jnp.argmax(distances)
        sampled_pt_val = points[sampled_pt_id, :]
        sampled_pt_ids = jnp.append(sampled_pt_ids, sampled_pt_id)

    return sampled_pt_ids


### Grouping Methods ###

def ball_query(points: jnp.ndarray, 
                radius: float,
                distance_metric: Callable) -> jnp.ndarray:
    """
    Implements ball query.
    """
    pass

def knn(points, centroid, k, distance_metric="euclidean"):
    """
    Implements k-nearest neighbors.
    """
    if distance_metric=="euclidean":
        centroids = jnp.tile(centroid, (points.shape[0], 1))
        distance = (centroids**2).sum(-1) + (points**2).sum(-1) - (2 * jnp.matmul(centroid, points.T).T)
    else:
        raise NotImplementedError
    
    return jax.lax.approx_max_k(-distance, k)[1]

    

### Bringing it all together (https://www.youtube.com/watch?v=73lj5qJbrms) ###

def sample_and_group(points: jnp.ndarray, num_samples: int, distance_metric: Callable, random_key):
    """
    Point cloud sampling and grouping.

    source: https://arxiv.org/pdf/2012.09688.pdf
    """
    # sample points
    sampled_points = farthest_point_sampling(
            points, 
            num_samples=num_samples, 
            distance_metric=distance_metric,
            random_key=random_key,
            )

    # group points
    #groups = vmap(knn, in_axes=(None, 0, None), out_axes=0)(sampled_points, points, k=32, distance_metric="euclidean")

    # generate group features through aggregation

    # calculate distance between sampled point and other points in the group

    # concatenate this term with 

    # apply linear batch norm and relu twice and complete with max pooling
    
    #return features
    
    pass


if __name__=="__main__":
    # generate a random key
    random_key = random.PRNGKey(1)

    # test euclidean distance
    #sample_point = jnp.array([1, 2, 3])
    #points = jnp.array([[1, 2, 5], [1, 2, 4], [1, 2, 3], [10,11,12]])
    #print(euclidean_distance(sample_point, points))
    
    # test farthest point sampling
    #num_samples = 2
    #sampled_points = farthest_point_sampling(points, num_samples, euclidean_distance, random_key)
    #print(sampled_points)


    # test knn 
    centroid = jnp.array([1, 2, 3], dtype=jnp.float32)
    points = jnp.array([[1, 2, 5], [1, 2, 4], [1, 2, 3], [10,11,12]], dtype=jnp.float32)
    k = 2
    #print(knn(points, centroid, k, distance_metric="euclidean"))

    # try to vmap knn
    centroids = jnp.array([[1, 2, 3], [10, 11, 12]], dtype=jnp.float32)
    print(centroids.shape)
    sampled_points = vmap(knn, (None, 0, None, None), 0)(points, centroids, 2, "euclidean")
    print(sampled_points)
