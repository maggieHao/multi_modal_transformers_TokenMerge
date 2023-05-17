"""
Implementation of a tokenizer for point cloud data.
"""

from typing import Callable

import chex
import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
from jax import pmap, random, vmap

# choose to enable/disable chex asserts
chex.disable_asserts()

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
    sampled_pt_ids = jnp.array([], dtype=jnp.int32) # concerned about efficient of this variable

    # draw random sample
    sampled_pt_id = random.choice(random_key, NUM_POINTS, replace=False)
    sampled_pt_val = points[sampled_pt_id, :]
    sampled_pt_ids = jnp.append(sampled_pt_ids, sampled_pt_id)

    for itr in range(num_samples - 1):
        print(itr)
        # calculate distance between sampled point and all points
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

    

### Creating the Sample and Group module ###
class SampleAndGroupModule(nn.Module):
    """
    Module to downsample and group point cloud data.
    """
    config: dict
    fps_distance_metric: Callable # fps = farthest point sampling
    is_training: bool
    
    @nn.compact
    def __call__(self, points, random_key):
        # unpack module parameters
        num_samples = self.config["num_samples"]
        num_neighbours_knn = self.config["num_neighbours_knn"]
        knn_distance_metric = self.config["knn_distance_metric"]
        embed_dim = self.config["embed_dim"]
        fps_distance_metric = self.fps_distance_metric
        is_training = self.is_training
        points_xyz = points[:, :3] # for sampling and grouping

        # sample points
        sampled_points = farthest_point_sampling(
            points_xyz, 
            num_samples=num_samples, 
            distance_metric=fps_distance_metric,
            random_key=random_key,
            )

        # group points
        centroids = jnp.take(points_xyz, sampled_points, axis=0)

        groups = vmap(
                knn, 
                in_axes=(None, 0, None, None), 
                out_axes=0)(points_xyz, centroids, num_neighbours_knn, knn_distance_metric)

        # aggregate features from groups
        def aggregate(points, group, centroid):
            # repeat centroid for each point in the group 
            centroid_repeated = jnp.tile(centroid, (group.shape[0], 1))

            # calculate distance between each point in the group and the centroid
            cluster_features = jnp.take(points, group, axis=0)
            delta = cluster_features - centroid_repeated

            # concatenate delta with cluster features
            cluster_features = jnp.concatenate((delta, cluster_features), axis=-1)
            return cluster_features
        
        features = vmap(aggregate, (None, 0, 0))(points, groups, sampled_points)

        # Linear -> BatchNorm -> ReLU
        config = self.config["LBR_1"]
        x = nn.DenseGeneral(
                features=config["DenseGeneral"]["features"],
                axis=config["DenseGeneral"]["axis"],
                batch_dims=config["DenseGeneral"]["batch_dims"],
                use_bias=config["DenseGeneral"]["bias"],
                kernel_init=nn.initializers.xavier_uniform(),
                name=config["DenseGeneral"]["name"],
                )(features)
        x = nn.BatchNorm(use_running_average=is_training)(x)
        x = nn.relu(x)

        # Linear -> BatchNorm -> ReLU
        config = self.config["LBR_2"]
        x = nn.DenseGeneral(
                features=config["DenseGeneral"]["features"],
                axis=config["DenseGeneral"]["axis"],
                batch_dims=config["DenseGeneral"]["batch_dims"],
                use_bias=config["DenseGeneral"]["bias"],
                kernel_init=nn.initializers.xavier_uniform(),
                name=config["DenseGeneral"]["name"],
                )(x)
        x = nn.BatchNorm(use_running_average=is_training)(x)
        x = nn.relu(x)

        return x

