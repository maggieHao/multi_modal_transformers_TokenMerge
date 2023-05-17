"""
Implementation of Point Cloud Transformer Architecture
"""

import os
import yaml

from absl.testing import absltest, parameterized

import chex
import jax
import jax.numpy as jnp
from jax import vmap
import flax
import flax.linen as nn

from attention.offset_attention import OffsetAttention
from tokenizers.point_cloud_tokenizer import euclidean_distance, SampleAndGroupModule

class PointCloudTransformer(nn.Module):
    """
    Point Cloud Transformer Architecture
    """
    model_config: dict
    
    @nn.compact
    def __call__(self, inputs, is_training: bool):

        # generate random key for sampling
        random_key = jax.random.PRNGKey(0)
        random_key_1, random_key_2 = jax.random.split(random_key)

        ### Linear -> BatchNorm -> ReLU ###
        config = self.model_config["LBR_1"]
        x = nn.DenseGeneral(
                features=config["DenseGeneral"]["features"],
                axis=config["DenseGeneral"]["axis"],
                batch_dims=config["DenseGeneral"]["batch_dims"],
                use_bias=config["DenseGeneral"]["bias"],
                kernel_init=nn.initializers.xavier_uniform(),
                name=config["DenseGeneral"]["name"],
                )(inputs)
        x = nn.BatchNorm(use_running_average=is_training)(x)
        x = nn.relu(x)
        
        ### Linear -> BatchNorm -> ReLU ###
        config = self.model_config["LBR_2"]
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

        ### Sample and Group Points ###
        config = self.model_config["SampleAndGroup1"]
        x = vmap(SampleAndGroupModule(
                config=config,
                fps_distance_metric=euclidean_distance,
                is_training=is_training,
                ), in_axes=(0, None), out_axes=0)(x, random_key_2)

        ### Sample and Group Points ###
        config = self.model_config["SampleAndGroup2"]
        x = vmap(SampleAndGroupModule(
                config=config,
                fps_distance_metric=euclidean_distance,
                is_training=is_training,
                ), in_axes=(0, None), out_axes=0)(x, random_key_2)

        ### Offset Attention Layers ###
        attention_ouputs = []
        for itr in range(4):
            config = self.model_config["OffsetAttention{}".format(itr+1)]
            x = OffsetAttention(
                    config=config,
                    )(x, is_training=True)
            attention_ouputs.append(x)
        
        ### Concatenate Attention Outputs ###
        x = jnp.concatenate(attention_ouputs, axis=-1)
        
        # TODO: adapt to suit contact_graspnet prediction task
        
        return x

class TestPointCloudTransformer(absltest.TestCase):
    def setUp(self):
        # read in model config 
        with open("model_configs/point_cloud_transformer.yaml", "r") as f:
            self.model_config = yaml.load(f, Loader=yaml.FullLoader)


    def test_model_instatiation(self):
        model = PointCloudTransformer(self.model_config)
        self.assertIsInstance(model, PointCloudTransformer)
    
    def test_forward_pass(self):
        # generate dummy pointcloud data
        BATCH_SIZE = 4
        NUM_POINTS = 3000
        NUM_FEATURES = 3
        IS_TRAINING = False

        pointcloud = jnp.ones((BATCH_SIZE, NUM_POINTS, NUM_FEATURES))

        # instantiate model
        model = PointCloudTransformer(self.model_config)
        params = model.init(jax.random.PRNGKey(0), pointcloud, IS_TRAINING)
        
        # run forward pass
        output = model.apply(params, pointcloud, IS_TRAINING)

        # check output shape
        self.assertEqual(
                output.shape, 
                (batch_size, 256, 128)
                )

if __name__ == "__main__":
    absltest.main()

    # profiling with snakeviz

    # read in model config 
    #with open("model_configs/point_cloud_transformer.yaml", "r") as f:
    #    model_config = yaml.load(f, Loader=yaml.FullLoader)

    # generate dummy pointcloud data
    #BATCH_SIZE = 4
    #NUM_POINTS = 150
    #NUM_FEATURES = 3
    #IS_TRAINING = False

    #pointcloud = jnp.ones((BATCH_SIZE, NUM_POINTS, NUM_FEATURES))

    # instantiate model
    #model = PointCloudTransformer(model_config)
    #params = model.init(jax.random.PRNGKey(0), pointcloud, IS_TRAINING)
        
    # run forward pass
    #output = model.apply(params, pointcloud, IS_TRAINING, mutable=["batch_stats", "batch_mean", "batch_var"])

