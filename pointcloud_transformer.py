"""
Implementation of Point Cloud Transformer Architecture
"""

import os
import yaml

from absl.testing import absltest, parameterised

import chex
import jax
import jax.numpy as jnp
import flax
import flax.linen as nn

from attention.offset_attention import OffsetAttention
from tokenizers.point_cloud_tokenizer import SampleAndGroupModule

class PointCloudTransformer(nn.Module):
    """
    Point Cloud Transformer Architecture
    """
    model_config: dict
    
    @nn.compact
    def __call__(self, inputs):
        # start with two linear, batch norm, relu layers
        x = nn.Dense(self.model_config["d_model"])(inputs)
        x = nn.BatchNorm()(x)
        x = nn.relu(x)
        x = nn.Dense(self.model_config["d_model"])(x)
        x = nn.BatchNorm()(x)
        x = nn.relu(x)

        # apply sample and group module
        x = SampleAndGroupModule(
                self.model_config["num_samples"], 
                self.model_config["radius"], 
                self.model_config["num_neighbors"])(x)

        # apply 4 stacked offset attention layers
        attention_ouputs = []
        for itr in range(4):
            x = OffsetAttention(
                    self.model_config["d_model"], 
                    self.model_config["num_heads"], 
                    self.model_config["num_neighbors"])(x)
            if itr != 3:
                attention_ouputs.append(x)

        # concatenate the outputs of the attention layers
        x = jnp.concatenate(attention_ouputs, axis=-1)

        # linear output layer
        x = nn.Dense(self.model_config["d_model"])(x)
        
        # TODO: adapt to suit contact_graspnet prediction task
        
        return x

class TestPointCloudTransformer(absltest.TestCase):
    def setUp(self):
        # read in model config 
        with open("configs/point_cloud_transformer.yaml", "r") as f:
            self.model_config = yaml.load(f, Loader=yaml.FullLoader)


    def TestModelInstatiation(self):
        model = PointCloudTransformer(self.model_config)
        self.assertIsInstance(model, PointCloudTransformer)
    
    def TestForwardPass(self):
        # generate dummy pointcloud data
        batch_size = 4
        num_points = 1024
        num_features = 3
        pointcloud = jnp.ones((batch_size, num_points, num_features))

        # instantiate model
        model = PointCloudTransformer(self.model_config)

        # run forward pass
        output = model(pointcloud)

        # check output shape
        self.assertEqual(
                output.shape, 
                (batch_size, num_points, self.model_config["d_model"])
                )

if __name__ == "__main__":
    absltest.main()
