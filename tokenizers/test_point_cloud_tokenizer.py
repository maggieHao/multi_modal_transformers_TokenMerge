"""
A basic unittesting script for the given tokenizers.
"""
from absl.testing import parameterised

import chex

from point_cloud_tokenizer import farthest_point_sampling

class FarthestPointSamplingTest(parameterized.TestCase):
    @parameterized.named_parameters(
        points=,
        num_samples=,
        distance_metric=,
    )
    def test(self):
        fn = farthest_point_sampling()
        self.assertEqual()

if __name__ == '__main__':
    absltest.main()
