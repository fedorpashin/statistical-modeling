from statistical_modeling.distributions.continuous_uniform import Distribution, Mean, Variance

from typing import Final

import unittest


class TestContinuousUniform(unittest.TestCase):
    d: Final = Distribution(0, 1)

    def test_distribution(self):
        self.assertEqual(self.d.a, 0)
        self.assertEqual(self.d.b, 1)

    def test_mean(self):
        self.assertAlmostEqual(
            Mean(self.d),
            0.5
        )

    def test_variance(self):
        self.assertAlmostEqual(
            Variance(self.d),
            0.08333,
            places=5
        )
