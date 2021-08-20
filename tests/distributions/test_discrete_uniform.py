from statistical_modeling.distributions.discrete_uniform import Distribution, Mean, Variance

from typing import Final

import unittest


class TestDiscreteUniform(unittest.TestCase):
    d: Final = Distribution(1, 100)

    def test_distribution(self):
        self.assertEqual(self.d.a, 1)
        self.assertEqual(self.d.b, 100)

    def test_mean(self):
        self.assertAlmostEqual(
            Mean(self.d),
            50.5
        )

    def test_variance(self):
        self.assertAlmostEqual(
            Variance(self.d),
            833.25
        )
